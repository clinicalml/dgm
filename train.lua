require "nn"
require "xlua"
require "nngraph"
require "utils"
require "cunn"
require "optim"
require "GaussianReparam"
disp = require "display"

torch.manualSeed(1)

---------------- Load Data ---------------
data=loadBinarizedMNIST(true)

---------------- Model Params. -----------
local dim_input = 784
local dim_hidden= 400
local dim_stochastic = 100
local nonlinearity   = nn.ReLU

--------- Recognition. Network -----------
local var_inp = nn.Identity()()
local dropped_inp = nn.Dropout()(var_inp)
local q_1 = nonlinearity()(nn.Linear(dim_input,dim_hidden)(dropped_inp))
local q_hid_1 = nonlinearity()(nn.Linear(dim_hidden,dim_hidden)(q_1))
local q_hid_2 = nonlinearity()(nn.Linear(dim_hidden,dim_hidden)(q_hid_1))
local mu  = nn.Linear(dim_hidden,dim_stochastic)(q_hid_2)
local logsigma  = nn.Linear(dim_hidden,dim_stochastic)(q_hid_2)
local reparam   = nn.GaussianReparam(dim_stochastic)
print (reparam.KL)
local z  = reparam({mu,logsigma})
local var_model = nn.gModule({var_inp},{z})

--------- Generative Network -------------
local gen_inp = nn.Identity()()
local hid1 = nonlinearity()(nn.Linear(dim_stochastic,dim_hidden)(gen_inp))
local hid2 = nonlinearity()(nn.Linear(dim_hidden,dim_hidden)(hid1))
local hid3 = nonlinearity()(nn.Linear(dim_hidden,dim_hidden)(hid2))
local hid4 = nonlinearity()(nn.Linear(dim_hidden,dim_hidden)(hid3))
local reconstr = nn.Sigmoid()(nn.Linear(dim_hidden,dim_input)(hid4))
local gen_model = nn.gModule({gen_inp},{reconstr})

----- Combining Models into Single MLP----
local inp = nn.Identity()()
mlp = nn.gModule({inp},{gen_model(var_model(inp))})
crit= nn.BCECriterion()
crit.sizeAverage = false

--------- Setup for Training/Viz.---------
img_format,format = setupDisplay()
setupFolder('./save')
mlp:cuda()
crit:cuda()
data.train_x = data.train_x:cuda()
data.test_x  = data.test_x:cuda()
parameters, gradients = mlp:getParameters()
config = {
    learningRate = 0.0001,
}
batchSize = 100
state = {}

---------  Sample from Gen. Model---------
function getsamples()
	local p = gen_model:forward(torch.randn(batchSize,dim_stochastic):typeAs(data.train_x))
	local s = torch.gt(p:double(),0.5)
	local samples = {}
	local mean_prob = {}
	for i=1,batchSize do 
		samples[i] = s[i]:float():reshape(28,28)
		mean_prob[i] = p[i]:float():reshape(28,28)
	end
	return samples,mean_prob
end
---------  Evaluate Likelihood   ---------
function eval(dataset)
	mlp:evaluate()
	local probs = mlp:forward(dataset)
	local nll   = crit:forward(probs,dataset)
	mlp:training()
	return (nll+reparam.KL)/dataset:size(1),probs
end
--------- Stitch Images Together ---------
function stitch(probs,batch)
	local imgs = {}
	for i = 1,batchSize do 
		imgs[i] = torch.cat(probs[i]:float():reshape(28,28),batch[i]:float():reshape(28,28),2)
	end
	return imgs
end

-------------- Training Loop -------------
for epoch =1,455 do 
    local upperbound = 0
	local trainnll = 0
    local time = sys.clock()
    local shuffle = torch.randperm(data.train_x:size(1))
	if epoch==100 then config.learningRate = 5e-5 end
	if epoch > 30 then config.learningRate = math.max(config.learningRate / 1.000005, 0.000001) end
    --Make sure batches are always batchSize
    local N = data.train_x:size(1) - (data.train_x:size(1) % batchSize)
    local N_test = data.test_x:size(1) - (data.test_x:size(1) % batchSize)
	local probs 
    local batch = torch.Tensor(batchSize,data.train_x:size(2)):typeAs(data.train_x)
	-- Pass through data
    for i = 1, N, batchSize do
        xlua.progress(i+batchSize-1, data.train_x:size(1))

        local k = 1
        for j = i,i+batchSize-1 do
            batch[k] = data.train_x[shuffle[j]]:clone() 
            k = k + 1
        end

        local opfunc = function(x)
            if x ~= parameters then
                parameters:copy(x)
            end
            mlp:zeroGradParameters()
            probs = mlp:forward(batch)
            local nll = crit:forward(probs, batch)
            local df_dw = crit:backward(probs, batch)
            mlp:backward(batch,df_dw)
            local upperbound = nll  + reparam.KL 
			trainnll = nll + trainnll
            return upperbound, gradients+(parameters*0.05)
        end

        parameters, batchupperbound = optim.rmsprop(opfunc, parameters, config, state)
        upperbound = upperbound + batchupperbound[1]
    end
    print("\nEpoch: " .. epoch .. " Upperbound: " .. upperbound/N .. " Time: " .. sys.clock() - time)
	--Display reconstructions and samples
	img_format.title="Train Reconstructions"
	img_format.win = id_reconstr
	id_reconstr = disp.images(stitch(probs,batch),img_format)
	local testnll,probs = eval(data.test_x)
	local b_test = torch.zeros(100,data.test_x:size(2)) 
	local p_test = torch.zeros(100,data.test_x:size(2)) 
	local shufidx = torch.randperm(data.test_x:size(1))
	for i=1,100 do
		p_test[i] = probs[shufidx[i]]:double()
		b_test[i] = data.test_x[shufidx[i]]:double()
	end
	img_format.title="Test Reconstructions"
	img_format.win = id_testreconstr
	id_testreconstr = disp.images(stitch(p_test,b_test),img_format)
	img_format.title="Model Samples"
	img_format.win = id_samples
	local s,mp = getsamples()
	id_samples =  disp.images(s,img_format)
	img_format.title="Mean Probabilities"
	img_format.win = id_mp
	id_mp =  disp.images(mp,img_format)
	print ("Train NLL:",trainnll/N,"Test NLL: ",testnll)
	
	--Save results
    if upperboundlist then
        upperboundlist = torch.cat(upperboundlist,torch.Tensor(1,1):fill(upperbound/N),1)
    else
        upperboundlist = torch.Tensor(1,1):fill(upperbound/N)
    end

    if epoch % 2 == 0 then
        torch.save('save/parameters.t7', parameters)
        torch.save('save/state.t7', state)
        torch.save('save/upperbound.t7', torch.Tensor(upperboundlist))
		local s,mp = getsamples()
		torch.save('save/samples.t7',s)
		torch.save('save/mean_probs.t7',mp)
    end
end
