require "nn"
require "xlua"
require "nngraph"
require "utils"
require "cunn"
require "cutorch"
require "optim"
require "GaussianReparam"
require 'Maxout'
require 'hdf5'
require 'rmsprop'
disp = require "display"

local devicenum = devicenum or 1
cutorch.setDevice(devicenum)
torch.setdefaulttensortype('torch.FloatTensor')
---------------- Train Params ------------
local savedir = savedir or 'save'
local optimMethod = optimMethod or 'adam'
local savebestmodel = savebestmodel or true
local datadir = datadir or './'
print('save directory = ' .. savedir)
print('data directory = ' .. datadir)

---------------- Load Data ---------------
data=loadBinarizedMNIST(true,datadir)

---------------- Model Params. -----------
local dim_input      = 784
local dim_hidden     = dim_hidden or 200
local dim_stochastic = dim_stochastic or 40
local dropout        = dropout or false
local maxoutWindow   = maxoutWindow or 4
local nonlinearity   = nonlinearity or nn.ReLU
local S              = S or 200
local init_std       = init_std or 1e-3

--------- Recognition. Network -----------
local var_inp = nn.Identity()()
if dropout then
    q_hid_1 = nn.Maxout(dim_input,dim_hidden,maxoutWindow)(nn.Dropout(dropout)(var_inp))
    q_hid_2 = nn.Maxout(dim_hidden,dim_hidden,maxoutWindow)(nn.Dropout(dropout)(q_hid_1))
else
    q_hid_1 = nn.Maxout(dim_input,dim_hidden,maxoutWindow)(var_inp)
    q_hid_2 = nn.Maxout(dim_hidden,dim_hidden,maxoutWindow)(q_hid_1)
end
mu  = nn.Linear(dim_hidden,dim_stochastic)(q_hid_2)
logsigma  = nn.Linear(dim_hidden,dim_stochastic)(q_hid_2)
reparam   = nn.GaussianReparam(dim_stochastic)
z = reparam({mu,logsigma})
local var_model = nn.gModule({var_inp},{z})

--------- Generative Network -------------
gen_inp = nn.Identity()()
if dropout then
	hid1 = nn.Maxout(dim_stochastic,dim_hidden,maxoutWindow)(nn.Dropout(dropout)(gen_inp))
	hid2 = nn.Maxout(dim_hidden,dim_hidden,maxoutWindow)(nn.Dropout(dropout)(hid1))
	reconstr = nn.Sigmoid()(nn.Linear(dim_hidden,dim_input)(hid2))
else
	hid1 = nn.Maxout(dim_stochastic,dim_hidden,maxoutWindow)(gen_inp)
	hid2 = nn.Maxout(dim_hidden,dim_hidden,maxoutWindow)(hid1)
	reconstr = nn.Sigmoid()(nn.Linear(dim_hidden,dim_input)(hid2))
end
gen_model = nn.gModule({gen_inp},{reconstr})

---------- Initialization ----------------
torch.manualSeed(1)
cutorch.manualSeed(1)
gp,gdp = gen_model:getParameters()
gp:copy(torch.randn(gp:size()):mul(init_std))
vp,vdp = var_model:getParameters()
vp:copy(torch.randn(vp:size()):mul(init_std))

----- Combining Models into Single MLP----
local inp = nn.Identity()()
mlp = nn.gModule({inp},{gen_model(var_model(inp))})
crit= nn.BCECriterion()
crit.sizeAverage = false

--------- Setup for Training/Viz.---------
img_format,format = setupDisplay()
setupFolder(savedir)
mlp:cuda()
crit:cuda()
data.train_x = data.train_x:cuda()
data.test_x  = data.test_x:cuda()
parameters, gradients = mlp:getParameters()
config = optimconfig or {
    learningRate = 1e-5,
}
print(config)
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
	local kl    = reparam.KL
    local N     = dataset:size(1)
	mlp:training()
	return (nll+kl)/N, probs, kl/N
end
--------- Test p(v) via importance sampling ---------
-- see appendix E in Rezende, D. J., Mohamed, S., and Wierstra, D. Stochastic backpropagation and approximate inference in deep generative models. In ICML, 2014.
function estimate_logpv(dataset,S)
        mlp:evaluate()
        local S = S or 200
        local logp = nil
        local sample_flow_output = torch.Tensor(dataset:size(1),S,dim_stochastic):float()
        for s = 1,S do
            collectgarbage()
            local probs = mlp:forward(dataset)
            -- ll = log(p(v|z_s)) ~ calc assumes each pixel output is conditional independent given latent states
            local ll = (torch.cmul(torch.log(probs+1e-12),dataset)+torch.cmul(torch.log(probs*(-1)+1+1e-12),dataset*(-1)+1)):sum(2)

            -- z_s = sampled z|v
            local z_s = z.data.module.output

            -- logprior is standard normal (note that the -n*0.5*log2pi will cancel out with logq)
            local logprior = torch.cmul(z_s,z_s):sum(2)*(-0.5)

            -- epsilon
            local eps = z.data.module.eps

            -- assume logsigma is log(sigma^2)
            local logsig2 = logsigma.data.module.output
            local logq = torch.cmul(eps,eps):sum(2)*(-0.5) + logsig2:sum(2)*(-0.5)

            -- sum the terms together
            local logp_s = ll+logprior-logq

            -- logp = [logp_1,...,logp_S]
            if logp ~= nil then
                    logp = nn.JoinTable(2):forward{logp_s:float(),logp}
            else
                    logp = logp_s:float()
            end
            sample_flow_output[{{},s,{}}]:copy(var_model.output:float())
        end
        local a = logp:max(2)
        -- computing logsumexp: https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
        local logsumexp = a + torch.log(torch.exp(logp-nn.Replicate(S,2):cuda():forward(a)):sum(2)) - math.log(S)
        local logpv = logsumexp:mean()
        mlp:training()
        return logpv, sample_flow_output
end
--------- Stitch Images Together ---------
function stitch(probs,batch)
	local imgs = {}
	for i = 1,batchSize do 
		imgs[i] = torch.cat(probs[i]:float():reshape(28,28),batch[i]:float():reshape(28,28),2)
	end
	return imgs
end

--------- Update lists of values ---------
function updateList(list,newval)
    local list = list
    if list then
        list = torch.cat(list,torch.Tensor(1,1):fill(newval),1)
    else
        list = torch.Tensor(1,1):fill(newval)
    end
    return list
end

-------------- Training Loop -------------
for epoch =1,5000 do 
	collectgarbage()
    local upperbound = 0
	local trainnll = 0
	local trainKL = 0
    local time = sys.clock()
    local shuffle = torch.randperm(data.train_x:size(1))
	--if epoch==100 then config.learningRate = 5e-5 end
	--if epoch > 30 then config.learningRate = math.max(config.learningRate / 1.000005, 0.000001) end
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
			local kl = reparam.KL
            local upperbound = nll  + kl
			trainKL = trainKL + kl
			trainnll = nll + trainnll
            return upperbound, gradients
        end

		if optimMethod == 'rmsprop' then
			parameters, batchupperbound = optim.rmsprop(opfunc, parameters, config, state)
		elseif optimMethod == 'adam' then
			parameters, batchupperbound = optim.adam(opfunc, parameters, config, state)
		else
			error('unknown optimization method')
		end
        upperbound = upperbound + batchupperbound[1]
    end
	
	--Save results
	upperboundlist = updateList(upperboundlist,upperbound/N)
    trainKLlist = updateList(trainKLlist,trainKL/N)

    if epoch % 10  == 0 then
    	print("\nEpoch: " .. epoch .. " Upperbound: " .. upperbound/N .. " Time: " .. sys.clock() - time)
        trainlogpv, train_sample_z = estimate_logpv(data.train_x,S)
        testlogpv, test_sample_z = estimate_logpv(data.test_x,S)
        trainlogpv = trainlogpv*(-1)
        testlogpv = testlogpv*(-1)
        print("logp = " .. testlogpv)
		if savebestmodel then
			if bestmodel == nil then
				bestmodel = testlogpv
			elseif testlogpv < bestmodel then
				local model_save = {
					mlp = mlp,
					flow = flow,
					mu = mu,
					logsigma = logsigma,
					z = z,
					var_model = var_model,
					gen_model = gen_model
				}
				torch.save(paths.concat(savedir,'model.t7'),model_save)
				bestmodel = testlogpv
			end
		end
		--Display reconstructions and samples
		img_format.title="Train Reconstructions"
		img_format.win = id_reconstr
		id_reconstr = disp.images(stitch(probs,batch),img_format)
		local testnll, probs, testKL = eval(data.test_x)
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
		print ("saving to directory: " .. savedir)

        testupperboundlist = updateList(testupperboundlist,testnll)
        testKLlist         = updateList(testKLlist,testKL)
        testlogpvlist      = updateList(testlogpvlist,testlogpv)
        trainlogpvlist     = updateList(trainlogpvlist,trainlogpv)

		sys.execute('mkdir -p ' .. savedir)
		torch.save(paths.concat(savedir,'parameters.t7'), parameters)
		torch.save(paths.concat(savedir,'state.t7'), state)
		torch.save(paths.concat(savedir,'upperbound.t7'), torch.Tensor(upperboundlist))
		torch.save(paths.concat(savedir,'upperbound_test.t7'),testupperboundlist)
		writeFile = hdf5.open(paths.concat(savedir,'upperbounds.h5'),'w')
		writeFile:write('train',upperboundlist)
		writeFile:write('test',testupperboundlist)
		writeFile:write('train_logpv',trainlogpvlist)
		writeFile:write('test_logpv',testlogpvlist)
        writeFile:write('trainKL',trainKLlist)
        writeFile:write('testKL',testKLlist)
		writeFile:close()
        if (epoch == 1) or (epoch % 100) then
            writeFile = hdf5.open(paths.concat(savedir,'sample_flow_output' .. epoch .. '.h5'),'w')
            writeFile:write('train_z',train_sample_z)
            writeFile:write('test_z',test_sample_z)
            writeFile:close()
        end
		local s,mp = getsamples()
		torch.save(paths.concat(savedir,'samples.t7'),s)
		torch.save(paths.concat(savedir,'mean_probs.t7'),mp)
    end
end
