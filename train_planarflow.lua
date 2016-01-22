require "nn"
require "xlua"
require "nngraph"
require "utils"
require "cunn"
require "cutorch"
require "optim"
require "GaussianReparam_normflow"
require "PlanarFlow"
require "Maxout"
require 'hdf5'
require 'rmsprop'
disp = require "display"

devicenum = devicenum or 1
cutorch.setDevice(devicenum)
torch.setdefaulttensortype('torch.FloatTensor')
---------------- Train Params ------------
local myname = myname or "planarflow1"
local savedir = savedir or "save_planarflow1"
local savebestmodel = savebestmodel or true
local annealing = annealing or false
local optimMethod = optimMethod or 'adam'
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
local nonlinearity   = nonlinearity or nn.ReLU
local maxoutWindow   = maxoutWindow or 4
local len_normflow   = len_normflow or 1
local deepParams     = deepParams or false
local S              = S or 200  -- num samples used to estimate -log(p(v)) in estimate_logpv
local init_std       = init_std or 1e-3
local flow_init      = flow_init or init_std

--------- Recognition. Network -----------
q0_inp = nn.Identity()()
if dropout then
	q_hid_1 = nn.Maxout(dim_input,dim_hidden,maxoutWindow)(nn.Dropout(dropout)(q0_inp))
	q_hid_2 = nn.Maxout(dim_hidden,dim_hidden,maxoutWindow)(nn.Dropout(dropout)(q_hid_1))
else
	q_hid_1 = nn.Maxout(dim_input,dim_hidden,maxoutWindow)(q0_inp)
	q_hid_2 = nn.Maxout(dim_hidden,dim_hidden,maxoutWindow)(q_hid_1)
end
mu  = nn.Linear(dim_hidden,dim_stochastic)(q_hid_2)
logsigma  = nn.Linear(dim_hidden,dim_stochastic)(q_hid_2)
reparam   = nn.GaussianReparam(dim_stochastic)
z  = reparam({mu,logsigma})
local q0_model = nn.gModule({q0_inp},{z})

-- add normalizing flow
flow_inp = nn.Identity()()
flow = {}
flow[0] = flow_inp
for k = 1,len_normflow do
	if k == len_normflow then logpz_flag = true else logpz_flag = false end
	if deepParams then
		w_in = nn.Maxout(dim_hidden,dim_stochastic,maxoutWindow)(nn.Dropout(dropout)(q_hid_2))
		b_in = nn.Maxout(dim_hidden,1,maxoutWindow)(nn.Dropout(dropout)(q_hid_2))
		u_in = nn.Maxout(dim_hidden,dim_stochastic,maxoutWindow)(nn.Dropout(dropout)(q_hid_2))
		flow[k] = nn.PlanarFlow(dim_stochastic,true,logpz_flag)({flow[k-1],w_in,b_in,u_in})
	else
		flow[k] = nn.PlanarFlow(dim_stochastic,false,logpz_flag)(flow[k-1])
	end
end
local flow_model = nn.gModule({flow_inp},{flow[len_normflow]})

local var_inp = nn.Identity()()
local var_model = nn.gModule({var_inp},{flow_model(q0_model(var_inp))})

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
gp, _ = gen_model:getParameters()
gp:copy(torch.randn(gp:size()):mul(init_std));
qp, _ = var_model:getParameters()
qp:copy(torch.randn(qp:size()):mul(init_std));
fp, _ = flow_model:getParameters()
fp:uniform(-0.5,0.5):mul(flow_init);

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
beta = 1 --annealing

---------  Aggregate contributions to KL in flow  ---------
function KL()
	local KL = 0
	for t=1,len_normflow do
		KL = flow[t].data.module.KL + KL
	end
	KL = reparam.KL + KL
	return KL:sum() 
end
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
	local kl    = KL()
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
		-- ll = p(v|z_s) ~ calc assumes each pixel output is conditional independent given latent states
		local ll = (torch.cmul(torch.log(probs+1e-12),dataset)+torch.cmul(torch.log(probs*(-1)+1+1e-12),dataset*(-1)+1)):sum(2)
		-- latent = z_s = sampled z|x
		local latent = flow[len_normflow].data.module.output
		-- logprior is standard normal
		local logprior = (torch.pow(latent,2):sum(2)+dim_stochastic*math.log(2*math.pi))*(-0.5)
		-- assume logsigma is log(sigma^2)
		local logsig = logsigma.data.module.output
		local m = mu.data.module.output
		local z0 = z.data.module.output
		local logq0 = ((torch.exp(logsig*(-1)):cmul(torch.pow(z0-m,2))):sum(2)+logsig:sum(2)+dim_stochastic*math.log(2*math.pi))*(-0.5)
		local logdetJ = 0
		for t=1,len_normflow do
			logdetJ = flow[t].data.module.logdetJacobian + logdetJ
		end
		local logq = logq0 + logdetJ
		-- log p_s = log(p(v|z_s)*p(z_s)/q(z_s|v))
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
local t_beta = 0 -- annealing index
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
            batch[k] = data.train_x[shuffle[j]] 
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
			if annealing then
				beta = math.min(1,0.01 + t_beta/100000) 
				df_dw = df_dw * beta
				flow[len_normflow].data.module:setAnnealing(beta)
				t_beta = t_beta+batchSize
			end
            mlp:backward(batch,df_dw)
			local kl = KL()
            local upperbound = nll + kl
			trainKL = trainKL + kl
			trainnll = nll + trainnll
            return upperbound, gradients
        end

        if optimMethod == 'rmsprop' then
				config.momentum = config.momentum or 0.9
				config.alpha = config.alpha or 0.9
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
                    z = z
                }
                torch.save(paths.concat(savedir,'model.t7'),model_save)
                bestmodel = testlogpv
            end
        end

		--Display reconstructions and samples
		img_format.title="Train Reconstructions" .. ": " .. myname
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
		img_format.title="Test Reconstructions" .. ": " .. myname
		img_format.win = id_testreconstr
		id_testreconstr = disp.images(stitch(p_test,b_test),img_format)
		img_format.title="Model Samples" .. ":" .. myname
		img_format.win = id_samples
		local s,mp = getsamples()
		id_samples =  disp.images(s,img_format)
		img_format.title="Mean Probabilities" .. ": " .. myname
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
