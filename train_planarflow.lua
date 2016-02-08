require "nn"
require "xlua"
require "nngraph"
require "utils"
require "cunn"
require "cutorch"
require "optim"
require 'hdf5'
require 'rmsprop'
require "PlanarFlow"
require "Maxout"
require 'NormalizingFlow'
disp = require "display"

---------------- Train Params ------------
opt               = opt or {}
opt.myname        = opt.myname or "planarflow1"
opt.savedir       = opt.savedir or "save_planarflow1"
opt.savebestmodel = opt.savebestmodel or true
opt.annealing     = opt.annealing or false
opt.optimMethod   = opt.optimMethod or 'adam'
opt.datadir       = opt.datadir or './'
opt.numepochs     = opt.numepochs or 833

opt.devicenum     = opt.devicenum or 1
cutorch.setDevice(opt.devicenum)
torch.setdefaulttensortype('torch.FloatTensor')
---------------- Load Data ---------------
data=loadBinarizedMNIST(true,opt.datadir)

---------------- Model Params. -----------
opt.dim_input      = 784
opt.dim_hidden     = opt.dim_hidden or 200
opt.dim_stochastic = opt.dim_stochastic or 40
opt.dropout        = opt.dropout or false
opt.maxoutWindow   = opt.maxoutWindow or 4
opt.deepParams     = opt.deepParams or false
opt.len_normflow   = opt.len_normflow or 1
opt.S              = opt.S or 200  -- num samples used to estimate -log(p(v)) in estimate_logpv
opt.init_std       = opt.init_std or 1e-3
opt.flow_init      = opt.flow_init or opt.init_std

print(opt)
--------- Recognition. Network -----------
if model then
	print('\nloading model\n')
	mu           = model.mu
	logsigma     = model.logsigma
	reparam      = model.reparam
	z            = model.z
	q0_model     = model.q0_model
	flow         = model.flow
	flow_model   = model.flow_model
	var_model    = model.var_model
	gen_model    = model.gen_model
	mlp          = model.mlp
	crit         = model.crit

	parameters   = model.parameters
	gradients    = model.gradients
else
	if opt.len_normflow > 0 then
		require "GaussianReparam_normflow"
	else
		require "GaussianReparam"
	end

	torch.manualSeed(1)
	cutorch.manualSeed(1)
	local dim_input       = opt.dim_input
	local dim_hidden      = opt.dim_hidden
	local dim_stochastic  = opt.dim_stochastic
	local maxoutWindow    = opt.maxoutWindow
	q0_inp = nn.Identity()()
	if opt.dropout then
		q_hid_1 = nn.Maxout(dim_input,dim_hidden,maxoutWindow)(nn.Dropout(opt.dropout)(q0_inp))
		q_hid_2 = nn.Maxout(dim_hidden,dim_hidden,maxoutWindow)(nn.Dropout(opt.dropout)(q_hid_1))
	else
		q_hid_1 = nn.Maxout(dim_input,dim_hidden,maxoutWindow)(q0_inp)
		q_hid_2 = nn.Maxout(dim_hidden,dim_hidden,maxoutWindow)(q_hid_1)
	end
	mu  = nn.Linear(dim_hidden,dim_stochastic)(q_hid_2)
	logsigma  = nn.Linear(dim_hidden,dim_stochastic)(q_hid_2)
	reparam   = nn.GaussianReparam(dim_stochastic)
	z = reparam({mu,logsigma})
	q0_model = nn.gModule({q0_inp},{z})
	qp, _ = q0_model:getParameters()
	qp:copy(torch.randn(qp:size()):mul(opt.init_std));

	-- add normalizing flow
	local var_inp = nn.Identity()()
	if opt.len_normflow > 0 then
		flow = nn.NormalizingFlow()
		for k = 1, opt.len_normflow do
			flow:add(nn.PlanarFlow(dim_stochastic))
		end
		fp, _ = flow:getParameters()
		fp:uniform(-0.5,0.5):mul(opt.flow_init);

		var_model = nn.gModule({var_inp},{flow(q0_model(var_inp))})
	else
		var_model = nn.gModule({var_inp},{q0_model(var_inp)})
	end

	--------- Generative Network -------------
	gen_inp = nn.Identity()()
	if opt.dropout then
		hid1 = nn.Maxout(dim_stochastic,dim_hidden,maxoutWindow)(nn.Dropout(opt.dropout)(gen_inp))
		hid2 = nn.Maxout(dim_hidden,dim_hidden,maxoutWindow)(nn.Dropout(opt.dropout)(hid1))
		reconstr = nn.Sigmoid()(nn.Linear(dim_hidden,dim_input)(hid2))
	else
		hid1 = nn.Maxout(dim_stochastic,dim_hidden,maxoutWindow)(gen_inp)
		hid2 = nn.Maxout(dim_hidden,dim_hidden,maxoutWindow)(hid1)
		reconstr = nn.Sigmoid()(nn.Linear(dim_hidden,dim_input)(hid2))
	end
	gen_model = nn.gModule({gen_inp},{reconstr})
	gp, _ = gen_model:getParameters()
	gp:copy(torch.randn(gp:size()):mul(opt.init_std));

	----- Combining Models into Single MLP----
	local inp = nn.Identity()()
	mlp = nn.gModule({inp},{gen_model(var_model(inp))})
	crit= nn.BCECriterion()
	crit.sizeAverage = false

	---------- Transfer to GPU ----------------
	mlp:cuda()
	crit:cuda()

	---------- Store Model ----------------
	model             = {}
	model.mu          = mu
	model.logsigma    = logsigma
	model.reparam     = reparam
	model.z           = z
	model.q0_model    = q0_model
	model.flow        = flow
	model.flow_model  = flow_model
	model.var_model   = var_model
	model.gen_model   = gen_model
	model.mlp         = mlp
	model.crit        = crit
	model.parameters  = parameters
	model.gradients   = gradients

	---------- Setup Parameters ----------------
	parameters, gradients = mlp:getParameters()
end
model.opt = opt

--------- Setup for Training/Viz.---------
img_format,format = setupDisplay()
setupFolder(opt.savedir)
data.train_x = data.train_x:cuda()
data.test_x  = data.test_x:cuda()
config = opt.optimconfig or {
    learningRate = 1e-5,
}
print(config)
batchSize = 100
state = {}
beta = 1 --annealing

---------  Aggregate contributions to KL in flow  ---------
function KL(beta)
	local KL = reparam.KL
	if flow then
		if beta ~= 1 then
			KL = KL + (flow:getKL(beta) or 0)
		else
			KL = KL + (flow.KL or 0)
		end
	end
	if type(KL) == 'number' then
		return KL
	else
		return KL:sum() 
	end
end

function LogDetJ()
	local logdetJ = flow:getlogdetJ() or 0
	if type(logdetJ) == 'number' then
		return logdetJ
	else
		return logdetJ:sum()
	end
end

function Logpz()
	local logpz = flow.logpz or 0
	if type(logpz) == 'number' then
		return logpz
	else
		return logpz:sum()
	end
end

function Logqz0()
	local logqz0 = reparam.KL
	if type(logqz0) == 'number' then
		return logqz0
	else
		return logqz0:sum()
	end
end

---------  Sample from Gen. Model---------
function getsamples()
	local p = gen_model:forward(torch.randn(batchSize,opt.dim_stochastic):typeAs(data.train_x))
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
--------- Estimate logp(v) via importance sampling ---------
-- see appendix E in Rezende, D. J., Mohamed, S., and Wierstra, D. Stochastic backpropagation and approximate inference in deep generative models. In ICML, 2014.
function estimate_logpv(dataset,S)
        mlp:evaluate()
        local S = S or 200
        local N = dataset:size(1)
        local batchSize = 500
        local logpv = 0
        local logp   = torch.FloatTensor(batchSize,S)
        for b = 1, N, batchSize do
            local start  = b
            local finish = math.min(N,b+batchSize-1)
            local x      = dataset[{{start,finish}}]
            logp:resize(finish-start+1,S)
            for s = 1,S do
                collectgarbage()
                local probs = mlp:forward(x)
                -- ll = log(p(v|z_s)) ~ calc assumes each pixel output is conditional independent given latent states
                local ll = (torch.cmul(torch.log(probs+1e-12),x)+torch.cmul(torch.log(probs*(-1)+1+1e-12),x*(-1)+1)):sum(2)

                -- z_s = sampled z|v
				local z_s = var_model.output

                -- logprior is standard normal (note that the -n*0.5*log2pi cancels out with logq)
                local logprior = torch.cmul(z_s,z_s):sum(2)*(-0.5)

                -- epsilon
                local eps = z.data.module.eps

                -- assume logsigma is log(sigma^2)
                local logsig2 = logsigma.data.module.output
                local logq0 = torch.cmul(eps,eps):sum(2)*(-0.5) + logsig2:sum(2)*(-0.5)
				
				local logq = logq0
				if flow then
					local logdetJ = flow:getlogdetJ() or 0
					logq = logq + logdetJ
				end

                -- sum the terms together
                local logp_s = ll+logprior-logq

                -- logp = [logp_1,...,logp_S]
                logp[{{},s}]:copy(logp_s:float())
            end
            local a = logp:max(2)
            --compute logsumexp: https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
            local logsumexp = a + torch.log(torch.exp(logp-nn.Replicate(S,2):cuda():forward(a)):sum(2)) - math.log(S)
            logpv = logsumexp:sum() + logpv
        end
        logpv = logpv / N
        mlp:training()
        return logpv
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
local avggrad = gradients:clone()
for epoch =1,opt.numepochs do 
	collectgarbage()
    local upperbound = 0
	local trainnll = 0
	local trainKL = 0
	avggrad:zero()
    local time = sys.clock()
    local shuffle = torch.randperm(data.train_x:size(1))
	--if epoch==100 then config.learningRate = 5e-5 end
	--if epoch > 30 then config.learningRate = math.max(config.learningRate / 1.000005, 0.000001) end
    --Make sure batches are always batchSize
    local N = data.train_x:size(1) - (data.train_x:size(1) % batchSize)
    local N_test = data.test_x:size(1) - (data.test_x:size(1) % batchSize)
	local probs 
	local logdetJ = 0
	local logpz = 0
	local logqz0 = 0
    local batch = torch.Tensor(batchSize,data.train_x:size(2)):typeAs(data.train_x)
	-- Pass through data
    for i = 1, N, batchSize do
		--parameters, gradients = flow_model:getParameters()
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
			--forward and backward
            mlp:zeroGradParameters()
            probs = mlp:forward(batch)
            local nll = crit:forward(probs, batch)
            local df_dw = crit:backward(probs, batch)
			--annealing
			if opt.annealing then
				beta = math.min(1,0.01 + t_beta/10000) 
				if flow then
					flow:setAnnealing(beta)
				else
					reparam:setAnnealing(beta)
				end
				t_beta = t_beta+1--batchSize
			end
            mlp:backward(batch,df_dw)
			local kl = KL()
            local batchupperbound = nll + kl
			upperbound = upperbound + batchupperbound
			trainKL = trainKL + kl
			trainnll = nll + trainnll
			logdetJ = LogDetJ() + logdetJ
			logpz = Logpz() + logpz
			logqz0 = Logqz0() + logqz0
			if opt.annealing then
				local kl = KL(beta)
				batchupperbound = nll*beta + kl
			end
			avggrad:add(gradients/N)
            return batchupperbound, gradients
        end

        if opt.optimMethod == 'rmsprop' then
				config.momentum = config.momentum or 0.9
				config.alpha = config.alpha or 0.9
                parameters, batchupperbound = optim.rmsprop(opfunc, parameters, config, state)
        elseif opt.optimMethod == 'adam' then
                parameters, batchupperbound = optim.adam(opfunc, parameters, config, state)
        else
                error('unknown optimization method')
        end
		--print(trainnll/(i+batchSize-1))
		--print(gradients:norm())
		--print(parameters:norm())
    end
	
	--Save results
	upperboundlist = updateList(upperboundlist,upperbound/N)
	trainKLlist = updateList(trainKLlist,trainKL/N)
	logdetJlist = updateList(logdetJlist,logdetJ/N)
	logpzlist = updateList(logpzlist,logpz/N)
	logqz0list = updateList(logqz0list,logqz0/N)
	pnormlist = updateList(pnormlist,parameters:norm())
	gradnormlist = updateList(gradnormlist,gradients:norm())
	avggradlist = updateList(avggradlist,avggrad:norm())

    if epoch % 10  == 0 then
    	print("\nEpoch: " .. epoch .. " Upperbound: " .. upperbound/N .. " Time: " .. sys.clock() - time)
		trainlogpv = estimate_logpv(data.train_x, opt.S)
		testlogpv = estimate_logpv(data.test_x, opt.S)
		trainlogpv = trainlogpv*(-1)
		testlogpv = testlogpv*(-1)
		print("logp = " .. testlogpv) 
        if opt.savebestmodel then
            if bestmodel == nil then
                bestmodel = testlogpv
            elseif testlogpv < bestmodel then
                torch.save(paths.concat(opt.savedir,'best_model.t7'),model)
                bestmodel = testlogpv
            end
        end
		model.optim = {}
		model.optim.config = config
		model.optim.state = state
		model.optim.epoch = epoch
		model.optim.batchSize = batchSize
		model.optim.t_beta = t_beta
		torch.save(paths.concat(opt.savedir,'model.t7'),model)

		--Display reconstructions and samples
		img_format.title="Train Reconstructions" .. ": " .. opt.myname
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
		img_format.title="Test Reconstructions" .. ": " .. opt.myname
		img_format.win = id_testreconstr
		id_testreconstr = disp.images(stitch(p_test,b_test),img_format)
		img_format.title="Model Samples" .. ":" .. opt.myname
		img_format.win = id_samples
		local s,mp = getsamples()
		id_samples =  disp.images(s,img_format)
		img_format.title="Mean Probabilities" .. ": " .. opt.myname
		img_format.win = id_mp
		id_mp =  disp.images(mp,img_format)
		print ("Train NLL:",trainnll/N,"Test NLL: ",testnll)
		print ("saving to directory: " .. opt.savedir)

		testupperboundlist = updateList(testupperboundlist,testnll)
		testKLlist         = updateList(testKLlist,testKL)
		testlogpvlist      = updateList(testlogpvlist,testlogpv)
		trainlogpvlist     = updateList(trainlogpvlist,trainlogpv)

		sys.execute('mkdir -p ' .. opt.savedir)
		torch.save(paths.concat(opt.savedir,'parameters.t7'), parameters)
		torch.save(paths.concat(opt.savedir,'state.t7'), state)
		torch.save(paths.concat(opt.savedir,'upperbound.t7'), torch.Tensor(upperboundlist))
		torch.save(paths.concat(opt.savedir,'upperbound_test.t7'),testupperboundlist)
		writeFile = hdf5.open(paths.concat(opt.savedir,'upperbounds.h5'),'w')
		writeFile:write('train',upperboundlist)
		writeFile:write('test',testupperboundlist)
		writeFile:write('train_logpv',trainlogpvlist)
		writeFile:write('test_logpv',testlogpvlist)
		writeFile:write('trainKL',trainKLlist)
		writeFile:write('testKL',testKLlist)
		writeFile:write('logdetJ',logdetJlist)
		writeFile:write('logpz',logpzlist)
		writeFile:write('logqz0',logqz0list)
		writeFile:write('pnorm',pnormlist)
		writeFile:write('gradnorm',gradnormlist)
		writeFile:write('avggrad',avggradlist)
		writeFile:close()
		local s,mp = getsamples()
		torch.save(paths.concat(opt.savedir,'samples.t7'),s)
		torch.save(paths.concat(opt.savedir,'mean_probs.t7'),mp)
    end
end
