require 'nn'
require 'hdf5'
require 'nngraph'
require 'sine'
require 'optim'
require "GaussianReparam_normflow"
require "PlanarFlow"
require "Maxout"
require 'hdf5'
require 'cunn'
require 'rmsprop'

cutorch.setDevice(device or 1)

len_normflow = len_normflow or 8
energy = energy or 1
batchSize = batchSize or 100
N = N or batchSize * 100
favorable_init = favorable_init or false
--[[
if suffix then
	suffix = '_' .. suffix
else
	suffix = ''
end
]]

savedir = savedir or 'synthetic_examples'
--savedir = paths.concat(savedir,'E' .. tostring(energy) .. '_' .. tostring(len_normflow)) .. '_b' .. tostring(batchSize) .. suffix
optimMethod = optimMethod or 'rmsprop'


init_std = init_std or 1e-3
annealing = annealing or true
begin_flow_train = begin_flow_train or 0

--[[
if annealing then
	savedir = savedir .. '_annealed'
end
if begin_flow_train > 0 then
	savedir = savedir .. '_iter' .. tostring(begin_flow_train)
end
]]
print(savedir)
numepochs = numepochs or 100

function w1g()
    local z_inp = nn.Identity()()
    local z1 = nn.Select(2,1)(z_inp)
    local val1 = nn.MulConstant(2*math.pi/4)(z1)
    local sin = nn.Sin()(val1)
    return nn.gModule({z_inp},{sin})
end
function w2g()
    local z_inp = nn.Identity()()
    local z1 = nn.Select(2,1)(z_inp)
    local val1 = nn.Square()(nn.MulConstant(1/0.6)(nn.AddConstant(-1)(z1)))
    local val2 = nn.MulConstant(3)(nn.Exp()(nn.MulConstant(-0.5)(val1)))
    return nn.gModule({z_inp},{val2})
end
function w3g()
    local z_inp = nn.Identity()()
    local z1 = nn.Select(2,1)(z_inp)
    local val1 = nn.MulConstant(1/0.3)(nn.AddConstant(-1)(z1))
    local sig = nn.Power(-1)(nn.AddConstant(1)(nn.Exp()(nn.MulConstant(-1)(val1))))
    local val2 = nn.MulConstant(3)(sig)
    return nn.gModule({z_inp},{val2})
end
function E1()
    local z_inp = nn.Identity()()
    local z1 = nn.Select(2,1)(z_inp)
    local v1 = nn.MulConstant(0.5)(nn.Square()(nn.MulConstant(1/0.4)(nn.AddConstant(-2)(nn.Sqrt()(nn.Sum(2)(nn.Square()(z_inp)))))))
    local v2 = nn.Exp()(nn.MulConstant(-0.5)(nn.Square()(nn.MulConstant(1/0.6)(nn.AddConstant(-2)(z1)))))
    local v3 = nn.Exp()(nn.MulConstant(-0.5)(nn.Square()(nn.MulConstant(1/0.6)(nn.AddConstant(2)(z1)))))
    local u = nn.CAddTable()({v1,nn.MulConstant(-1)(nn.Log()(nn.CAddTable()({v2,v3})))})
    return nn.gModule({z_inp},{u})
end
function E2()
    local z_inp = nn.Identity()()
    local z2 = nn.Select(2,2)(z_inp)
    local w1 = w1g()(z_inp)
    local val1 = nn.MulConstant(1/0.4)(nn.CAddTable()({z2,nn.MulConstant(-1)(w1)}))
    local val2 = nn.MulConstant(0.5)(nn.Square()(val1))
    return nn.gModule({z_inp},{val2})
end
function E3()
    local z_inp = nn.Identity()()
    local z2 = nn.Select(2,2)(z_inp)
    local w1 = w1g()(z_inp)
    local w2 = w2g()(z_inp)
    local val1 = nn.CAddTable()({z2,nn.MulConstant(-1)(w1)})
    local val2 = nn.MulConstant(-0.5)(nn.Square()(nn.MulConstant(1/0.35)(val1)))
    local val3 = nn.CAddTable()({z2,nn.MulConstant(-1)(w1),w2})
    local val4 = nn.MulConstant(-0.5)(nn.Square()(nn.MulConstant(1/0.35)(val3)))
    local val5 = nn.Exp()(val2)
    local val6 = nn.Exp()(val4)
    local val7 = nn.MulConstant(-1)(nn.Log()(nn.CAddTable()({val5,val6})))
    return nn.gModule({z_inp},{val7})
end
function E4()
    local z_inp = nn.Identity()()
    local z2 = nn.Select(2,2)(z_inp)
    local w1 = w1g()(z_inp)
    local w3 = w3g()(z_inp)
    local val1 = nn.CAddTable()({z2,nn.MulConstant(-1)(w1)})
    local val2 = nn.MulConstant(-0.5)(nn.Square()(nn.MulConstant(1/0.4)(val1)))
    local val3 = nn.CAddTable()({z2,nn.MulConstant(-1)(w1),w3})
    local val4 = nn.MulConstant(-0.5)(nn.Square()(nn.MulConstant(1/0.35)(val3)))
    local val5 = nn.Exp()(val2)
    local val6 = nn.Exp()(val4)
    local val7 = nn.MulConstant(-1)(nn.Log()(nn.CAddTable()({val5,val6})))
    return nn.gModule({z_inp},{val7})
end


flow = nil

optimConfig0 = {}
if optimConfig then
	for k,v in pairs(optimConfig) do
		optimConfig0[k] = v
	end
else
	optimConfig0 = {
		learningRate = 5e-3,
		momentum = 0.9
	}
end

for stage = 1,len_normflow do
	print('------- stage ' .. tostring(stage) .. ' -------')
	optimConfig = {}
	for k,v in pairs(optimConfig0) do
		optimConfig[k] = v
	end
	if learningRates then
		optimConfig.learningRate = learningRates[stage] or 1e-3
	end
	
	state = nil

	q0_inp = nn.Identity()()
	mu = nn.CMul(2)(q0_inp)
	logsigma = nn.CMul(2)(q0_inp)
	reparam = nn.GaussianReparam(2)
	z0 = reparam({mu,logsigma})
	q0_model = nn.gModule({q0_inp},{z0})
	qp,qdp = q0_model:getParameters()
	qp:fill(0)

	if stage > 1 then
		prev_flow = flow
	end

	flow_inp = nn.Identity()()
	flow = {}
	flow[0] = flow_inp
	for k = 1,stage do
		flow[k] = nn.PlanarFlow(2,false,false)(flow[k-1])
	end
	flow_model = nn.gModule({flow_inp},{flow[stage]})

	-- copy flow parameters from previous training iteration
	for k = 1,stage-1 do
		prev_fp,_ = prev_flow[k].data.module:getParameters()
		fp,_ = flow[k].data.module:getParameters()
		fp:copy(prev_fp)
	end
		
	fp,_ = flow[stage].data.module:getParameters()
	--fp:copy(torch.randn(fp:size())*init_std)
	fp:copy(torch.Tensor(fp:size()):uniform(-0.5,0.5)*init_std)--,1)*init_std)


	var_inp = nn.Identity()()
	var_model = nn.gModule({var_inp},{flow_model(q0_model(var_inp))})

	if energy == 1 then
		gen_model = E1()
	elseif energy == 2 then
		gen_model = E2()
	elseif energy == 3 then
		gen_model = E3()
	else
		gen_model = E4()
	end
	mlp_inp = nn.Identity()()
	mlp = nn.gModule({mlp_inp},{gen_model(var_model(mlp_inp))})
	crit = nn.Sum():cuda()
	mlp = mlp:cuda()

	if fixed then
		parameters, gradients = flow[stage].data.module:getParameters()
	else
		parameters, gradients = mlp:getParameters()
	end

	upperboundlist = nil
	trainnlllist = nil
	log_KL = nil
	log_NLL = nil
	log_gradnorm = nil
	params = nil
	function KL()
		local KL = 0
		for t=1,stage do
			KL = flow[t].data.module.KL + KL
		end
		KL = reparam.KL + KL
		return KL:sum()
	end

	function evaluate(epoch)
		local num_evals = 10000
		local probs = mlp:forward(torch.ones(num_evals,2):cuda())
		local nll = crit:forward(probs)[1]
		z = {}
		for i = 0,stage do
			z[i] = flow[i].data.module.output:float()
		end
		sys.execute('mkdir -p ' .. savedir)
		writeFile = hdf5.open(paths.concat(savedir,'q_k' .. tostring(stage) .. '_t' .. tostring(epoch) .. '.h5'),'w')
		for i = 0,stage do
			writeFile:write('z' .. tostring(i),z[i])
		end
		writeFile:close()

		local kl = KL()/num_evals
		local nll_eval = nll/num_evals, kl/num_evals

		if log_KL then
			log_KL = torch.cat(log_KL,torch.Tensor(1,1):fill(kl),1)
			log_NLL = torch.cat(log_NLL,torch.Tensor(1,1):fill(nll_eval),1)
			params = torch.cat(params,torch.Tensor(1,parameters:size(1)):copy(parameters:float()),1)
			log_gradnorm = torch.cat(log_gradnorm,torch.Tensor(1,1):fill(gradients:norm()),1)
		else
			log_KL = torch.Tensor(1,1):fill(kl)
			log_NLL = torch.Tensor(1,1):fill(nll_eval)
			params = torch.Tensor(1,parameters:size(1)):copy(parameters:float())
			log_gradnorm = torch.Tensor(1,1):fill(gradients:norm())
		end
		--sys.execute('rm ' .. paths.concat(savedir,'log.h5'))
		filepath = paths.concat(savedir,'stage' .. stage .. '_log.h5')
		writeFile = hdf5.open(filepath,'w')
		writeFile:write('eval_KL',log_KL:float())
		writeFile:write('eval_nll',log_NLL:float())
		writeFile:write('params',params)
		writeFile:close()
		torch.save(paths.concat(savedir,'flow.t7'),flow)

		return nll_eval, kl
	end

	evaluate(0)
	mlp:training()
	t_beta = 0 -- annealing index
	batch = torch.ones(batchSize,2):cuda()
	ones = torch.ones(1):cuda()
	print(optimConfig)
	for epoch = 1,numepochs do

		collectgarbage()
		local upperbound = 0
		local trainnll = 0
		-- Pass through data
		for i = 1, N, batchSize do
			--xlua.progress(i+batchSize-1, N)

			-- learning rate decay
			local t = (epoch-1)*N + i
			--config.learningRate = math.max(1e-5,config.lr0/(1+t*5e-5)^0.5)
			--config.momentum = config.mom0/(1+math.exp(10-t*2.5e-5))

			local opfunc = function(x)
				if x ~= parameters then
					parameters:copy(x)
				end
				mlp:zeroGradParameters()
				local probs = mlp:forward(batch)
				local nll = crit:forward(probs)[1]
				local df_dw = crit:backward(probs, ones)
				if annealing then
					t_beta = t_beta+1
					beta = math.min(1,0.01 + t_beta/(10000/batchSize))
					df_dw = df_dw * beta
					flow[stage].data.module:setAnnealing(beta)
				end
				mlp:backward(batch,df_dw)
				local upperbound = nll + KL()
				--trainnll = nll + trainnll
				return upperbound, gradients
			end

			if optimMethod == 'rmsprop' then
					optimConfig.momentum = optimConfig.momentum or 0.9
					optimConfig.alpha = optimConfig.alpha or 0.9
					parameters, batchupperbound = optim.rmsprop(opfunc, parameters, optimConfig, state)
			elseif optimMethod == 'adam' then
					parameters, batchupperbound = optim.adam(opfunc, parameters, optimConfig, state)
			else
					error('unknown optimization method')
			end
			upperbound = upperbound + batchupperbound[1]
		end
		if epoch % 10 == 0 then
			local nll, kl = evaluate(epoch)
			print(epoch, kl, gradients:norm())
		end
		collectgarbage()
	end
end



