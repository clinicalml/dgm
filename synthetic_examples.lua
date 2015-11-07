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

len_normflow = len_normflow or 1
energy = energy or 1
batchSize = batchSize or 1000
N = N or batchSize * 10
favorable_init = favorable_init or false
--if suffix then
--	suffix = '_' .. suffix
--else
--	suffix = ''
--end

savedir = savedir or 'synthetic_examples'
--savedir = paths.concat(savedir,'E' .. tostring(energy) .. '_' .. tostring(len_normflow)) .. '_b' .. tostring(batchSize) .. suffix
optimMethod = optimMethod or 'rmsprop'
config = optimConfig or {
    learningRate = 1e-5,
    momentum = 0.9
}
init_std = init_std or 1e-3
annealing = annealing or true
--if annealing then
--	savedir = savedir .. '_annealed'
--end
begin_flow_train = begin_flow_train or 0
--if begin_flow_train > 0 then
--	savedir = savedir .. '_iter' .. tostring(begin_flow_train)
--end
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

-- q0
q0_inp = nn.Identity()()
mu = nn.CMul(2)(q0_inp)
logsigma = nn.CMul(2)(q0_inp)
reparam = nn.GaussianReparam(2)
z0 = reparam({mu,logsigma})
q0_model = nn.gModule({q0_inp},{z0})

var_inp = nn.Identity()()
if len_normflow > 0 then
	-- add normalizing flow
	flow_inp = nn.Identity()()
	flow = {}
	flow[0] = flow_inp
	for k = 1,len_normflow do
		--if k == len_normflow then logpz_flag = true else logpz_flag = false end
		flow[k] = nn.PlanarFlow(2,false,false)(flow[k-1])
	end
	flow_model = nn.gModule({flow_inp},{flow[len_normflow]})

	var_model = nn.gModule({var_inp},{flow_model(q0_model(var_inp))})
else
	var_model = nn.gModule({var_inp},{q0_model(var_inp)})
end

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

qp,qdp = q0_model:getParameters()
--qp:copy(torch.randn(qp:size())*init_std)
qp:fill(0)
fp,fdp = flow_model:getParameters()
fp,_ = flow_model:getParameters()
--fp:copy(torch.randn(fp:size())*init_std)
fp:copy(torch.Tensor(fp:size()):uniform(-0.5,0.5)*init_std)
--fp:copy(torch.randn(fp:size())*5)

if favorable_init then
	flow[1].data.module.w.data.module.weight[1] = 8
	flow[1].data.module.u.data.module.weight[1] = 1
	flow[1].data.module.b.data.module.weight:fill(0)
end

parameters, gradients = flow_model:getParameters()

upperboundlist = nil
trainnlllist = nil
log_KL = nil
log_NLL = nil
params = nil

function KL()
    local KL = 0
	if len_normflow > 0 then
		for t=1,len_normflow do
			KL = flow[t].data.module.KL + KL
		end
	end
    KL = reparam.KL + KL
    return KL:sum()
end

function evaluate(epoch)
	local num_evals = 100000
	local probs = mlp:forward(torch.ones(num_evals,2):cuda())
	local nll = crit:forward(probs)[1]
	z = {}
	for i = 0,len_normflow do
		z[i] = flow[i].data.module.output:float()
	end
	sys.execute('mkdir -p ' .. savedir)
	local writeFile = hdf5.open(paths.concat(savedir,'q_t' .. tostring(epoch) .. '.h5'),'w')
	for i = 0,len_normflow do
		writeFile:write('z' .. tostring(i),z[i])
	end
	writeFile:close()
	local kl = KL()/num_evals
	local nll_eval = nll/num_evals, kl/num_evals

	if log_KL then
		log_KL = torch.cat(log_KL,torch.Tensor(1,1):fill(kl),1)
		log_NLL = torch.cat(log_NLL,torch.Tensor(1,1):fill(nll_eval),1)
		params = torch.cat(params,torch.Tensor(1,parameters:size(1)):copy(parameters:float()),1)
	else
		log_KL = torch.Tensor(1,1):fill(kl)
		log_NLL = torch.Tensor(1,1):fill(nll_eval)
		params = torch.Tensor(1,parameters:size(1)):copy(parameters:float())
	end
	writeFile = hdf5.open(paths.concat(savedir,'log.h5'),'w')
	writeFile:write('eval_KL',log_KL:float())
	writeFile:write('eval_nll',log_NLL:float())
	writeFile:write('params',params)
	--writeFile:write('upperboundlist',upperboundlist:float())
	--writeFile:write('trainnlllist',trainnlllist:float())
	writeFile:close()
	torch.save(paths.concat(savedir,'flow.t7'),flow)

	return nll_eval, kl
end

evaluate(0)
mlp:training()
t_beta = 0 -- annealing index
batch = torch.ones(batchSize,2):cuda()
ones = torch.ones(1):cuda()
config.lr0 = config.learningRate
config.mom0 = config.momentum
print(config)
for epoch = 1,numepochs do
	if begin_flow_train > 0 then
		if epoch == begin_flow_train then
			if config.m then
				config.m = nil
				config.v = nil
				config.tmp = nil
			end
			parameters, gradients = mlp:getParameters()
			config.lr0 = 1e-3
			config.learningRate = 1e-3
		else
			parameters, gradients = q0_model:getParameters()
			t_beta = 0
		end
	end

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
            if annealing and len_normflow > 0 then
                t_beta = t_beta+1
                beta = math.min(1,0.01 + t_beta/(10000/batchSize))
                df_dw = df_dw * beta
                flow[len_normflow].data.module:setAnnealing(beta)
            end
            mlp:backward(batch,df_dw)
            local upperbound = nll + KL()
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
    if upperboundlist then
        upperboundlist = torch.cat(upperboundlist,torch.Tensor(1,1):fill(upperbound/N),1)
        trainnlllist = torch.cat(trainnlllist,torch.Tensor(1,1):fill(trainnll/N),1)
    else
        upperboundlist = torch.Tensor(1,1):fill(upperbound/N)
        trainnlllist = torch.Tensor(1,1):fill(trainnll/N)
    end
    if epoch % 10 == 0 then
		local nll_eval, kl = evaluate(epoch)
        print(epoch)
    end
	collectgarbage()
end




