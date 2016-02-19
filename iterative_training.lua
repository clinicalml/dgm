require "nn"
require "nngraph"
require "cunn"
require "GaussianReparamFlow"
require "PlanarFlow"
require "NormalizingFlow"

function deepcopy(x,clonetensor)
	local clonetensor = clonetensor or true
	if type(x) == 'table' then
		local y = {}
		for k,v in pairs(x) do
			y[k] = deepcopy(v,clonetensor)
		end
		return y
	else
		if clonetensor and torch.isTensor(x) then
			return x:clone()
		else
			return x
		end
	end
end

function iterative_training(opt)
	opt.modelfile     = opt.modelfile or 'model.lua'
	opt.len_normflow0 = opt.len_normflow
	opt.optimconfig0  = deepcopy(opt.optimconfig,true)
	opt.len_normflow  = 0
	opt.numepochs     = opt.epochs[1]

	dofile(opt.modelfile)
	model = loadmodel(opt)

	print('***************************')
	print('                   L =',0)
	dofile(opt.trainfile)

	local rec_model = model.rec_model
	local reparam   = nn.GaussianReparamFlow(opt.dim_stochastic)
	local x         = nn.Identity()()
	local z         = reparam(rec_model(x))
	local q0_model  = nn.gModule({x},{z})

	local flow = nn.NormalizingFlow()
	for L = 1, opt.len_normflow0 do
		print('***************************')
		print('                   L =',L)
		-- new flow length
		opt.len_normflow = L

		-- reset optimconfig
		opt.optimconfig  = deepcopy(opt.optimconfig0,true)

		-- set number of epochs for each iteration
		if L < opt.len_normflow0 then
			opt.numepochs = opt.epochs[1]
		else
			opt.numepochs = opt.epochs[2]
		end
	
		-- add next flow layer
		local nextlayer = nn.PlanarFlow(opt.dim_stochastic)
		flow:add(nextlayer)
		local fp,_ = nextlayer:getParameters()
		fp:uniform(-0.5,0.5):mul(opt.flow_init)
		
		-- setup variational model
		local var_inp = nn.Identity()()
		local var_model = nn.gModule({var_inp},{flow(q0_model(var_inp))})

		-- setup VAE
		local gen_model = model.gen_model
		local mlp_inp = nn.Identity()()
		local mlp = nn.gModule({mlp_inp},{gen_model(var_model(mlp_inp))})

		-- transfer to cuda
		mlp:cuda()
		
		-- setup for training
		parameters, gradients = mlp:getParameters()		
		gradients:zero()

		model.reparam = reparam
		model.z = z
		model.q0_model = q0_model
		model.flow = flow
		model.var_model = var_model
		model.gen_model = gen_model
		model.mlp = mlp
		model.parameters = parameters
		model.gradients = gradients

		collectgarbage()

		-- train
		dofile(opt.trainfile)
	end
end


		


