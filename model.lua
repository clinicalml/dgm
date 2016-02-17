require "nn"
require "nngraph"
require "Maxout"
require 'NormalizingFlow'
require "GaussianReparam"
require "GaussianReparamFlow"
require 'cunn'

function loadmodel(opt)
	local opt          = opt or {}
	opt.devicenum      = opt.devicenum or 1 
	opt.dim_input      = opt.dim_input or 784
	opt.genlayers      = opt.genlayers or 3
	opt.gendim         = opt.gendim or 200
	opt.varlayers      = opt.varlayers or 3
	opt.vardim         = opt.vardim or 200
	opt.dim_stochastic = opt.dim_stochastic or 40
	opt.dropout        = opt.dropout or false
	opt.maxoutWindow   = opt.maxoutWindow or 4
	opt.len_normflow   = opt.len_normflow or 10
	opt.init_std       = opt.init_std or 1e-3
	opt.flow_init      = opt.flow_init or opt.init_std
	opt.flow_type      = opt.flow_type or 'PlanarFlow'
	opt.reparamnoise   = opt.reparamnoise or 0.05


	cutorch.setDevice(opt.devicenum)
	torch.setdefaulttensortype('torch.FloatTensor')
	torch.manualSeed(1)
	cutorch.manualSeed(1)

	local rec_base = nn.Sequential()
	for i = 1, opt.varlayers do
		if opt.dropout then
			rec_base:add(nn.Dropout(opt.dropout))
		end
		local dim
		if i == 1 then
			dim = opt.dim_input
		else
			dim = opt.vardim
		end
		rec_base:add(nn.Maxout(dim,opt.vardim,opt.maxoutWindow))
	end
	print('Recognition Network')
	print(rec_base)
	local x = nn.Identity()()
	local mu        = nn.Linear(opt.vardim,opt.dim_stochastic)(rec_base(x))
	local logsigma  = nn.Linear(opt.vardim,opt.dim_stochastic)(rec_base(x))
	local rec_model = nn.gModule({x},{mu,logsigma})

	local reparam
	if opt.len_normflow > 0 then
		reparam     = nn.GaussianReparamFlow(opt.dim_stochastic,opt.reparamnoise)
	else
		reparam     = nn.GaussianReparam(opt.dim_stochastic,opt.reparamnoise)
	end

	local x         = nn.Identity()()
	local z         = reparam(rec_model(x))
	local q0_model  = nn.gModule({x},{z})
	local qp, _     = q0_model:getParameters()
	qp:randn(qp:size()):mul(opt.init_std);


	-- add normalizing flow
	local var_inp = nn.Identity()()
	if opt.len_normflow > 0 then
		require(opt.flow_type)
		flow = nn.NormalizingFlow()
		for k = 1, opt.len_normflow do
			flow:add(nn.PlanarFlow(opt.dim_stochastic))
		end
		fp, _ = flow:getParameters()
		fp:uniform(-0.5,0.5):mul(opt.flow_init);

		var_model = nn.gModule({var_inp},{flow(q0_model(var_inp))})
	else
		var_model = nn.gModule({var_inp},{q0_model(var_inp)})
	end

	--------- Generative Network -------------
	local gen_model = nn.Sequential()
	for i = 1, opt.genlayers do
		if opt.dropout then
			gen_model:add(nn.Dropout(opt.dropout))
		end
		local dim
		if i == 1 then
			dim = opt.dim_stochastic
		else
			dim = opt.gendim
		end
		gen_model:add(nn.Maxout(dim,opt.gendim,opt.maxoutWindow))
	end
	gen_model:add(nn.Linear(opt.gendim,opt.dim_input))
	gen_model:add(nn.Sigmoid())

	print('Generative Network')
	print(gen_model)

	local gp, _ = gen_model:getParameters()
	gp:randn(gp:size()):mul(opt.init_std);

	----- Combining Models into Single MLP----
	local inp = nn.Identity()()
	mlp = nn.gModule({inp},{gen_model(var_model(inp))})
	crit= nn.BCECriterion()
	crit.sizeAverage = false

	mlp:cuda()
	crit:cuda()

	---------- Setup Parameters ----------------
	parameters, gradients = mlp:getParameters()

	---------- Store Model ----------------
	model             = {}
	model.mu          = mu
	model.logsigma    = logsigma
	model.reparam     = reparam
	model.z           = z
	model.q0_model    = q0_model
	model.flow        = flow
	model.flow_model  = flow_model
	model.rec_model   = rec_model
	model.var_model   = var_model
	model.gen_model   = gen_model
	model.mlp         = mlp
	model.crit        = crit
	model.parameters  = parameters
	model.gradients   = gradients
	model.opt    = opt

	return model
end

