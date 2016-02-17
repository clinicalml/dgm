require "nn"
require "xlua"
require "nngraph"
require "utils"
require "cunn"
require "cutorch"
require "optim"
require "GaussianReparam_normflow"
require "PlanarFlow"
require "NormalizingFlow"
require "Maxout"
require 'hdf5'
disp = require "display"

opt.devicenum = opt.devicenum or 2
cutorch.setDevice(opt.devicenum)
torch.manualSeed(1)
cutorch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')
---------------- Experiment Params ------------
opt               = opt or {}
opt.myname        = opt.myname or "iterative_training"
opt.savedir       = opt.savedir or "/scratch/jmj/dgm_maxout40/save_iterative_training"
opt.savebestmodel = opt.savebestmodel or true
opt.datadir       = opt.datadir or '/scratch/jmj/dgm_maxout40/'

opt.optimMethod   = opt.optimMethod or 'adam'
opt.optimconfig0  = opt.optimconfig0 or {learningRate = 5e-3,
										momentum = 0.9}
opt.epochs        = opt.epochs or {50,500}

---------------- Model Params. -----------
opt.dim_input      = 784
opt.dim_hidden     = opt.dim_hidden or 200
opt.dim_stochastic = opt.dim_stochastic or 40
opt.dropout        = opt.dropout or false
opt.maxoutWindow   = opt.maxoutWindow or 4
opt.deepParams     = opt.deepParams or false
opt.len_normflow   = opt.len_normflow or 10
opt.annealing      = opt.annealing or false
opt.S              = opt.S or 200  -- num samples used to estimate -log(p(v)) in estimate_logpv
opt.init_std       = opt.init_std or 1e-3
opt.flow_init      = opt.flow_init or 3

------------- Save optimization parameters ----------
function deepcopy(x)
	if type(x) == 'table' then
		local y = {}
		for k,v in pairs(x) do
			y[k] = deepcopy(v)
		end
		return y
	else
		return x
	end
end

for L = 1, opt.len_normflow do
	print('=============> L =',L)
	--------- Reset optimization parameters --------
	opt.optimconfig = deepcopy(opt.optimconfig0)

	--------- Set number of epochs for each iteration ----------
	if L < opt.len_normflow then
		opt.numepochs = opt.epochs[1]
	else
		opt.numepochs = opt.epochs[2]
	end
	--------- Load Params from Previous Training Iteration -----------
	local bqp,bfp,bgp
	if L > 1 then
		bqp,_ = model.q0_model:getParameters()
		bfp = {}
		for t = 1, L-1 do
			bfp[t],_ = model.flow.modules[t]:getParameters()
		end
		bgp,_ = model.gen_model:getParameters()
	end

	--------- Recognition. Network -----------
	local dim_input       = opt.dim_input
	local dim_hidden      = opt.dim_hidden
	local dim_stochastic  = opt.dim_stochastic
	local maxoutWindow    = opt.maxoutWindow
	q0_inp = nn.Identity()()
    if opt.dropout then
        q_hid_1 = nn.Maxout(dim_input,dim_hidden,maxoutWindow)(nn.Dropout(opt.dropout)(q0_inp))
        q_hid_2 = nn.Maxout(dim_hidden,dim_hidden,maxoutWindow)(nn.Dropout(opt.dropout)(q_hid_1))
        q_hid_3 = nn.Maxout(dim_hidden,dim_hidden,maxoutWindow)(nn.Dropout(opt.dropout)(q_hid_2))
    else
        q_hid_1 = nn.Maxout(dim_input,dim_hidden,maxoutWindow)(q0_inp)
        q_hid_2 = nn.Maxout(dim_hidden,dim_hidden,maxoutWindow)(q_hid_1)
        q_hid_3 = nn.Maxout(dim_hidden,dim_hidden,maxoutWindow)(q_hid_2)
    end
	mu  = nn.Linear(dim_hidden,dim_stochastic)(q_hid_3)
	logsigma  = nn.Linear(dim_hidden,dim_stochastic)(q_hid_3)
	reparam   = nn.GaussianReparam(dim_stochastic)
	z  = reparam({mu,logsigma})
	q0_model = nn.gModule({q0_inp},{z})

	-- initialize recognition network by copying baseline model parameters
	qp,_ = q0_model:getParameters()
	if L > 1 then
		qp:copy(bqp)
	else
		qp:copy(torch.randn(qp:size()):mul(opt.init_std))
	end

	--------- Normalizing Flow -------------
	if L > 0 then
		flow_inp = nn.Identity()()
		flow = {}
		flow[0] = flow_inp
		flow = nn.NormalizingFlow()
		for k = 1,L do
			flow:add(nn.PlanarFlow(dim_stochastic))
		end
		for k = 1,L-1 do
			local fp,fdp = flow.modules[k]:getParameters()
			fp:copy(bfp[k])
		end
		local fpL, _ = flow:getParameters()
		fpL:uniform(-0.5,0.5):mul(opt.flow_init)

		local var_inp = nn.Identity()()
		var_model = nn.gModule({var_inp},{flow(q0_model(var_inp))})
	else
		local var_inp = nn.Identity()()
		var_model = nn.gModule({var_inp},{q0_model(var_inp)})
	end

	--------- Generative Network -------------
	gen_inp = nn.Identity()()
    if opt.dropout then
        hid1 = nn.Maxout(dim_stochastic,dim_hidden,maxoutWindow)(nn.Dropout(opt.dropout)(gen_inp))
        hid2 = nn.Maxout(dim_hidden,dim_hidden,maxoutWindow)(nn.Dropout(opt.dropout)(hid1))
        hid3 = nn.Maxout(dim_hidden,dim_hidden,maxoutWindow)(nn.Dropout(opt.dropout)(hid2))
        hid4 = nn.Maxout(dim_hidden,dim_hidden,maxoutWindow)(nn.Dropout(opt.dropout)(hid3))
        reconstr = nn.Sigmoid()(nn.Linear(dim_hidden,dim_input)(hid4))
    else
        hid1 = nn.Maxout(dim_stochastic,dim_hidden,maxoutWindow)(gen_inp)
        hid2 = nn.Maxout(dim_hidden,dim_hidden,maxoutWindow)(hid1)
        hid3 = nn.Maxout(dim_hidden,dim_hidden,maxoutWindow)(hid2)
        hid4 = nn.Maxout(dim_hidden,dim_hidden,maxoutWindow)(hid3)
        reconstr = nn.Sigmoid()(nn.Linear(dim_hidden,dim_input)(hid4))
    end
	gen_model = nn.gModule({gen_inp},{reconstr})

	-- copy baseline model parameters
	gp,_ = gen_model:getParameters()
	if L > 1 then
		gp:copy(bgp)
	else
		gp:copy(torch.randn(gp:size()):mul(opt.init_std))
	end

	----- Combining Models into Single MLP----
	local inp = nn.Identity()()
	mlp = nn.gModule({inp},{gen_model(var_model(inp))})
	crit= nn.BCECriterion()
	crit.sizeAverage = false
	mlp:cuda()
	crit:cuda()


	---------- Setup Parameters ----------------
	parameters, gradients = mlp:getParameters()
	gradients:zero()

	---------- Store Model ----------------
	model             = {}
	model.mu          = mu
	model.logsigma    = logsigma
	model.reparam     = reparam
	model.z           = z
	model.q0_model    = q0_model
	model.flow        = flow
	model.var_model   = var_model
	model.gen_model   = gen_model
	model.mlp         = mlp
	model.crit        = crit
	model.parameters  = parameters
	model.gradients   = gradients

	baseline = nil
	collectgarbage()

	---------- Run ----------------
	dofile 'train_planarflow.lua'
end
