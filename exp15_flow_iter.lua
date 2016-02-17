cmd = torch.CmdLine()
cmd:option('-devicenum', 1)
cmd:option('-genlayers', 1)
cmd:option('-gendim', 200)
cmd:option('-varlayers', 1)
cmd:option('-vardim', 200)
cmd:option('-dim_stochastic', 40)
cmd:option('-len_normflow',  10)
cmd:option('-iter', 150)
cmd:option('-finalepochs',1000)
cmd:option('-reparamnoise',0.05)
cmd:option('-learningrate',5e-4)
cmd:option('-annealing',false)


opt = cmd:parse(arg)
--opt.devicenum      = 1
--opt.genlayers      = 1
--opt.gendim         = 200
--opt.varlayers      = 1
--opt.vardim         = 200
--opt.dim_stochastic = 40
--opt.len_normflow   = 10
opt.flow_init      = 3
--opt.annealing      = false
opt.dropout        = false
opt.epochs         = {opt.iter,opt.finalepochs}

opt.rootdir        = '/scratch/jmj/dgm_maxout40/'
opt.experimentdir  = 'exp15'

opt.experiment     = tostring(opt.genlayers) .. 'g' .. tostring(opt.gendim) .. '_'
opt.experiment     = opt.experiment .. tostring(opt.varlayers) .. 'v' .. tostring(opt.vardim) .. '_'
opt.experiment     = opt.experiment .. tostring(opt.dim_stochastic) .. 's_'
opt.experiment     = opt.experiment .. 'flow' .. tostring(opt.len_normflow) 
opt.experiment     = opt.experiment .. '_iter' .. tostring(opt.epochs[1])
opt.experiment     = opt.experiment .. '_noise' .. tostring(opt.reparamnoise)
opt.experiment     = opt.experiment .. '_lr' .. tostring(opt.learningrate)

if opt.annealing then
	opt.experiment = opt.experiment .. '_annealed'
end

opt.savedir        = paths.concat(opt.rootdir,opt.experimentdir,opt.experiment)
opt.datadir        = '/scratch/jmj/dgm_maxout40'

opt.optimMethod = 'adam'
opt.optimconfig = {
	learningRate = opt.learningrate,
	momentum = 0.9
}

sys.execute('mkdir -p ' .. opt.savedir)

opt.trainfile = 'train_planarflow.lua'
opt.modelfile = 'model.lua'
opt.runfile   = string.sub(tostring(debug.getinfo(1,'S').source),2)
opt.iterfile  = 'iterative_training2.lua'

sys.execute('cp ' .. opt.trainfile .. ' ' .. opt.savedir)
sys.execute('cp ' .. opt.runfile   .. ' ' .. opt.savedir)
sys.execute('cp ' .. opt.modelfile .. ' ' .. opt.savedir)
sys.execute('cp ' .. opt.iterfile  .. ' ' .. opt.savedir)

dofile(opt.iterfile)

iterative_training(opt)

