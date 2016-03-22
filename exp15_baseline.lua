cmd = torch.CmdLine()
cmd:option('-devicenum', 1)
cmd:option('-genlayers', 1)
cmd:option('-gendim', 200)
cmd:option('-varlayers', 1)
cmd:option('-vardim', 200)
cmd:option('-dim_stochastic', 40)
cmd:option('-reparamnoise',0.05)
cmd:option('-learningrate',5e-4)
cmd:option('-lrdecay',0)
cmd:option('-numepochs',2000)
cmd:option('-beta1',0.9)
cmd:option('-beta2',0.999)
cmd:option('-init_std',1e-1)
cmd:option('-numsamples',1)
cmd:option('-warmup',false)
cmd:option('-warmup_epochs',200)


opt = cmd:parse(arg)
--opt.devicenum      = 1
--opt.genlayers      = 1
--opt.gendim         = 200
--opt.varlayers      = 1
--opt.vardim         = 200
--opt.dim_stochastic = 40
opt.len_normflow   = 0
opt.annealing      = false
opt.dropout        = false

opt.rootdir        = '/scratch/jmj/dgm_maxout40/'
opt.experimentdir  = 'exp15'

opt.experiment     = tostring(opt.genlayers) .. 'g' .. tostring(opt.gendim) .. '_'
opt.experiment     = opt.experiment .. tostring(opt.varlayers) .. 'v' .. tostring(opt.vardim) .. '_'
opt.experiment     = opt.experiment .. tostring(opt.dim_stochastic) .. 's_'
opt.experiment     = opt.experiment .. 'baseline'
opt.experiment     = opt.experiment .. '_noise' .. tostring(opt.reparamnoise)
opt.experiment     = opt.experiment .. '_lr' .. tostring(opt.learningrate)
if opt.lrdecay > 0 then
    opt.experiment = opt.experiment .. 'decay' .. tostring(opt.lrdecay)
end
opt.experiment     = opt.experiment .. '_' .. tostring(opt.beta1) .. 'b1'
opt.experiment     = opt.experiment .. '_' .. tostring(opt.beta2) .. 'b2'
opt.experiment     = opt.experiment .. '_init' .. tostring(opt.init_std)
opt.experiment     = opt.experiment .. '_samples' .. tostring(opt.numsamples)
if opt.warmup then
	opt.experiment = opt.experiment .. '_warmup' .. opt.warmup_epochs
end

opt.savedir        = paths.concat(opt.rootdir,opt.experimentdir,opt.experiment)
opt.datadir        = '/scratch/jmj/dgm_maxout40'

opt.optimMethod = 'adam'
opt.optimconfig = {
    learningRate = opt.learningrate,
    learningRate0 = opt.learningrate,
    learningRateDecay = opt.lrdecay,
	beta1 = opt.beta1,
	beta2 = opt.beta2
}

sys.execute('mkdir -p ' .. opt.savedir)

opt.trainfile = 'train_planarflow.lua'
opt.modelfile = 'model.lua'
opt.runfile   = string.sub(tostring(debug.getinfo(1,'S').source),2)

sys.execute('cp ' .. opt.trainfile .. ' ' .. opt.savedir)
sys.execute('cp ' .. opt.runfile   .. ' ' .. opt.savedir)
sys.execute('cp ' .. opt.modelfile .. ' ' .. opt.savedir)

dofile(opt.modelfile)

model = loadmodel(opt)

dofile(opt.trainfile)

