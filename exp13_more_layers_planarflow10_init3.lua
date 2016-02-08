opt = {}
opt.devicenum = 2
opt.len_normflow = 10
opt.deepParams = false
opt.myname = 'planarflow10_init3_PF3_more_layers'
opt.savedir = '/scratch/jmj/dgm_maxout40/exp13_more_layers_planarflow10_init3_PF3'
opt.datadir = '/scratch/jmj/dgm_maxout40/'
opt.optimMethod = 'adam'
opt.annealing = false
opt.optimconfig = {
    learningRate = 5e-4,
	momentum = 0.9
	}
opt.flow_init = 3
opt.dim_hidden = 100
opt.dim_stochastic = 40

opt.trainfile = 'train_planarflow.lua'
sys.execute('mkdir -p ' .. opt.savedir)
sys.execute('cp ' .. opt.trainfile .. ' ' .. opt.savedir)
dofile(opt.trainfile)

