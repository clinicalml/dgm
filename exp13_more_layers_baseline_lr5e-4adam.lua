devicenum = 1
opt = {}
opt.savedir = '/scratch/jmj/dgm_maxout40/exp13_baseline_morelayers_lr5e-4adam'
opt.datadir = '/scratch/jmj/dgm_maxout40'
opt.optimMethod = 'adam'
opt.optimconfig = {
	learningRate = 5e-4,
	momentum = 0.9
}
opt.len_normflow = 0
opt.dim_hidden = 100
opt.dim_stochastic = 40

opt.trainfile = 'train_planarflow.lua'
sys.execute('mkdir -p ' .. opt.savedir)
sys.execute('cp ' .. opt.trainfile .. ' ' .. opt.savedir)
dofile(opt.trainfile)

