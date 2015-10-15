devicenum = 2
len_normflow = 5
deepParams = false
myname = 'planarflow5'
savedir = '/scratch/jmj/dgm_maxout40/save_planarflow5'
optimMethod = 'adam'
annealing = false
optimconfig = {
    learningRate = 5e-5,
	}
dofile 'train_planarflow.lua'

