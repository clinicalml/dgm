devicenum = 1
len_normflow = 5
deepParams = true
myname = 'planarflow5deep'
savedir = '/scratch/jmj/dgm_maxout40/save_planarflow5deep'
optimMethod = 'adam'
annealing = false
optimconfig = {
    learningRate = 5e-5,
	}
dofile 'train_planarflow.lua'

