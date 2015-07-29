require "paths"
-- Initial file to process command line options
print '==> processing options'
cmd = torch.CmdLine()
cmd:text()
cmd:text('Training DGM')
cmd:text()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'Random Seed')
cmd:option('-threads', 2, '# of threads')
-- data:
cmd:option('-data', 'MNIST', 'dataset: MNIST (no others currently)')
-- model:
cmd:option('-layers', 2, 'Number of stochastic layers in DGM')
cmd:option('-dimhid', 200,'Dimensionality of hidden units')
cmd:option('-dimstoc', 200,'Dimensionality of hidden units')
-- training:
cmd:option('-save', './checkpoint', 'subdirectory to save/log experiments in')
cmd:option('-opt', 'RMSPROP', 'optimization method: ADAGRAD | RMSPROP')
cmd:option('-lr', 1e-4, 'learning rate at t=0')
cmd:option('-decay_rate', 0.95, 'decay of learning rate')
cmd:option('-batch', 200, 'mini-batch size (1 = pure stochastic)')
cmd:option('-gpu', 1, '0 (CPU) | 1 | 2 ... ')
cmd:option('-nonlinearity','Tanh','ReLU|Tanh')
cmd:option('-expt', 'expt', 'Experiment name')
cmd:option('-epochs',1000,'Maximum number of epochs')
cmd:text()
opt = cmd:parse(arg or {})
--Add gpu number to experiment name 
opt.expt = opt.expt .. '_maxepochs' .. opt.epochs .. '_layers' .. opt.layers .. '_dimhidden' .. opt.dimhid ..'_dimstoc'..opt.dimstoc..'_lr'..opt.lr..'_q_embed'..tostring(opt.q_embed)
--Setup for cuda 
if opt.gpu>0 then 
	opt.cuda = true 
else
	opt.cuda = false 
end
if opt.cuda then 
	require "cunn"
	require "cutorch"
	cutorch.setDevice(opt.gpu)
end
torch.setnumthreads(opt.threads)
torch.manualSeed(opt.seed)
--Start training 
print (opt)
dofile("train-dgm.lua")
