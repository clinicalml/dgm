--Load modules --
require "nn"
require 'sys'
require 'xlua'
require 'paths'
disp = require 'display'
require "utils"
require "optim"
require "image"
require "dgm"
if not opt then
    require "starter"
end
--Dataset--
if opt.data == 'MNIST' then 
    data = loadMNIST(opt.cuda)
else
	error('Invalid dataset specified')
end
-- Get model 
dgm = DGM(opt.layers,data.dim_input,opt.dimhid,opt.dimstoc)
-- Optimization --
N = data.train_x:size(1) - (data.train_x:size(1) % opt.batch)
batch = torch.Tensor(opt.batch,data.train_x:size(2)):typeAs(data.train_x)
paramx,paramdx = dgm:getParameters()
config = {
    learningRate = opt.lr,
	alpha = opt.decay_rate
}
state = {}
-- Setup to save results -- 
setupFolder(opt.save)
img_format,format = setupDisplay()
local infostr = 'la'..opt.layers..'hid'..opt.dimhid..'stoc'..opt.dimstoc..'lr'..opt.lr..'nlin'..opt.nonlinearity
-- Train model --
print ('Beginning training.....')
for epoch = 1,opt.epochs do
	-- Statistics to track 
    local var_ub,KL,nll = 0,0,0
    local time = sys.clock()
    local shuffle = torch.randperm(data.train_x:size(1))
	-- Pass through data --
    for i = 1, N, opt.batch do
        xlua.progress(i+opt.batch-1, N)
        k=1
        for j = i,i+opt.batch-1 do
            batch[k] = data.train_x[shuffle[j]]:clone()
            k = k + 1
        end
        local opfunc = function(x)
            if x ~= paramx then
                paramx:copy(x)
            end
            stats,fx,dfdx = dgm:forwardAndBackward(batch)
            -- Track stats -- 
            probs = stats.probs
            KL = KL + stats.KL
            nll = nll + stats.nll
            return fx,dfdx 
        end
        -- Choice of optimizer --
        if opt.opt == 'ADAGRAD' then 
            paramx, batchlowerbound = optim.adagrad(opfunc, paramx, config, state)
        elseif opt.opt == 'RMSPROP' then
            paramx, batchlowerbound = optim.rmsprop(opfunc, paramx, config, state)
        else
            error('Invalid optimization specified')
        end
        -- Track ub --
        var_ub = var_ub + batchlowerbound[1]
    end
    --Track statistics and output 
    local time_taken = sys.clock() - time
    KL,nll,var_ub = KL/N,nll/N,var_ub/N
    print("\nEpoch: " .. epoch .. " Lowerbound: " .. var_ub .." Time: " .. time_taken)
	print ("||w|| : ",torch.norm(paramx)," ||dw||: ",torch.norm(paramdx))
	print (" Avg. KL: "..KL.. " Avg. nll: "..nll)

    KLlist = appendToTensor(KLlist,KL)
    timelist = appendToTensor(timelist,time_taken)
    ublist = appendToTensor(ublist,var_ub)

    --Display output to server and save parameters
    if epoch % 5 == 0 then 
		local randidx = torch.randperm(opt.batch)
		local trainimgs = {} 
		for ctr = 1,20 do  
			idx = randidx[ctr]
			trainimgs[ctr] = torch.cat(probs[idx]:reshape(28,28):float(),batch[idx]:reshape(28,28):float(),2)
		end
		img_format.title = 'Trainlayers'..infostr
        img_format.win = id_trainimgs
        id_trainimgs = disp.images(trainimgs,img_format)

        --Track variational bound
        local len = ublist:size(1)
        format.win = id_vb 
        format.title = 'Var. Bound'
        id_vb = disp.plot(torch.cat(torch.linspace(1,len,len),ublist:reshape(len),2), format)

        --Sample from the model 
        local samples = dgm:genSamples(math.min(25),true)
        img_format.win = id_samples
        img_format.title = 'Samples | '..infostr 
        id_samples = disp.images(samples,img_format)

        --Track predictions on test set
        local nll,predictions = dgm:getNLL(data.test_x)
        print ("Test NLL: ",nll)
        testnll = appendToTensor(testnll,nll)
        len = testnll:size(1)
        format.win = id_testnll 
        format.title = 'Test NLL'
        id_testnll = disp.plot(torch.cat(torch.linspace(1,len,len),testnll:reshape(len),2), format)
    end

    if epoch %50 == 0 then 
        --Save tensor 
		local savedata = {}
		savedata.titlestr = infostr
		savedata.time_taken = torch.Tensor(timelist)
		savedata.upperbound = torch.Tensor(ublist)
        local samples = dgm:genSamples(25)
		savedata.samples    = samples
        torch.save(opt.save..'/data_'.. opt.expt .. '.t7',savedata)
		savedata = nil
        collectgarbage()
    end
end
