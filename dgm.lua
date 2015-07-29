require "nn"
require "nngraph"
require "GaussianReparam"
clone_utils = require "cloneUtils.lua"

local DGM = torch.class("DGM")
function DGM:__init(layers,dim_input,dim_hidden,dim_stochastic)
    print ('----------Creating DGM with : '..layers..' layers '..' dim_hidden: '..dim_hidden..' dim_stochastic: '..dim_stochastic..'--------')
	-- Setup Constants --
    self.layers = layers
    self.dim_hidden = dim_hidden 
    self.dim_stochastic = dim_stochastic
    self.nonlinearity   = nn.Tanh
    if opt.nonlinearity == 'ReLU' then 
        self.nonlinearity = nn.ReLU
    end
	-- Setup Models --
    self.var_model = self:getVarModel(dim_input,dim_hidden,dim_stochastic) 
    self.gen_model = self:getGenModel(dim_input,dim_hidden,dim_stochastic) 
    -- Criterion --
    self.crit = nn.BCECriterion()
    self.crit.sizeAverage = false
	if opt.cuda then 
		print('Putting model on GPU')
		self.gen_model:cuda()
		self.var_model:cuda()
		self.crit:cuda()
	end
	-- Combine Parameters --
	self.w, self.dw = clone_utils.combine_all_parameters(self.gen_model,self.var_model)
	-- Initialize Weights --
    self.w:uniform(-0.05,0.05) 
	collectgarbage()
end

--Setup variational model
function DGM:getVarModel(dim_input,dim_hidden,dim_stochastic)
    local split = nn.ConcatTable()
    split:add(nn.Linear(dim_hidden,dim_stochastic))
    split:add(nn.Linear(dim_hidden,dim_stochastic))
    local rpm_mod = nn.Sequential()
    rpm_mod:add(split:clone())
    rpm_mod:add(nn.GaussianReparam(dim_stochastic))
    --Variational Model 
    local input_var = nn.Identity()()
    local dropped_in = nn.Dropout(0.1)(input_var)
    local xis = {} 
    local hid = {}
    for i=1,self.layers do 
        if i==1 then 
			local l1 = self.nonlinearity()(nn.Linear(dim_input,dim_hidden)(dropped_in))
            local l2 = nn.Dropout()(self.nonlinearity()(nn.Linear(dim_hidden,dim_hidden)(l1)))
            local l3 = self.nonlinearity()(nn.Linear(dim_hidden,dim_hidden)(l2))
            hid[i] = l3
        else
           	hid[i] = self.nonlinearity()(nn.Linear(dim_hidden,dim_hidden)(hid[i-1]))
        end
        xis[i] = rpm_mod:clone()(hid[i])
    end
	return nn.gModule({input_var},xis)
end

--Setup generative model
function DGM:getGenModel(dim_input,dim_hidden,dim_stochastic)
    --Generative model
    local mlp_gen = nn.Sequential()
    mlp_gen:add(nn.Linear(dim_hidden,dim_hidden))
    mlp_gen:add(self.nonlinearity())
    local input_gen = nn.Identity()()
    if self.layers>1 then 
        xi_in = {input_gen:split(self.layers)} 
    else
        xi_in = {input_gen} 
    end
    local hid = {}
    for i=self.layers,1,-1 do
        if i==self.layers then
            hid[i] = nn.Linear(dim_stochastic,dim_hidden)(xi_in[i])
        else
            hid[i] = nn.CAddTable()({mlp_gen:clone()(hid[i+1]),nn.Linear(dim_stochastic,dim_hidden)(xi_in[i])})
        end
    end
    local output_gen 
    if self.layers > 1 then 
		local inp = hid[1]
		local l1  = self.nonlinearity()(nn.Linear(dim_stochastic,dim_hidden)(inp))
		local l2  = self.nonlinearity()(nn.Linear(dim_hidden,dim_hidden)(l1))
		local l3  = self.nonlinearity()(nn.Linear(dim_hidden,dim_input)(l2))
        output_gen = nn.Sigmoid()(l3)
    else 
		local inp = xi_in[1]
		local l1  = self.nonlinearity()(nn.Linear(dim_stochastic,dim_hidden)(inp))
		local l2  = self.nonlinearity()(nn.Linear(dim_hidden,dim_hidden)(l1))
		local l3  = self.nonlinearity()(nn.Linear(dim_hidden,dim_hidden)(l2))
		local l4  = self.nonlinearity()(nn.Linear(dim_hidden,dim_input)(l3))
        output_gen = nn.Sigmoid()(l4)
    end
    return  nn.gModule({input_gen},{output_gen})
end

--Compute forward and backward pass
function DGM:forwardAndBackward(batch)
    --Zero gradients
    self.gen_model:zeroGradParameters()
    self.var_model:zeroGradParameters()
    --Do forward and backward pass
    local etas_and_kl
	local batch_fwd 
	batch_fwd = batch
	etas_and_kl = self.var_model:forward(batch_fwd)
    local KL = 0
    local etas = {}
    if self.layers >1 then 
        for k=1,self.layers do
            etas[k] = etas_and_kl[k][1]
            local stats = etas_and_kl[k][2]
            KL = KL + stats.KL 
        end
    else
        etas = etas_and_kl[1]
        local stats = etas_and_kl[2]
        KL = KL + stats.KL 
    end
    local probs= self.gen_model:forward(etas)
    local nll = self.crit:forward(probs,batch)
    local grad_gen_model_output = self.crit:backward(probs,batch) 
    local grad_gen_model = self.gen_model:backward(etas,grad_gen_model_output)
    local grad_var_model = self.var_model:backward(batch_fwd,grad_gen_model)
    --lower bound 
    local lb = nll + KL 
    local stats = {}
	stats.probs = probs
    stats.KL = KL 
    stats.nll = nll 
    return stats,lb,self.dw
end

--Compute NLL on dataset 
function DGM:getNLL(dataset)
    --Do forward  pass
    local etas_and_kl
	local batch = dataset
    etas_and_kl = self.var_model:forward(batch)
    local KL = 0
    local etas = {}
    if self.layers>1 then 
        for k=1,self.layers do
            etas[k] = etas_and_kl[k][1]
        end
    else
        etas = etas_and_kl[1]
    end
    local probs= self.gen_model:forward(etas)
    local nll = self.crit:forward(probs,dataset)
    return nll/dataset:size(1),probs
end

--Generate samples from model 
function DGM:genSamples(nsamples,gen_probs)
    local sample = {}
    for ctr =1,nsamples do
        local xis_sample = {}
        if self.layers>1 then 
            for l=1,self.layers do
            	xis_sample[l] = torch.randn(self.dim_stochastic):typeAs(self.dw)
            end
        else
          	xis_sample= torch.randn(self.dim_stochastic):typeAs(self.dw)
        end
        sample[ctr] = self.gen_model:forward(xis_sample)
        if gen_probs then 
            sample[ctr] = sample[ctr]:double():reshape(28,28)
        else
            sample[ctr] = torch.bernoulli(sample[ctr]:float())
        end
    end
    return sample 
end

--Get handles to parameters
function DGM:getParameters()
    return self.w,self.dw
end

