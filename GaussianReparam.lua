require 'nn'

local GaussianReparam, parent = torch.class('nn.GaussianReparam', 'nn.Module')
--Based on JoinTable
function GaussianReparam:__init(dimension,noiseparam)
    parent.__init(self)
    self.size = torch.LongStorage()
    self.dimension = dimension
    self.gradInput = {}
	self.noiseparam = noiseparam or 0.05
	self.train = true
	self.KL = 0
	self.warmup = 1
end 

function GaussianReparam:fixeps(fixed)
	self.fixed = fixed
end


function GaussianReparam:setWarmUp(warmup)
	self.warmup = warmup or 1
end

--Forward pass
function GaussianReparam:updateOutput(input)
	local warmup = self.warmup or 1
	if input[1]:dim()==1 then  --SGD setting 
		if not self.dimension then self.dimension = input[1]:size(1) end
	elseif input[1]:dim()==2 then --Batch setting 
		if not self.dimension then self.dimension = input[1]:size(2) end 
	else
		error('Input must be a vector or a matrix')
	end	
	--Treat input[2] as log sigma^2
	if (not self.fixed) or (not self.eps) then
		self.eps = torch.randn(input[1]:size()):typeAs(input[1])
	end

	self.output = torch.exp(input[2]*0.5):cmul(self.eps):add(input[1])
	local kl = (input[2] + 1):mul(-1):add(torch.pow(input[1],2)):add(torch.exp(input[2]))
	self.KL = kl:sum()*0.5*warmup

	--Add noise to output during training 
	if self.train then
		local noise = torch.randn(input[1]:size()):typeAs(input[1])
		self.output:add(noise*self.noiseparam)
	end
    return self.output
end


--Backward pass
function GaussianReparam:updateGradInput(input, gradOutput)
	local warmup = self.warmup or 1
	--Gradient with respect to mean
	self.gradInput[1]= (gradOutput+input[1]*warmup)
	--Gradient with respect to R
	self.gradInput[2]=torch.mul(input[2],0.5):exp():mul(0.5):cmul(self.eps):cmul(gradOutput)
	local grad_R = (torch.exp(input[2])-1)*(0.5*warmup)
	self.gradInput[2]:add(grad_R)
    return self.gradInput
end
