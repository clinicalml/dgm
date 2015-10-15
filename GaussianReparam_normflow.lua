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
end 

--Forward pass
function GaussianReparam:updateOutput(input)	
	if input[1]:dim()==1 then  --SGD setting 
		if not self.dimension then self.dimension = input[1]:size(1) end
	elseif input[1]:dim()==2 then --Batch setting 
		if not self.dimension then self.dimension = input[1]:size(2) end 
	else
		error('Input must be a vector or a matrix')
	end	
	--Treat input[2] as log sigma^2
	self.eps = torch.randn(input[1]:size()):typeAs(input[1])
	local noise = torch.randn(input[1]:size()):typeAs(input[1])

	self.output = torch.exp(input[2]*0.5):cmul(self.eps):add(input[1])
	local kl = (input[2] + 1):mul(-1)
	self.KL = kl:sum(2)*0.5 - 0.5*self.dimension*math.log(2*math.pi)

	--Add noise to output during training 
	if not self.train then
		noise:fill(0)
	end
	self.output:add(noise*self.noiseparam)
    return self.output
end

--Backward pass
function GaussianReparam:updateGradInput(input, gradOutput)
	--Gradient with respect to mean
	self.gradInput[1]= gradOutput
	--Gradient with respect to R
	self.gradInput[2]=torch.mul(input[2],0.5):exp():mul(0.5):cmul(self.eps):cmul(gradOutput)
	local grad_R = torch.ones(input[2]:size())*(-0.5)
	self.gradInput[2]:add(grad_R:type(input[2]:type()))
    return self.gradInput
end
