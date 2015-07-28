require 'nn'

local GaussianReparam, parent = torch.class('nn.GaussianReparam', 'nn.Module')
--Based on JoinTable
function GaussianReparam:__init(dimension,noiseparam)
    parent.__init(self)
    self.size = torch.LongStorage()
    self.dimension = dimension
    self.gradInput = {}
	self.noiseparam = noiseparam or 0.05
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
	local kl = (input[2] + 1):mul(-1):add(torch.pow(input[1],2)):add(torch.exp(input[2]))
	self.KL = kl:sum()*0.5

	--Add noise to output 
	self.output:add(noise*self.noiseparam)
	--Also output statistics
	local stats = {}
	stats.norm_mu = torch.norm(input[1])
	stats.norm_R  =  torch.norm(input[2]-1)
	stats.KL = self.KL 
    return {self.output,stats}
end

--Backward pass
function GaussianReparam:updateGradInput(input, gradOutput)
	--Gradient with respect to mean
	self.gradInput[1]= gradOutput+input[1]
	--Gradient with respect to R
	self.gradInput[2]=torch.mul(input[2],0.5):exp():mul(0.5):cmul(self.eps):cmul(gradOutput)
	local grad_R = (torch.exp(input[2])-1)*0.5
	self.gradInput[2]:add(grad_R)
    return self.gradInput
end
