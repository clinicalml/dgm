require 'nn'

local Maxout, parent = torch.class('nn.Maxout', 'nn.Module')

function Maxout:__init(inputDimension,outputDimension,window)
	parent.__init(self)
	self.window = window or 4
	self.inputDim = inputDimension
	self.outputDim = outputDimension

	assert(inputDimension ~= nil, "must specify input dimension")
	assert(outputDimension ~= nil, "must specify output dimension")
	
	local maxout = nn.Sequential()
	maxout:add(nn.Linear(self.inputDim,self.outputDim*self.window))
	maxout:add(nn.View(self.outputDim*self.window,1))
	maxout:add(nn.TemporalMaxPooling(self.window,self.window))
	maxout:add(nn.View(self.outputDim))
	self.maxout = maxout	
end 

function Maxout:parameters()
	return self.maxout:parameters()
end

function Maxout:cuda()
	self.maxout = self.maxout:cuda()
	return self:type('torch.CudaTensor')
end

--Forward pass
function Maxout:updateOutput(input)	
	self.output = self.maxout:forward(input)
    return self.output
end


--Backward pass
function Maxout:updateGradInput(input, gradOutput)
	self.gradInput = self.maxout:backward(input,gradOutput)
    return self.gradInput
end





