require 'nn';

local NormalizingFlow, parent = torch.class('nn.NormalizingFlow', 'nn.Sequential')

function NormalizingFlow:__init()
	parent.__init(self)

	self.KL = torch.Tensor()
	self.logdetJ = torch.Tensor()
	self.logpz = torch.Tensor()
	self.beta = 1
	self.gradOutput = torch.Tensor()
end


function NormalizingFlow:getlogpz()
	local output = self.output
	if output:nDimension() == 1 then
		local dim = output:size(1)
		self.logpz:resize(1)
		self.logpz:fill(torch.dot(output,output)):add(dim*torch.log(2*math.pi)):mul(0.5)
	elseif output:nDimension() == 2 then
		local batchsize = output:size(1)
		local dim = output:size(2)
		self.logpz:resize(batchsize)
		self.logpz:copy(torch.pow(output,2):sum(2)):add(dim*torch.log(2*math.pi)):mul(0.5)
	else
		error('cannot handle 3d output')
	end
	return self.logpz
end

function NormalizingFlow:getlogdetJ()
	if self.output:nDimension() == 1 then
		self.logdetJ:resize(1):zero()
	elseif self.output:nDimension() == 2 then
		self.logdetJ:resize(self.output:size(1)):zero()
	else
		error('cannot handle 3d output')
	end
	for i=1,#self.modules do
		if self.modules[i].logdetJ then
			self.logdetJ:add(self.modules[i].logdetJ)
		end
	end
	return self.logdetJ
end

function NormalizingFlow:getKL(beta)
	if self.output:nDimension() == 1 then
		self.KL:resize(1):zero()
	elseif self.output:nDimension() == 2 then
		self.KL:resize(self.output:size(1)):zero()
	else
		error('cannot handle 3d output')
	end
	for i=1,#self.modules do
		if self.modules[i].KL then
			self.KL:add(self.modules[i].KL)
		end
	end
	if beta == 1 or beta == nil then
		self.KL:add(self:getlogpz())
	else
		self.KL:add(self:getlogpz()*beta)
	end
	return self.KL
end

function NormalizingFlow:setAnnealing(beta)
	-- this only impacts gradient calculations
	self.beta = beta or 1
end

function NormalizingFlow:updateOutput(input)
	self.output = parent.updateOutput(self,input)
	-- update KL
	self:getKL();
	return self.output
end

function NormalizingFlow:updateGradInput(input, gradOutput)
	-- add gradient logpz w.r.t. z
	self.beta = self.beta or 1
	self.gradOutput:resizeAs(gradOutput):copy(gradOutput)
	if self.beta ~= 1 then
		self.gradOutput:add(self.beta,self.output)
	else
		self.gradOutput:add(self.output)
	end
	return parent.updateGradInput(self,input,self.gradOutput)
end

function NormalizingFlow:accGradParameters(input, gradOutput)
	-- add gradient logpz w.r.t. z
	self.beta = self.beta or 1
	self.gradOutput:resizeAs(gradOutput):copy(gradOutput)
	if self.beta ~= 1 then
		self.gradOutput:add(self.beta,self.output)
	else
		self.gradOutput:add(self.output)
	end
	return parent.accGradParameters(self,input,self.gradOutput)
end

function NormalizingFlow:backward(input, gradOutput, scale)
	-- add gradient logpz w.r.t. z
	self.beta = self.beta or 1
	self.gradOutput:resizeAs(gradOutput):copy(gradOutput)
	if self.beta ~= 1 then
		self.gradOutput:add(self.beta,self.output)
	else
		self.gradOutput:add(self.output)
	end
	return parent.backward(self,input,self.gradOutput, scale)
end

function NormalizingFlow:accUpdateGradParameters(input, gradOutput, lr)
	-- add gradient logpz w.r.t. z
	self.beta = self.beta or 1
	self.gradOutput:resizeAs(gradOutput):copy(gradOutput)
	if self.beta ~= 1 then
		self.gradOutput:add(self.beta,self.output)
	else
		self.gradOutput:add(self.output)
	end
	return parent.accUpdateGradParameters(self,input,self.gradOutput, lr)
end

