require 'nn';

local PlanarFlow, parent = torch.class('nn.PlanarFlow', 'nn.Module')

function PlanarFlow:__init(dimension)
	parent.__init(self)
	assert(dimension ~= nil,"must specify dimension")
	assert(type(dimension) == 'number',"dimension must be a number")
	self.dimension = dimension
	self.train = true
	self.KL = 0
	self.logdetJ = torch.Tensor(1)
	self.weight = torch.Tensor(dimension*2)
	self.bias = torch.Tensor(1)
	self.gradWeight = torch.Tensor(dimension*2)
	self.gradBias = torch.Tensor(1)
	self.output = torch.Tensor(dimension)
	self.gradInput = torch.Tensor(dimension)
	self.warmup = 1
end

function PlanarFlow:getweights()
	local u,w = unpack(self.weight:split(self.dimension))
	local b = self.bias[1]
	local du,dw = unpack(self.gradWeight:split(self.dimension))
	local db = self.gradBias
	return u,w,b,du,dw,db
end

function PlanarFlow:setWarmUp(warmup)
	self.warmup = warmup or 1
end


function PlanarFlow:updateOutput(input)
	assert(input ~= nil, "input cannot be nil")
	assert(input:nDimension() <= 2, "input must be a 1d (single) or 2d (batch) tensor")

	-- declare local variables
	local output = self.output:resizeAs(input)
	local logdetJ = self.logdetJ
	local u,w,b,_,_,_ = self:getweights()

	-- calculate f(z) and log|J|
	local u_hat = u:clone()
	local wTu = torch.dot(w,u) 
	local m = -1 + torch.log(1+torch.exp(wTu))
	local wTw = torch.dot(w,w) + 1e-12
	u_hat:add(w*((m-wTu)/wTw))

	if input:nDimension() == 2 then
		local h = torch.mv(input,w)
		h:add(b)
		h:tanh()
		logdetJ:resize(input:size(1)):copy(h)
		output:ger(h,u_hat)
	else
		local h = torch.tanh(torch.dot(input,w)+b)
		logdetJ:resize(1):fill(h)
		output:copy(u_hat):mul(h)
	end
	output:add(input)
	logdetJ:pow(2):mul(-1):add(1):mul(torch.dot(w,u_hat)):add(1+1e-12):abs():log():mul(-1)
	self.KL = logdetJ
	self.logdetJ = logdetJ

	self.output = output
	return output
end

function PlanarFlow:updateGradInput(input, gradOutput)
	self.warmup = self.warmup or 1
	local u,w,b,_,_,_ = self:getweights()

	-- calculate u_hat
	local u_hat = u:clone()
	local wTu = torch.dot(w,u) 
	local m = -1 + torch.log(1+torch.exp(wTu))
	local wTw = torch.dot(w,w) + 1e-12
	u_hat:copy(u)
	u_hat:add(w*(m-wTu)/wTw)

	local gradInput = self.gradInput:resizeAs(input)
	if input:nDimension() == 2 then
		local batchSize = input:size(1)
		local h = torch.mv(input,w)
		h:add(b)
		h:tanh()
		local dh = torch.pow(h,2):mul(-1):add(1)
		-- gradient w.r.t. log(p(x|z))
		gradInput:ger(torch.mv(gradOutput,u_hat):cmul(dh),w)
		gradInput:add(gradOutput)

		local wTu_hat = torch.dot(w,u_hat)
		-- gradient w.r.t. log|J|
		if self.warmup == 1 then
			gradInput:add(torch.ger(h:cmul(dh):mul(2*wTu_hat):cdiv(dh:mul(wTu_hat):add(1+1e-12)),w))
		else
			gradInput:add(self.warmup,torch.ger(h:cmul(dh):mul(2*wTu_hat):cdiv(dh:mul(wTu_hat):add(1+1e-12)),w))
		end

	else
		local h = torch.tanh(torch.dot(input,w)+b)
		local dh = 1-h^2
		-- gradient w.r.t. log(p(x|z))
		gradInput:mv(torch.ger(w,u_hat),gradOutput):mul(dh)
		gradInput:add(gradOutput)

		local wTu_hat = torch.dot(w,u_hat)
		-- gradient w.r.t. log|J|
		if self.warmup == 1 then
			gradInput:add(w*2*wTu_hat*h*dh/(1+1e-12+wTu_hat*dh))
		else
			gradInput:add(self.warmup,w*2*wTu_hat*h*dh/(1+1e-12+wTu_hat*dh))
		end
	end

	self.gradInput = gradInput
	return gradInput
end


function PlanarFlow:accGradParameters(input, gradOutput, scale)
	self.warmup = self.warmup or 1
	scale = scale or 1

	local u,w,b,du,dw,db = self:getweights()

	local u_hat = u:clone()
	local wTu = torch.dot(w,u) 
	local m = -1 + torch.log(1+torch.exp(wTu))
	local wTw = torch.dot(w,w) + 1e-12
	u_hat:copy(u)
	u_hat:add(w*(m-wTu)/wTw)
	local wTu_hat = torch.dot(w,u_hat)

	if input:nDimension() == 2 then
		local batchSize = input:size(1)
		local h = torch.mv(input,w)
		h:add(b)
		h:tanh()
		local dh = torch.pow(h,2):mul(-1):add(1)

		-- gradient log(p(x|z)) w.r.t. u
		local hdf = torch.mv(gradOutput:t(),h)
		du:add(hdf*scale)
		du:add(-w*(torch.dot(w,hdf)/wTw*torch.sigmoid(-wTu)*scale))

		-- gradient log|J| w.r.t. u
		local dlogJdu = dh*wTu_hat
		dlogJdu:add(1+1e-12)
		dlogJdu:cdiv(dh,dlogJdu)
		du:add(-w*(dlogJdu:sum()*torch.sigmoid(wTu)*self.warmup*scale))

		-- gradient log(p(x|z)) w.r.t. w
		local wThdf = torch.dot(w,hdf)
		dw:add(u*(-torch.sigmoid(-wTu)/wTw*wThdf*scale))
		local val = (m-wTu)/wTw
		dw:add(w*(-2*val*wThdf/wTw*scale))
		dw:add(hdf*(val*scale))
		dw:addmv(scale,input:t(),torch.mv(gradOutput,u_hat):cmul(dh))

		-- gradient log|J| w.r.t. w
		local val = dlogJdu:sum()
		dw:add(u*(val*(torch.sigmoid(-wTu)-1)*self.warmup*scale))
		dw:addmv(2*wTu_hat*self.warmup*scale,input:t(),torch.cmul(dlogJdu,h))

		-- gradient log(p(x|z)) w.r.t. b
		db:add(torch.dot(u_hat,torch.mv(gradOutput:t(),dh))*scale)
		-- gradient log|J| w.r.t. b
		db:add(2*torch.dot(dlogJdu,h)*wTu_hat*self.warmup*scale)



	else
		error('this has not been implemented yet for 1d inputs')
		local h = torch.tanh(torch.dot(input,w)+b)
		local dh = 1-h^2
		-- gradient w.r.t. log(p(x|z))

		local wTu_hat = torch.dot(w,u_hat)
		-- gradient w.r.t. log|J|
	end

end


