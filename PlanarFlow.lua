require 'nn'
require 'nngraph'

local PlanarFlow, parent = torch.class('nn.PlanarFlow', 'nn.Module')


local function DotProduct(x,y)
	return nn.Sum(2)(nn.CMulTable()({x,y})) --TODO only works for batch right now
end

function PlanarFlow:__init(dimension,deep_params,logpz_flag,beta)
	parent.__init(self)
	self.size = torch.LongStorage()
	self.dimension = dimension
	self.train = true
	self.logpz_flag = logpz_flag or false
	self.beta = beta or 1 -- annealing
	self.KL = 0
	self.logdetJacobian = 0
	self.logpz = 0
	self.deep_params = deep_params or false

	assert(dimension ~= nil, "must specify dimension")

	-- define graph of flow
	local z = nn.Identity()()
	local w_in = nn.Identity()()
	local b_in = nn.Identity()()
	local u_in = nn.Identity()()
	local w,b,u
	if deep_params then
		w = w_in
		b = b_in
		u = u_in
	else
		w = nn.CMul(self.dimension)(w_in)
		b = nn.CMul(1)(b_in)
		u = nn.CMul(self.dimension)(u_in)
	end
	local wTz = DotProduct(w,z)
	local wTz_b = nn.CAddTable()({wTz,b})
	local h = nn.Replicate(self.dimension,1,0)(nn.Tanh()(wTz_b)) --TODO would this work for SGD?
	-- enforce w*u_hat >= -1
	local wTu = DotProduct(w,u)
	local m = nn.MulConstant(-1)(nn.AddConstant(1)(nn.LogSigmoid()(nn.MulConstant(-1)(wTu))))
	local w_normsq = nn.Sum(2)(nn.Square()(w)) --TODO only works for batch right now
	local val1 = nn.CSubTable()({m,wTu})
	local val2 = nn.CDivTable()({val1,w_normsq})
	local val3 = nn.Replicate(self.dimension,1,0)(val2) --TODO would this work for SGD?
	local val4 = nn.CMulTable()({w,val3})
	local u_hat = nn.CAddTable()({u,val4})
	local uh = nn.CMulTable()({u_hat,h})
	-- add stuff to self for debugging
	self.u = u
	self.u_hat = u_hat
	self.w = w
	self.uh = uh
	self.b = b
	self.h = h
	-- z_out = z + u_hat * h
	local z_out = nn.CAddTable()({z,uh})
	-- logdetJacobian = log|1+u*psi(z)|
	-- psi = dhdz * w
	local psi = nn.CMulTable()({w,nn.AddConstant(1)(nn.MulConstant(-1)(nn.Square()(h)))})
	local logdetJ = nn.MulConstant(-1)(nn.Log()(nn.Abs()(nn.AddConstant(1)(DotProduct(psi,u_hat)))))
	-- logpz = Elog(p(z)) = E[log(zTz)]
	if self.logpz_flag then
		local logpz = nn.AddConstant(0.5*self.dimension*math.log(2*math.pi))(nn.MulConstant(0.5)(nn.Sum(2)(nn.Square()(z_out)))) --TODO only works for batch
		self.flow = nn.gModule({z,w_in,b_in,u_in},{z_out,logdetJ,logpz})
	else
		self.flow = nn.gModule({z,w_in,b_in,u_in},{z_out,logdetJ})
	end
end 

function PlanarFlow:parameters()
	return self.flow:parameters()
end

function PlanarFlow:cuda()
	self.flow = self.flow:cuda()
	return self:type('torch.CudaTensor')
end

function PlanarFlow:setAnnealing(beta)
	self.beta = beta or 1
end

--Forward pass
function PlanarFlow:updateOutput(input)	
	local z,w_in,b_in,u_in
	if type(input) == 'table' then
		errmsg = 'if input to PlanarFlow is table, then input must contain 4 tensors: z,w,b,u'
		assert(input[1] ~= nil, errmsg)
		assert(input[2] ~= nil, errmsg)
		assert(input[3] ~= nil, errmsg)
		assert(input[4] ~= nil, errmsg)
		z = input[1]
		w_in = input[2]
		b_in = input[3]
		u_in = input[4]
	else
		z = input
		w_in = torch.ones(z:size()):type(z:type())
		b_in = torch.ones(z:size(1)):type(z:type()) -- TODO: only works in batch setting
		u_in = torch.ones(z:size()):type(z:type())
	end
	assert(z:nDimension() == 2, 'z must be 2 dimensional: batchSize x num_latent_states')
	local one = torch.ones(z:size(1)):type(z:type())
	local z_out, logdetJ, logpz = unpack(self.flow:forward({z,w_in,b_in,u_in}))
	self.logdetJacobian = logdetJ
	self.logpz = logpz or 0
	self.output = z_out
	-- accumulate KL
	self.KL = logdetJ + self.logpz
	
    return self.output
end

--Backward pass
function PlanarFlow:updateGradInput(input, gradOutput)
	local z,w_in,b_in,u_in
	if type(input) == 'table' then
		errmsg = 'if input to PlanarFlow is table, then input must contain 4 tensors: z,w,b,u'
		assert(input[1] ~= nil, errmsg)
		assert(input[2] ~= nil, errmsg)
		assert(input[3] ~= nil, errmsg)
		assert(input[4] ~= nil, errmsg)
		z = input[1]
		w_in = input[2] 
		b_in = input[3]
		u_in = input[4] 
	else
		z = input
		w_in = torch.ones(z:size()):type(z:type())
		b_in = torch.ones(z:size(1)):type(z:type()) -- TODO: only works in batch setting
		u_in = torch.ones(z:size()):type(z:type())
	end
	assert(z:nDimension() == 2, 'z must be 2 dimensional: batchSize x num_latent_states')
	local one = torch.ones(z:size(1)):type(z:type())
	if self.logpz_flag then 
		dz, dw, db, du = unpack(self.flow:backward({z,w_in,b_in,u_in},{gradOutput,one,one*self.beta}))
	else
		dz, dw, db, du = unpack(self.flow:backward({z,w_in,b_in,u_in},{gradOutput,one}))
	end
	if type(input) == 'table' then
		self.gradInput = {dz,dw,db,du}
	else
		self.gradInput = dz
	end
    return self.gradInput
end





