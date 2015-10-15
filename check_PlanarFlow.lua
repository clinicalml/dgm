require 'nn'
require 'nngraph'
require 'PlanarFlow'

--[[

> th check_PlanarFlow.lua

Script checks a model containing 
  * a flow of PlanarFlow modules.

User must adjust batchsize, dim_stochastic, len_normflow, deep.
The most important parameter to adjust is deep, since this fundamentally
changes the architecture of the flow.

]]

batchsize = 10
dim_stochastic = 5
len_normflow = 5
deep = true

-- define input to model
z0 = torch.randn(batchsize,dim_stochastic)


-- define model
input = nn.Identity()()
inputs = {input}
flow = {}
flow[0] = input
for k = 1,len_normflow do
	if k == len_normflow then logpz_flag = true else logpz_flag = false end
	if deep then
		local w_in = nn.Identity()()
		local b_in = nn.Identity()()
		local u_in = nn.Identity()()
		table.insert(inputs,w_in)
		table.insert(inputs,b_in)
		table.insert(inputs,u_in)
		flow[k] = nn.PlanarFlow(dim_stochastic,deep,logpz_flag)({flow[k-1],w_in,b_in,u_in})
	else
		flow[k] = nn.PlanarFlow(dim_stochastic,deep,logpz_flag)(flow[k-1])
	end
end
m = nn.gModule(inputs,{flow[len_normflow]})

if not deep then
	-- set parameters of model
	p,dp = m:getParameters()
	p:copy(torch.rand(p:size()))
	p = (p-p:mean())/p:std()
end

function getinputs(z)
	if deep then
		local w_in = torch.randn(batchsize,dim_stochastic)
		local b_in = torch.randn(batchsize,1)
		local u_in = torch.randn(batchsize,dim_stochastic)
		my_inputs = {z}
		for k=1,len_normflow do
			table.insert(my_inputs,w_in) -- w
			table.insert(my_inputs,b_in)  -- b
			table.insert(my_inputs,u_in) -- u
		end
		return my_inputs
	else
		return z
	end
end
	

print('\n-----------------------------------------------------')
print('check forward propagation\n')

out = m:forward(getinputs(z0))
p,dp = m:parameters()

function normflowcheck(z0)
	local z_list = {}
	local logdetJ_list = {}
	local logpz_list = {}
	local KL_list = {}
	for i = 1,batchsize do
		local z = torch.Tensor(z0[i]:size()):copy(z0[i])
		local logdetJ = 0
		for k = 1,len_normflow do
			local w = torch.Tensor(z:size()):copy(flow[k].data.module.w.data.module.output[i])
			local b = flow[k].data.module.b.data.module.output[i]
			local u = torch.Tensor(z:size()):copy(flow[k].data.module.u.data.module.output[i])
			if deep then b = b[1] end
			local wTz_b = torch.dot(w,z)+b
			local h = math.tanh(wTz_b)
			local wTu = torch.dot(w,u)
			local m = -1 + math.log(1+math.exp(wTu))
			local w_norm2 = torch.dot(w,w)
			local u_hat = u + w*(m-wTu)/w_norm2
			
			local psi = w*(1-h^2)
			logdetJ = logdetJ - 1*math.log(math.abs(1+torch.dot(psi,u_hat)))

			z = z + u_hat*h
		end
		local C = 0.5*dim_stochastic*math.log(2*math.pi)
		local logpz = C+0.5*torch.dot(z,z)
		local KL = logdetJ+logpz
		z_list[i] = z
		logdetJ_list[i] = logdetJ
		logpz_list[i] = logpz
		KL_list[i] = KL
	end
	z = nn.JoinTable(1):forward(z_list)
	logdetJ = torch.Tensor(logdetJ_list)
	logpz = torch.Tensor(logpz_list)
	KL = torch.Tensor(KL_list)
	return {z,logdetJ,logpz,KL}
end


function flow_KL(f)
    local logdetJ = 0
    local logpz = 0
	local KL = 0
    for t=1,len_normflow do
        logdetJ = f[t].data.module.logdetJacobian + logdetJ
		logpz = (f[t].data.module.logpz or 0) + logpz
		KL = f[t].data.module.KL + KL
    end
    return {logdetJ,logpz,KL}
end
m.logdetJacobian, m.logpz, m.KL = unpack(flow_KL(flow))
--[[
check_z = {}
check_logdetJ = {}
check_logpz = {}
check_KL = {}
for i=1,batchsize do
	check_z[i],check_logdetJ[i],check_logpz[i],check_KL[i] = unpack(normflowcheck(z0[i]))
end
check_z = nn.JoinTable(1):forward(check_z)
check_logdetJ = torch.Tensor(check_logdetJ)
check_logpz = torch.Tensor(check_logpz)
check_KL = torch.Tensor(check_KL)
]]
check_z,check_logdetJ,check_logpz,check_KL = unpack(normflowcheck(z0))
print('PlanarFlow.output')
print(out)
print('checkOutput:')
print(check_z)
print('PlanarFlow.logdetJacobian:')
print(m.logdetJacobian)
print('check_logdetJ:')
print(check_logdetJ)
print('PlanarFlow.logpz:')
print(m.logpz)
print('check_logpz:')
print(check_logpz)
diff_z = out-check_z
diff_logdetJ = m.logdetJacobian - check_logdetJ
diff_logpz = m.logpz - check_logpz
diff_KL = m.KL - check_KL
print('PlanarFlowOutput - checkOutput:')
print(diff_z)
print('PlanarFlow.logdetJacobian - check_logdetJ')
print(diff_logdetJ)
print('PlanarFlow.logpz - check_logpz')
print(diff_logpz)
print('||PlanarFlowOutput - checkOutput|| = ' .. tostring(diff_z:norm()))
print('||PlanarFlow.logdetJacobian - check_logdetJ|| = ' .. tostring(diff_logdetJ:norm()))
print('||PlanarFlow.logpz - check_logpz|| = ' .. tostring(diff_logpz:norm()))
print('||PlanarFlow.KL - check_KL|| = ' .. tostring(diff_KL:norm()))
maxdiff = math.max(diff_z:norm(),diff_logdetJ:norm(),diff_logpz:norm(),diff_KL:norm())
if maxdiff <= 1e-6 then 
	output_testmsg = 'Forward Test PASSED ' .. tostring(maxdiff)
else
	output_testmsg = 'Forward Test FAILED ' .. tostring(maxdiff)
end
print(output_testmsg)


print('\n-----------------------------------------------------')
print('check gradInput\n')
function check_gradInput(perturbInputs,input,gradInput)

	dfdz = {}
	eps = 1e-6
	for i = 1,input:size(2) do
		e = torch.zeros(input:size())
		e[{{},{i}}] = eps
		up = m:forward(perturbInputs(e)):clone()
		m.logdetJacobian, m.logpz, m.KL = unpack(flow_KL(flow))
		--up_logdetJ = m.logdetJacobian:clone()
		--up_logpz = m.logpz:clone()
		up_KL = m.KL:clone()
		down = m:forward(perturbInputs(-e)):clone()
		m.logdetJacobian, m.logpz, m.KL = unpack(flow_KL(flow))
		--down_logdetJ = m.logdetJacobian:clone()
		--down_logpz = m.logpz:clone()
		down_KL = m.KL:clone()
		dfdz[i] = ((up-down)/(2*eps)):cmul(gradOutput):sum(2) 
		--dfdz[i] = dfdz[i] + (up_logdetJ - down_logdetJ)/2/eps
		--dfdz[i] = dfdz[i] + (up_logpz - down_logpz)/2/eps 
		dfdz[i] = dfdz[i] + (up_KL - down_KL)/2/eps
	end
	dfdz = nn.JoinTable(2):forward(dfdz)
	print('PlanarFlow.gradInput:')
	print(gradInput)
	print('approx gradient:')
	print(dfdz)
	print('gradInput - approx_gradInput:')
	gradInput_diff = gradInput - dfdz
	print(gradInput_diff)
	print('||dfdz - approx_dfdz|| = ' .. tostring((gradInput_diff):norm()))

	if gradInput_diff:norm() <= 1e-6 then
			gradInput_testmsg = 'gradInput Test PASSED ' .. tostring(gradInput_diff:norm())
	else
			gradInput_testmsg = 'gradInput Test FAILED ' .. tostring(gradInput_diff:norm())
	end
	print(gradInput_testmsg)	
	return gradInput_testmsg
end


gradInput_testmsgs = {}
if deep then
	numinputs = 1+3*len_normflow
	inputs = {}
	inputs[1] = torch.randn(batchsize,dim_stochastic)
	for k=1,len_normflow do
		table.insert(inputs,torch.randn(batchsize,dim_stochastic))
		table.insert(inputs,torch.randn(batchsize,1))
		table.insert(inputs,torch.randn(batchsize,dim_stochastic))
	end
	local copy = function(x)
		newcopy = {}
		for j=1,#x do
			newcopy[j] = x[j]:clone()
		end
		return newcopy
	end
	gradOutput = torch.rand(batchsize,dim_stochastic)
	m:forward(inputs)
	m:backward(inputs,gradOutput)
	for i=1,numinputs do
		local perturb = function(e)
			input_clone = copy(inputs)
			input_clone[i] = input_clone[i]+e
			return input_clone
		end
		gradInput_testmsgs[i] = '[' .. tostring(i) .. '] ' .. check_gradInput(perturb,inputs[i],m.gradInput[i])
	end
else
	inputs = torch.randn(batchsize,dim_stochastic)
	gradOutput = torch.rand(batchsize,dim_stochastic)
	m:forward(inputs)
	m:backward(inputs,gradOutput)
	local perturb = function(e)
		return inputs+e
	end
	gradInput_testmsgs[1] = check_gradInput(perturb,inputs,m.gradInput)
end		



print('\n-----------------------------------------------------')
print('check gradParameters\n')
p,dp = m:getParameters()
if p:nDimension() == 0 then
	gradParam_testmsg = 'no gradParameters to test'
else
	dp:zero();
	gradOutput = torch.rand(batchsize,dim_stochastic)
	m:forward(getinputs(z0))
	m:backward(z0,gradOutput)
	p,dp = m:getParameters()
	p0 = p:clone()

	dfdp = {}
	eps = 1e-6
	np = p:size(1)
	for i = 1,np do
		e = torch.zeros(np)
		e[i] = eps
		p:copy(p0+e)
		up = m:forward(getinputs(z0)):clone()
		m.logdetJacobian, m.logpz, m.KL = unpack(flow_KL(flow))
		--up_logdetJ = m.logdetJacobian:clone()
		--up_logpz = m.logpz:clone()
		up_KL = m.KL:clone()
		p:copy(p0-e)
		down = m:forward(getinputs(z0)):clone()
		m.logdetJacobian, m.logpz, m.KL = unpack(flow_KL(flow))
		--down_logdetJ = m.logdetJacobian:clone()
		--down_logpz = m.logpz:clone()
		down_KL = m.KL:clone()
		dfdp[i] = (((up-down)/(2*eps)):cmul(gradOutput):sum(2)) 
		--dfdp[i] = dfdp[i] + (up_logdetJ - down_logdetJ)/2/eps
		--dfdp[i] = (dfdp[i] + (up_logpz - down_logpz)/2/eps):sum()
		dfdp[i] = (dfdp[i] + (up_KL - down_KL)/2/eps):sum()
	end
	dfdp = torch.Tensor(dfdp)
	print('PlanarFlow.gradParameters:')
	print(dp)
	print('approx gradient:')
	print(dfdp)
	print('gradParameters - approx_gradParameters:')
	gradParam_diff = dp - dfdp
	print(gradParam_diff)
	print('||dfdp - approx_dfdp|| = ' .. tostring((gradParam_diff):norm()))

	if gradParam_diff:norm() <= 1e-6 then
			gradParam_testmsg = 'gradParameters Test PASSED ' .. tostring(gradParam_diff:norm())
	else
			gradParam_testmsg = 'gradParameters Test FAILED ' .. tostring(gradParam_diff:norm())
	end
	print(gradParam_testmsg)	
end



print('\n-----------------------------------------------------')
print(output_testmsg)
for i = 1,#gradInput_testmsgs do
	print(gradInput_testmsgs[i])
end
print(gradParam_testmsg)

