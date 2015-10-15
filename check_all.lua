require 'nn'
require 'nngraph'
require 'PlanarFlow'
require 'GaussianReparam_normflow'

--[[

> th check_all.lua

Script checks a model containing 
  * 1 GaussianReparam_normflow module
  * a flow of PlanarFlow modules

User must adjust batchsize, dim_stochastic, len_normflow, deep.
The most important parameter to adjust is deep, since this fundamentally
changes the architecture of the flow.

]]

batchsize = 10
dim_stochastic = 5
len_normflow = 5
deep = false

testmsg = {}
for _, deep in pairs({true,false}) do
	testmsg[deep] = {}
	-- define model
	mu = nn.Identity()()
	logsigma = nn.Identity()()
	reparam = nn.GaussianReparam(dim_stochastic)
	z_q0 = reparam({mu,logsigma})

	inputs = {mu,logsigma}

	flow = {}
	flow[0] = z_q0
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

	-- set parameters of model
	p,dp = m:getParameters()
	p:copy(torch.rand(p:size()))

	function forward(mu,logsigma)
		my_inputs = {mu,logsigma}
		torch.manualSeed(1)
		z_q0.data.module.train = false
		if deep then
			ones = torch.ones(batchsize,dim_stochastic)
			one = torch.ones(batchsize,1)
			for k=1,len_normflow do
				table.insert(my_inputs,ones) -- w
				table.insert(my_inputs,one)  -- b
				table.insert(my_inputs,ones) -- u
			end
		end
		return m:forward(my_inputs)
	end
	function backward(mu,logsigma,gradOutputs)
		my_inputs = {mu,logsigma}
		torch.manualSeed(1)
		if deep then
			ones = torch.ones(batchsize,dim_stochastic)
			one = torch.ones(batchsize,1)
			for k=1,len_normflow do
				table.insert(my_inputs,ones) -- w
				table.insert(my_inputs,one)  -- b
				table.insert(my_inputs,ones) -- u
			end
		end
		return m:backward(my_inputs,gradOutputs)
	end
		

	print('\n-----------------------------------------------------')
	print('check forward propagation\n')

	mu_input = torch.rand(batchsize,dim_stochastic)
	logsigma_input = torch.rand(batchsize,dim_stochastic)
	m:evaluate()
	out = forward(mu_input,logsigma_input)
	p,dp = m:parameters()

	function normflowcheck(batchnum)
		local eps = z_q0.data.module.eps[batchnum]
		local mu_ = mu.data.module.output[batchnum]
		local sigma = torch.pow(torch.exp(logsigma.data.module.output[batchnum]),0.5)
		local z = mu_ + torch.cmul(sigma,eps)
		local logdetJ = 0
		for k = 1,len_normflow do
			local w = torch.Tensor(z:size()):copy(flow[k].data.module.w.data.module.output[batchnum])
			local b = flow[k].data.module.b.data.module.output[batchnum]
			local u = torch.Tensor(z:size()):copy(flow[k].data.module.u.data.module.output[batchnum])
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
		local q0_KL = -C-0.5*((torch.log(torch.pow(sigma,2))+1):sum())
		local KL = q0_KL + logdetJ + logpz
		return {z,logdetJ,logpz,q0_KL,KL}
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
		local q0_KL = z_q0.data.module.KL
		KL = q0_KL + KL
		return {logdetJ,logpz,q0_KL,KL}
	end
	m.logdetJacobian, m.logpz, m.q0_KL, m.KL = unpack(flow_KL(flow))

	check_z = {}
	check_logdetJ = {}
	check_logpz = {}
	check_q0_KL = {}
	check_KL = {}
	for i=1,batchsize do
		check_z[i],check_logdetJ[i],check_logpz[i],check_q0_KL[i],check_KL[i] = unpack(normflowcheck(i))
	end
	check_z = nn.JoinTable(1):forward(check_z)
	check_logdetJ = torch.Tensor(check_logdetJ)
	check_logpz = torch.Tensor(check_logpz)
	check_q0_KL = torch.Tensor(check_q0_KL)
	check_KL = torch.Tensor(check_KL)
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
	diff_q0_KL = m.q0_KL - check_q0_KL
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
	print('||reparam.KL - check_q0_KL|| = ' .. tostring(diff_q0_KL:norm()))
	print('||KL - check_KL|| = ' .. tostring(diff_KL:norm()))
	if diff_z:norm() <= 1e-6 and diff_logdetJ:norm() <= 1e-6 and diff_logpz:norm() <= 1e-6 and diff_q0_KL:norm() <= 1e-6 and diff_KL:norm() <= 1e-6 then
		output_testmsg = 'Forward Test PASSED'
	else
		output_testmsg = 'Forward Test FAILED'
	end
	print(output_testmsg)


	print('\n-----------------------------------------------------')
	print('check gradInput\n')

	function forward(inputs)
		torch.manualSeed(1)
		return m:forward(inputs)
	end

	function check_gradInput(perturbInputs,input,gradInput)

		dfdz = {}
		eps = 1e-6
		for i = 1,input:size(2) do
			e = torch.zeros(input:size())
			e[{{},{i}}] = eps
			up = forward(perturbInputs(e)):clone()
			m.logdetJacobian, m.logpz, m.q0_KL, m.KL = unpack(flow_KL(flow))
			--up_logdetJ = m.logdetJacobian:clone()
			--up_logpz = m.logpz:clone()
			up_KL = m.KL:clone()
			down = forward(perturbInputs(-e)):clone()
			m.logdetJacobian, m.logpz, m.q0_KL, m.KL = unpack(flow_KL(flow))
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


	function getinputs()
		inputs = {}
		inputs[1] = torch.randn(batchsize,dim_stochastic)
		inputs[2] = torch.randn(batchsize,dim_stochastic)
		if deep then
			for k=1,len_normflow do
				table.insert(inputs,torch.randn(batchsize,dim_stochastic))
				table.insert(inputs,torch.randn(batchsize,1))
				table.insert(inputs,torch.randn(batchsize,dim_stochastic))
			end
		end
		return inputs
	end

	gradInput_testmsgs = {}
	inputs = getinputs()
	numinputs = #inputs
	copy = function(x)
				newcopy = {}
				for j=1,#x do
					newcopy[j] = x[j]:clone()
				end
				return newcopy
			end
	gradOutput = torch.rand(batchsize,dim_stochastic)
	forward(inputs)
	m:backward(inputs,gradOutput)
	for i=1,numinputs do
		local perturb = function(e)
			input_clone = copy(inputs)
			input_clone[i] = input_clone[i]+e
			return input_clone
		end
		gradInput_testmsgs[i] = '[' .. tostring(i) .. '] ' .. check_gradInput(perturb,inputs[i],m.gradInput[i])
	end



	print('\n-----------------------------------------------------')
	print('check gradParameters\n')
	p,dp = m:getParameters()
	if p:nDimension() == 0 then
		gradParam_testmsg = 'no gradParameters to test'
	else
		dp:zero();
		gradOutput = torch.rand(batchsize,dim_stochastic)
		forward(getinputs(z0))
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
			up = forward(inputs):clone()
			m.logdetJacobian, m.logpz, m.q0_KL, m.KL = unpack(flow_KL(flow))
			--up_logdetJ = m.logdetJacobian:clone()
			--up_logpz = m.logpz:clone()
			up_KL = m.KL:clone()
			p:copy(p0-e)
			down = forward(inputs):clone()
			m.logdetJacobian, m.logpz, m.q0_KL, m.KL = unpack(flow_KL(flow))
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
	table.insert(testmsg[deep],output_testmsgs)
	for i = 1,#gradInput_testmsgs do
		table.insert(testmsg[deep],gradInput_testmsgs[i])
	end
	table.insert(testmsg[deep],gradParam_testmsg)
	print(testmsg[deep])
end


print('\n-------------------- TEST SUMMARY ----------------------')

print('deep = true')
print(testmsg[true])
print('deep = false')
print(testmsg[false])
	


