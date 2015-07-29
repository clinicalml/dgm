disp = require "display"
require "paths"
require "tgm"
require "cunn"
cmd = torch.CmdLine()
cmd:text()
cmd:text('Visualizing saved information in generative model')
cmd:text()
cmd:text('Options')
-- model params
cmd:option('-path','./checkpoint', 'File name to load and display')
cmd:option('-file',nil, 'File name to load and display')
cmd:text()
-- parse input params
opt = cmd:parse(arg)

print ('Options')
print (opt)


function parse(data)	
	local title = data.titlestr
	if not ids then 
		ids = {}
		ctr = 1
	end
	ctr = 1
	for k,v in pairs(data) do 
		if string.match(k,'img') then
			print ('Displaying.. '..k)
			ids[ctr] = disp.images(data[k],{title = k..title,win = ids[ctr]})
			ctr = ctr + 1
		elseif type(v)~='table' then
			print (k,v)
		end
	end
end
--Parse results
if opt.all=='false' then
	local data = torch.load(opt.path..'/'..opt.file)
	parse(data)
else
	for f in paths.files(opt.path) do 
		print('Processing '..opt.path..'/'..f)
		if string.match(f,'epoch150_') or string.match(f,'epoch70_') then 
			local data = torch.load(opt.path..'/'..f)
			parse(data)
   			io.write("continue with this operation (y/n)? ")
   			io.flush()
   			local answer=io.read()
	    	if answer=="n" then
				break
			end
		end
	end
end
