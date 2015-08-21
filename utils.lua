require "cunn"
require "paths"
mnist = require "mnist"
--Loading binarized MNIST dataset
function loadBinarizedMNIST(cuda)
	local dataset = {}
	local trainset = mnist.traindataset()
	local testset = mnist.testdataset()
	dataset.train_x = torch.zeros(trainset.size,784)
	dataset.test_x  = torch.zeros(testset.size ,784)
	for it = 1,trainset.size do 
		dataset.train_x[it] = torch.gt(trainset.data[it]:view(784):clone(),127):double()
		if it<=testset.size then
			dataset.test_x[it] = torch.gt(testset.data[it]:view(784):clone(),127):double()
		end
	end
	if cuda then 
		dataset.train_x = dataset.train_x:cuda()
		dataset.test_x  = dataset.test_x:cuda()
	end
	dataset.dim_input = 784
	--Clean memory
	trainset = nil
	testset  = nil
	collectgarbage()
	return dataset
end
--Check if folder exists, if not create it
function setupFolder(folder)
    if not paths.dirp(folder) then
        paths.mkdir(folder)
    end
end
--Append to datalist if not nil
function appendToTensor(datalist,data)
    if datalist then
        datalist = torch.cat(datalist,torch.Tensor(1,1):fill(data),1)
    else
        datalist = torch.Tensor(1,1):fill(data)
    end
    return datalist
end

--variables for displaying results
function setupDisplay()
    local img_format = {width = 800,height = 800,title = ''}
    local format = {win = '',width=500,height=500,xLabel='Iterations',title = ''}
    return img_format,format
end

--Convert from string to bool
function stringToBool(str)
    if str == 'true' or  str=='T' or str=='t' or str =='True' or str=='TRUE' then
        return true
    else
        return false
    end
end
