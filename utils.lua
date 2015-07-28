require "hdf5"
require "cunn"
require "paths"

function loadMNIST(cuda)
	local dataset = {}
	local f = hdf5.open('mnist.hdf5')
	dataset.train_x = torch.cat(f:read('x_train'):all():float(),f:read('x_valid'):all():float(),1)
	dataset.test_x  = f:read('x_test'):all():float()
	if cuda then 
		dataset.train_x = dataset.train_x:cuda()
		dataset.test_x  = dataset.test_x:cuda()
	end
	dataset.dim_input = dataset.train_x:size(2)
	return dataset
end

function setupFolder(folder)
    if not paths.dirp(folder) then
        paths.mkdir(folder)
    end
end

--Append to Tensor
function appendToTensor(datalist,data)
    if datalist then
        datalist = torch.cat(datalist,torch.Tensor(1,1):fill(data),1)
    else
        datalist = torch.Tensor(1,1):fill(data)
    end
    return datalist
end

--Setup variables for displaying results
function setupDisplay()
    local img_format = {width = 500,height = 500,title = ''}
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