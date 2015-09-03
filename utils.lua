require "cunn"
require "paths"
mnist = require "mnist"
--Loading MNIST dataset (NON standard dataset. This is not the one to be used for comparison of NLL)
function loadMNIST(cuda)
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

--Read MNIST file downloaded and return Tensor  
function readMNISTfile(fname,lines)
	local data = torch.Tensor(lines,784):fill(0)
	local f    = torch.DiskFile(fname,'r')
	for i=1,lines do 
		data[i] = torch.Tensor(f:readDouble(784))
	end
	return data
end

--Download data and setup directory
function getBinarizedMNIST()
	--Get train & valid. Append them
	if not paths.dirp('./binarizedMNIST') then 
		paths.mkdir('./binarizedMNIST')
	end
	print ('Downloading data...')
	os.execute('wget -O ./binarizedMNIST/binarized_mnist_train.amat http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_train.amat')		
	os.execute('wget -O ./binarizedMNIST/binarized_mnist_valid.amat http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_valid.amat') 
	os.execute('wget -O ./binarizedMNIST/binarized_mnist_test.amat http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_test.amat')
	print ('Converting data to torch format...')
	test  = readMNISTfile('./binarizedMNIST/binarized_mnist_test.amat',10000)
	train = readMNISTfile('./binarizedMNIST/binarized_mnist_train.amat',50000)
	valid = readMNISTfile('./binarizedMNIST/binarized_mnist_valid.amat',10000)
	print ('Saving data...')
	torch.save('./binarizedMNIST/train.t7',train)
	torch.save('./binarizedMNIST/test.t7',test)
	torch.save('./binarizedMNIST/valid.t7',valid)
end

--Load standard MNIST data
function loadBinarizedMNIST()
	if not paths.dirp('./binarizedMNIST') or not paths.filep('./binarizedMNIST/valid.t7') or not paths.filep('./binarizedMNIST/test.t7') or not paths.filep('./binarizedMNIST/train.t7') then 
		getBinarizedMNIST()
	end
	print ('Loading Binarized MNIST dataset')
	local train = torch.load('./binarizedMNIST/train.t7')
	local test  = torch.load('./binarizedMNIST/test.t7')
	local valid = torch.load('./binarizedMNIST/valid.t7')
	return torch.cat(train,valid,1),test
end


--[[
function testGetData()
	train,test = loadBinarizedMNIST()
	print (train:size())
	print (test:size())
end

function testRead()
	disp = require "display"
	print('Reading test')
	test = readMNISTfile('./binarized_mnist_test.amat',10000)
	print ('Displaying test',test:size())
	local samples = {}
	local shuffle = torch.randperm(10000)
	for i=1,100 do 
		samples[i] = test[shuffle[i] ]:reshape(28,28)
	end
	disp.images(samples,{title='Test'})
	
	print('Reading train')
	train= readMNISTfile('./binarized_mnist_train.amat',50000)
	print ('Displaying train',train:size())
	local samples = {}
	local shuffle = torch.randperm(10000)
	for i=1,100 do 
		samples[i] =train[shuffle[i] ]:reshape(28,28)
	end
	disp.images(samples,{title='Train'})
	
	print('Reading valid')
	valid = readMNISTfile('./binarized_mnist_valid.amat',10000)
	print ('Displaying valid',valid:size())
	local samples = {}
	local shuffle = torch.randperm(10000)
	for i=1,100 do
		samples[i] =valid[shuffle[i] ]:reshape(28,28)
	end
	disp.images(samples,{title='Valid'})
	print ('Done')
end
--]]
