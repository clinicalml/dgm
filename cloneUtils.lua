-- Taken from: https://github.com/karpathy/char-rnn
-- which in turn is 
-- adapted from https://github.com/wojciechz/learning_to_execute
-- 1) utilities for combining/flattening parameters in a model
-- 2) utilities for parameter sharing across time 
-- 3) cloning lists
local clone_utils = {}
function clone_utils.combine_all_parameters(...)
    --[[ like module:getParameters, but operates on many modules ]]--
    --Put all the parameters in a table and then flatten them 
    -- get parameters
    local networks = {...}
    local parameters = {}
    local gradParameters = {}
    print (#networks)
	--First collect all the parameters in the networks
    for i = 1, #networks do
        local net_params, net_grads = networks[i]:parameters()
        --iterate through each networks parameters 
        if net_params then
            for _, p in pairs(net_params) do
                parameters[#parameters + 1] = p
            end
            for _, g in pairs(net_grads) do
                gradParameters[#gradParameters + 1] = g
            end
        end
    end

    --Function to check if the object is in set indexed by unique torch pointer 
	--set is a table that stores in the format: key (pointer to storage) value (tuple of storage and nParameters)
    local function storageInSet(set, storage)
		--torch.pointer returns a unique ID of storage
        local storageAndOffset = set[torch.pointer(storage)]
        if storageAndOffset == nil then
            return nil
        end
        local _, offset = unpack(storageAndOffset)
        return offset
    end

    -- this function flattens arbitrary lists of parameters,
    -- even complex shared ones
    local function flatten(parameters)
		--If null or no parameters then return empty tensor
        if not parameters or #parameters == 0 then
            return torch.Tensor()
        end
		--Constructor to create a new tensor whose datatype corresponds to that 
		--of parameters[1] 
        local Tensor = parameters[1].new
        local storages = {}
        local nParameters = 0
        --Go through the table of parameters 
        for k = 1,#parameters do
			--parameters[k] could be n-dim tensors. Instead work with its storage (1dim tensor)
            local storage = parameters[k]:storage()
            --map from pointer to storage to {storage,number of parameters in storage}
            if not storageInSet(storages, storage) then
                storages[torch.pointer(storage)] = {storage, nParameters}
                nParameters = nParameters + storage:size()
            end
        end
        --vector to house all params 
        local flatParameters = Tensor(nParameters):fill(1)
        local flatStorage = flatParameters:storage()
        for k = 1,#parameters do
            --Go through the parameters again and get the storage offset 
            local storageOffset = storageInSet(storages, parameters[k]:storage())
            parameters[k]:set(flatStorage, --set parameters[k]'s storag to be flatStorage 
                storageOffset + parameters[k]:storageOffset(), --set st pos. storageOffset assures cumulativeness
                parameters[k]:size(), --set size to be same as original parameters 
                parameters[k]:stride()) --set stride to be the same as the original parameters 
            parameters[k]:zero()
        end
		--So flat parameters was all initially set to be 1. As we go along, we zero out the parameters
		--that have been used. However, if in the storage, for whatever reason, the memory is not 
		--contiguous, then there will be holes (denoted by 1's in the flat storage). Therefore
		--we can save space by shifting over the parameters over the holes since we are guaranteed
		--that they will not be used. 
        local maskParameters=  flatParameters:float():clone()
        local cumSumOfHoles = flatParameters:float():cumsum(1)
        local nUsedParameters = nParameters - cumSumOfHoles[#cumSumOfHoles]
        local flatUsedParameters = Tensor(nUsedParameters)
        local flatUsedStorage = flatUsedParameters:storage()

        for k = 1,#parameters do
            local offset = cumSumOfHoles[parameters[k]:storageOffset()]
            parameters[k]:set(flatUsedStorage,
                parameters[k]:storageOffset() - offset, --shift the parameters' storageoffset by the number of holes
                parameters[k]:size(), --leave size and stride unchanged
                parameters[k]:stride())
        end

        for _, storageAndOffset in pairs(storages) do
			--Go through all the storages in the set "storages"
            local k, v = unpack(storageAndOffset) 
			--k contains the "original" storage of params[k] and v contains the cumulative number of parameters
			--seen thus far. This operation sets locations in flatParameters to the value in the storage
			--of the original parameters 
            flatParameters[{{v+1,v+k:size()}}]:copy(Tensor():set(k))
        end
		--Finally, set it up to use only flatUsedParameters and return that
        if cumSumOfHoles:sum() == 0 then
            flatUsedParameters:copy(flatParameters)
        else
            local counter = 0
            for k = 1,flatParameters:nElement() do
                if maskParameters[k] == 0 then
                    counter = counter + 1
                    flatUsedParameters[counter] = flatParameters[counter+cumSumOfHoles[k]]
                end
            end
            assert (counter == nUsedParameters)
        end
        return flatUsedParameters
    end

    -- flatten parameters and gradients
    local flatParameters = flatten(parameters)
    local flatGradParameters = flatten(gradParameters)

    -- return new flat vector that contains all discrete parameters
    return flatParameters, flatGradParameters
end




function clone_utils.clone_many_times(net, T)
    local clones = {}

    local params, gradParams
    if net.parameters then
        params, gradParams = net:parameters()
        if params == nil then
            params = {}
        end
    end

    local paramsNoGrad
    if net.parametersNoGrad then
        paramsNoGrad = net:parametersNoGrad()
    end

    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net) --write network into file 

    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject() --clone of net from file 
        reader:close()

        if net.parameters then
            local cloneParams, cloneGradParams = clone:parameters()
            local cloneParamsNoGrad
            for i = 1, #params do
                cloneParams[i]:set(params[i])
                cloneGradParams[i]:set(gradParams[i])
            end
            if paramsNoGrad then
                cloneParamsNoGrad = clone:parametersNoGrad()
                for i =1,#paramsNoGrad do
                    cloneParamsNoGrad[i]:set(paramsNoGrad[i])
                end
            end
        end

        clones[t] = clone
        collectgarbage()
    end

    mem:close()
    return clones
end

function clone_utils.clone_list(tensor_list, zero_too)
    -- Takes a list of tensors and returns a list of cloned tensors
    local out = {}
    for k,v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    return out
end
return clone_utils
