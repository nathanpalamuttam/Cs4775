#!/usr/bin/env th

require 'hdf5'

require 'postprocess'

----------------------------------------------------------------
-- parse arguments
----------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('DNA ConvNet testing')
cmd:text()
cmd:text('Arguments')
cmd:argument('model_file')
cmd:argument('data_file')
cmd:argument('out_file')
cmd:text()
cmd:text('Options:')
cmd:option('-batch_size', 128, 'Batch size')
cmd:option('-cuda', false, 'Run on GPGPU')
cmd:option('-norm', false, 'Normalize all targets to a level plane')
cmd:option('-pool', 10, 'Pool adjacent positions for filter outputs')
cmd:text()
opt = cmd:parse(arg)

-- set cpu/gpu
cuda = opt.cuda
require 'convnet'

----------------------------------------------------------------
-- load data
----------------------------------------------------------------
local data_open = hdf5.open(opt.data_file, 'r')
local test_seqs = data_open:read('test_in')
local test_targets = data_open:read('test_out')

local num_seqs = test_seqs:dataspaceSize()[1]
local seq_len = test_seqs:dataspaceSize()[4]
local num_targets = test_targets:dataspaceSize()[2]

----------------------------------------------------------------
-- construct model
----------------------------------------------------------------
-- initialize
local convnet = ConvNet:__init()

-- load from saved parameters
local convnet_params = torch.load(opt.model_file)
convnet:load(convnet_params)

-- pooling
local nn_pool = nn.SpatialMaxPooling(opt.pool, 1)
local pseq_len = math.floor((seq_len-convnet.conv_filter_sizes[1]+1) / opt.pool)

----------------------------------------------------------------
-- predict and test
----------------------------------------------------------------
-- guarantee evaluate mode
convnet.model:evaluate()

-- track statistics across batches
local preds = torch.Tensor(num_seqs, num_targets)
local filter_outs = torch.Tensor(num_seqs, convnet.conv_filters[1], pseq_len)
local si = 1

-- maintain variables
local preds_batch
local filter_outs_batch
local filter_outs_pool

-- initialize batcher
local batcher = Batcher:__init(test_seqs, nil, opt.batch_size)

-- next batch
local Xb = batcher:next()

-- while batches remain
while Xb ~= nil do
    -- cuda
    if cuda then
        Xb = Xb:cuda()
    end

    -- predict
    preds_batch = convnet.model:forward(Xb)

    -- get filter outputs
    filter_outs_batch = convnet:get_nonlinearity(1).output:squeeze()

    filter_outs_pool = nn_pool:updateOutput(filter_outs_batch)

    -- copy into full Tensors
    for i = 1,(#preds_batch)[1] do
        preds[{si,{}}] = preds_batch[{i,{}}]:float()
        filter_outs[{si,{}}] = filter_outs_pool[{i,{}}]:float()
        si = si + 1
    end

    -- next batch
    Xb = batcher:next()

    collectgarbage()
end

-- normalize
if opt.norm then
    preds = troy_norm(preds, convnet.pred_means)
end

-- close HDF5
data_open:close()

----------------------------------------------------------------
-- dump to file
----------------------------------------------------------------
local hdf_out = hdf5.open(opt.out_file, 'w')
hdf_out:write("preds", preds)
hdf_out:write("filter_outs", filter_outs)
hdf_out:close()
