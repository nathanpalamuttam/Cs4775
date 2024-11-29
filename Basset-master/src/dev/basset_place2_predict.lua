#!/usr/bin/env th

require 'hdf5'

require 'batcherT'
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

----------------------------------------------------------------
-- construct model
----------------------------------------------------------------
-- initialize
local convnet = ConvNet:__init()

-- load from saved parameters
local convnet_params = torch.load(opt.model_file)
convnet:load(convnet_params)

-- guarantee evaluate mode
convnet.model:evaluate()

-- move to gpu
-- if cuda then
--     convnet.model:cuda()
-- end

----------------------------------------------------------------
-- predict and test
----------------------------------------------------------------
-- measure accuracy on a test set
local scores = convnet:predict_scores(test_seqs, opt.batch_size)

-- close HDF5
data_open:close()

----------------------------------------------------------------
-- dump to file
----------------------------------------------------------------
local hdf_out = hdf5.open(opt.out_file, 'w')
hdf_out:write("scores", scores)
hdf_out:close()
