#!/usr/bin/env th

require 'hdf5'

require 'batcher'
require 'convnet_io'
require 'convnet'

----------------------------------------------------------------
-- parse arguments
----------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('DNA ConvNet hidden layer visualizations')
cmd:text()
cmd:text('Arguments')
cmd:argument('model_file')
cmd:argument('data_file')
cmd:argument('out_file')
cmd:text()
opt = cmd:parse(arg)

-- fix seed
torch.manualSeed(1)

----------------------------------------------------------------
-- load data
----------------------------------------------------------------
local convnet_params = torch.load(opt.model_file)
local convnet = ConvNet:__init()
convnet:load(convnet_params)

-- get convolution filter stats
local conv_filters = convnet.conv_filters[1]
local filter_size = convnet.conv_filter_sizes[1]

-- open HDF5 and get test sequences
local data_open = hdf5.open(opt.data_file, 'r')
local test_seqs = data_open:read('test_in')

local num_seqs = test_seqs:dataspaceSize()[1]

----------------------------------------------------------------
-- predict
----------------------------------------------------------------
-- predict seqs
convnet.model:evaluate()
convnet:predict(test_seqs, num_seqs)

-- The weights will be difficult to interpret as is. There may be
-- a neat way to relate them to the underlying sequence though.

-- get convolution filter params
local filter_weights = convnet.model.modules[1].weight:squeeze()

-- get convolution filter outputs
local filter_outs = convnet:get_nonlinearity(2).output:squeeze()

-- dump to file, load into python.
local hdf_out = hdf5.open(opt.out_file, 'w')
hdf_out:write('weights', filter_weights)
hdf_out:write('outs', filter_outs)
hdf_out:close()

data_open:close()