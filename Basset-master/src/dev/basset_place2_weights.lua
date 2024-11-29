#!/usr/bin/env th

require 'hdf5'

require 'batcher'
require 'convnet_io'

----------------------------------------------------------------
-- parse arguments
----------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('DNA ConvNet hidden layer visualizations')
cmd:text()
cmd:text('Arguments')
cmd:argument('model_file')
cmd:argument('out_file')
cmd:text()
cmd:option('-cuda', false, 'Run on GPGPU')
cmd:text()
opt = cmd:parse(arg)

-- fix seed
torch.manualSeed(1)

-- set cpu/gpu
cuda = opt.cuda
require 'convnet'

----------------------------------------------------------------
-- load data
----------------------------------------------------------------
local convnet_params = torch.load(opt.model_file)
local convnet = ConvNet:__init()
convnet:load(convnet_params)

-- get convolution filter params
local filter_weights = convnet.model.modules[1].weight:squeeze()

-- dump to file, load into python.
local hdf_out = hdf5.open(opt.out_file, 'w')
hdf_out:write('weights', filter_weights)
hdf_out:close()
