#!/usr/bin/env th

require 'hdf5'

require 'convnet_io'
require 'postprocess'

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
cmd:text('Options:')
cmd:option('-batch_size', 300, 'Maximum batch size')
cmd:option('-cuda', false, 'Run on GPGPU')
opt = cmd:parse(arg)

-- fix seed
torch.manualSeed(1)

cuda = opt.cuda
require 'convnet'

----------------------------------------------------------------
-- load data
----------------------------------------------------------------
local convnet_params = torch.load(opt.model_file)
local convnet = ConvNet:__init()
convnet:load(convnet_params)
convnet:decuda()

-- open HDF5 and get test sequences
local data_open = hdf5.open(opt.data_file, 'r')
local test_seqs = data_open:read('test_in')

local num_seqs = test_seqs:dataspaceSize()[1]
local seq_len = test_seqs:dataspaceSize()[4]
local nts = {'A','C','G','T'}

----------------------------------------------------------------
-- predict
----------------------------------------------------------------
-- predict seqs
convnet.model:evaluate()
local preds, scores, reprs = convnet:predict_reprs(test_seqs, opt.batch_size)

data_open:close()

----------------------------------------------------------------
-- dump to file, load into python
----------------------------------------------------------------
local hdf_out = hdf5.open(opt.out_file, 'w')
for l = 1,#reprs do
    local repr_name = string.format("reprs%d", l)
    hdf_out:write(repr_name, reprs[l])
end
hdf_out:close()
