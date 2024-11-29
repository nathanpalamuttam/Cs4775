#!/usr/bin/env th

require 'hdf5'

require 'convnet_io'
require 'postprocess'

----------------------------------------------------------------
-- parse arguments
----------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('DNA ConvNet response to DB motifs')
cmd:text()
cmd:text('Arguments')
cmd:argument('motif')
cmd:argument('model_file')
cmd:argument('data_file')
cmd:argument('out_file')
cmd:text()
cmd:text('Options:')
cmd:option('-batch_size', 256, 'Maximum batch size')
cmd:option('-cuda', false, 'Run on GPGPU')
cmd:option('-pool', false, 'Take representation after pooling')
opt = cmd:parse(arg)

cuda = opt.cuda
require 'convnet'

----------------------------------------------------------------
-- load data
----------------------------------------------------------------
-- construct model
local convnet_params = torch.load(opt.model_file)
local convnet = ConvNet:__init()
convnet:load(convnet_params)
convnet.model:evaluate()

-- open HDF5 and get test sequences
local data_open = hdf5.open(opt.data_file, 'r')
local test_seqs = data_open:read('test_in'):all()
data_open:close()

-- save stats
local num_seqs = (#test_seqs)[1]
local seq_len = (#test_seqs)[4]
local seq_mid = math.floor(seq_len/2 - #opt.motif/2) - 1

----------------------------------------------------------------
-- pre-predict
----------------------------------------------------------------
-- local pre_preds = convnet:predict(test_seqs, opt.batch_size,
local pre_preds, pre_scores, pre_reprs = convnet:predict_reprs(test_seqs, opt.batch_size, true, opt.pool, {1})

----------------------------------------------------------------
-- modify and re-predict
----------------------------------------------------------------
function nt_index(nt)
    if nt == "A" then
        return 1
    elseif nt == "C" then
        return 2
    elseif nt == "G" then
        return 3
    elseif nt == "T" then
        return 4
    else
        error("ACGT only")
    end
end

-- write the motif into the test seqences
for si = 1,num_seqs do
    for pi = 1,#opt.motif do
        -- zero the pos
        for ni = 1,4 do
            test_seqs[si][ni][1][seq_mid+pi] = 0
        end

        -- set the nt
        local nt = opt.motif:sub(pi,pi)
        local ni = nt_index(nt)
        test_seqs[si][ni][1][seq_mid+pi] = 1
    end
end

-- predict
local preds, scores, reprs = convnet:predict_reprs(test_seqs, opt.batch_size, true, opt.pool, {1})

----------------------------------------------------------------
-- dump to file, load into python
----------------------------------------------------------------
local hdf_out = hdf5.open(opt.out_file, 'w')
hdf_out:write('pre_preds', pre_preds)
hdf_out:write('preds', preds)
hdf_out:write('scores', scores)
hdf_out:write('filter_outs', reprs[1])
hdf_out:write('pre_filter_outs', pre_reprs[1])
hdf_out:close()
