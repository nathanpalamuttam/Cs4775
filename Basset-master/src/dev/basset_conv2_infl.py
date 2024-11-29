#!/usr/bin/env python
from optparse import OptionParser
import os
import random
import subprocess

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

################################################################################
# basset_conv2_infl.py
#
# Visualize the 2nd convolution layer of a CNN.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <model_file> <test_hdf5_file>'
    parser = OptionParser(usage)
    parser.add_option('-d', dest='model_hdf5_file', default=None, help='Pre-computed model output as HDF5.')
    parser.add_option('-o', dest='out_dir', default='.')
    parser.add_option('-s', dest='sample', default=None, type='int', help='Sample sequences from the test set [Default:%default]')
    (options,args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide Basset model file and test data in HDF5 format.')
    else:
        model_file = args[0]
        test_hdf5_file = args[1]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    #################################################################
    # load data
    #################################################################
    # load sequences
    test_hdf5_in = h5py.File(test_hdf5_file, 'r')
    seq_vecs = np.array(test_hdf5_in['test_in'])
    seq_targets = np.array(test_hdf5_in['test_out'])
    target_labels = list(test_hdf5_in['target_labels'])
    test_hdf5_in.close()


    #################################################################
    # sample
    #################################################################
    if options.sample is not None:
        # choose sampled indexes
        sample_i = np.array(random.sample(xrange(seq_vecs.shape[0]), options.sample))

        # filter
        seq_vecs = seq_vecs[sample_i]
        seq_targets = seq_targets[sample_i]

        # create a new HDF5 file
        sample_hdf5_file = '%s/sample.h5' % options.out_dir
        sample_hdf5_out = h5py.File(sample_hdf5_file, 'w')
        sample_hdf5_out.create_dataset('test_in', data=seq_vecs)
        sample_hdf5_out.create_dataset('test_out', data=seq_targets)
        sample_hdf5_out.close()

        # update test HDF5
        test_hdf5_file = sample_hdf5_file


    #################################################################
    # Torch predict
    #################################################################
    if options.model_hdf5_file is None:
        options.model_hdf5_file = '%s/model_out.h5' % options.out_dir
        # TEMP
        torch_cmd = './basset_convs_infl.lua -layer 2 %s %s %s' % (model_file, test_hdf5_file, options.model_hdf5_file)
        print torch_cmd
        subprocess.call(torch_cmd, shell=True)

    # load model output
    model_hdf5_in = h5py.File(options.model_hdf5_file, 'r')
    filter_means = np.array(model_hdf5_in['filter_means'])
    filter_stds = np.array(model_hdf5_in['filter_stds'])
    filter_infl = np.array(model_hdf5_in['filter_infl'])
    filter_infl_targets = np.array(model_hdf5_in['filter_infl_targets'])
    model_hdf5_in.close()

    # store useful variables
    num_filters = filter_means.shape[0]
    num_targets = filter_infl_targets.shape[1]

    #############################################################
    # print filter influence table
    #############################################################
    # loss change table
    table_out = open('%s/table_loss.txt' % options.out_dir, 'w')
    for fi in range(num_filters):
        cols = (fi, filter_infl[fi], filter_means[fi], filter_stds[fi])
        print >> table_out, '%3d  %7.4f  %6.4f  %6.3f' % cols
    table_out.close()

    # target change table
    table_out = open('%s/table_target.txt' % options.out_dir, 'w')
    for fi in range(num_filters):
        for ti in range(num_targets):
            cols = (fi, ti, target_labels[ti], filter_infl_targets[fi,ti])
            print >> table_out, '%-3d  %3d  %20s  %7.4f' % cols
    table_out.close()


def plot_filter_heat(weight_matrix, out_pdf):
    ''' Plot a heatmap of the filter's parameters.

    Args
        weight_matrix: np.array of the filter's parameter matrix
        out_pdf
    '''
    weight_range = abs(weight_matrix).max()

    sns.set(font_scale=0.8)
    plt.figure(figsize=(2,12))
    sns.heatmap(weight_matrix, cmap='PRGn', linewidths=0.05, vmin=-weight_range, vmax=weight_range, yticklabels=False)
    ax = plt.gca()
    ax.set_xticklabels(range(1,weight_matrix.shape[1]+1))
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def plot_output_density(f_outputs, out_pdf):
    ''' Plot the output density and compute stats.

    Args
        f_outputs: np.array of the filter's outputs
        out_pdf
    '''
    sns.set(font_scale=1.3)
    plt.figure()
    sns.distplot(f_outputs, kde=False)
    plt.xlabel('ReLU output')
    plt.savefig(out_pdf)
    plt.close()

    return f_outputs.mean(), f_outputs.std()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
    #pdb.runcall(main)
