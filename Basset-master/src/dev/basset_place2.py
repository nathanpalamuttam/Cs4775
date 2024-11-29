#!/usr/bin/env python
from optparse import OptionParser
import os
import random
import subprocess
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import BayesianRidge, LassoCV, LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor

import dna_io
import stats

################################################################################
# basset_place2.py
#
# Place pairs of filters into N's and study the predictions.
#
# Did not work well. Either N's confuse some models or there was a float bug.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <model_file>'
    parser = OptionParser(usage)
    parser.add_option('-c', dest='center_dist', default=10, type='int', help='Distance between the motifs and sequence center [Default: %default]')
    parser.add_option('-d', dest='model_hdf5_file', default=None, help='Pre-computed model output as HDF5 [Default: %default]')
    parser.add_option('-g', dest='cuda', default=False, action='store_true', help='Run on the GPGPU [Default: %default]')
    parser.add_option('-l', dest='seq_length', default=600, type='int', help='Sequence length [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='heat', help='Output directory [Default: %default]')
    parser.add_option('-t', dest='targets', default='0', help='Comma-separated list of target indexes to plot (or -1 for all) [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 1:
        parser.error('Must provide Basset model file')
    else:
        model_file = args[0]

    out_targets = [int(ti) for ti in options.targets.split(',')]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    random.seed(1)

    # torch options
    cuda_str = ''
    if options.cuda:
        cuda_str = '-cuda'

    #################################################################
    # place filter consensus motifs
    #################################################################
    # determine filter consensus motifs
    filter_consensus = get_filter_consensus(model_file, options.out_dir, cuda_str)

    seqs_1hot = []
    num_filters = len(filter_consensus)
    # num_filters = 40
    filter_len = filter_consensus[0].shape[1]

    # position the motifs
    left_i = options.seq_length/2 - options.center_dist - filter_len
    right_i = options.seq_length/2 + options.center_dist

    ns_1hot = np.zeros((4,options.seq_length)) + 0.25
    # ns_1hot = np.zeros((4,options.seq_length))
    # for i in range(options.seq_length):
    #     nt_i = random.randint(0,3)
    #     ns_1hot[nt_i,i] = 1

    for i in range(num_filters):
        for j in range(num_filters):
            # copy the sequence of N's
            motifs_seq = np.copy(ns_1hot)

            # write them into the one hot coding
            motifs_seq[:,left_i:left_i+filter_len] = filter_consensus[i]
            motifs_seq[:,right_i:right_i+filter_len] = filter_consensus[j]

            # save
            seqs_1hot.append(motifs_seq)

    # make a full array
    seqs_1hot = np.array(seqs_1hot)

    # reshape for spatial
    seqs_1hot = seqs_1hot.reshape((seqs_1hot.shape[0],4,1,options.seq_length))


    #################################################################
    # place filter consensus motifs
    #################################################################
    # save to HDF5
    seqs_file = '%s/motif_seqs.h5' % options.out_dir
    h5f = h5py.File(seqs_file, 'w')
    h5f.create_dataset('test_in', data=seqs_1hot)
    h5f.close()

    # predict scores
    scores_file = '%s/motif_seqs_scores.h5' % options.out_dir
    torch_cmd = 'th basset_place2_predict.lua %s %s %s %s' % (cuda_str, model_file, seqs_file, scores_file)
    subprocess.call(torch_cmd, shell=True)

    # load in scores
    hdf5_in = h5py.File(scores_file, 'r')
    motif_seq_scores = np.array(hdf5_in['scores'])
    hdf5_in.close()

    #################################################################
    # analyze
    #################################################################
    for ti in out_targets:
        #################################################################
        # compute pairwise expectations
        #################################################################
        # X = np.zeros((motif_seq_scores.shape[0],num_filters))
        # xi = 0
        # for i in range(num_filters):
        #     for j in range(num_filters):
        #         X[xi,i] += 1
        #         X[xi,j] += 1
        #         xi += 1

        X = np.zeros((motif_seq_scores.shape[0],2*num_filters))
        xi = 0
        for i in range(num_filters):
            for j in range(num_filters):
                X[xi,i] += 1
                X[xi,num_filters+j] += 1
                xi += 1

        # fit model
        model = BayesianRidge()
        model.fit(X, motif_seq_scores[:,ti])

        # predict pairwise expectations
        motif_seq_preds = model.predict(X)
        print model.score(X, motif_seq_scores[:,ti])

        # print filter coefficients
        coef_out = open('%s/coefs_t%d.txt' % (options.out_dir,ti), 'w')
        for i in range(num_filters):
            print >> coef_out, '%3d  %6.2f' % (i,model.coef_[i])
        coef_out.close()

        #################################################################
        # normalize pairwise predictions
        #################################################################
        filter_interaction = np.zeros((num_filters,num_filters))
        table_out = open('%s/table_t%d.txt' % (options.out_dir,ti), 'w')

        si = 0
        for i in range(num_filters):
            for j in range(num_filters):
                filter_interaction[i,j] = motif_seq_scores[si,ti] - motif_seq_preds[si]
                cols = (i, j, motif_seq_scores[si,ti], motif_seq_preds[si], filter_interaction[i,j])
                print >> table_out, '%3d  %3d  %6.3f  %6.3f  %6.3f' % cols
                si += 1

        table_out.close()

        scores_abs = abs(filter_interaction.flatten())
        max_score = stats.quantile(scores_abs, .999)
        print 'Limiting scores to +-%f' % max_score
        filter_interaction_max = np.zeros((num_filters, num_filters))
        for i in range(num_filters):
            for j in range(num_filters):
                filter_interaction_max[i,j] = np.min([filter_interaction[i,j], max_score])
                filter_interaction_max[i,j] = np.max([filter_interaction_max[i,j], -max_score])

        # plot heat map
        plt.figure()
        sns.heatmap(filter_interaction_max, xticklabels=False, yticklabels=False)
        plt.savefig('%s/heat_t%d.pdf' % (options.out_dir,ti))


def get_filter_consensus(model_file, out_dir, cuda_str=''):
    ''' Determine filter consensus sequences '''

    weights_file = '%s/weights.h5' % out_dir

    # get the weights from torch
    torch_cmd = 'th basset_place2_weights.lua %s %s %s' % (cuda_str, model_file, weights_file)
    subprocess.call(torch_cmd, shell=True)

    # load model output
    hdf5_in = h5py.File(weights_file, 'r')
    filter_weights = np.array(hdf5_in['weights'])
    hdf5_in.close()

    # determine consensus
    filter_consensus = np.zeros(filter_weights.shape)
    for i in range(filter_weights.shape[0]):
        for pos in range(filter_weights.shape[2]):
            nt_col = filter_weights[i,:,pos]
            weight_max = np.max(nt_col)
            filter_consensus[i,:,pos] = 1*(nt_col == weight_max)
            filter_consensus[i,:,pos] /= filter_consensus[i,:,pos].sum()

            '''
            if nt_col[0] == weight_max:
                filter_consensus[i] += 'A'
            elif nt_col[1] == weight_max:
                filter_consensus[i] += 'C'
            elif nt_col[2] == weight_max:
                filter_consensus[i] += 'G'
            else:
                filter_consensus[i] += 'T'
            '''

    return filter_consensus


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
