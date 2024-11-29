#!/usr/bin/env python
from optparse import OptionParser
import copy
import math
import os
import random
import subprocess
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

sns_colors = sns.color_palette('deep')

from dna_io import one_hot_set, vecs2dna

################################################################################
# basset_anchor.py
#
# Anchor a motif in the center of a set of sequences.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <motif> <model_file> <test_hdf5_file>'
    parser = OptionParser(usage)
    parser.add_option('-d', dest='model_hdf5_file', default=None, help='Pre-computed model output as HDF5.')
    parser.add_option('-f', dest='filters', default=None, help='Filters to plot length analysis [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='.')
    parser.add_option('-p', dest='pool', default=False, action='store_true', help='Take representation after pooling [Default: %default]')
    parser.add_option('-s', dest='sample', default=None, type='int', help='Sequences to sample [Default: %default]')
    parser.add_option('-t', dest='targets', default=None, help='Comma-separated list of targets to analyze in more depth [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 3:
        parser.error('Must provide motif, Basset model file, and test data in HDF5 format.')
    else:
        motif = args[0]
        model_file = args[1]
        test_hdf5_file = args[2]

    random.seed(2)

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    #################################################################
    # load data
    #################################################################
    # load sequences
    test_hdf5_in = h5py.File(test_hdf5_file, 'r')
    seq_vecs = np.array(test_hdf5_in['test_in'])
    seq_targets = np.array(test_hdf5_in['test_out'])
    seq_headers = np.array(test_hdf5_in['test_headers'])
    target_labels = np.array(test_hdf5_in['target_labels'])
    test_hdf5_in.close()


    #################################################################
    # sample
    #################################################################
    if options.sample is not None and options.sample < seq_vecs.shape[0]:
        # choose sampled indexes
        sample_i = np.array(random.sample(xrange(seq_vecs.shape[0]), options.sample))

        # filter
        seq_vecs = seq_vecs[sample_i]
        seq_targets = seq_targets[sample_i]
        seq_headers = seq_headers[sample_i]

        # create a new HDF5 file
        sample_hdf5_file = '%s/sample.h5' % options.out_dir
        sample_hdf5_out = h5py.File(sample_hdf5_file, 'w')
        sample_hdf5_out.create_dataset('test_in', data=seq_vecs)
        sample_hdf5_out.close()

        # update test HDF5
        test_hdf5_file = sample_hdf5_file


    #################################################################
    # write in motif
    #################################################################
    # this code must match the Torch code
    seq_len = seq_vecs.shape[3]
    seq_mid = math.floor(seq_len/2.0 - len(motif)/2.0) - 1
    for si in range(seq_vecs.shape[0]):
        for pi in range(len(motif)):
            one_hot_set(seq_vecs[si], seq_mid+pi, motif[pi])

    # get fasta
    seq_dna = vecs2dna(seq_vecs)

    #################################################################
    # Torch predict
    #################################################################
    if options.model_hdf5_file is None:
        pool_str = ''
        if options.pool:
            pool_str = '-pool'

        options.model_hdf5_file = '%s/model_out.h5' % options.out_dir

        torch_cmd = 'basset_anchor_predict.lua %s %s %s %s %s' % (pool_str, motif, model_file, test_hdf5_file, options.model_hdf5_file)
        print torch_cmd
        subprocess.call(torch_cmd, shell=True)

    # load model output
    model_hdf5_in = h5py.File(options.model_hdf5_file, 'r')
    pre_preds = np.array(model_hdf5_in['pre_preds'])
    preds = np.array(model_hdf5_in['preds'])
    scores = np.array(model_hdf5_in['scores'])
    seq_filter_outs = np.array(model_hdf5_in['filter_outs'])
    pre_seq_filter_outs = np.array(model_hdf5_in['pre_filter_outs'])
    model_hdf5_in.close()

    # pre-process
    seq_filter_means = seq_filter_outs.mean(axis=2)
    filter_means = seq_filter_means.mean(axis=0)
    filter_msds = seq_filter_means.std(axis=0) + 1e-6

    num_seqs = seq_filter_means.shape[0]
    num_filters = seq_filter_means.shape[1]
    num_targets = len(target_labels)

    if options.filters is None:
        options.filters = range(num_filters)
    else:
        options.filters = [int(fi) for fi in options.filters.split(',')]

    if options.targets is None:
        options.targets = range(num_targets)
    else:
        options.targets = [int(ti) for ti in options.targets.split(',')]

    #################################################################
    # scatter plot prediction changes
    #################################################################
    sns.set(style='ticks', font_scale=1.5)
    lim_eps = 0.02

    for ti in options.targets:
        if num_seqs > 500:
            isample = np.array(random.sample(range(num_seqs), 500))
        else:
            isample = np.array(range(num_seqs))

        plt.figure(figsize=(8,8))

        g = sns.jointplot(pre_preds[isample,ti], preds[isample,ti], color='black', stat_func=None, alpha=0.5, space=0)

        ax = g.ax_joint
        ax.plot([0,1], [0,1], c='black', linewidth=1, linestyle='--')

        ax.set_xlim((0-lim_eps, 1+lim_eps))
        ax.set_ylim((0-lim_eps, 1+lim_eps))

        ax.set_xlabel('Pre-insertion accessibility')
        ax.set_ylabel('Post-insertion accessibility')
        ax.grid(True, linestyle=':')

        ax_x = g.ax_marg_x
        ax_x.set_title(target_labels[ti])

        plt.savefig('%s/scatter_t%d.pdf' % (options.out_dir, ti))
        plt.close()


    #################################################################
    # plot sequences
    #################################################################
    for ti in options.targets:
        # sort sequences by score
        seqsi = np.argsort(scores[:,ti])[::-1]

        # print a fasta file with uniformly sampled sequences
        unif_i = np.array([int(sp) for sp in np.arange(0,num_seqs,num_seqs/200.0)])
        seqsi_uniform = seqsi[unif_i]
        fasta_out = open('%s/seqs_t%d.fa' % (options.out_dir,ti), 'w')
        for si in seqsi_uniform:
            print >> fasta_out, '>%s_gc%.2f_p%.2f\n%s' % (seq_headers[si], gc(seq_dna[si]), preds[si,ti], seq_dna[si])
        fasta_out.close()

        # print their filter/pos activations to a table
        #  this is slow and big, and I only need it when I'm trying
        #  to find a specific example.
        table_out = open('%s/seqs_t%d_table.txt' % (options.out_dir, ti), 'w')
        for si in seqsi_uniform:
            for fi in range(num_filters):
                for pi in range(seq_filter_outs.shape[2]):
                    cols = (seq_headers[si], fi, pi, seq_filter_outs[si,fi,pi])
                    print >> table_out, '%-25s  %3d  %3d  %5.2f' % cols
        table_out.close()

        # sample fewer for heat map
        unif_i = np.array([int(sp) for sp in np.arange(0,num_seqs,num_seqs/200.0)])
        seqsi_uniform = seqsi[unif_i]

        ''' these kinda suck
        # plot heat map
        plt.figure()
        n = 20
        ax_sf = plt.subplot2grid((1,n), (0,0), colspan=n-1)
        ax_ss = plt.subplot2grid((1,n), (0,n-1))

        # filter heat
        sf_norm = seq_filter_means[seqsi_uniform,:] - filter_means
        # sf_norm = np.divide(seq_filter_means[seqsi_uniform,:] - filter_means, filter_msds)

        sns.heatmap(sf_norm, vmin=-.04, vmax=.04, xticklabels=False, yticklabels=False, ax=ax_sf)
        # scores heat
        sns.heatmap(scores[seqsi_uniform,ti].reshape(-1,1), xticklabels=False, yticklabels=False, ax=ax_ss)

        # this crashed the program, and I don't know why
        # plt.tight_layout()
        plt.savefig('%s/seqs_t%d.pdf' % (options.out_dir, ti))
        plt.close()
        '''


    #################################################################
    # filter mean correlations
    #################################################################
    # compute and print
    table_out = open('%s/table.txt' % options.out_dir, 'w')
    filter_target_cors = np.zeros((num_filters,num_targets))
    for fi in range(num_filters):
        for ti in range(num_targets):
            cor, p = spearmanr(seq_filter_means[:,fi], scores[:,ti])
            cols = (fi, ti, cor, p)
            print >> table_out, '%-3d  %3d  %6.3f  %6.1e' % cols
            if np.isnan(cor):
                cor = 0
            filter_target_cors[fi,ti] = cor
    table_out.close()

    # plot
    ftc_df = pd.DataFrame(filter_target_cors, columns=target_labels)
    plt.figure()
    g = sns.clustermap(ftc_df)
    for tick in g.ax_heatmap.get_xticklabels():
        tick.set_rotation(-45)
        tick.set_horizontalalignment('left')
        tick.set_fontsize(3)
    for tick in g.ax_heatmap.get_yticklabels():
        tick.set_fontsize(3)
    plt.savefig('%s/filters_targets.pdf' % options.out_dir)
    plt.close()


    #################################################################
    # filter position correlation
    #################################################################
    sns.set(style='ticks', font_scale=1.7)

    table_out = open('%s/filter_pos.txt' % options.out_dir, 'w')

    for fi in options.filters:
        for ti in options.targets:
            print 'Plotting f%d versus t%d' % (fi,ti)

            # compute correlations
            pos_cors = []
            pos_cors_pre = []
            nans = 0
            for pi in range(seq_filter_outs.shape[2]):
                # motif correlation
                cor, p = spearmanr(seq_filter_outs[:,fi,pi], preds[:,ti])
                if np.isnan(cor):
                    cor = 0
                    p = 1
                    nans += 1
                pos_cors.append(cor)

                # pre correlation
                cor_pre, p_pre = spearmanr(pre_seq_filter_outs[:,fi,pi], pre_preds[:,ti])
                if np.isnan(cor_pre):
                    cor_pre = 0
                    p_pre = 1
                pos_cors_pre.append(cor_pre)

                cols = (fi, pi, ti, cor, p, cor_pre, p_pre)
                print >> table_out, '%-3d  %3d  %3d  %6.3f  %6.1e  %6.3f  %6.1e' % cols

            if nans < 50:
                # plot
                # df_pc = pd.DataFrame({'Position':range(len(pos_cors)), 'Correlation':pos_cors})
                plt.figure(figsize=(9,6))
                plt.title(target_labels[ti])
                # sns.regplot(x='Position', y='Correlation', data=df_pc, lowess=True)
                plt.scatter(range(len(pos_cors)), pos_cors_pre, c=sns_colors[2], alpha=0.8, linewidths=0, label='Before motif insertion')
                plt.scatter(range(len(pos_cors)), pos_cors, c=sns_colors[1], alpha=0.8, linewidths=0, label='After motif insertion')
                plt.axhline(y=0, linestyle='--', c='grey', linewidth=1)

                ax = plt.gca()
                ax.set_xlim(0, len(pos_cors))
                ax.set_xlabel('Position')
                ax.set_ylabel('Activation vs Prediction Correlation')
                ax.grid(True, linestyle=':')

                sns.despine()
                plt.legend()
                plt.tight_layout()
                plt.savefig('%s/f%d_t%d.pdf' % (options.out_dir,fi,ti))
                plt.close()

    table_out.close()


def gc(seq):
    ''' Return GC% '''
    gc_count = 0
    for nt in seq:
        if nt == 'C' or nt == 'G':
            gc_count += 1
    return gc_count/float(len(seq))

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
