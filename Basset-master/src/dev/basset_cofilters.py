#!/usr/bin/env python
from optparse import OptionParser
import copy, math, os, pdb, random, subprocess, sys, time

import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, roc_curve

import dna_io
from seq_logo import seq_logo

sns_colors = sns.color_palette('deep')

################################################################################
# basset_cofilters.py
#
# Study accuracy and co-filter activations from each individual filter's
# perspective.
#
# Very underdeveloped.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <model_file> <test_hdf5_file>'
    parser = OptionParser(usage)
    parser.add_option('-a', dest='act_t', default=0.5, type='float', help='Activation threshold as a percentage of the max used to anchor a filter occurrence [Default: %default]')
    parser.add_option('-d', dest='model_hdf5_file', default=None, help='Pre-computed model output as HDF5 [Default: %default]')
    parser.add_option('-f', dest='filters', default='-1', help='Comma-separated list of filter indexes to plot (or -1 for all) [Default: %default]')
    parser.add_option('-e', dest='draw_heat', default=False, action='store_true', help='Draw heat maps [Default: %default]')
    parser.add_option('-m', dest='motifs_sample', default=1000, type='int', help='Sample motif occurrences for the heatmaps [Default: %default]')
    parser.add_option('-n', dest='center_nt', default=50, type='int', help='Center nt to mutate and plot in the heat map [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='heat', help='Output directory [Default: %default]')
    parser.add_option('-p', dest='pool', default=4, type='int', help='Pool adjacent positions of filter outputs [Default: %default]')
    parser.add_option('-s', dest='sample', default=30000, type='int', help='Sample sequences from the test set [Default:%default]')
    parser.add_option('-t', dest='targets', default='0', help='Comma-separated list of target indexes to plot (or -1 for all) [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide Basset model file and test data in an HDF file')
    else:
        model_file = args[0]
        test_hdf5_file = args[1]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    random.seed(1)

    #################################################################
    # parse input file
    #################################################################
    # load (sampled) test data from HDF5
    hdf5_in = h5py.File(test_hdf5_file, 'r')
    seqs_1hot = np.array(hdf5_in['test_in'])
    seq_targets = np.array(hdf5_in['test_out'])
    seq_headers = np.array(hdf5_in['test_headers'])
    # target_labels = np.array(hdf5_in['target_labels'])
    hdf5_in.close()

    # sample
    if options.sample:
        sample_i = np.array(random.sample(xrange(seqs_1hot.shape[0]), options.sample))
        seqs_1hot = seqs_1hot[sample_i]
        seq_headers = seq_headers[sample_i]
        seq_targets = seq_targets[sample_i]

        # write sampled data to a new HDF5 file
        test_hdf5_file = '%s/model_in.h5'%options.out_dir
        h5f = h5py.File(test_hdf5_file, 'w')
        h5f.create_dataset('test_in', data=seqs_1hot)
        h5f.create_dataset('test_out', data=seq_targets)
        h5f.close()

    # convert to ACGT sequences
    # seqs = dna_io.vecs2dna(seqs_1hot)


    #################################################################
    # Torch predict modifications
    #################################################################
    if options.model_hdf5_file is None:
        options.model_hdf5_file = '%s/model_out.h5' % options.out_dir
        torch_cmd = 'basset_cofilters_predict.lua -norm -pool %d %s %s %s' % (options.pool, model_file, test_hdf5_file, options.model_hdf5_file)
        print torch_cmd
        subprocess.call(torch_cmd, shell=True)


    #################################################################
    # load Torch output
    #################################################################
    hdf5_in = h5py.File(options.model_hdf5_file, 'r')
    preds = np.array(hdf5_in['preds'])
    filter_outs = np.array(hdf5_in['filter_outs'])
    hdf5_in.close()

    num_seqs = filter_outs.shape[0]
    num_filters = filter_outs.shape[1]
    pool_len = filter_outs.shape[2]
    num_targets = seq_targets.shape[1]

    if options.filters == '-1':
        study_filters = range(num_filters)
    else:
        study_filters = [int(fi) for fi in options.filters.split(',')]

    if options.targets == '-1':
        study_targets = range(num_targets)
    else:
        study_targets = [int(t) for t in options.targets.split(',')]


    #################################################################
    # classify sequences by middle filters
    #################################################################
    # determine the threshold to call an occurrence
    filter_means = filter_outs.mean(axis=2).mean(axis=0)
    filter_maxes = filter_outs.max(axis=2).max(axis=0)
    filter_thresh = options.act_t*(filter_maxes - filter_means) + filter_means

    # map out sequence centers
    seq_len = pool_len*options.pool
    sides_len = max(0, seq_len-options.center_nt)
    center_start = int(math.floor(sides_len/2 / options.pool))
    center_end = int(math.ceil((sides_len/2 + options.center_nt) / options.pool))
    filter_outs_cmax = filter_outs[:,:,center_start:center_end].max(axis=2)

    # initialize lists of sequences with each filter
    filter_seqs = []
    for fi in range(num_filters):
        filter_seqs.append([])

    # fill in lists of sequences with each filter
    for si in range(num_seqs):
        for fi in range(num_filters):
            if filter_outs_cmax[si,fi] > filter_thresh[fi]:
                filter_seqs[fi].append(si)

    # sample update to array for easier slicing
    for fi in range(num_filters):
        if len(filter_seqs[fi]) > options.motifs_sample:
            filter_seqs[fi] = np.array(random.sample(filter_seqs[fi], options.motifs_sample))
        else:
            filter_seqs[fi] = np.array(filter_seqs[fi])


    #################################################################
    # measure accuracy
    #################################################################
    auc_out = open('%s/aucs.txt' % options.out_dir, 'w')
    for fi in range(num_filters):
        for ti in range(num_targets):
            if len(filter_seqs[fi]) > 10 and 3 < sum(seq_targets[filter_seqs[fi],ti]) < len(filter_seqs[fi])-3:
                auc = roc_auc_score(seq_targets[filter_seqs[fi],ti], preds[filter_seqs[fi],ti])
                print >> auc_out, '%-3d  %3d  %3d  %.4f' % (fi, ti, len(filter_seqs[fi]), auc)

                if fi in study_filters and ti in study_targets:
                    fpr, tpr, thresh = roc_curve(seq_targets[filter_seqs[fi],ti], preds[filter_seqs[fi],ti])

                    plt.figure(figsize=(6,6))
                    plt.scatter(fpr, tpr, s=8, linewidths=0, c=sns_colors[0])
                    plt.plot([0,1], [0,1], c='black', linewidth=1, linestyle='--')
                    plt.xlabel('False positive rate')
                    plt.ylabel('True positive rate')
                    plt.xlim((0,1))
                    plt.ylim((0,1))
                    plt.grid(True)
                    plt.tight_layout()

                    out_pdf = '%s/roc_f%d_t%d.pdf' % (options.out_dir, fi, ti)
                    plt.savefig(out_pdf)
                    plt.close()
            else:
                print >> auc_out, '%-3d  %3d  %3d  %.4f' % (fi, ti, len(filter_seqs[fi]), 0)

    auc_out.close()

    exit()


    #################################################################
    # study filters
    #################################################################
    preds_means = preds.mean(axis=0)

    table_out = open('%s/filter_target_anchor.txt' % options.out_dir, 'w')
    table2_out = open('%s/filter2_target_anchor.txt' % options.out_dir, 'w')

    for ti in study_targets:
        # pre-compute correlations between the filter's mean
        #  activation and the predictions for this target.
        tpreds = preds[:,ti]
        fcors = np.zeros(num_filters)
        for fi in range(num_filters):
            fouts_mean = filter_outs[:,fi,:].mean(axis=1)
            cor, p = pearsonr(fouts_mean, tpreds)
            fcors[fi] = cor

        # save matrix of conditional correlations
        ff_cor = np.zeros((num_filters,num_filters))

        for fi in study_filters:
            print 'filter%d (%d)' % (fi,len(filter_seqs[fi]))
            if len(filter_seqs[fi]) > 10:
                ###########################################
                # measure filter influence
                ###########################################
                fa_outs = filter_outs[filter_seqs[fi],fi,:]
                fa_preds = preds[filter_seqs[fi],ti]
                fa_targets = seq_targets[filter_seqs[fi],ti]

                fa_preds_mean = fa_preds.mean()

                cols = (fi, ti, len(fa_preds), fa_preds_mean, preds_means[ti], fa_preds_mean-preds_means[ti])
                print >> table_out, '%3d  %3d  %6d  %.4f  %.4f  %7.4f' % cols


                ###########################################
                # plot the filter activations against
                #  predictions
                ###########################################
                if options.draw_heat:
                    # sort sequences by predictions
                    heat_indexes = np.argsort(fa_preds)[::-1]

                    # plot activation heatmap
                    plt.figure(figsize=(4,12))
                    sns.heatmap(fa_outs[heat_indexes], xticklabels=False, yticklabels=False)
                    plt.savefig('%s/f%d_t%d_fheat.pdf' % (options.out_dir,fi,ti))
                    plt.close()

                    pt = np.vstack([fa_targets[heat_indexes], fa_preds[heat_indexes]]).T

                    # plot targets and predictions heatmap
                    plt.figure(figsize=(0.5,12))
                    sns.heatmap(pt, xticklabels=False, yticklabels=False)
                    plt.savefig('%s/f%d_t%d_pheat.pdf' % (options.out_dir,fi,ti))
                    plt.close()


                    # could try for a meta-plot where we plot the
                    # correlation of activation at each position
                    # with the predictions.


                ###########################################
                # compute filter interaction stats
                ###########################################
                # undo the sigmoid
                fa_z = -np.log(fa_preds**-1 - 1)

                # take the filter activation mean across the sequences
                fa_outs_mean = fa_outs.mean(axis=1).squeeze()

                # scatter plot
                plt.figure()
                sns.jointplot(fa_outs_mean, fa_z)
                plt.savefig('%s/f%d_t%d_lm.pdf' % (options.out_dir,fi,ti))
                plt.close()

                # estimate the accessibility from this filter
                cor, p = pearsonr(fa_outs_mean, fa_z)

                # measure accuracy
                ff_cor[fi,fi] = cor

                cols = (fi, fi, ti, cor, p, fcors[fi])
                print >> table2_out, '%3d  %3d  %3d  %7.4f  %6.1e  %7.4f' % cols

                # for each other filter
                # TEMP TEMP
                # for fi2 in range(num_filters):
                for fi2 in [5]:
                    if fi != fi2:
                        # get filter2 outputs
                        f2_outs = filter_outs[filter_seqs[fi],fi2,:]

                        # take the mean
                        f2_outs_mean = f2_outs.mean(axis=1)# .reshape(-1,1)

                        # model
                        cor, p = pearsonr(f2_outs_mean, fa_z)
                        ff_cor[fi,fi2] = cor

                        cols = (fi, fi2, ti, cor, p, fcors[fi2])
                        print >> table2_out, '%3d  %3d  %3d  %7.4f  %6.1e  %7.4f' % cols

                        if options.draw_heat:
                            # plot activation heatmap
                            plt.figure(figsize=(4,12))
                            sns.heatmap(f2_outs[heat_indexes], xticklabels=False, yticklabels=False)
                            plt.savefig('%s/f%d_t%d_f%d_fheat.pdf' % (options.out_dir,fi,ti,fi2))
                            plt.close()

                            plt.figure()
                            sns.lmplot('f2_outs_mean', 'fa_z', pd.DataFrame({'f2_outs_mean':f2_outs_mean, 'fa_z':fa_z}))
                            plt.savefig('%s/f%d_t%d_f%d_lm.pdf' % (options.out_dir, fi, ti, fi2))
                            plt.close()

        '''
        plt.figure()
        sns.clustermap(ff_r2)
        plt.savefig('%s/ff_t%d_heat.pdf' % (options.out_dir,ti))
        plt.close()
        '''

    table_out.close()
    table2_out.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
    #pdb.runcall(main)
