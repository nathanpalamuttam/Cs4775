#!/usr/bin/env python
from optparse import OptionParser
import copy
import math
import os
import random
import string
import subprocess
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.stats import spearmanr
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

sns_colors = sns.color_palette('deep')

from dna_io import one_hot_set, vecs2dna

################################################################################
# basset_kmers.py
#
# Generate random sequences and study scores by k-mers.
#
# Draw as graph:
#  -construct the graph w/ all single edits as edges.
#  -perform a force-directed layout.
#  -label the k-mers.
#  -color by score.
#  -http://networkx.github.io/documentation/latest/gallery.html
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <model_file>'
    parser = OptionParser(usage)
    parser.add_option('-a', dest='targets_file', default=None, help='File labelings targets in the second column [Default: %default]')
    parser.add_option('-c', dest='center_nt', default=50, help='Center nt to consider kmers from [Default: %default]')
    parser.add_option('-d', dest='model_out_file', default=None, help='Pre-computed model output table.')
    parser.add_option('-k', dest='kmer', default=8, type='int', help='K-mer length [Default: %default]')
    parser.add_option('-l', dest='seq_len', default=1000, type='int', help='Input sequence length [Default: %default]')
    parser.add_option('-n', dest='num_seqs', default=100000, type='int', help='Number of sequences to predict [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='.')
    parser.add_option('-r', dest='rc', default=False, action='store_true', help='Consider k-mers w/ their reverse complements [Default: %default]')
    parser.add_option('-t', dest='targets', default=None, help='Comma-separated list of targets to analyze in more depth [Default: %default]')
    parser.add_option('--top', dest='top_num', default=100, type='int', help='Number of sequences with which to make a multiple sequence alignment')
    (options,args) = parser.parse_args()

    if len(args) != 1:
        parser.error('Must provide Basset model file.')
    else:
        model_file = args[0]

    random.seed(2)

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    if options.model_out_file is not None:
        seq_dna = []
        for line in open('%s/seqs.fa' % options.out_dir):
            if line[0] == '>':
                seq_dna.append('')
            else:
                seq_dna[-1] += line.rstrip()

    else:
        #################################################################
        # generate random sequences
        #################################################################
        # random sequences
        seq_vecs = np.zeros((options.num_seqs,4,1,options.seq_len), dtype='float16')
        for si in range(options.num_seqs):
            for li in range(options.seq_len):
                ni = random.randint(0,3)
                seq_vecs[si,ni,0,li] = 1

        # create a new HDF5 file
        seq_hdf5_file = '%s/seqs.h5' % options.out_dir
        seq_hdf5_out = h5py.File(seq_hdf5_file, 'w')
        seq_hdf5_out.create_dataset('test_in', data=seq_vecs)
        seq_hdf5_out.close()

        # get fasta
        seq_dna = vecs2dna(seq_vecs)

        # print to file
        fasta_out = open('%s/seqs.fa' % options.out_dir, 'w')
        for i in range(len(seq_dna)):
            print >> fasta_out, '>%d\n%s' % (i,seq_dna[i])
        fasta_out.close()

        #################################################################
        # Torch predict
        #################################################################
        options.model_out_file = '%s/model_out.txt' % options.out_dir

        torch_cmd = 'basset_predict.lua -scores %s %s %s' % (model_file, seq_hdf5_file, options.model_out_file)
        print torch_cmd
        subprocess.call(torch_cmd, shell=True)

        # clean up sequence HDF5
        os.remove(seq_hdf5_file)

    # load scores
    seq_scores = np.loadtxt(options.model_out_file, dtype='float32')

    # read target labels
    if options.targets_file:
        target_labels = [line.split()[1] for line in open(options.targets_file)]
    else:
        target_labels = ['t%d'%(ti+1) for ti in range(seq_scores.shape[1])]

    if options.targets is None:
        options.targets = range(seq_scores.shape[1])
    else:
        options.targets = [int(ti) for ti in options.targets.split(',')]


    #################################################################
    # process and output
    #################################################################
    kmers_start = (options.seq_len - options.center_nt) / 2

    for ti in options.targets:
        print 'Working on target %d' % ti

        ##############################################
        # hash scores by k-mer
        ##############################################
        kmer_scores_raw = {}

        for si in range(len(seq_dna)):
            # get score
            sscore = seq_scores[si,ti]

            # hash to each center kmer
            for ki in range(kmers_start, kmers_start + options.center_nt):
                kmer = seq_dna[si][ki:ki+options.kmer]
                if options.rc:
                    kmer = consider_rc(kmer)

                kmer_scores_raw.setdefault(kmer,[]).append(sscore)

        ##############################################
        # compute means and print table
        ##############################################
        table_out = open('%s/table%d.txt' % (options.out_dir,ti), 'w')

        kmer_means_raw = {}
        for kmer in kmer_scores_raw:
            kmer_means_raw[kmer] = np.mean(kmer_scores_raw[kmer])
            kmer_n = len(kmer_scores_raw[kmer])
            cols = (kmer, kmer_n, kmer_means_raw[kmer], np.std(kmer_scores_raw[kmer])/math.sqrt(kmer_n))
            print >> table_out, '%s  %4d  %6.3f  %6.3f' % cols

        table_out.close()

        ##############################################
        # plot density
        ##############################################
        plt.figure()
        sns.distplot(kmer_means_raw.values(), kde=False)
        plt.savefig('%s/density%d.pdf' % (options.out_dir,ti))
        plt.close()

        ##############################################
        # top k-mers distance matrix
        ##############################################
        kmer_means = {}
        kmer_means_mean = np.mean(kmer_means_raw.values())
        for kmer in kmer_means_raw:
            kmer_means[kmer] = kmer_means_raw[kmer] - kmer_means_mean

        # score by score
        scores_kmers = [(kmer_means[kmer],kmer) for kmer in kmer_means]
        scores_kmers.sort(reverse=True)

        # take top k-mers
        top_kmers = []
        top_kmers_scores = []
        for score, kmer in scores_kmers[:options.top_num]:
            top_kmers.append(kmer)
            top_kmers_scores.append(score)
        top_kmers = np.array(top_kmers)
        top_kmers_scores = np.array(top_kmers_scores)

        # compute distance matrix
        top_kmers_dists = np.zeros((options.top_num, options.top_num))
        for i in range(options.top_num):
            for j in range(i+1,options.top_num):
                if options.rc:
                    top_kmers_dists[i,j] = kmer_distance_rc(top_kmers[i], top_kmers[j])
                else:
                    top_kmers_dists[i,j] = kmer_distance(top_kmers[i], top_kmers[j])
                top_kmers_dists[j,i] = top_kmers_dists[i,j]

        # clip the distances
        np.clip(top_kmers_dists, 0, 3, out=top_kmers_dists)

        # plot
        plot_kmer_dists(top_kmers_dists, top_kmers_scores, top_kmers, '%s/top_kmers_heat%d.pdf'%(options.out_dir,ti))

        # cluster and plot
        cluster_kmer_dists(top_kmers_dists, top_kmers_scores, top_kmers, '%s/top_kmers_clust%d.pdf'%(options.out_dir,ti))


def consider_rc(kmer):
    rc_kmer = rc(kmer)
    if rc_kmer < kmer:
        return rc_kmer
    else:
        return kmer


def kmer_distance(x, y, max_shifts=1):
    ''' Compute the edit distance between two kmers

    Might consider trying scikit-bio global_pairwise_align_nucleotide.

    '''

    # no shifts
    min_d = 0
    for i in range(len(x)):
        if x[i] != y[i]:
            min_d += 1

    # shifts
    for s in range(1,max_shifts+1):
        # shift x
        d = 1
        for i in range(len(x)-s):
            if x[s+i] != y[i]:
                d += 1
        if d < min_d:
            min_d = d

        # shift y
        d = 1
        for i in range(len(y)-s):
            if x[i] != y[s+i]:
                d += 1
        if d < min_d:
            min_d = d

    return min_d


def kmer_distance_rc(x, y):
    ''' Compute the edit distance between two kmers,
        considering the reverse complements. '''

    d_fwd = kmer_distance(x, y)
    d_rc = kmer_distance(x, rc(y))
    return min(d_fwd, d_rc)


def plot_kmer_dists(kmers_dists, kmers_scores, kmers, out_pdf):
    ''' Plot a heatmap of k-mer distances and scores.'''

    # shape scores
    kmers_scores = kmers_scores.reshape((-1,1))

    cols = 20
    plt.figure()
    ax_dist = plt.subplot2grid((1,cols), (0,0), colspan=cols-1)
    ax_score = plt.subplot2grid((1,cols), (0,cols-1), colspan=1)

    sns.heatmap(kmers_dists, cmap=sns.cubehelix_palette(n_colors=(1+kmers_dists.max()), reverse=True, as_cmap=True), ax=ax_dist, yticklabels=kmers, xticklabels=False)
    for tick in ax_dist.get_yticklabels():
        if kmers_dists.shape[0] <= 100:
            tick.set_fontsize(4)
        elif kmers_dists.shape[0] <= 250:
            tick.set_fontsize(2.5)
        else:
            tick.set_fontsize(2)

    score_max = kmers_scores.max()
    sns.heatmap(kmers_scores, cmap = 'RdBu_r', vmin=-score_max, vmax=score_max, ax=ax_score, yticklabels=False, xticklabels=False)

    plt.savefig(out_pdf)
    plt.close()


def cluster_kmer_dists(kmers_dists, kmers_scores, kmers, out_pdf):
    ''' Plot a clustered heatmap of k-mer distances and scores.'''

    # cluster
    kmer_cluster = hierarchy.linkage(kmers_dists, method='single', metric='euclidean')
    order = hierarchy.leaves_list(kmer_cluster)

    # re-order distance matrix
    kmers_dists_reorder = kmers_dists[order,:]
    kmers_dists_reorder = kmers_dists_reorder[:,order]

    # plot
    plot_kmer_dists(kmers_dists_reorder, kmers_scores[order], kmers[order], out_pdf)


def rc(seq):
    ''' Reverse complement sequence'''
    return seq.translate(string.maketrans("ATCGatcg","TAGCtagc"))[::-1]


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
