#!/usr/bin/env python
from optparse import OptionParser
import copy, os, pdb, random, subprocess, sys

import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns

import dna_io
from seq_logo import seq_logo

################################################################################
# basset_net.py
#
# Visualize the network internals for the given test sequences.
################################################################################

################################################################################
# main
############################s####################################################
def main():
    usage = 'usage: %prog [options] <model_file> <input_file>'
    parser = OptionParser(usage)
    parser.add_option('-a', dest='input_activity_file', help='Optional activity table corresponding to an input FASTA file')
    parser.add_option('-d', dest='model_hdf5_file', default=None, help='Pre-computed model output as HDF5 [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='heat', help='Output directory [Default: %default]')
    parser.add_option('-r', dest='rng_seed', default=1, type='float', help='Random number generator seed [Default: %default]')
    parser.add_option('-s', dest='sample', default=None, type='int', help='Sample sequences from the test set [Default:%default]')
    parser.add_option('-t', dest='targets', default='0', help='Comma-separated list of target indexes to plot (or -1 for all) [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 2:
        parser.error('Must provide Basset model file and input sequences (as a FASTA file or test data in an HDF file')
    else:
        model_file = args[0]
        input_file = args[1]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    random.seed(options.rng_seed)

    #################################################################
    # parse input file
    #################################################################
    try:
        # input_file is FASTA

        # load sequences and headers
        seqs = []
        seq_headers = []
        for line in open(input_file):
            if line[0] == '>':
                seq_headers.append(line[1:].rstrip())
                seqs.append('')
            else:
                seqs[-1] += line.rstrip()

        model_input_hdf5 = '%s/model_in.h5'%options.out_dir

        if options.input_activity_file:
            # one hot code
            seqs_1hot, targets = dna_io.load_data_1hot(input_file, options.input_activity_file, mean_norm=False, whiten=False, permute=False, sort=False)

            # read in target names
            target_labels = open(options.input_activity_file).readline().strip().split('\t')

        else:
            # load sequences
            seqs_1hot = dna_io.load_sequences(input_file, permute=False)
            targets = None
            target_labels = None

        # sample
        if options.sample:
            sample_i = np.array(random.sample(xrange(seqs_1hot.shape[0]), options.sample))
            seqs_1hot = seqs_1hot[sample_i]
            seq_headers = seq_headers[sample_i]
            if targets is not None:
                targets = targets[sample_i]

        # reshape sequences for torch
        seqs_1hot = seqs_1hot.reshape((seqs_1hot.shape[0],4,1,seqs_1hot.shape[1]/4))

        # write as test data to a HDF5 file
        h5f = h5py.File(model_input_hdf5, 'w')
        h5f.create_dataset('test_in', data=seqs_1hot)
        h5f.close()

    except (IOError, IndexError):
        # input_file is HDF5

        try:
            model_input_hdf5 = input_file

            # load (sampled) test data from HDF5
            hdf5_in = h5py.File(input_file, 'r')
            seqs_1hot = np.array(hdf5_in['test_in'])
            targets = np.array(hdf5_in['test_out'])
            try: # TEMP
                seq_headers = np.array(hdf5_in['test_headers'])
                target_labels = np.array(hdf5_in['target_labels'])
            except:
                seq_headers = None
                target_labels = None
            hdf5_in.close()

            # sample
            if options.sample:
                sample_i = np.array(random.sample(xrange(seqs_1hot.shape[0]), options.sample))
                seqs_1hot = seqs_1hot[sample_i]
                seq_headers = seq_headers[sample_i]
                targets = targets[sample_i]

                # write sampled data to a new HDF5 file
                model_input_hdf5 = '%s/model_in.h5'%options.out_dir
                h5f = h5py.File(model_input_hdf5, 'w')
                h5f.create_dataset('test_in', data=seqs_1hot)
                h5f.close()

            # convert to ACGT sequences
            seqs = dna_io.vecs2dna(seqs_1hot)

        except IOError:
            parser.error('Could not parse input file as FASTA or HDF5.')


    #################################################################
    # Torch predict modifications
    #################################################################
    if options.model_hdf5_file is None:
        options.model_hdf5_file = '%s/model_out.h5' % options.out_dir
        torch_cmd = 'basset_net_predict.lua %s %s %s' % (model_file, model_input_hdf5, options.model_hdf5_file)
        print torch_cmd
        subprocess.call(torch_cmd, shell=True)


    #################################################################
    # load modification predictions
    #################################################################
    hdf5_in = h5py.File(options.model_hdf5_file, 'r')
    reprs = []
    l = 1
    while 'reprs%d'%l in hdf5_in.keys():
        reprs.append(np.array(hdf5_in['reprs%d'%l]))
        l += 1
    hdf5_in.close()


    #################################################################
    # plot
    #################################################################
    print len(reprs)
    for l in range(len(reprs)):
        for si in range(len(seq_headers)):
            plt.figure()

            # just write the sequence out above it
            # or maybe I'll ultimately want to write an
            # influence version. yea probably.

            print reprs[l][si].shape
            sns.heatmap(reprs[l][si], linewidths=0, xticklabels=False)
            plt.savefig('%s/%s_l%d.pdf' % (options.out_dir, header_filename(seq_headers[si]), l))
            plt.close()


def header_filename(header):
    ''' Revise the FASTA header to be ba better filename '''
    # no colons
    header = header.replace(':','_')

    # no parentheses
    header = header.replace('(','_')
    header = header.replace(')','')

    return header


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
    #pdb.runcall(main)
