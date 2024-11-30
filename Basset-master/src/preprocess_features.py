#!/usr/bin/env python
from optparse import OptionParser
import gzip
import os
import re
import subprocess
import sys

import h5py
import numpy as np

################################################################################
# preprocess_features.py
#
# Preprocess a set of feature BED files for Basset analysis, potentially adding
# them to an existing database of features, specified as a BED file with the
# target activities comma-separated in column 4 and a full activity table file.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] <target_beds_file>'
    parser = OptionParser(usage)
    parser.add_option('-a', dest='db_act_file', default=None, help='Existing database activity table.')
    parser.add_option('-b', dest='db_bed', default=None, help='Existing database BED file.')
    parser.add_option('-c', dest='chrom_lengths_file', help='Table of chromosome lengths')
    parser.add_option('-i', dest='ignore_auxiliary', default=False, action='store_true', help='Ignore auxiliary chromosomes that don\'t match "chr\\d+ or chrX" [Default: %default]')
    parser.add_option('-m', dest='merge_overlap', default=200, type='int', help='Overlap length (after extension to feature_size) above which to merge features [Default: %default]')
    parser.add_option('-n', dest='no_db_activity', default=False, action='store_true', help='Do not pass along the activities of the database sequences [Default: %default]')
    parser.add_option('-o', dest='out_prefix', default='features', help='Output file prefix [Default: %default]')
    parser.add_option('-s', dest='feature_size', default=600, type='int', help='Extend features to this size [Default: %default]')
    parser.add_option('-y', dest='ignore_y', default=False, action='store_true', help='Ignore Y chromosome features [Default: %default]')
    (options, args) = parser.parse_args()
    if len(args) != 1:
        parser.error('Must provide file labeling the targets and providing BED file paths.')
    else:
        target_beds_file = args[0]

    # determine whether we'll add to an existing DB
    db_targets = []
    db_add = False
    if options.db_bed is not None:
        db_add = True
        if not options.no_db_activity:
            if options.db_act_file is None:
                parser.error('Must provide both activity table or specify -n if you want to add to an existing database')
            else:
                # read db target names
                with open(options.db_act_file) as db_act_in:
                    db_targets = db_act_in.readline().strip().split('\t')

    # read in targets and assign them indexes into the db
    target_beds = []
    target_dbi = []
    for line in open(target_beds_file):
        a = line.split()
        if len(a) != 2:
            print(a, file=sys.stderr)
            print('Each row of the target BEDS file must contain a label and BED file separated by whitespace', file=sys.stderr)
            exit(1)
        target_dbi.append(len(db_targets))
        db_targets.append(a[0])
        target_beds.append(a[1])

    # read in chromosome lengths
    chrom_lengths = {}
    if options.chrom_lengths_file:
        for line in open(options.chrom_lengths_file):
            a = line.split()
            chrom_lengths[a[0]] = int(a[1])
    else:
        print('Warning: chromosome lengths not provided, so regions near ends may be incorrect.', file=sys.stderr)

    #################################################################
    # print peaks to chromosome-specific files
    #################################################################
    chrom_files = {}
    chrom_outs = {}

    peak_beds = target_beds
    if db_add:
        peak_beds.append(options.db_bed)

    for bi in range(len(peak_beds)):
        if peak_beds[bi][-3:] == '.gz':
            peak_bed_in = gzip.open(peak_beds[bi])
        else:
            peak_bed_in = open(peak_beds[bi])
        change = False
        for line in peak_bed_in:
            if change == False:
                change = True
                continue
            if not line.startswith('#'):
                a = line.split('\t')
                a[-1] = a[-1].rstrip()

                # hash by chrom/strand
                chrom = a[0]
                strand = '+'
                if len(a) > 5 and a[5] in '+-':
                    strand = a[5]
                chrom_key = (chrom, strand)

                # adjust coordinates to midpoint
                print(a)
                start = int(a[1])
                end = int(a[2])
                mid = find_midpoint(start, end)
                a[1] = str(mid)
                a[2] = str(mid + 1)

                # open chromosome file
                if chrom_key not in chrom_outs:
                    chrom_files[chrom_key] = f'{options.out_prefix}_{chrom}_{strand}.bed'
                    chrom_outs[chrom_key] = open(chrom_files[chrom_key], 'w')

                # write to file
                if db_add and bi == len(peak_beds) - 1:
                    if options.no_db_activity:
                        a[6] = '.'
                        print('\t'.join(a[:7]), file=chrom_outs[chrom_key])
                    else:
                        print(line.rstrip(), file=chrom_outs[chrom_key])
                else:
                    while len(a) < 7:
                        a.append('')
                    a[5] = strand
                    a[6] = str(target_dbi[bi])
                    print('\t'.join(a[:7]), file=chrom_outs[chrom_key])

        peak_bed_in.close()

    # close chromosome-specific files
    for chrom_key in chrom_outs:
        chrom_outs[chrom_key].close()

    # ignore Y
    if options.ignore_y:
        for orient in '+-':
            chrom_key = ('chrY', orient)
            if chrom_key in chrom_files:
                print(f'Ignoring chrY {orient}', file=sys.stderr)
                os.remove(chrom_files[chrom_key])
                del chrom_files[chrom_key]

    # ignore auxiliary
    if options.ignore_auxiliary:
        primary_re = re.compile('chr\\d+$')
        for chrom_key in list(chrom_files.keys()):
            chrom, strand = chrom_key
            primary_m = primary_re.match(chrom)
            if not primary_m and chrom != 'chrX':
                print(f'Ignoring {chrom} {strand}', file=sys.stderr)
                os.remove(chrom_files[chrom_key])
                del chrom_files[chrom_key]

    #################################################################
    # sort chromosome-specific files
    #################################################################
    for chrom_key in chrom_files:
        chrom, strand = chrom_key
        chrom_sbed = f'{options.out_prefix}_{chrom}_{strand}_sort.bed'
        sort_cmd = f'sortBed -i {chrom_files[chrom_key]} > {chrom_sbed}'
        subprocess.call(sort_cmd, shell=True)
        os.remove(chrom_files[chrom_key])
        chrom_files[chrom_key] = chrom_sbed

    #################################################################
    # construct/update activity table
    #################################################################
    final_act_out = open(f'{options.out_prefix}_act.txt', 'w')

    # print header
    cols = [''] + db_targets
    print('\t'.join(cols), file=final_act_out)

    # print sequences
    for line in open(f'{options.out_prefix}.bed'):
        a = line.rstrip().split('\t')
        peak_id = f'{a[0]}:{a[1]}-{a[2]}({a[5]})'

        peak_act = [0] * len(db_targets)
        for ai in a[6].split(','):
            if ai != '.':
                peak_act[int(ai)] = 1

        cols = [peak_id] + peak_act
        print('\t'.join(map(str, cols)), file=final_act_out)

    final_act_out.close()


def find_midpoint(start, end):
    return (start + end) // 2


if __name__ == '__main__':
    main()
