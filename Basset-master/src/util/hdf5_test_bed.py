#!/usr/bin/env python
from optparse import OptionParser
import re
import h5py

################################################################################
# hdf5_test_bed.py
#
# Extract a BED of test sequences from an HDF5 file with test_headers that
# specify the region.
################################################################################

################################################################################
# main
############################s####################################################
def main():
    usage = 'usage: %prog [options] <hdf5_file>'
    parser = OptionParser(usage)
    parser.add_option('-c', dest='center_nt', type='int', default=None, help='Center nucleotides to consider [Default: %default]')
    (options,args) = parser.parse_args()

    if len(args) != 1:
        parser.error('Must provide HDF file')
    else:
        hdf5_file = args[0]

    bed_re = re.compile('(chr\w+):(\d+)-(\d+)\(([+-])\)')

    hdf5_in = h5py.File(hdf5_file, 'r')

    for header in hdf5_in['test_headers']:
        bed_m = bed_re.search(header)
        chrom = bed_m.group(1)
        start = int(bed_m.group(2))
        end = int(bed_m.group(3))
        strand = bed_m.group(4)

        if options.center_nt is None:
            cstart = start
            cend = end
        else:
            mid = (end + start)/2
            cstart = mid - options.center_nt/2
            cend = mid + options.center_nt/2

        print '%s\t%d\t%d\t.\t1\t%s' % (chrom,cstart,cend,strand)

    hdf5_in.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
    #pdb.runcall(main)
