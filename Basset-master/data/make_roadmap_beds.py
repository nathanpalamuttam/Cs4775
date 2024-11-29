#!/usr/bin/env python
from __future__ import print_function
from optparse import OptionParser
import os
import pandas as pd

################################################################################
# make_roadmap_beds.py
#
# Extract names for samples from the spreadsheet.
################################################################################

################################################################################
# main
################################################################################
def main():
    usage = 'usage: %prog [options] arg'
    parser = OptionParser(usage)
    #parser.add_option()
    (options,args) = parser.parse_args()

    df = pd.read_excel('jul2013.roadmapData.qc.xlsx', sheet_name='Consolidated_EpigenomeIDs_summa')

    beds_out = open('roadmap_beds.txt', 'w')

    for i in range(df.shape[0]):
    	eid = df.iloc[i,1]

    	peaks_bed = 'roadmap/%s-DNase.hotspot.fdr0.01.peaks.bed.gz' % eid
    	if os.path.isfile(peaks_bed):
    		print('%s\t%s' % (df.iloc[i,5], peaks_bed), file=beds_out)

    beds_out.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
