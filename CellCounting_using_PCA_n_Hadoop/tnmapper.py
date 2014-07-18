#!/usr/bin/env python
# encoding: utf-8

# Training the sample files and produce Eigenvector + and -
# input: Depth Volume ID set
# Output: Eigenvector + and - or excel file?

# Author: Chul Sung
# Updated by: Jongwook Woo, jwoo5@calstatela.edu
# Updated by: Chul Sung, daniel.c.sung@gmail.com
# Date: 02/03/2013

import sys, os
import scipy
#import pylab

#sys.path.append("/home/hduser/hadoop-samples/cell-counting/src/")

from cellneuronpca import *
from cellvalidationpca import *

if __name__ ==  '__main__':
    
    #directory = "/home/hduser/hadoop-samples/cell-counting/n_data/"
    directory = ""
    
    l_radius = 5

    testing_files = []

    #01: training05_n
    #02: training08b_n
    #03: training10c_n

    tn_name = ""
    c_index = 0
    # MR: input is new sub-volume
    for arg in sys.stdin:
        c_index = np.int(arg)
        if c_index == 1:
            tn_name = "training05_n"
        elif c_index == 2:
            tn_name = "training08b_n"
        elif c_index == 3:
            tn_name = "training10c_n"
        else: break
        testing_files.append(directory + tn_name)
        posi_coeff,posi_meanvector,neg_coeff,neg_meanvector = NeuronPCA(testing_files,l_radius)

        pevrow, pevcol = np.shape(posi_coeff)
        p_eigenv_str = ' '.join([np.str(np.round(pevi.real,7)) for pevi in posi_coeff.flatten()])
        p_meanv_str = ' '.join([np.str(np.round(pmvi.real,7)) for pmvi in posi_meanvector])

        nevrow, nevcol = np.shape(neg_coeff)
        n_eigenv_str = ' '.join([np.str(np.round(nevi.real,7)) for nevi in neg_coeff.flatten()])
        n_meanv_str = ' '.join([np.str(np.round(nmvi.real,7)) for nmvi in neg_meanvector])

        #print "shape"
        #print np.shape(posi_coeff)
        #print posi_coeff
        #prow, pcol = np.shape(posi_coeff)
        #new_posi_coeff = posi_coeff.flatten()
        #prev_posi_coeff = new_posi_coeff.reshape(prow, pcol)
        #print np.shape(prev_posi_coeff)
        #print prev_posi_coeff

        #print p_eigenv_str
        #p_array_d = [np.float(di) for di in p_eigenv_str.split()]
        #print p_array_d
        for expdepth in range(5,45):
            sys.stdout.write('%d,%d,%d\t%d\t%d\t%s\t%s\n' % (expdepth, c_index, 1, pevrow, pevcol, p_eigenv_str, p_meanv_str))
            sys.stdout.write('%d,%d,%d\t%d\t%d\t%s\t%s\n' % (expdepth, c_index, 0, nevrow, nevcol, n_eigenv_str, n_meanv_str))
