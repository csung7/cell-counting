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

    for line in sys.stdin:
        data = line.rstrip().split('\t')
        # perform some error checking
        if len(data) < 5:
            continue

        expdepth, vindex, cclass= data[0].rstrip().split(',')
        eigenvstr_row = data[1]
        eigenvstr_col = data[2]
        eigenvstr = data[3]
        meanvectorstr = data[4]

        eigenvarrayflatl = [np.float(di) for di in eigenvstr.split()]
        eigenvarrayflat = np.array(eigenvarrayflatl)
        eigenvarray = eigenvarrayflat.reshape(np.int(eigenvstr_row), np.int(eigenvstr_col))
        meanvectorarrayl = [np.float(di) for di in meanvectorstr.split()]
        meanvectorarray = np.array(meanvectorarrayl)

        #directory = "/home/hduser/hadoop-samples/cell-counting/n_data/"
        directory = ""
    
        l_radius = 5

        validation_files = []

        validation_files.append(directory + "training06_n")

        depth_from_param = np.int(expdepth)
	pca_input_arr, pca_input_value_arr, rand_pointarrays, proximity_pointarrays = ValidationPCA(validation_files,l_radius,depth_from_param)

	pca_databack = np.zeros((pca_input_arr.shape))

	for index in range(pca_input_arr.shape[0]):
	    l_pr = np.dot(pca_input_arr[index,:]-meanvectorarray,eigenvarray)
	    pca_databack[index,:] = np.dot(l_pr,eigenvarray.T).real + meanvectorarray

	dist_err = np.zeros((pca_input_arr.shape[0],))

	######################### euclidean distance #############################
	for ei in range(pca_input_arr.shape[0]):
	    dist_err[ei] = np.linalg.norm(pca_input_arr[ei,:] - pca_databack[ei,:])
	##########################################################################

	#### mr ############################################################
	# MR: print the result that should be the input to the reducer
	for i in range(pca_input_arr.shape[0]):
	    sys.stdout.write('%d,%d,%d\t%d\t%d\t%0.4f\t%0.4f\n' % (rand_pointarrays[i,0], rand_pointarrays[i,1], rand_pointarrays[i,2], np.int(vindex), np.int(cclass), dist_err[i], proximity_pointarrays[i]))
	#####################################################################
