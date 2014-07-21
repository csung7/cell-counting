#!/usr/bin/env python
# encoding: utf-8

# kpcabound(data,sigma,numev)
#
# kpcabound demonstrates 'Kernel PCA for novelty detection'
#
# kpcabound plots the reconstruction error in feature space into the 
# original space and plots the decision boundary enclosing all data points
# 
# input:
#
# data: array of data points, one row for each data point
# sigma: width of Gaussian kernel
# numev: number of eigenvalues to be extracted

import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.sparse.linalg import eigs 
from scipy import diag

# k=kernel(x,y,param)
#
# Kernel function for KPCA. Here, a Gaussian kernel is implemented.
# You may changed this function to try other kernels.
#
# x,y: two data points
# param: parameter of kernel function
def kernel(x,y,param):
    diff = x-y
    return np.exp(-np.dot(diff,diff.T)*param)

# err = recerr(x,data,param,alpha,alphaK_row_mean,sumalpha,K_mean)
# 
# This function computes the reconstruction error of x in feature
# space.
#
# x: test point
# data: data set from which KPCA was computed
# param: kernel parameter
# alpha,alphaK_row_mean,sumalpha,K_mean: resulting data from KPCA
# (see kpcabound.m)
def recerr(x,data,param,v_alpha,alphaK_row_mean,sum_alpha,K_mean):
    
    n = data.shape[0]
    k = np.zeros((n,))
    for j in range(n):
        k[j] = kernel(x,data[j,:],param)

    # projections:
    f = np.dot(k,v_alpha) - sum_alpha*(np.sum(k)/n - K_mean) - alphaK_row_mean
    
    # reconstruction error:
    err = kernel(x,x,param) - 2*np.sum(k)/n + K_mean - np.dot(f,f.T)
    return err 
   
def kpcabound(data,sigma,numev):
   
    n,d = data.shape
    # n : number of data points
    # d : dimension of data points

    # kernel matrix:
    K = np.zeros((n,n))
    
    # kernel parameter:
    param = 0.5/(sigma*sigma)
    
    #print 'computing kernel matrix K\n'
    
    for i in range(n):
        for j in range(i,n):
            K[i,j] = kernel(data[i,:],data[j,:],param)
            K[j,i] = K[i,j]
 
    # correct K for non-zero center of data in feature space:
#    onemat = (np.ones((n,n))/n)
#    K_norm = K-np.dot(onemat,K) - np.dot(K,onemat) + np.dot(np.dot(onemat,K),onemat)

    # correct K for non-zero center of data in feature space:
    K_row_mean = np.sum(K,0)/n
    K_mean = np.sum(K_row_mean)/n
    
    for i in range(n):
        for j in range(n):
            K[i,j] = K[i,j] - K_row_mean[i] - K_row_mean[j] + K_mean

    # print 'extracting eigenvectors of K\n'
    if numev <= 0 or numev >= n:
        raise ValueError("numev must be between 1 and rank(A)-1")

    # print K
#    w_lambda, v_alpha = eigsh(K, k = numev)
#    
#    # sort eignenvectors in descending order
#    indices = w_lambda.argsort()[::-1]
#    w_lambda = w_lambda[indices]
#    v_alpha = v_alpha[:, indices]
#
##    w_lambda, v_alpha = eigs(K, numev)
##    idx = np.argsort(w_lambda)     # sorting the eigenvalues
##    idx = idx[::-1]             # in ascending order
##    
##    # sorting eigenvectors according to the sorted eigenvalues
##    w_lambda = w_lambda[idx] # sorting eigenvalues
##    v_alpha = v_alpha[:,idx]
#
#    # remove eigenvectors with a zero eigenvalue
#    w_lambda = w_lambda[w_lambda > 0]
#    v_alpha = v_alpha[:, w_lambda > 0]
    u, w_lambda, v_alpha = np.linalg.svd(K)
    # w_lambda, v_alpha = np.linalg.eig(K)
    
    v_alpha = v_alpha.T
    
    idx = np.argsort(w_lambda)     # sorting the eigenvalues
    idx = idx[::-1]             # in ascending order
    # sorting eigenvectors according to the sorted eigenvalues
    v_alpha = v_alpha[:,idx]
    w_lambda = w_lambda[idx] # sorting eigenvalues
    v_alpha = v_alpha[:,range(numev)] # cutting some PCs
    w_lambda = w_lambda[range(numev)]
  

    # residual variance:
    resvar = np.trace(K) - np.sum(w_lambda)

    # normalize alpha:
#    for iv in range(v_alpha.shape[1]):
#        v_alpha[:,iv] = v_alpha[:,iv] / np.sqrt(w_lambda[iv])

    v_alpha = np.dot(v_alpha, np.linalg.inv(np.sqrt(diag(w_lambda))))
    
    #linalg.inv(a)
    
    # compute some helper vectors:
    sum_alpha = np.sum(v_alpha,0)
    alphaK_row_mean = np.dot(K_row_mean, v_alpha)
    
    # print 'evaluating reconstruction error for all data points\n'
    err = np.zeros((n,))

#    print param
#    print v_alpha
#    print alphaK_row_mean
#    print sum_alpha
#    print K_mean

    for i in range(n):
        x = data[i,:]   # test point
        err[i] = recerr(x,data,param,v_alpha,alphaK_row_mean,sum_alpha,K_mean)

    serr = np.round(np.sort(err)[::-1],4)
    
    # print serr

    return err

    
x = [2.5, 0.5, 2.2, 1.9, 3.1, 2.3, 2, 1, 1.5, 1.1]
y = [2.4, 0.7, 2.9, 2.2, 3.0, 2.7, 1.6, 1.1, 1.6, 0.9]
pca_input_mtx = []

pca_input_mtx.append(x)
pca_input_mtx.append(y)

pca_input_arr = np.array(pca_input_mtx, 'd')

A = pca_input_arr.T


kpcabound(A,0.4,7)

a1 = np.array([1,0,3,4,3])
a2 = np.array([2,1,2,3,1])

dist = np.linalg.norm(a2 - a1)
# print dist

