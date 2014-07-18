#!/usr/bin/env python
# encoding: utf-8
#
# Author: Chul Sung
# Updated by: Chul Sung, daniel.c.sung@gmail.com
# Date: 02/03/2013

import numpy as np
import math
import matplotlib.pyplot as plt

def entropy(counts):
    '''Compute entropy.'''
    ps = counts/np.float(np.sum(counts))  # coerce to float and normalize
    ps = ps[np.nonzero(ps)]            # toss out zeros
    H = -np.sum(ps * np.log2(ps))   # compute entropy
    
    return H

def mi(x, y, bins):
    '''Compute mutual information'''
    counts_xy = np.histogram2d(x, y, bins=bins)[0]
    counts_x  = np.histogram(  x,    bins=bins)[0]
    counts_y  = np.histogram(  y,    bins=bins)[0]
    
    H_xy = entropy(counts_xy)
    H_x  = entropy(counts_x)
    H_y  = entropy(counts_y)
    
    return H_x + H_y - H_xy

''' Volume loader '''
def LoadVolume(filename):
    """ Returns a numpy array """
    vol_header = np.fromfile(file=filename, dtype=np.uint32, count=3) 
    x_size = vol_header[0]
    y_size = vol_header[1]
    z_size = vol_header[2]
    
    all_buff = np.fromfile(filename, dtype=np.uint8)
 
    vol_data = np.frombuffer(all_buff, dtype=np.uint8, offset=12)
    
    vol_data.shape = (z_size,y_size,x_size)
    
    ending_msg = "[volume loading]:" + filename
    
    return vol_data

''' Volume saver '''
def SaveVolume(vol_data, filename):
    vol_header = np.uint32([vol_data.shape[2],vol_data.shape[1],vol_data.shape[0]])

    file_id = open(filename, 'w')
    
    file_id.write(vol_header)
    file_id.write(np.uint8(vol_data))
    
    file_id.close()
    ending_msg = "[volume saving]:" + filename

''' Read *.cel file and loading the list of cell points '''
def OBJToPoints(filename):
    ''' Return a numpy.ndarray '''
    pointarrays = np.loadtxt(filename, dtype='d', comments='p', delimiter=' ', usecols=(1,2,3))
    
    return pointarrays

def gauss_kern(r_size=5,sigma=1.5):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    h1 = r_size
    h2 = r_size
    
    x, y = np.mgrid[0:h2, 0:h1]
    x = x-h2/2
    y = y-h1/2

    G = 1/((2*np.pi)*sigma**2 ) * np.exp( - ( x**2 + y**2 ) / (2*sigma**2))
       
    return G/G.sum()

def gauss_kern3D(r_size=3, k_sigma=0.5):
    """ Returns a normalized 3D gauss kernel array for convolutions """
    h0 = r_size
    h1 = r_size
    h2 = r_size
    
    x, y, z = np.mgrid[0:h2, 0:h1, 0:h0]
    
    x = x-h2/2
    y = y-h1/2
    z = z-h0/2

    H = 1/(np.sqrt(2*np.pi)*(2*np.pi)*k_sigma**3 ) * np.exp( - ( x**2 + y**2 + z**2 ) / (2*k_sigma**2))
    
    return H/H.sum()

"""
2D image filter of numpy arrays, via FFT.

Connelly Barnes, public domain 2007.

>>> filter([[1,2],[3,4]], [[0,1,0],[1,1,1],[0,1,0]])    # Filter greyscale image
array([[  8.,  11.],
       [ 14.,  17.]])
>>> A = [[[255,0,0],[0,255,0]],[[0,0,255],[255,0,0]]]   # Filter RGB image
>>> filter(A, gaussian())                               # Sigma: 0.5 (default)
array([[[ 206.4667464 ,   24.2666268 ,   24.2666268 ],
        [  48.5332536 ,  203.57409357,    2.89265282]],

       [[  48.5332536 ,    2.89265282,  203.57409357],
        [ 206.4667464 ,   24.2666268 ,   24.2666268 ]]])
>>> K = gaussian(3)                                     # Sigma: 3
>>> K = gaussian(3, 6)                                  # Sigma: 3 with 6x6 support
(A 6x6 numpy array)

>>> # Load image via PIL, filter, and show result
>>> I = numpy.asarray(Image.open('test.jpg'))
>>> I = filter(I, gaussian(4, (10,10)))
>>> Image.fromarray(numpy.uint8(I)).show()
"""


def gaussian(sigma=0.5, shape=None):
    """
    Gaussian kernel numpy array with given sigma and shape.
    
    The shape argument defaults to ceil(6*sigma).
    """
    sigma = max(abs(sigma), 1e-10)
    if shape is None:
        shape = max(int(6*sigma+0.5), 1)
    if not isinstance(shape, tuple):
        shape = (shape, shape)
    x = np.arange(-(shape[0]-1)/2.0, (shape[0]-1)/2.0+1e-8)
    y = np.arange(-(shape[1]-1)/2.0, (shape[1]-1)/2.0+1e-8)
    Kx = np.exp(-x**2/(2*sigma**2))
    Ky = np.exp(-y**2/(2*sigma**2))
    ans = np.outer(Kx, Ky) / (2.0*np.pi*sigma**2)
    return ans/sum(sum(ans))


def arrayresize(imagetable, colscale, rowscale):
    col, row = imagetable.shape
    colmax = np.uint16(math.floor(col*colscale))
    rowmax = np.uint16(math.floor(row*rowscale))
    
    newtable = np.zeros((colmax, rowmax))
    for i in range(colmax):
        for j in range(rowmax):
            c_col = np.uint16(math.ceil(1/colscale * i))
            c_row = np.uint16(math.ceil(1/rowscale * j))
            newtable[i,j] = imagetable[c_col,c_row]
    
    return newtable

def MakeTargetVolume(point_list, shape, l_radius):
    """
        In this setting, out of boundary in less than 0.4
        arguments:
        * point_list: the list of center points
        * shape: the size of volume (z, y, x)
        * l_radius: local volume radius

        return:
        * gauss_volume: it contained center probability values around center points 

    """
    
    gauss_template = gauss_kern3D(l_radius*2+1, k_sigma=1.5)
    
    gauss_template = gauss_template/np.max(gauss_template)  # scaling gauss_template
    
    #print gauss_template 

    # Create an empty volume
    gauss_volume = np.zeros(shape, np.dtype('d'))
    z_size, y_size, x_size = gauss_volume.shape

    z_max = z_size - 1
    y_max = y_size - 1
    x_max = x_size - 1
    
    count = 0
    for point in point_list:
        x = point[0]
        y = point[1]
        z = point[2]

        c_x_min = x - l_radius
        c_y_min = y - l_radius
        c_z_min = z - l_radius
        
        c_x_max = x + l_radius
        c_y_max = y + l_radius
        c_z_max = z + l_radius
        
        if (c_x_min >= 0) and (c_y_min >= 0) and (c_z_min >= 0) and (c_x_max <= x_max) and (c_y_max <= y_max) and (c_z_max <= z_max):
            gauss_volume[c_z_min:c_z_max+1,c_y_min:c_y_max+1,c_x_min:c_x_max+1] = gauss_template
            count += 1
    
    return gauss_volume
   
def extern_pca(data,k):
    """
        Performs the eigen decomposition of the covariance matrix based
        on the eigen decomposition of the exterior product matrix.
        
        
        arguments:
        * data: 2D numpy array where each row is a sample and
                each column a feature.
        * k: number of principal components to keep.
        
        return:
        * w: the eigen values of the covariance matrix sorted in from 
              highest to lowest.
        * U: the corresponding eigen vectors. u[:,i] is the vector
             corresponding to w[i]
             
        Notes: This function computes PCA, based on the exterior product
               matrix (C = X*X.T/(n-1)) instead of the covariance matrix
               (C = X.T*X) and uses relations based of the singular
               value decomposition to compute the corresponding the
               final eigen vectors. While this can be much faster when 
               the number of samples is much smaller than the number
               of features, it can lead to loss of precisions.
               
               The (centered) data matrix X can be decomposed as:
                    X.T = U * S * v.T
               On computes the eigen decomposition of :
                    X * X.T = v*S^2*v.T
               and the eigen vectors of the covariance matrix are
               computed as :
                    U = X.T * v * S^(-1)
    """
    data_m = data - data.mean(0)
    K = np.dot(data_m,data_m.T)
    w,v = eigsh(K,k = k,which = 'LA')
    U = np.dot(data.T,v/np.sqrt(w))
    return w[::-1]/(len(data)-1),U[:,::-1]

def princomp(A,numpc=2):
    """
    Returns: 
    coeff - sorted normalized eigenvectors of the covariance matrix,
    latent - sorted normalized eigenvalues of the covariance matrix,
    meanvector - mean vector of input data matrix
    """
    
    n = A.shape[0]

    if numpc <= 0 or numpc >= n:
        raise ValueError("numpc must be between 1 and rank(A)-1")
    
    latent,coeff = extern_pca(A,numpc)
    meanvector = A.mean(0)

    # reconstruction formula
    # score = np.dot(A - A.mean(0),coeff)
    # backdata = np.dot(score,v.T) + A.mean(0)
    
    return coeff,latent,meanvector

def pcadataback(A,numpc=0, auto=True, maxratio=0.7):
    """
    Returns: 
    coeff - sorted normalized eigenvectors of the covariance matrix,
    latent - sorted eigenvalues of the covariance matrix,
    Amean - mean vector of input data matrix
    """
    # computing eigenvalues and eigenvectors of covariance matrix
    Amean = np.mean(A.T,axis=1)
    M = (A-Amean).T # subtract the mean (along columns)
    [latent,coeff] = np.linalg.eig(np.cov(M))
    p = np.size(coeff,axis=1)
    idx = np.argsort(latent) # sorting the eigenvalues
    idx = idx[::-1]       # in ascending order
    
    # sorting eigenvectors according to the sorted eigenvalues
    coeff = coeff[:,idx]
    latent = latent[idx] # sorting eigenvalues
    
    sum_latent = sum(latent)
    
    if(auto == True):
        numpc = 0
        accum_latent = 0
        for eigenvalue in latent:
            accum_latent = accum_latent + eigenvalue
            if accum_latent/sum_latent <= maxratio:
                numpc = numpc + 1
            else:
                break
    
    #print "============= # of pc ============="
    #print numpc
    #print "==================================="
    
#    fig = plt.figure() # plot
#    ax1 = fig.add_subplot(211)
#    ax1.set_yscale('log')
#    ax1.set_ylabel('log')
#    ax1.plot(latent, '*r', label = 'A Matrix')
#    plt.show()
    
    if numpc < p or numpc >= 0:
        coeff = coeff[:,range(numpc)] # cutting less important vectors
        latent = latent[:,range(numpc)] # cutting less important evalues
    
    return coeff,latent,Amean

#def pcadataback(A,numpc=2):
#    """
#    Returns: 
#    databack - return pca reduced data
#    """
#    # w - reduced sorted eigenvalues
#    # v - the corresponding eigenvectors 
#    
#    n = A.shape[0]
#
#    if numpc <= 0 or numpc >= n:
#        raise ValueError("numpc must be between 1 and rank(A)-1")
#    
#    w,v = extern_pca(A,numpc)
#
#########  bar graph #######################    
#    num_eigenvals = w.shape[0]
#    
#    fig = plt.figure()
#    ax1 = fig.add_subplot(111)
#
#    x = range(1, num_eigenvals+1)[::-1]
#
#    ax1.bar(x, w[::-1], width=.5, color='#6699CC', edgecolor='#6699CC', align='center')
#
#    #xtick_labels = ['ev1', 'ev2', 'ev3']
#    pos = np.arange(num_eigenvals)
#    #plt.xticks(pos, xtick_labels, color='#4B5320', weight='regular', size='small')
#    plt.xticks(pos, color='#4B5320', weight='regular', size='small')
#    #plt.axis('off')
#    plt.show()
#########  bar graph #######################
#
##    fig = plt.figure() # plot
##    ax1 = fig.add_subplot(211)
##    ax1.set_yscale('log')
##    ax1.set_ylabel('log')
##    ax1.plot(w, '*r', label = 'A Matrix')
##    plt.show()
#
#    pr = np.dot(A - A.mean(0),v)
#    return np.dot(pr,v.T) + A.mean(0)
