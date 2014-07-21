from cellcommon import *
from cellkpcabound import *
from sklearn.decomposition.pca import PCA
from sklearn.decomposition.kernel_pca import KernelPCA
from pylab import *
import random

def Positive_PCA_Cal(posi_pca_input_arr,l_radius,planenum,planeselect):

#    ################ Kernel PCA Reconstruction Error
#    numev = np.round((posi_pca_input_arr.shape[0]/5)*4)
#    # numev = posi_pca_input_arr.shape[0]-1
#    err = kpcabound(posi_pca_input_arr,0.4,numev)
#    # print err
#    
#    print posi_pca_input_arr.shape
#    
#    p_index = np.zeros((posi_pca_input_arr.shape[0],))
#    for ik in range (posi_pca_input_arr.shape[0]):
#        p_index[ik] = ik+1
#    
#    bad_data = err[err >= 0.5]
#    good_data = err[err < 0.5]
#    print bad_data.__len__()
#    print good_data.__len__()
#    accuracy = (np.float(good_data.__len__())/np.float(err.__len__()))*100
#    print accuracy
#
#    f6 = figure(frameon=False)
#    ax1 = f6.add_subplot(2,1,1)
#    #ax1.plot(err[err > 0.5], 'ro')
#    #ax1.plot(err[err < 0.5], 'go')
#    ax1.bar(p_index[err >= 0.5],err[err >= 0.5], width=.5, color='red', align='center')
#    ax1.bar(p_index[err < 0.5],err[err < 0.5], width=.5, color='green', align='center')
#    
#    ax1.legend(['err >= 0.5','err < 0.5'], loc='upper right')
#    # line, = ax1.semilogy(err, color='blue', lw=2)
#    show()
#    ###################################################
    
    
#    e_indices = err.argsort()[::-1]
#    err = err[e_indices]
#    posi_pca_input_arr = posi_pca_input_arr[e_indices, :]
#
#    # remove outlier
#    outlier_thr = 0.4
#    err = err[err < outlier_thr]
#    posi_pca_input_arr = posi_pca_input_arr[err < outlier_thr, :]
#    
#    print posi_pca_input_arr.shape
#     
#    numev = np.round((posi_pca_input_arr.shape[0]/10)*8)
#    # numev = posi_pca_input_arr.shape[0]-1
#    err2 = kpcabound(posi_pca_input_arr,1.5,numev)
#    print err2
#    
#    p_index = np.zeros((posi_pca_input_arr.shape[0],))
#    for ik in range (posi_pca_input_arr.shape[0]):
#        p_index[ik] = ik+1
#    
#    f61 = figure(frameon=False)
#    ax1 = f61.add_subplot(2,1,1)
#    ax1.plot(err2, 'ro')
#    # line, = ax1.semilogy(err2, color='red', lw=2)
#    show()

    #posi_coeff,posi_latent,posi_meanvector = pcadataback(posi_pca_input_arr,0,True,maxratio=0.4)
    posi_coeff,posi_latent,posi_meanvector = pcadataback(posi_pca_input_arr,5,False,maxratio=0.4)
#    posi_databack = np.zeros((posi_pca_input_arr.shape))
#    for p_index in range(posi_pca_input_arr.shape[0]):
#        l_pr = np.dot(posi_pca_input_arr[p_index,:]-posi_meanvector,posi_coeff)
#        posi_databack[p_index,:] = np.dot(l_pr,posi_coeff.T) + posi_meanvector

#    dist_err = np.zeros((posi_databack.shape[0],))
#    e_index = np.zeros((posi_databack.shape[0],))
#    
#    ######################### euclidean distance #############################
#    for ei in range(posi_pca_input_arr.shape[0]):
#        dist_err[ei] = np.linalg.norm(posi_pca_input_arr[ei,:] - posi_databack[ei,:])
#        e_index[ei] = ei + 1 
#    ##########################################################################
    
    ######################### mutual information #############################
#    for ei in range(posi_pca_input_arr.shape[0]):
#        org_xyzimg = np.uint8(posi_pca_input_arr[ei,:].round().reshape((l_radius*2+1)**2, planenum))
#        cvt_xyzimg = np.uint8(posi_databack[ei,:].round().reshape((l_radius*2+1)**2, planenum))
#        dist_err[ei] = mi(org_xyzimg[:,0],cvt_xyzimg[:,0],256)
#        #print dist_err[ei] 
#        e_index[ei] = ei + 1
    ##########################################################################
    
    #################### reconstruction graphs ###############################
#    f3 = figure(frameon=False)
#    f3.suptitle('test')
#    ax1 = f3.add_subplot(2,1,1)
#    ax1.bar(e_index,dist_err, width=.5, color='green', align='center')
#    ax1.legend(['+ mi'], loc='upper right')
#    xlabel('index of cells')
#    ylabel('mutual information')
#    show()
    ##########################################################################
    
    ###########################
    ## KernelPCA
    ###########################
    # eigen_solver in ("auto", "dense", "arpack"):
    # kernel in ("linear", "rbf", "poly"):
#    kpca = KernelPCA(kernel="rbf", gamma=0.5, fit_inverse_transform=True, eigen_solver='auto')
#    X_kpca = kpca.fit_transform(posi_pca_input_arr)
#    posi_databack = kpca.inverse_transform(X_kpca)

    ###########################
    ## General PCA
    ###########################
#        pca = PCA(n_components=2)
#        pca.fit(pca_input_arr)
#        #eigenvectors
#        # print pca.components_
#        
#        #get value proportion of each eigenvalue:
#        # eva2 = np.cumsum(pca.explained_variance_/np.sum(pca.explained_variance_))
#        # print eva2
#        
#        trans_data = pca.transform(posi_pca_input_arr)
#        databack = pca.inverse_transform(trans_data)
    
    
    
    #################### pca_transformed cell bodies #########################
#    posi_databack = posi_databack.round()
#
#    planeimgsize = (l_radius*2+1)**2
#
#    f2 = figure(frameon=False)
#
#    if planeselect == 0:
#        f2.suptitle('X-Y planes with PCA dimensionality reduction', fontsize=16)
#    elif planeselect == 1:
#        f2.suptitle('X-Z planes with PCA dimensionality reduction', fontsize=16)
#    else:
#        f2.suptitle('Y-Z planes with PCA dimensionality reduction', fontsize=16)
#    fig_col = math.ceil(posi_databack.shape[0]/10)
#
#    print posi_databack.shape
#    ccc = 1
#    for i in range(posi_databack.shape[0]):
#        f2.add_subplot(fig_col+1, 10, ccc)  # this line outputs images on top of each other
#        xyzimg = np.uint8(posi_databack[i,:].reshape(planeimgsize, planenum))
#        xyimg = xyzimg[:,0].reshape(l_radius*2+1,l_radius*2+1)
#        
#        xzimg = xyzimg[:,1].reshape(l_radius*2+1,l_radius*2+1)
#        yzimg = xyzimg[:,2].reshape(l_radius*2+1,l_radius*2+1)
#        if planeselect == 0:
#            imshow(xyimg,cmap=cm.Greys_r)
#        elif planeselect == 1:
#            imshow(xzimg,cmap=cm.Greys_r)
#        else:
#            imshow(yzimg,cmap=cm.Greys_r)
#        axis('off')
#        ccc += 1
#    show()
    ##########################################################################    
    return posi_coeff,posi_meanvector


def Negative_PCA_Cal(neg_pca_input_arr,l_radius,planenum,planeselect):
    # Linear PCA
#    neg_coeff,neg_meanvector = princomp(neg_pca_input_arr,2)
#    neg_databack = np.zeros((neg_pca_input_arr.shape))
#    for n_index in range(neg_pca_input_arr.shape[0]):
#        l_pr = np.dot(neg_pca_input_arr[n_index,:]-neg_meanvector,neg_coeff)
#        neg_databack[n_index,:] = np.dot(l_pr,neg_coeff.T) + neg_meanvector


    #neg_coeff,neg_latent,neg_meanvector = pcadataback(neg_pca_input_arr,0,True,maxratio=0.4)
    neg_coeff,neg_latent,neg_meanvector = pcadataback(neg_pca_input_arr,3,False,maxratio=0.4)

    return neg_coeff,neg_meanvector
#
#    
#    numev = np.round((neg_pca_input_arr.shape[0]/4)*3)
#    # numev = neg_pca_input_arr.shape[0]-1
#    err = kpcabound(neg_pca_input_arr,1.5,numev)
#    #print err
#    
#    p_index = np.zeros((neg_pca_input_arr.shape[0],))
#    for ik in range (neg_pca_input_arr.shape[0]):
#        p_index[ik] = ik+1
#    
#    ###########################
#    ## KernelPCA
#    ###########################
#    kpca = KernelPCA(kernel="rbf", gamma=0.5, fit_inverse_transform=True, eigen_solver='auto')
#    X_kpca = kpca.fit_transform(neg_pca_input_arr)
#    neg_databack = kpca.inverse_transform(X_kpca)
#
#    neg_databack = neg_databack.round()
#    planeimgsize = (l_radius*2+1)**2
#    
#    f4 = figure(frameon=False)
#
#    if planeselect == 0:
#        f4.suptitle('X-Y planes with neg-PCA dimensionality reduction', fontsize=16)
#    elif planeselect == 1:
#        f4.suptitle('X-Z planes with neg-PCA dimensionality reduction', fontsize=16)
#    else:
#        f4.suptitle('Y-Z planes with neg-PCA dimensionality reduction', fontsize=16)
#    fig_col = math.ceil(neg_databack.shape[0]/10)
#
#    ccc = 1
#    for i in range(neg_databack.shape[0]):
#        f4.add_subplot(fig_col+1, 10, ccc)  # this line outputs images on top of each other
#        xyzimg = np.uint8(neg_databack[i,:].reshape(planeimgsize, planenum))
#        xyimg = xyzimg[:,0].reshape(l_radius*2+1,l_radius*2+1)
#        xzimg = xyzimg[:,1].reshape(l_radius*2+1,l_radius*2+1)
#        yzimg = xyzimg[:,2].reshape(l_radius*2+1,l_radius*2+1)
#        if planeselect == 0:
#            imshow(xyimg,cmap=cm.Greys_r)
#        elif planeselect == 1:
#            imshow(xzimg,cmap=cm.Greys_r)
#        else:
#            imshow(yzimg,cmap=cm.Greys_r)
#        axis('off')
#        ccc += 1
#    
#    show()
    
    
def Negative_Generate_Points_ND_PCA_Cal(vol_data,pointarrays,pos_pnum,target_volume,l_radius,outofbound_thr):
    z_size, y_size, x_size = vol_data.shape
    
    z_max = z_size - 1
    y_max = y_size - 1
    x_max = x_size - 1
    
    planeimgsize = (l_radius*2+1)**2
    planenum = 3
    
    neg_pca_input_mtx = []
    
    planeselect = 0 # default: xy, xz = 1, yz = 2
    f3 = figure(frameon=False)
    fig_col = math.ceil(pos_pnum/10)
    
    if planeselect == 0:
        f3.suptitle('X-Y planes at the negative points in unit volumes', fontsize=16)
    elif planeselect == 1:
        f3.suptitle('X-Z planes at the negative points in unit volumes', fontsize=16)
    else:
        f3.suptitle('Y-Z planes at the negative points in unit volumes', fontsize=16)

    ## Generate the list of negative points using target_volume
    neg_pointarrays = np.ndarray(shape=(pos_pnum,3), dtype='uint8')
    neg_values = np.ndarray(shape=(pos_pnum,), dtype='d')
    neg_pnum = 0
    while (neg_pnum < pos_pnum):
        cand_z = random.randint(0, z_max)
        cand_y = random.randint(0, y_max)
        cand_x = random.randint(0, x_max)

        nx_min = cand_x - l_radius
        ny_min = cand_y - l_radius
        nz_min = cand_z - l_radius
        
        nx_max = cand_x + l_radius
        ny_max = cand_y + l_radius
        nz_max = cand_z + l_radius
        
        if (nx_min >= 0) and (ny_min >= 0) and (nz_min >= 0) and (nx_max <= x_max) and (ny_max <= y_max) and (nz_max <= z_max):
            if(target_volume[cand_z,cand_y,cand_x] < outofbound_thr):
                f3.add_subplot(fig_col+1, 10, neg_pnum+1)  # this line outputs images on top of each other
                
                neg_values[neg_pnum] = round(target_volume[cand_z,cand_y,cand_x], 3)
                
                neg_pointarrays[neg_pnum,0] = cand_x 
                neg_pointarrays[neg_pnum,1] = cand_y
                neg_pointarrays[neg_pnum,2] = cand_z
                
                #X-Y Planes data
                xyimg = vol_data[cand_z,ny_min:ny_max+1,nx_min:nx_max+1]
    
                #X-Z Planes data
                xzimg = vol_data[nz_min:nz_max+1,cand_y,nx_min:nx_max+1]
                
                #Y-Z Planes data
                yzimg = vol_data[nz_min:nz_max+1,ny_min:ny_max+1,cand_x]
        
                if planeselect == 0:
                    imshow(xyimg,cmap=cm.Greys_r)
                elif planeselect == 1:
                    imshow(xzimg,cmap=cm.Greys_r)
                else:
                    imshow(yzimg,cmap=cm.Greys_r)
                axis('off')
        
                xyimg = xyimg.flatten()
                xzimg = xzimg.flatten()
                yzimg = yzimg.flatten()
    
                xyzimg = zeros((planeimgsize, planenum))
                xyzimg[:,0] = xyimg
                xyzimg[:,1] = xzimg
                xyzimg[:,2] = yzimg
                
                neg_pca_input_mtx.append(xyzimg.flatten())
        
                neg_pnum += 1
                
                del xyimg, xzimg, yzimg, xyzimg

    neg_pca_input_arr = array(neg_pca_input_mtx, 'd')
    del neg_pca_input_mtx

    # Linear PCA
#    neg_coeff,neg_meanvector = princomp(neg_pca_input_arr,2)
#    neg_databack = np.zeros((neg_pca_input_arr.shape))
#    for n_index in range(neg_pca_input_arr.shape[0]):
#        l_pr = np.dot(neg_pca_input_arr[n_index,:]-neg_meanvector,neg_coeff)
#        neg_databack[n_index,:] = np.dot(l_pr,neg_coeff.T) + neg_meanvector

    numev = np.round((neg_pca_input_arr.shape[0]/4)*3)
    # numev = neg_pca_input_arr.shape[0]-1
    err = kpcabound(neg_pca_input_arr,1.5,numev)
    #print err
    
    p_index = np.zeros((neg_pca_input_arr.shape[0],))
    for ik in range (neg_pca_input_arr.shape[0]):
        p_index[ik] = ik+1
    
    f7 = figure(frameon=False)
    ax1 = f7.add_subplot(2,1,1)
    # ax1.plot(err, 'b-')
    line, = ax1.semilogy(err, color='blue', lw=2)
    show()
    
    ###########################
    ## KernelPCA
    ###########################
    kpca = KernelPCA(kernel="rbf", gamma=0.5, fit_inverse_transform=True, eigen_solver='auto')
    X_kpca = kpca.fit_transform(neg_pca_input_arr)
    neg_databack = kpca.inverse_transform(X_kpca)

    neg_databack = neg_databack.round()
    
    f4 = figure(frameon=False)

    if planeselect == 0:
        f4.suptitle('X-Y planes with neg-PCA dimensionality reduction', fontsize=16)
    elif planeselect == 1:
        f4.suptitle('X-Z planes with neg-PCA dimensionality reduction', fontsize=16)
    else:
        f4.suptitle('Y-Z planes with neg-PCA dimensionality reduction', fontsize=16)
    fig_col = math.ceil(neg_databack.shape[0]/10)

    ccc = 1
    for i in range(neg_databack.shape[0]):
        f4.add_subplot(fig_col+1, 10, ccc)  # this line outputs images on top of each other
        xyzimg = np.uint8(neg_databack[i,:].reshape(planeimgsize, planenum))
        xyimg = xyzimg[:,0].reshape(l_radius*2+1,l_radius*2+1)
        xzimg = xyzimg[:,1].reshape(l_radius*2+1,l_radius*2+1)
        yzimg = xyzimg[:,2].reshape(l_radius*2+1,l_radius*2+1)
        if planeselect == 0:
            imshow(xyimg,cmap=cm.Greys_r)
        elif planeselect == 1:
            imshow(xzimg,cmap=cm.Greys_r)
        else:
            imshow(yzimg,cmap=cm.Greys_r)
        axis('off')
        ccc += 1
    
    show()

    # check negative points
#   for n_point in neg_pointarrays:
#       print target_volume[n_point[2],n_point[1],n_point[0]] 
