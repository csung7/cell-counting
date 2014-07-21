from cellcommon import *
from cellcalculatepca import *
import scipy
from scipy import ndimage

def Generate_PCA_Matrix(vol_data,pointarrays,l_radius,planenum,planeselect):

    z_size, y_size, x_size = vol_data.shape
    point_num = pointarrays.shape[0]
    
    z_max = z_size - 1
    y_max = y_size - 1
    x_max = x_size - 1
    
    planeimgsize = (l_radius*2+1)**2

    # positive point count
    pos_pnum = 0
    
    #f1 = figure(frameon=False)
    #fig_col = math.ceil(point_num/10)        
    
#    if planeselect == 0:
#        f1.suptitle('X-Y planes at the center points in unit volumes', fontsize=16)
#    elif planeselect == 1:
#        f1.suptitle('X-Z planes at the center points in unit volumes', fontsize=16)
#    else:
#        f1.suptitle('Y-Z planes at the center points in unit volumes', fontsize=16)

    local_posi_pca_input_mtx = np.zeros((point_num,planeimgsize*planenum))
    
    for point in pointarrays:
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
            #f1.add_subplot(fig_col+1, 10, pos_pnum+1)  # this line outputs images on top of each other
            #X-Y Planes data
            xyimg = vol_data[z,c_y_min:c_y_max+1,c_x_min:c_x_max+1]

            #X-Z Planes data
            xzimg = vol_data[c_z_min:c_z_max+1,y,c_x_min:c_x_max+1]
            
            #Y-Z Planes data
            yzimg = vol_data[c_z_min:c_z_max+1,c_y_min:c_y_max+1,x]

#            if planeselect == 0:
#                imshow(xyimg,cmap=cm.Greys_r)
#            elif planeselect == 1:
#                imshow(xzimg,cmap=cm.Greys_r)
#            else:
#                imshow(yzimg,cmap=cm.Greys_r)
#            axis('off')
            
            xyimg = xyimg.flatten()
            xzimg = xzimg.flatten()
            yzimg = yzimg.flatten()

            xyzimg = zeros((planeimgsize, planenum))
            xyzimg[:,0] = xyimg
            xyzimg[:,1] = xzimg
            xyzimg[:,2] = yzimg
            
            local_posi_pca_input_mtx[pos_pnum,:] = xyzimg.flatten()
            pos_pnum += 1
            
            del xyimg, xzimg, yzimg, xyzimg
    
    # truncate empty array
    local_posi_pca_input_mtx = local_posi_pca_input_mtx[range(pos_pnum),:]
    
    return local_posi_pca_input_mtx, pos_pnum


def Generate_Neg_PCA_Matrix(vol_data,pointarrays,l_radius,pos_pnum,target_volume,outofbound_thr,planenum,planeselect):
    z_size, y_size, x_size = vol_data.shape
    
    z_max = z_size - 1
    y_max = y_size - 1
    x_max = x_size - 1
    
    planeimgsize = (l_radius*2+1)**2
    
    local_neg_pca_input_mtx = np.zeros((pos_pnum,planeimgsize*planenum))
    
#    f3 = figure(frameon=False)
#    fig_col = math.ceil(pos_pnum/10)
#    
#    if planeselect == 0:
#        f3.suptitle('X-Y planes at the negative points in unit volumes', fontsize=16)
#    elif planeselect == 1:
#        f3.suptitle('X-Z planes at the negative points in unit volumes', fontsize=16)
#    else:
#        f3.suptitle('Y-Z planes at the negative points in unit volumes', fontsize=16)

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
                #f3.add_subplot(fig_col+1, 10, neg_pnum+1)  # this line outputs images on top of each other
                
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
        
#                if planeselect == 0:
#                    imshow(xyimg,cmap=cm.Greys_r)
#                elif planeselect == 1:
#                    imshow(xzimg,cmap=cm.Greys_r)
#                else:
#                    imshow(yzimg,cmap=cm.Greys_r)
#                axis('off')
        
                xyimg = xyimg.flatten()
                xzimg = xzimg.flatten()
                yzimg = yzimg.flatten()
    
                xyzimg = zeros((planeimgsize, planenum))
                xyzimg[:,0] = xyimg
                xyzimg[:,1] = xzimg
                xyzimg[:,2] = yzimg
                
                local_neg_pca_input_mtx[neg_pnum,:] = xyzimg.flatten()
                neg_pnum += 1
                
                del xyimg, xzimg, yzimg, xyzimg

    # truncate empty array
    local_neg_pca_input_mtx = local_neg_pca_input_mtx[range(neg_pnum),:]

    return local_neg_pca_input_mtx
    
    
def NeuronPCA(testing_files,l_radius):

    # local volume radius
    planenum = 3
    planeselect = 0 # default: xy, xz = 1, yz = 2

    
    posi_pca_input_mtx = []
    neg_pca_input_mtx = []
        
    for testing_file in testing_files:
        vol_filename = testing_file + ".vol"
        point_filename = testing_file + ".cel"
    
        #print vol_filename

        vol_data = LoadVolume(vol_filename)
        
        # 2 times scale down!
        d = 2
        vol_data = ndimage.convolve(np.uint16(vol_data), np.ones((d,d,d)))[::d,::d,::d]/8
        
        z_size, y_size, x_size = vol_data.shape
        
        #g = gauss_kern3D(r_size=3, k_sigma=0.5)
        #vol_data = ndimage.convolve(np.uint16(vol_data),g)

        #print vol_data[0,:,:]
        #plt.imshow(vol_data[0,:,:])
        #plt.gray()
        #plt.show()
        
        #scipy.misc.imsave('test2.png', vol_data[0,:,:])
        
        #SaveVolume(vol_data, 'chul.vol')
        
        pointarrays = OBJToPoints(point_filename)
        
        for point in pointarrays:
            point[0] = math.floor(point[0] * x_size)
            point[1] = math.floor(point[1] * y_size)
            point[2] = math.floor(point[2] * z_size)
       
        vol_pca_input_list, pos_pnum = Generate_PCA_Matrix(vol_data,pointarrays,l_radius,planenum,planeselect)

        posi_pca_input_mtx.extend(vol_pca_input_list)
        del vol_pca_input_list

        ###############################
        # This generates target volume with gauss templete
        target_volume = MakeTargetVolume(pointarrays, vol_data.shape, l_radius)
        ###############################
        
        # debug for target_volume values
#        g_c = 0
#        g_r = 1
#        for point in pointarrays:
#            x = point[0]
#            y = point[1]
#            z = point[2]
#            if(target_volume[z,y,x] > 0.9):
#                g_c += 1
#                print target_volume[z-g_r:z+g_r+1,y-g_r:y+g_r+1,x-g_r:x+g_r+1]
#        print g_c

        # out of cell boundary threshold
        outofbound_thr = 0.1
        
        neg_vol_pca_input_list = Generate_Neg_PCA_Matrix(vol_data,pointarrays,l_radius,pos_pnum,target_volume,outofbound_thr,planenum,planeselect)
        
        neg_pca_input_mtx.extend(neg_vol_pca_input_list)
        
        del neg_vol_pca_input_list
        del target_volume
        del vol_data
        del pointarrays

        ###############################
        # Negative_Generate_Points_ND_PCA_Cal(vol_data,pointarrays,pos_pnum,target_volume,l_radius,outofbound_thr)
        ###############################

    posi_pca_input_arr = array(posi_pca_input_mtx, 'd')
    del posi_pca_input_mtx
    
    posi_coeff,posi_meanvector = Positive_PCA_Cal(posi_pca_input_arr,l_radius,planenum,planeselect)
    
    neg_pca_input_arr = array(neg_pca_input_mtx, 'd')
    del neg_pca_input_mtx
    
    neg_coeff,neg_meanvector = Negative_PCA_Cal(neg_pca_input_arr,l_radius,planenum,planeselect)

    return posi_coeff,posi_meanvector,neg_coeff,neg_meanvector


