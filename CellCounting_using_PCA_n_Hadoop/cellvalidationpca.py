#!/usr/bin/env python
# encoding: utf-8
#
# Author: Chul Sung
# Updated by: Chul Sung, daniel.c.sung@gmail.com
# Date: 02/03/2013

from cellcommon import *
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy import ndimage
import random

def Generate_RAND_PCA_Matrix(vol_data,rand_pointarrays,l_radius,planenum,planeselect,target_volume):

    z_size, y_size, x_size = vol_data.shape
    point_num = rand_pointarrays.shape[0]
    
    z_max = z_size - 1
    y_max = y_size - 1
    x_max = x_size - 1
    
    planeimgsize = (l_radius*2+1)**2

#    f1 = figure(frameon=False)
#    fig_col = math.ceil(point_num/10)        
#    
#    if planeselect == 0:
#        f1.suptitle('X-Y planes at the center points in unit volumes', fontsize=16)
#    elif planeselect == 1:
#        f1.suptitle('X-Z planes at the center points in unit volumes', fontsize=16)
#    else:
#        f1.suptitle('Y-Z planes at the center points in unit volumes', fontsize=16)

    local_pca_input_mtx = np.zeros((point_num,planeimgsize*planenum))
    
    local_pca_input_values = np.zeros((point_num))
    
    # positive point count
    pos_pnum = 0
    
    #f2 = plt.figure(frameon=False)
    
    for point in rand_pointarrays:
        x = point[0]
        y = point[1]
        z = point[2]

        c_x_min = x - l_radius
        c_y_min = y - l_radius
        c_z_min = z - l_radius

        c_x_max = x + l_radius
        c_y_max = y + l_radius
        c_z_max = z + l_radius
        
        fig_col = math.ceil(rand_pointarrays.shape[0]/10)
        
        if (c_x_min >= 0) and (c_y_min >= 0) and (c_z_min >= 0) and (c_x_max <= x_max) and (c_y_max <= y_max) and (c_z_max <= z_max):
            #f2.add_subplot(fig_col+1, 10, pos_pnum+1)  # this line outputs images on top of each other
            
            #X-Y Planes data
            xyimg = vol_data[z,c_y_min:c_y_max+1,c_x_min:c_x_max+1]
            #plt.imshow(xyimg,cmap=plt.cm.Greys_r)

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

            xyzimg = np.zeros((planeimgsize, planenum))
            xyzimg[:,0] = xyimg
            xyzimg[:,1] = xzimg
            xyzimg[:,2] = yzimg
            
            local_pca_input_mtx[pos_pnum,:] = xyzimg.flatten()
            local_pca_input_values[pos_pnum] = target_volume[z,y,x]
            
            pos_pnum += 1
            
            del xyimg, xzimg, yzimg, xyzimg
            #plt.axis('off')
            
    #plt.show()
    # truncate empty array
    local_pca_input_mtx = local_pca_input_mtx[range(pos_pnum),:]
        
    local_pca_input_values = local_pca_input_values[range(pos_pnum)]
    
    return local_pca_input_mtx, local_pca_input_values


def ValidationPCA(validation_files,l_radius,depth_from_param):
    
    selected_point_num = 2000
    
    pca_input_mtx = []
    pca_input_values = []
    rand_pointarrays_total = []
    proximity_pointarrays_total = []

    for validation_file in validation_files:
        vol_filename = validation_file + ".vol"
        point_filename = validation_file + ".cel"

        #print vol_filename
        vol_data = LoadVolume(vol_filename)
        
        # 2 times scale down!
        d = 2
        vol_data = ndimage.convolve(np.uint16(vol_data), np.ones((d,d,d)))[::d,::d,::d]/8
        
        #SaveVolume(vol_data, "training06z.vol")
        
        z_size, y_size, x_size = vol_data.shape
        
#        g = gauss_kern3D()
#        vol_data = ndimage.convolve(vol_data,g)
        
        #################### read cell points ####################
        pointarrays = OBJToPoints(point_filename)
        
        for point in pointarrays:
            point[0] = math.floor(point[0] * x_size)
            point[1] = math.floor(point[1] * y_size)
            point[2] = math.floor(point[2] * z_size)
        ##########################################################
        
        ################### make target volume ###################
        target_volume = MakeTargetVolume(pointarrays, vol_data.shape, l_radius)
        #target_volume.tofile("target.vol")
        ##########################################################
        
#        ################## create random points ##################
#        rand_pointarrays = np.zeros((selected_point_num,3))
#
#        z_max = z_size - 1
#        y_max = y_size - 1
#        x_max = x_size - 1
#    
#        rand_pnum = 0
#        while (rand_pnum < selected_point_num):
#            cand_z = random.randint(0, z_max)
#            cand_y = random.randint(0, y_max)
#            cand_x = random.randint(0, x_max)
#    
#            nx_min = cand_x - l_radius
#            ny_min = cand_y - l_radius
#            nz_min = cand_z - l_radius
#            
#            nx_max = cand_x + l_radius
#            ny_max = cand_y + l_radius
#            nz_max = cand_z + l_radius
#            
#            if (nx_min >= 0) and (ny_min >= 0) and (nz_min >= 0) and (nx_max <= x_max) and (ny_max <= y_max) and (nz_max <= z_max):
##                if (target_volume[cand_z,cand_y,cand_x] > 0.3):
##                    continue
#                rand_pointarrays[rand_pnum,0] = cand_x 
#                rand_pointarrays[rand_pnum,1] = cand_y
#                rand_pointarrays[rand_pnum,2] = cand_z
#                rand_pnum += 1
#        ##########################################################

        ################## create random points ##################
        rand_pointarrays = np.zeros((8100,3))
        proximity_pointarrays = np.zeros((8100,))

        z_max = z_size - 1
        y_max = y_size - 1
        x_max = x_size - 1
    
        rand_pnum = 0
        for x_i in range(5,x_size-5):
          for y_i in range(5,y_size-5):
            cand_z = depth_from_param
            cand_y = y_i
            cand_x = x_i
    
            nx_min = cand_x - l_radius
            ny_min = cand_y - l_radius
            nz_min = cand_z - l_radius
            
            nx_max = cand_x + l_radius
            ny_max = cand_y + l_radius
            nz_max = cand_z + l_radius
            
            if (nx_min >= 0) and (ny_min >= 0) and (nz_min >= 0) and (nx_max <= x_max) and (ny_max <= y_max) and (nz_max <= z_max):
#                if (target_volume[cand_z,cand_y,cand_x] > 0.3):
#                    continue
                proximity_pointarrays[rand_pnum] = target_volume[cand_z,cand_y,cand_x] 
                rand_pointarrays[rand_pnum,0] = cand_x 
                rand_pointarrays[rand_pnum,1] = cand_y
                rand_pointarrays[rand_pnum,2] = cand_z
                rand_pnum += 1
        ##########################################################
        
        # local volume radius
        planenum = 3
        planeselect = 0 # default: xy, xz = 1, yz = 2
        
        local_pca_input_mtx, local_pca_input_values = Generate_RAND_PCA_Matrix(vol_data,rand_pointarrays,l_radius,planenum,planeselect,target_volume)
        
        pca_input_mtx.extend(local_pca_input_mtx)
        del local_pca_input_mtx
        pca_input_values.extend(local_pca_input_values)
        del local_pca_input_values
        rand_pointarrays_total.extend(rand_pointarrays)
        del rand_pointarrays
        proximity_pointarrays_total.extend(proximity_pointarrays)
        del proximity_pointarrays
      
    pca_input_arr = np.array(pca_input_mtx, 'd')
    del pca_input_mtx
    
    pca_input_value_arr = np.array(pca_input_values, 'd')
    del pca_input_values

    rand_pointarrays_array = np.array(rand_pointarrays_total, 'd')
    del rand_pointarrays_total
    
    proximity_pointarrays_array = np.array(proximity_pointarrays_total, 'd')
    del proximity_pointarrays_total
    
    return pca_input_arr, pca_input_value_arr, rand_pointarrays_array, proximity_pointarrays_array 
