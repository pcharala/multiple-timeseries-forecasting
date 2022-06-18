# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 18:18:38 2022

Guangzhou traffic speed dataset pre-processing

@author: Christos Tzagkarakis <tzagarak@ics.forth.gr>
"""

import numpy as np
import scipy.io


def preprocess(input_filepath,
               output_filepath):
    
    mat = scipy.io.loadmat(input_filepath[0])
    data = mat['tensor']
    
    # Based on the official dataset description, there are some missing values. We decided
    #to keep all the road segments that contain only non-zero values.
    nz_val = []
    tmp = []
    cnt = 0
    for col in range(0, data.shape[0]):
        nw = (np.where(data[col,:,:] == 0))
        if (len(nw[0])+len(nw[1])) == 0:
            nz_val.append(col)
        tmp.append(nw)
        
        cnt = cnt + len(nw[0])
    
    # Keep the road segments that contain only non-zero values
    data_new = data[nz_val,:,:]
    
    # Here, we can see that the remaining tensor does not contain any NaN values
    mis_val = []
    for col in range(0, data_new.shape[0]):
        if (np.isnan(data_new[col,:,:]).any()) == True:
            mis_val.append(col)
    
    
    data_new_mean = np.zeros((data_new.shape[0], data_new.shape[1], int(data_new.shape[2]/6)))
    for col in range(0, data_new.shape[0]):
        arr = data_new[col,:,:]
        mean_arr = np.mean(arr.reshape(-1, 6), axis=1)
        data_new_mean[col,:,:] = mean_arr.reshape(data_new.shape[1], int(data_new.shape[2]/6))
    
    
    data_concat = np.zeros((data_new_mean.shape[0], data_new_mean.shape[1]*data_new_mean.shape[2]))
    for col in range(0, data_new_mean.shape[0]):
        data_concat[col,:] = data_new_mean[col,:,:].reshape(-1, data_new_mean.shape[1]*data_new_mean.shape[2])
    
    # the time spannig is:  61 days from August 1, 2016 to September 30, 2016
    np.savetxt(output_filepath, data_concat, delimiter=",")
