# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 16:29:17 2025

@author: User
"""

import tifffile
import scipy.io as sio
#import nrrd

def load_tiff_stack(filepath):
    return tifffile.imread(filepath)

def save_mat(filepath, data_dict):
    sio.savemat(filepath, data_dict)

def load_mat(filepath, key=None):
    mat = sio.loadmat(filepath)
    if key:
        return mat[key]
    return mat

#def load_nrrd(filepath):
    #data, header = nrrd.read(filepath)
    #return data, header