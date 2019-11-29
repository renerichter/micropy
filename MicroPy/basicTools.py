# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 12:39:42 2019

@author: rene
"""
import numpy as np
import NanoImagingPack as nip
import os
import math


# %% get_range
def get_range(dim_im, dim_goal):
    # only for even sizes
    dim_goal = np.transpose(np.array(dim_goal),)
    pos_center = np.squeeze(np.array(dim_im)//2)
    if len(pos_center) == 3 and len(dim_goal) < len(pos_center):
        pos_center = np.delete(pos_center, 0)
    return [pos_center[0]-dim_goal[0]//2, pos_center[0]+dim_goal[0]//2+dim_goal[0] % 2, pos_center[1]-dim_goal[1]//2, pos_center[1]+dim_goal[1]//2+dim_goal[0] % 2]


def memory_adress(var_list=[], name_list=[]):
    # test if list
    if not isinstance(var_list, list):
        var_list = [var_list, ]
    if not isinstance(name_list, list):
        name_list = [name_list, ]
    # test if list iterate and print
    if len(name_list) == len(var_list):
        for myc in range(len(var_list)):
            print("Adress of {0}: {1}".format(
                name_list[myc], hex(id(var_list[myc]))))
    else:
        for myc in range(len(var_list)):
            print(hex(id(var_list[myc])))


def getIterationProperties(load_experiment, offset=0, file_limit=10, batch_size=10, frame_interleave=0):
    '''
    Calculates iteration properties

    Param:
        load_experiment: Path to set(folder) of images
        file_limit: maximum number of files to be read in 
        batch_size: size of each read in batch (basically limited by system memory given calculation overhead in other routines)
    '''
    file_list = os.listdir(load_experiment)
    if frame_interleave >= len(file_list):
        file_list = file_list[-1]
    elif frame_interleave > 0:
        file_list = file_list[0::frame_interleave]
    len_fl = len(file_list)
    offset = abs(offset)
    if offset > len_fl:
        offset = len_fl
        file_limit = 0
    file_total = file_limit if file_limit else (len_fl - offset)
    batch_iterations = math.floor(file_total/batch_size)
    batch_last_iter = file_total % batch_size
    last_file = offset + file_total
    return file_list[offset:last_file], file_total, batch_iterations, batch_last_iter


# %% My Version from NIP!
