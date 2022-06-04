# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 12:39:42 2019

@author: rene
"""
import os
import math
from .general_imports import *

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


# %%
# ------------------------------------------------------------------
#                    Safety Checks on Structures 
# ------------------------------------------------------------------

def sanityCheck_structure(pstruct,params={'psfdet_array':None,'shift_offset':[2,2],'nbr_det':[3,3],'fmodel':'rft'},isclass=False,show_results=False):
    '''
    Checks whether given parameters are variables in the structure and adds them if not. 

    TODO: 1) implement for list,tuple

    :PARAMS:
    ========
    :pstruct:      (STRUCTURE) to be tested
    :params:        (LIST) of STRINGS to use for element testing

    :OUTPUT:
    =======
    :params:        (LIST) updated list

    :EXAMPLE:
    =========
    psf_params = nip.PSF_PARAMS()
    params={'wavelength':488,'pol':'lin','aplanar':True,'mytest':20}
    psf_params = sanityCheck_structure(psf_params,params)
    '''
    res_had = f"I found the following entries:\n----------------------------\n"
    res_add = "I added the following entries:\n----------------------------\n"
    if type(pstruct) == nip.util.struct or isclass:
        for m in params:
            if hasattr(pstruct,m):
                res_had += f"{m:<20}= " + str(getattr(pstruct,m)) + "\n"
            else: 
                setattr(pstruct,m,params[m])
                res_add += f"{m:<20}= {params[m]}\n"
    elif type(pstruct) == dict:
        for m in params:
            if m in pstruct:
                res_had += f"{m:<20}= " + str(pstruct[m]) + "\n"
            else:
                pstruct[m] = params[m]
                res_add += f"{m:<20}= {params[m]}\n"
    else:
        raise Exception(f'Type={type(pstruct)} not implemented yet.')

    # display results
    if show_results:
        print(res_had)
        print(res_add)

    # done?
    return pstruct