import time
import os
import re
import NanoImagingPack as nip
import numpy as np


# %%
# ---------------------------------------------------------------------------------------------------------
#                                           LOAD


def get_filelist(load_path='.', fn_proto='jpg'):
    ''' 
    Gets a list of file according to a prototype.

    :param:
    =======
    load_path: absolute path to load directory
    :fn_proto: prototype-filename to select basic structure on and use e.g. regex on -> not implemented yet

    :out:
    =====
    fl: file_list

    '''
    # better, but traversing and checking not necessary for now
    # fl = os.listdir(load_path).sort(reverse=False) # automatically excludes '.' and '..'
    # if os.path.isdir(fname)

    # for now -> short solution
    from NanoImagingPack import get_sorted_file_list as gsfl
    fl = gsfl(load_path, fn_proto, sort='integer_key', key='0')
    return fl


def get_batch_numbers(filelist=['', ], batch_size=100):
    '''
    Calculates the number of batches

    :param:DEFAULT:
    =======
    :filelist:['',]: (LIST) list of files 
    :batch_size:100: (INT) number of images per batch  

    :out:
    =====
    :fl_len: length of filelist
    :fl_iter: number of batches for full stack
    :fl_lastiter: size of last batch
    '''
    #
    from math import floor
    #
    fl_len = len(filelist)
    fl_iter = floor(fl_len/batch_size)
    fl_lastiter = fl_len % batch_size
    print("{} files will be split into {} iterations with {} objects in the last iteration using a batch_size of {}.".format(
        fl_len, fl_iter, fl_lastiter, batch_size))
    return fl_len, fl_iter, fl_lastiter


def loadStack(file_list, ids=None, ide=None, channel=1):
    '''
    loads an image stack from a file-list (containing the absolute path to a file) given a start and end-number. 
    Caution: Errors in file_list or exp_path not catched yet

    Param:
        exp_path: Path to experiment Folder (containing all the images)
        file_list: list of all images to be loaded for this stack
        ids: start index
        ide: end index
        channel: which channel to be read in
        not_prepent standard if file-list already contains full path
    Return:
        im1: stack of read images
        read_list: read list from stack (empty images skipped)
    '''
    # define vars -----------------------------------------------------------
    read_list = []
    prepent_path = False
    # catch some errors -----------------------------------------------------
    fll = len(file_list)  # =maximum iterated entry in list
    if ids == None:
        ids = 0
    if ide == None:
        ide = fll - 1
    if ids < 0:
        raise ValueError('Start-Index is negative. Make ids positive.')
    if ide > fll:
        # raise ValueError('Batch-length too long. Make ide smaller.')
        ide = fll-1
    if not (isinstance(ide, int) or isinstance(ids, int)):
        raise ValueError('ide or ids is not of type int.')
    # print("ids={0}, ide={1}".format(ids,ide))
    if ide < ids:
        raise ValueError('Make sure: ide >= ids.')
    # try:
    #    if exp_path[0] == file_list[0][0]:
    #        prepent_path = False
    # except Exception as e:
    #    print(e)
    # read in ----------------------------------------------------------------
    # start list
    im1, myh = loadChkNonEmpty(file_list, idx=ids, myh=0, ide=ide,
                               channel=channel, prepent_path=prepent_path)  # might still be empty
    myc = ids+myh
    read_list.append(myc)
    im1 = im1[np.newaxis]
    myc += 1
    # iterate through images
    if myc <= ide:
        while myc <= ide:
            # print('myc={}'.format(myc))
            myh = 0
            im1h, myh = loadChkNonEmpty(
                file_list, idx=myc, myh=myh, ide=ide, channel=channel, prepent_path=prepent_path)
            # print("Load-Stack function -- mystep={0}".format(myh))
            read_list.append(myc + myh)
            myc += myh if myh > 0 else 1  # next step
            try:
                im1 = np.concatenate((im1, im1h[np.newaxis]), 0)
            except Exception as e:
                print(e)
                return im1, read_list
            # print("Load-Stack function -- myc={0}".format(myc))
    if isinstance(im1, tuple):
        im1 = nip.image(im1)
    elif isinstance(im1, np.ndarray):
        im1 = nip.image(im1)
    return im1, read_list


def loadChkNonEmpty(file_list, idx=0, myh=0, ide=100, channel=1, prepent_path=False):
    '''
     avoid empty start-image -> so concatenation is possible
    '''
    im1 = loadPrepent(file_list, idx+myh, channel, prepent_path)
    while im1.shape == ():
        myh += 1
        im1 = loadPrepent(file_list, idx+myh, channel, prepent_path)
        if myh == ide:
            break
        return im1
    return im1, myh  # return myh, but is already overwritten in place


def loadPrepent(file_list, idx=0, channel=1, prepent_path=False, exp_path=None):
    '''
    channel_limit: only 3 color-channel exist and hence 3 marks 'all'
    Only implemented for image-structure: [Y,X,channel] -> raspi images
    '''
    # print('idx={}'.format(idx));sys.stdout.flush()
    channel_limit = 3
    if prepent_path:  # somehow nip.readim is not working properly yet
        im1 = np.squeeze(nip.readim(exp_path + file_list[idx]))
        # im1 = np.squeeze(tff.(exp_path + file_list[idx]))
    else:
        im1 = np.squeeze(nip.readim(file_list[idx]))
    if channel < channel_limit:
        im1 = im1[channel, :, :]
    return im1

# %%
# ---------------------------------------------------------------------------------------------------------
#                                           FILE CHANGES


def rename_files(file_dir, version=1):
    '''
    Renames numbered stack and inserts 0s so that readin it in works better. 
    Leaves out image #9. Why?
    '''
    tstart = time.time()
    # glob(load_experiment + '*.' + file_extension
    file_list = os.listdir(file_dir)
    file_list.sort(key=len)  # sorts the string-list ascending
    index_max_nbr = len(file_list)
    file_max_length = len(file_list[-1])  # should now yield the max length
    # if numbers smaller e15
    # index_length = math.floor(math.log10(index_max_nbr))+1
    # numbers are enclosed by _ -> use regex
    for myc in range(0, index_max_nbr-1):
        file_len = len(file_list[myc])
        if(file_len < file_max_length):
            if version == 0:  # for older measurements structure was 'yyyy-mm-dd_techique_nbr_TECH_NIQUE.jpg'
                pos_help = re.search('_[0-9]+_', file_list[myc])
            elif version == 1:  # for new structure, e.g '2019-07-12_Custom_7114.jpg'
                pos_help = re.search('_[0-9]+.', file_list[myc])
            else:  # for new structure, e.g '20190815-TYPE-Technique--00001.jpg'
                pos_help = re.search('--[0-9]+.', file_list[myc])
            string_help = str(0)*(file_max_length-file_len)
            os.rename(file_dir + file_list[myc], file_dir + file_list[myc]
                      [0:pos_help.start()+2] + string_help + file_list[myc][pos_help.start()+2:])
    print('Renaming took: {0}s.'.format(time.time()-tstart))


# %%
