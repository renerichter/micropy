import time
import os
import re
import NanoImagingPack as nip
import numpy as np
import matplotlib.pyplot as plt

# %% -------------------------------------------
#       LOGGING
# ----------------------------------------------


def add_logging(logger_filepath='./logme.log', start_logger='RAWprocessor'):
    '''
    adds logging to an environment. Deletes all existing loggers and creates a stream and a file logger based on setting the root logger and later adding a file logger 'logger'. 
    '''
    import logging
    from sys import stdout
    # get root
    root = logging.getLogger()
    # set formatters
    fromage = logging.Formatter(
        datefmt='%Y%m%d %H:%M:%S', fmt='[ %(levelname)-8s ] [ %(name)-13s ] [ %(asctime)s ] %(message)s')
    # set handlers
    strh = logging.StreamHandler(stdout)
    strh.setLevel(logging.DEBUG)
    strh.setFormatter(fromage)
    fh = logging.FileHandler(logger_filepath)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fromage)
    if 'logger' in locals():
        if len(logger.handlerse):
            for loghdl in logger.handlers:
                logger.removeHandler(loghdl)
    if len(root.handlers):
        while len(root.handlers):  # make sure that root does not contain any more handlers
            for loghdl in root.handlers:  # it seems it only can delete 1 type of handlers and then leaves the others if multiple are existing
                #print("deleting handler={}".format(loghdl))
                root.removeHandler(loghdl)
    # set root levels
    root.setLevel(logging.DEBUG)
    root.addHandler(strh)
    root.addHandler(fh)
    # add first new handler -> root levels are automatically applied
    logger = logging.getLogger('RAWprocessor')
    logger.setLevel(logging.DEBUG)
    return root, logger


def logger_switch_output(str_message, logger=False):
    '''
    switches between printing and logging messages

    :param:
    ======
    :logger: has to be an object pointing to existing logger and correct level, e.g. Logger.warn
    '''
    if logger:
        logger(str_message)
    else:
        print(str_message)
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
    fl.sort()
    return fl


def filelist2path(file_list, file_path):
    '''
    Prepends the file_path to every element of a file_list.
    '''
    res = [file_path + x for x in file_list]
    return res


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
    fl_iter = floor(fl_len/batch_size) + (1 if fl_len % batch_size > 0 else 0)
    fl_lastiter = batch_size if (
        fl_len % batch_size == 0 and fl_iter > 0) else fl_len % batch_size
    print("{} files will be split into {} iterations with {} objects in the last iteration using a batch_size of {}.".format(
        fl_len, fl_iter, fl_lastiter, batch_size))
    return fl_len, fl_iter, fl_lastiter


def loadStackfast(file_list, logger=False, colorful=1):
    '''
    Easy alternative to load stack fast and make useable afterwards.
    '''
    from cv2 import imread
    from tifffile import imread as timread
    im = []
    rl = []
    for m in range(len(file_list)):
        try:
            if colorful:
                imh = imread(file_list[m])
            else:
                imh = imread(file_list[m], 0)
                if not type(imh) == np.ndarray:
                    # if imh == None:
                    imh = timread(file_list[m])
            if type(imh) == np.ndarray:
                im.append(imh)
                rl.append(m)
            else:
                logger_switch_output("Readin of {} is of type {} and thus was discarded.".format(
                    file_list[m], type(imh)), logger=logger)
        except Exception as ex:
            logger_switch_output(
                "Exception ---->{}<---- occured.".format(ex), logger=logger)
    im = np.array(im)
    if im.ndim > 3:
        im = np.transpose(im, [0, -1, 1, 2])
    return np.array(im), rl


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
    brename = False
    # glob(load_experiment + '*.' + file_extension
    file_list = os.listdir(file_dir)
    file_list.sort(key=len)  # sorts the string-list ascending by length
    index_max_nbr = len(file_list)
    file_max_length = len(file_list[-1])  # should now yield the max length
    # if numbers smaller e15
    # index_length = math.floor(math.log10(index_max_nbr))+1
    # numbers are enclosed by _ -> use regex
    if not len(file_list[0]) == file_max_length:
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
        brename = True
    tdelta = time.time()-tstart
    print('Renaming took: {0}s.'.format(tdelta))
    return tdelta, brename


# %% ------------------------------------------------------
# ---                        Plotting                   ---
# ---------------------------------------------------------
#
def print_stack2subplot(imstack, plt_raster=[4, 4], plt_format=[8, 6], title=None, titlestack=None, colorbar=True, axislabel=False):
    '''
    Plots an 3D-Image-stack as set of subplots
    Based on this: https://stackoverflow.com/a/46616645
    '''
    if type(imstack) == list:
        imstack_len = len(imstack)
    # needs NanoImagingPack imported as nip
    elif type(imstack) == nip.image or type(imstack) == np.array:
        imstack_len = imstack.shape
    else:
        raise TypeError("Unexpected Data-type.")
    if not(imstack_len > 0):
        raise ValueError("Image size not fitting!")
    # check for title
    from datetime import datetime
    if title == None:
        title = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(
        nrows=plt_raster[0], ncols=plt_raster[1], figsize=plt_format)
    # plot simple raster image on each sub-plot
    for m, axm in enumerate(ax.flat):
        ima = axm.imshow(imstack[m], alpha=1.0,
                         interpolation='none')  # alpha=0.25)
        # write row/col indices as axes' title for identification
        if colorbar:
            if ax.ndim == 1:
                fig.colorbar(ima, ax=ax[m])
            elif ax.ndim == 2:
                fig.colorbar(ima, ax=ax[m // plt_raster[1], m % plt_raster[1]])
            else:
                raise ValueError('Too many axes!')
        if not axislabel:
            axm.set_xlabel('PIX [a.u.]')
            axm.set_ylabel('PIX [a.u.]')
        if not titlestack == None:
            axm.set_title(titlestack[m])
        else:
            axm.set_title(
                "Row:"+str(m // plt_raster[1])+", Col:"+str(m % plt_raster[1]))
        if m >= imstack_len-1:
            break
    if title:
        fig.suptitle(title)
    return fig


def plot_save(ppointer, save_name, save_format='png'):
    '''
    Just an easy wrapper.
    '''
    ppointer.savefig(save_name+'.'+save_format, dpi=300, bbox_inches='tight')


# %% Directory and file-structure


def dir_test_existance(mydir):
    try:
        if not os.path.exists(mydir):
            os.makedirs(mydir)
            # logger.debug(
            #    'Folder: {0} created successfully'.format(mydir))
    finally:
        # logger.debug('Folder check done!')
        pass
