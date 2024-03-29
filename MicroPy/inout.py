import time
import os
import re
import socket
from datetime import datetime
import NanoImagingPack as nip
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpl_toolkits as mptk
from mpl_toolkits.axes_grid1 import make_axes_locatable, AxesGrid
from matplotlib.patches import Arrow, Rectangle
from matplotlib.ticker import FormatStrFormatter
from tifffile import imread as tifimread
from typing import Optional, Tuple, List, Union, Generator, Callable
from colorsys import rgb_to_hls, hls_to_rgb

import subprocess
from io import StringIO
import untangle
from copy import deepcopy


# mipy imports
from .utility import normNoff, add_multi_newaxis, transpose_arbitrary, fill_dict1_with_dict2
from .transformations import radial_sum
from .microscopyCalculations import calculate_resolution, convert_x_to_k
from .filters import moving_average_2d

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
        if len(logger.handlers):
            for loghdl in logger.handlers:
                logger.removeHandler(loghdl)
    if len(root.handlers):
        while len(root.handlers):  # make sure that root does not contain any more handlers
            for loghdl in root.handlers:  # it seems it only can delete 1 type of handlers and then leaves the others if multiple are existing
                # print("deleting handler={}".format(loghdl))
                root.removeHandler(loghdl)
    # set root levels
    root.setLevel(logging.DEBUG)
    root.addHandler(strh)
    root.addHandler(fh)
    # add first new handler -> root levels are automatically applied
    logger = logging.getLogger(start_logger)
    logger.setLevel(logging.DEBUG)
    return root, logger


def add_logger(save_path, loggername):
    tbegin = time.time()
    tnow = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    logger_filepath = save_path + 'log-' + tnow + '.log'
    dir_test_existance(save_path)
    # set-Logger
    if not 'logger' in locals():
        logger_root, logger = add_logging(
            logger_filepath, start_logger=loggername)
        logger.debug('Processing running on DEVICE = {}'.format(
            socket.gethostname()))
        logger.debug(
            'Logger created and started. Saving to: {}'.format(logger_filepath))
    return logger


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
# -----------------------------------------------
#                      STORAGE
# -----------------------------------------------

def load_data(param_dict, load_data_dict=False):
    store_dict = np.load(param_dict['save_path']+param_dict['save_name'] +
                         '_data.npz', allow_pickle=True)['arr_0'].item()
    ret = [store_dict['proc_dict'], store_dict['param_dict']]

    if load_data_dict:
        ret += store_dict['data_dict']

    return ret


def store_data(param_dict, proc_dict, data_dict=None):
    store_dict = {'proc_dict': proc_dict, 'param_dict': param_dict}
    if not data_dict is None:
        store_dict['data_dict'] = data_dict
    np.savez(param_dict['save_path']+param_dict['save_name']+'_data.npz', store_dict)

def fix_dict_pixelsize(dictin, pixelsize, fixlist=[], verbose=True):
    # get fixlist
    if fixlist==[]:
        if 'pixel_fixlist' in dictin:
            fixlist=dictin['pixel_fixlist'] 
        else:
            fixlist=dictin['pixel_fixlist'] = list(
                filter(None, [(key if type(dictin[key]) == nip.image else '') for key in dictin]))
    
    # fix pixelsizes
    for val in fixlist:
        if val in dictin:
            if not type(dictin[val]) == nip.image:
                dictin[val] = nip.image(dictin[val])
            dictin[val].pixelsize = pixelsize
        else:
            print(f"WARNING: key={val} not in prd on load.")

    if verbose:
        print("Fixed pixelsize property for keys: "+', '.join(fixlist)+'.')

    # done?
    return dictin

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
    try:
        from NanoImagingPack import get_sorted_file_list as gsfl
        fl = gsfl(load_path, fn_proto, sort='integer_key', key='0')
    except Exception as e:
        print('Exception in Get_Filelist: {}'.format(e))
        from os import listdir
        from os.path import isfile, join, getmtime
        fl = [(f, getmtime(join(load_path, f)))
              for f in listdir(load_path) if isfile(join(load_path, f))]
        fl = list(filter(lambda x: x[0].find(fn_proto) >= 0, fl))
    if type(fl[0]) == tuple:
        fl = [m[0] for m in fl]  # make sure that only filenames are left over
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

        # try to read in
        try:
            if colorful:
                imh = imread(file_list[m])
            else:
                imh = imread(file_list[m], 0)
                if not type(imh) == np.ndarray:
                    # if imh == None:
                    imh = timread(file_list[m])
        except Exception as ex:
            logger_switch_output(
                "Exception ---->{}<---- occured. Trying again with tiffile.".format(ex), logger=logger)
            try:
                imh = timread(file_list[m])
            except Exception as ex2:
                logger_switch_output(
                    "2nd try failed with Exception:  ---->{}<---- occured. Trying again with tiffile.".format(ex2), logger=logger)

        # check formatting
        if type(imh) == np.ndarray:
            im.append(imh)
            rl.append(m)
        else:
            logger_switch_output("Readin of {} is of type {} and thus was discarded.".format(
                file_list[m], type(imh)), logger=logger)
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


class SPEloader(object):
    '''
    From:  https://scipy-cookbook.readthedocs.io/items/Reading_SPE_files.html
    extended with: https://github.com/stuwilkins/pyspec/blob/7300a7f9753f28b504ce5e2dab9c0762a6b36008/pyspec/ccd/files.py#L175
    meta-data and parsing from: https://github.com/ashirsch/spe2py/blob/cf5534bb997774e5edf2fa4f7385214ea835cd8b/spe2py.py#L134
    '''

    _datastart = 4100
    _datemax = 10
    _timemax = 7

    def __init__(self, fname):
        self._fid = open(fname, 'rb')
        self._load_size()
        #self._load_date_time()

        
        self._read_footer()
        self._get_dims()
        self._get_roi_info()
        self._get_coords()

    def _load_size(self):
        self._xdim = np.int64(self.read_at(42, 1, np.int16)[0])
        self._ydim = np.int64(self.read_at(656, 1, np.int16)[0])
        self._zdim = np.int64(self.read_at(1446, 1, np.uint32)[0])
        self.nframes =self._zdim
        dxdim = np.int64(self.read_at(6, 1, np.int16)[0])
        dydim = np.int64(self.read_at(18, 1, np.int16)[0])
        vxdim = np.int64(self.read_at(14, 1, np.int16)[0])
        vydim = np.int64(self.read_at(16, 1, np.int16)[0])
        self._size = (self._zdim, self._ydim, self._xdim)
        self._chipSize = (dydim, dxdim)
        self._vChipSize = (vydim, vxdim)
        dt = np.int16(self.read_at(108, 1, np.int16)[0])
        data_types = (np.float32, np.int32, np.int16, np.uint16)
        if (dt > 3) or (dt < 0):
            raise Exception("Unknown data type")
        self._dataType = data_types[dt]

    def _get_dims(self):
        """
        Returns the x and y dimensions for each region as stored in the XML footer
        """
        self._xdim_block = [int(block["width"]) for block in self.footer.SpeFormat.DataFormat.DataBlock.DataBlock]
        self._ydim_block = [int(block["height"]) for block in self.footer.SpeFormat.DataFormat.DataBlock.DataBlock]

        return self._xdim_block, self._ydim_block

    def _readAtString(self, pos, size):
        self._fid.seek(pos)
        return self._fid.read(size).rstrip(chr(0))

    def _read_footer(self):
        """
        Loads and parses the source file's xml footer metadata to an 'untangle' object.
        """
        footer_pos = self.read_at(678, 8, np.uint64)[0]

        self._fid.seek(footer_pos)
        xmltext = self._fid.read()

        parser = untangle.make_parser()
        sax_handler = untangle.Handler()
        parser.setContentHandler(sax_handler)

        parser.parse(StringIO(xmltext.decode('utf-8')))

        self.footer=sax_handler.root

        return self.footer

    def _get_roi_info(self):
        """
        Returns region of interest attributes and numbers of regions of interest
        """
        try:
            camerasettings = self.footer.SpeFormat.DataHistories.DataHistory.Origin.Experiment.Devices.Cameras.Camera
            regionofinterest = camerasettings.ReadoutControl.RegionsOfInterest.CustomRegions.RegionOfInterest
        except AttributeError:
            print("XML Footer was not loaded prior to calling _get_roi_info")
            raise

        if isinstance(regionofinterest, list):
            self.nroi = len(regionofinterest)
            self.roi = regionofinterest
        else:
            self.nroi = 1
            self.roi = [regionofinterest]

        self.roi

        return self.roi, self.nroi

    def _get_coords(self):
        """
        Returns x and y pixel coordinates. Used in cases where xdim and ydim do not reflect image dimensions
        (e.g. files containing frames with multiple regions of interest)
        """
        xcoord = [[] for _ in range(0, self.nroi)]
        ycoord = [[] for _ in range(0, self.nroi)]

        for roi_ind in range(0, self.nroi):
            working_roi = self.roi[roi_ind]
            ystart = int(working_roi['y'])
            ybinning = int(working_roi['yBinning'])
            yheight = int(working_roi['height'])
            ycoord[roi_ind] = range(ystart, (ystart + yheight), ybinning)

        for roi_ind in range(0, self.nroi):
            working_roi = self.roi[roi_ind]
            xstart = int(working_roi['x'])
            xbinning = int(working_roi['xBinning'])
            xwidth = int(working_roi['width'])
            xcoord[roi_ind] = range(xstart, (xstart + xwidth), xbinning)

        self.xcoord = xcoord
        self.ycoord = ycoord

        return self.xcoord, self.ycoord

    def _read_data(self):
        """
        Loads raw image data into an nframes X nroi list of arrays.
        """
        self._fid.seek(4100)

        frame_stride = int(self.footer.SpeFormat.DataFormat.DataBlock['stride'])
        frame_size = int(self.footer.SpeFormat.DataFormat.DataBlock['size'])
        metadata_size = frame_stride - frame_size
        if metadata_size != 0:
            metadata_dtypes, metadata_names = None,None #not implemented now: self._get_meta_dtype()
            metadata = np.zeros((self.nframes, len(metadata_dtypes)))
        else:
            metadata_dtypes, metadata_names = None, None
            metadata = None

        data = [[0 for _ in range(self.nroi)] for _ in range(self.nframes)]
        for frame in range(0, self.nframes):
            for region in range(0, self.nroi):
                if self.nroi > 1:
                    data_xdim = len(self.xcoord[region])
                    data_ydim = len(self.ycoord[region])
                else:
                    data_xdim = np.asarray(self._xdim_block[region], np.uint32)
                    data_ydim = np.asarray(self._ydim_block[region], np.uint32)
                data[frame][region] = np.fromfile(self._fid, self._dataType, data_xdim * data_ydim).reshape(data_ydim, data_xdim)
            if metadata_dtypes is not None:
                for meta_block in range(len(metadata_dtypes)):
                    metadata[frame, meta_block] = np.fromfile(self._fid, dtype=metadata_dtypes[meta_block], count=1)

        return data, metadata, metadata_names

    def _load_date_time(self):
        rawdate = self.read_at(20, self._datemax, np.int8)
        rawtime = self.read_at(172, self._timemax, np.int8)
        strdate = ''
        for ch in rawdate:
            strdate += chr(ch)
        for ch in rawtime:
            strdate += chr(ch)
        self._date_time = time.strptime(strdate, "%d%b%Y%H%M%S")

    def get_size(self):
        return self._size

    def get_datetime(self):
        return self._date_time

    def read_at(self, pos, size, ntype):
        self._fid.seek(pos)
        return np.fromfile(self._fid, ntype, size)

    def _readArray(self):
        self._fid.seek(self._datastart)
        self._array = np.fromfile(self._fid, dtype = self._dataType, count = -1)
        size_diff=np.abs(len(self._array)-np.prod(self._size))
        if size_diff>0:
            self._array_leftover=self._array[-size_diff:]
            self._array=self._array[:-size_diff]

        self._array = self._array.reshape(self._size)

    def load_img(self):
        img = self.read_at(4100, self._xdim * self._ydim * self._zdim, np.uint16)
        return img.reshape((self._zdim,self._ydim, self._xdim))

    def close(self):
        self._fid.close()


def load_SPE(fname):
    '''
    Load function for SPE-class.
    '''
    fid = SPEloader(fname)
    img = fid.load_img()
    fid.close()
    return nip.image(img)


def osaka_load_2Dscan(im_path='',im=None,imd=[128, 128], overscan=[1, 1.25], nbr_det=[16, 16], reader=2):
    """Takes Scan-overhead into account. As SPE-is not working right now, assumes TIF-stacks.

    Parameters
    ----------
    im_path : str
        path of image to be loaded
    imd : list, optional
        image dimensions of scan, by default [128,128]
    overscan : list, optional
        scanning overhead per axis, by default [1,1.25]
    nbr_det : list, optional
        number of detectors per dimension used for scanning, by default [16,16]
    reader : int, optional
        readin method to be used, by default 2
            1: nip.readim
            2: tifffile.imread

    Returns
    -------
    im : image
        read in 2D-image with 1 pinhole dimension reduced to 1, shape [N,M,L] with M,L = scan directions

    See Also
    --------
    convert2matlab_Osaka
    """
    # which reader to use?
    if im is None:
        if reader == 1:
            im = np.squeeze(nip.readim(im_path))
        else:
            im = nip.image(tifimread(im_path))

    # take overscan into account and throw out
    imshape = [imd[0]*overscan[0],
               int(imd[1]*overscan[1]), nbr_det[0]*nbr_det[1]]
    im = np.reshape(im, imshape)[:imd[0], :imd[1]]

    # reshape properly
    im = np.transpose(im, [2, 0, 1])

    # done?
    return im


def osaka_convert2matlab(fd, imd=[128, 128], overscan=[1, 1.25], nbr_det=[16, 16], reader=2):
    """Load 2D-overscan OSAKA-Data and save into 1D-list to be read by MATLAB for PSF estimation.

    Parameters
    ----------
    fd : dict
        Dictionary of files (and paths) to be loaded
    imd : list, optional
        image dimensions of scan, by default [128,128]
    overscan : list, optional
        scanning overhead per axis -> see load_osakaScan2D, by default [1,1.25]
    nbr_det : list, optional
        number of detectors per dimension used for scanning -> see load_osakaScan2D, by default [16,16]
    reader : int, optional
        readin method to be used -> see load_osakaScan2D, by default 2

    Raises
    ------
    ValueError
        Loaded and stored data are not the same -> error with output

    Example
    -------
    >>> load_path = os.getcwd()
    >>> save_path = os.getcwd()
    >>> beads = {'low': os.path.join(load_path,'beads_SAX_low_10um_128_200_EM1_metamorph.tif'), \
    'high': os.path.join(load_path,'beads_SAX_high_10um_128_200_EM1_metamorph.tif')}
    >>> mipy.convert2matlab_Osaka(beads,save_path,imd=[128,128])

    See Also
    --------
    load_osakaScan2D, load_SPE

    TODO
    ----
    1) add physical data to store directly into files from experiment
    """
    # traverse dictionary
    for m in fd:
        # parameter
        load_name = fd[m]
        save_name = fd[m][:-4] + '_forMATLAB.tif'

        # load and reshape data to list
        dat = load_osakaScan2D(
            load_name, imd=imd, overscan=overscan, nbr_det=nbr_det, reader=reader)
        dat = np.reshape(dat, (np.prod(dat.shape[:2]),) + dat.shape[2:])

        # put in row for easier processing
        nip.imsave(dat, load_name[:-4] + '_forMATLAB.tif', form='tif',
                   rescale=False, truncate=False, Floating=True)

        # readin again
        if reader == 2:
            dat2 = tifimread(save_name)
        else:
            dat2 = nip.readim(save_name)

        # compare datasets
        if not np.allclose(dat, dat2):
            raise ValueError('Stored and loaded data are not the same.')
    # done?
    return 'done'

# %% ------------------------------------------------------
# ---            Directory&Filestructure                ---
# ---------------------------------------------------------
#


def rename_files(file_dir, extension='jpg', version=1):
    '''
    Renames numbered stack and inserts 0s so that readin it in works better.
    Leaves out image #9. Why?

    Note:
        09.07.2020 -> version_parameter not needen anymore due to updated regex for file_search.
    '''
    tstart = time.time()
    brename = False

    # load file_list without path and exclude files of wrong extension
    # glob(load_experiment + '*.' + extension)
    file_list = os.listdir(file_dir)
    file_list = [m for m in file_list if m[-len(extension):] == extension]

    # sorts the string-list ascending by length
    if len(file_list):
        file_list.sort(key=len)
        index_max_nbr = len(file_list)
        file_max_length = len(file_list[-1])

        # do renaming
        if not len(file_list[0]) == file_max_length:
            for myc in range(0, index_max_nbr-1):
                file_len = len(file_list[myc])
                if(file_len < file_max_length):
                    # if version == 0:  # for older measurements structure was 'yyyy-mm-dd_techique_nbr_TECH_NIQUE.jpg'
                    #    pos_help = re.search('_[0-9]+_', file_list[myc])
                    # elif version == 1:  # for new structure, e.g '2019-07-12_Custom_7114.jpg'
                    #    pos_help = re.search('_[0-9]+.', file_list[myc])
                    # elif version == 2:  # for new structure, e.g '20190815-TYPE-Technique--00001.jpg'
                    #    pos_help = re.search('--[0-9]+.', file_list[myc])
                    # else:  # for new structure, e.g '20190815-TYPE-Technique-00001.jpg'
                    #    pos_help = re.search('-[0-9]+.', file_list[myc])
                    try:
                        pos_help = re.search(
                            '(_|--|-)[0-9]+(.|_)', file_list[myc])
                        string_help = str(0)*(file_max_length-file_len)
                        offset = pos_help.start()+pos_help.lastindex-1
                        os.rename(file_dir + file_list[myc], file_dir + file_list[myc]
                                  [0:offset] + string_help + file_list[myc][offset:])
                    except Exception as e:
                        print(
                            "Input file myc={}, hence: {} has wrong formatting. Exception: -->{}<-- raised. ".format(myc, file_list[myc], e))

            brename = True
        tdelta = time.time()-tstart
        print('Renaming took: {0}s.'.format(tdelta))

    else:
        print('No file with extension >{}< found in directory {}.'.format(
            extension, file_dir))
        tdelta = 0
        brename = False

    # done?
    return tdelta, brename


def dir_test_existance(mydir):
    try:
        if not os.path.exists(mydir):
            os.makedirs(mydir)
            # logger.debug(
            #    'Folder: {0} created successfully'.format(mydir))
    finally:
        # logger.debug('Folder check done!')
        pass


def delete_files_in_path(load_path):
    '''
    Deletes all files from a path, but leaves directories.
    '''
    for root_path, dirs, files in os.walk(load_path):
        for file in files:
            os.remove(os.path.join(root_path, file))


def fill_zeros(nbr, max_nbr):
    '''
    Fills pads zeros according to max_nbr in front of a number and returns it as string.
    '''
    return str(nbr).zfill(int(np.log10(max_nbr))+1)


def paths_from_dict(path_dict):
    '''
    generates full paths from dict:

    Example:
    ========
    if system == 'linux':
        base_drive = '/media/rene/Rene_work_backup/'
    else:
        base_drive = 'D:/'
    path_dict = {'e001': {'base_drive': base_drive, 'base_raw': 'Data/01_Fluidi/data/' + 'raw/', 'base_processed': 'Data/01_Fluidi/data/' +
                      'processed/', 'base_device': 'Inkubator-Setup03-UC2_Inku_405nm/', 'base_date': '20191114/', 'base_experiment': 'expt_001/', 'base_device_short': 'Inku03'}}
    load_path, save_path = paths_from_dict(path_dict)
    '''
    load_path = []
    save_path = []
    for m in path_dict:
        m = path_dict[m]
        load_path.append(m['base_drive'] + m['base_raw'] +
                         m['base_device'] + m['base_date'] + m['base_experiment'])
        save_path.append(m['base_drive'] + m['base_processed'] +
                         m['base_device'] + m['base_date'] + m['base_experiment'])
    return load_path, save_path


# %% ------------------------------------------------------
# ---                        Plotting                   ---
# ---------------------------------------------------------
#
def add_coordinate_axis(ax, apos=[[120, 10], [120, 10]], alen=[[-40, 0], [0, 40]], acolor=[1, 1, 1], text=['y', 'x'], tcolor=[1, 1, 1], tsize=None):
    for p, pos in enumerate(apos):
        alenA = alen[p]
        patch_size = np.sum(abs(np.array(alenA)))//4
        tsize = 5*patch_size if tsize is None else tsize
        ax.add_patch(Arrow(y=pos[0], x=pos[1], dy=alenA[0],
                           dx=alenA[1], width=patch_size, color=acolor))
        ax.text(x=pos[1]+1.1*alenA[1], y=pos[0]+1.1*alenA[0],
                s=text[p], color=tcolor, fontsize=tsize)

def plot_value_in_box(ax, ys, box_text, box_prop={}, text_prop={}):
    # default
    box_xy = [ys[-1]//3, ys[-2]//8]
    box_prop_default = {'xy': [ys[-1]-box_xy[0], ys[-2]-box_xy[1]], 'width': box_xy[0] -
                        3, 'height': box_xy[1]-3, 'color': 'white', 'facecolor': 'none', 'fill': False}
    text_prop_default = {'x': box_prop_default['xy'][0]+1, 'y': ys[-2] -
                         6, 'color': 'white', 'fontsize': 40}

    bp = box_prop_default if box_prop == {} else fill_dict1_with_dict2(
        box_prop_default, box_prop)
    tp = text_prop_default if box_prop == {
    } else fill_dict1_with_dict2(text_prop_default, text_prop)
    tpp = deepcopy(tp)
    del tpp['x'], tpp['y']

    for m, bname in enumerate(box_text):
        ax[m].add_patch(Rectangle(**bp))  # 'none
        ax[m].text(tp['x'], tp['y'], bname, **tpp)

    return ax


def default_grid_param(plt_raster):
    return {'pos': 111, 'nrows_ncols': (
        plt_raster[0], plt_raster[1]), 'axes_pad': 0.1, 'cbar_mode': 'single', 'cbar_location': 'right', 'cbar_pad': 0.1,'cbar_font_size':0.5,'cbar_va':'top','cbar_rotation':90,'cbar_label':''}


def default_coord_axis(imshape):
    return {'ax': 0, 'apos': [[int(imshape[-2]*0.95), int(imshape[-1]*0.05)], ]*2, 'alen': [[-imshape[-2]//3, 0], [0, imshape[-2]//3]], 'acolor': [1, 1, 1], 'text': ['y', 'x'], 'tcolor': [1, 1, 1], 'tsize': None}


def print_stack2subplot(imstack, im_minmax=[None, None], imdir='row', inplace=False, plt_raster=[4, 4], plt_format=[8, 6], title=None, titlestack=True, colorbar=True, axislabel=True, laytight=True, nbrs=True, nbrs_list=97, nbrs_color=[1, 1, 1], nbrs_size=None, nbrs_offsets=None, xy_norm=None, aspect=None, use_axis=None, plt_show=False, gridspec_kw=None, yx_ticks=None, yx_labels=None,axticks_format=None,ax_extent=False, grid_param=None, norm_imstack=False, coord_axis=[],ax_ret=False):
    '''
    Plots an 3D-Image-stack as set of subplots
    Based on this: https://stackoverflow.com/a/46616645
    '''
    # sanity checks
    if type(imstack) == list:
        imstack_len = len(imstack)
    # needs NanoImagingPack imported as nip
    elif type(imstack) == nip.image or type(imstack) == np.ndarray:
        imstack_len = imstack.shape[0]
    else:
        raise TypeError("Unexpected Data-type.")
    if not(imstack_len > 0):
        raise ValueError("Image size not fitting!")
    if not inplace:
        imstack = nip.image(np.array(imstack))
    # check for title
    from datetime import datetime
    if title == None:
        title = datetime.now().strftime("%d.%m.%Y %H:%M:%S")

    # imstack alignment in final image
    if type(nbrs_list) == int:
        use_gen_nbrs_list = True
        if imdir == 'col':
            nbrs_list = np.transpose(np.reshape(
                np.arange(np.prod(plt_raster)), plt_raster[::-1])).flatten()+nbrs_list

        elif type(imdir) in [list, tuple, np.ndarray]:
            # rethink this option!
            # for m, imd in enumerate(imdir):
            #    imstack_list.append(imstack[imd])
            nbrs_list = np.array(imdir)
        else:
            nbrs_list = np.arange(np.prod(plt_raster))+nbrs_list
    else:
        use_gen_nbrs_list = False

    # create figure (fig), and array of axes (ax)-> gridspec_kw = {'height_ratios':[2,2,1,1]}
    if colorbar == 'global':
        # sanity
        grid_param_default = default_grid_param(plt_raster)
        gp = grid_param = grid_param_default if grid_param is None else fill_dict1_with_dict2(
            grid_param_default,grid_param)

        # generate grid
        fig = plt.figure(figsize=plt_format)
        grid = AxesGrid(fig, gp['pos'],
                        nrows_ncols=gp['nrows_ncols'],
                        axes_pad=gp['axes_pad'],
                        cbar_mode=gp['cbar_mode'],
                        cbar_location=gp['cbar_location'],
                        cbar_pad=gp['cbar_pad']
                        )
        ax = np.array(grid.axes_all)

        if norm_imstack:
            imstack = normNoff(imstack, dims=(-2, -1), method='max', direct=False,)

        # assure proper results
        imstack[np.isnan(imstack)] = 0
        xy_norm_h = [650, 650]

    else:
        fig, ax = plt.subplots(
            nrows=plt_raster[0], ncols=plt_raster[1], figsize=plt_format, sharex=True, sharey=False,
            gridspec_kw=gridspec_kw)
        xy_norm_h = [250.0, 250.0]

    if not type(ax) == np.ndarray:
        ax = np.array([ax, ])

    xy_norm = xy_norm_h if xy_norm is None else xy_norm

    # rescaling parameteres
    xy_extent=None
    if ax_extent:
        xy_extent = None if yx_ticks is None else [
            yx_ticks[1][0], yx_ticks[1][-1], yx_ticks[0][0], yx_ticks[0][-1]]
        if aspect is None:
            aspect = 'auto' if xy_extent is None else (
                xy_extent[1]-xy_extent[0])/(xy_extent[3]-xy_extent[2])
    
    # parameter
    ax_meas = [ax.flat[0].bbox.width, ax.flat[0].bbox.height]
    if nbrs_offsets is None:
        x_offset = np.round(3*ax_meas[0] / xy_norm[1]) .astype('uint16')
        y_offset = np.round(21*ax_meas[1]/xy_norm[1]).astype('uint16')
    else:
        y_offset, x_offset = nbrs_offsets

    # prepare variables to rescale graph
    nbrs_psize = 50*0.2 if nbrs_size is None else 50*nbrs_size
    if ax_extent:#if not yx_ticks is None:
        dx = xy_extent[1]-xy_extent[0]
        dy = xy_extent[3]-xy_extent[2]
        x_offset *= dx/ax_meas[0]
        y_offset = xy_extent[3] - 2*dy/ax_meas[1]*y_offset

    # plot simple raster image on each sub-plot
    for m, axm in enumerate(ax.flat):
        ima = axm.imshow(imstack[m], alpha=1.0,
                         interpolation='none', extent=xy_extent, aspect=aspect, vmin=im_minmax[0], vmax=im_minmax[1])  # alpha=0.25)
        # write row/col indices as axes' title for identification
        if colorbar == True:
            divider = make_axes_locatable(axm)
            colorbar_axes = divider.append_axes("right", size="10%", pad=0.1)
            cbar = fig.colorbar(ima, cax=colorbar_axes)
            imstp = [np.min(imstack[m]), np.max(imstack[m])]
            cbar.set_ticks(np.arange(imstp[0], imstp[1], (imstp[1]-imstp[0])/5))

        if axislabel == True:
            axm.set_xlabel('[PIX] = a.u.')
            axm.set_ylabel('[PIX] = a.u.')
        elif axislabel == False:
            pass
        else:
            axm.set_xlabel(axislabel[0], fontsize=nbrs_psize)
            axm.set_ylabel(axislabel[1], fontsize=nbrs_psize)

        if titlestack == True:
            axm.set_title(
                "Row:"+str(m // plt_raster[1])+", Col:"+str(m % plt_raster[1]))
        elif titlestack == False:
            pass
        else:
            axm.set_title(titlestack[m], fontsize=nbrs_psize*4/5)

        # add numbers to panels
        if nbrs:
            aletter = chr(nbrs_list[m]) if use_gen_nbrs_list else nbrs_list[m]
            axm.text(x_offset, y_offset, aletter+')', fontsize=nbrs_psize,
                     color=nbrs_color, fontname='Helvetica', weight='normal')

        # set axis format
        if not axticks_format is None:
            axm.xaxis.set_major_formatter(FormatStrFormatter(axticks_format[0]))
            axm.yaxis.set_major_formatter(FormatStrFormatter(axticks_format[1]))

        if not yx_ticks is None:
            if yx_labels is None:
                yx_labels=[[str(mitem) for mitem in yx_ticks[0]],[str(mitem) for mitem in yx_ticks[1]]]
            #axm.set_xticks(yx_ticks[1])
            #axm.set_xticklabels(yx_labels[1],fontsize=nbrs_psize/2)
            axm.xaxis.set_ticks(yx_ticks[1])
            axm.xaxis.set_ticklabels(yx_labels[1],fontsize=nbrs_psize*4/5)
            #axm.yaxis.set_tick_params(labelsize=nbrs_psize)
            #axm.set_yticks(yx_ticks[0])
            #axm.set_yticklabels(yx_labels[0],fontsize=nbrs_psize/2)
            axm.yaxis.set_ticks(yx_ticks[0])
            axm.yaxis.set_ticklabels(yx_labels[0],fontsize=nbrs_psize*4/5)

        # plot axis according to selection -> in case of 'bottom','left' axis-names are only plotted at bottom-left panel
        if use_axis == None:
            axm.axis('off')
        else:
            if 'bottom' in use_axis:
                if m//plt_raster[1] < (plt_raster[0]-1):
                    axm.xaxis.set_visible(False)
                else:
                    if 'left' in use_axis:
                        if not m%plt_raster[1] == 0:
                            axm.xaxis.set_ticklabels("")
                            axm.set_xlabel('')
                    elif 'right' in use_axis:
                        if not m%plt_raster[1] == plt_raster[1]-1:
                            axm.xaxis.set_ticklabels("")
                            axm.set_xlabel('')
            elif 'top' in use_axis:
                if m//plt_raster[1] > 0:
                    axm.xaxis.set_visible(False)
                else:
                    if 'left' in use_axis:
                        if not m%plt_raster[1] == 0:
                            axm.xaxis.set_ticklabels("")
                            axm.set_xlabel('')
                    elif 'right' in use_axis:
                        if not m%plt_raster[1] == plt_raster[1]-1:
                            axm.xaxis.set_ticklabels("")
                            axm.set_xlabel('')
            else:
                pass
            if 'left' in use_axis:
                if m % plt_raster[1] > 0:
                    axm.yaxis.set_visible(False)
                else:
                    if 'top' in use_axis:
                        if not m//plt_raster[1] == 0:
                            axm.yaxis.set_ticklabels("")
                            axm.set_ylabel('')
                    elif 'bottom' in use_axis:
                        if not m//plt_raster[1] == plt_raster[0]-1:
                            axm.yaxis.set_ticklabels("")
                            axm.set_ylabel('')
            elif 'right' in use_axis:
                if m % plt_raster[1] < plt_raster[1]-1:
                    axm.yaxis.set_visible(False)
                else:
                    if 'top' in use_axis:
                        if not m//plt_raster[1] == 0:
                            axm.yaxis.set_ticklabels("")
                            axm.set_ylabel('')
                    elif 'bottom' in use_axis:
                        if not m//plt_raster[1] == plt_raster[0]-1:
                            axm.yaxis.set_ticklabels("")
                            axm.set_ylabel('')
            else:
                pass

        # add coordinate axis
        if not coord_axis == []:
            for ca in coord_axis:
                if ca['ax'] == m:
                    ca['ax'] = axm
                    add_coordinate_axis(**ca)

        if m >= imstack_len-1:
            break

    if colorbar == 'global':
        cbar = grid.cbar_axes[0].colorbar(ima)
        cbar.ax.tick_params(labelsize=gp['cbar_font_size']*2/3)
        #cbar.ax.set_ylim()
        cbar.ax.set_ylabel(gp['cbar_label'], rotation=gp['cbar_rotation'], va=gp['cbar_va'], fontsize=gp['cbar_font_size'])

    # delete empty axes
    while m+1 < (plt_raster[0]*plt_raster[1]):
        m += 1
        fig.delaxes(ax.flatten()[m])

    # add topic
    if title:
        fig.suptitle(title)
    # fix layout
    if laytight:
        plt.tight_layout()

    if plt_show:
        fig.show()
    
    if ax_ret:
        return fig,ax
    else:
        return fig

# not finished yet and maybe even a bad idea...
# class Draw():
#    """General Drawing class to generate basic plots for typical image-stacks that are generated within the different projects. It incorporates the plotting #functions:
#    * print_stack2subplot
#    * stack2plot
#    * plot_3dstacks
#
#    and tries to overcome all the positioning limitations of legends etc
#    """
#
#    # Dimensions
#    is_2D = True
#
#    # Figure and canvas dimensions
#    figure_size = [8, 8]
#    figure_dpi = 300
#    rows = 1
#    cols = 1
#
#    # labels
#    xlabel = 'Pixel [a.u.]'
#    ylabel = 'Pixel [a.u.]'
#    zlabel = 'Pixel [a.u.]'
#    label_share_axes = True
#
#    figure_title = ''
#    subplot_title = ''
#    subplot_nbrs_show = False
#    subplot_nbrs_colors = [0, 0, 0]
#    subplot_nbrs_size = 0.2
#    subplot_shape = [2, 3]
#
#    label_fontname = 'Helvetica'
#    label_weight = 'normal'
#
#    # Axes
#    axis_show = False
#    projection = 'rectilinear'
#    active_ax = None
#
#    # Colors
#    colors = None
#    colors_cmap = cm.viridis
#
#    # Lines and Markers
#    marker_style = ''
#    marker_color = ''
#    marker_size = ''
#    line_style = '-'
#    line_width = 2.0
#    line_color = 'b'
#
#    # Graphic Options
#    aliased = False
#
#    # Legend and colorbar
#    legend = False
#    legend_loc = None
#    legend_pos = [1.05, 1.2]
#    colorbar = False
#    colorbar_pos = 'right'
#    colorbar_size = '10%'
#    colorbar_pad = '0.1'
#
#    # Layout
#    tight_layout = False
#
#    # Data
#    datx = None
#    daty = None
#    datz = None
#    dat_stack = daty
#    dat_type = 'g'  # 'g'=graph, 'im'=image
#
#    # output options
#    show_plot = False
#
#    def __init__(self, **kwargs):
#        # populate basic_attr_dict with user-input
#        [setattr(self, m, kwargs[m]) for m in kwargs if getattr(self, m)]
#
#    def create_canvas(self):
#        self.fig = mpl.figure.Figure(figsize=self.figsize, dpi=self.dpi)
#
#    @staticmethod
#    def is_list(obj):
#        if type(obj) in [list, tuple, np.ndarray]:
#            return True
#        else:
#            return False
#
#    def make_listable(self, obj_list, ldim_list):
#        for m, obj in enumerate(obj_list):
#            if not self.is_list(getattr(self, obj)):
#                setattr(self, obj, [obj, ] * len(ldim_list[m]))
#
#    def draw_data(self):
#        self.make_listable(['projection', 'dat_type', 'is_2D'], [len(self.dat_stack), ]*3)
#        self.projection = self.make_listable(self.projection, len(self.dat_stack))
#        for m, dat in enumerate(self.dat_stack):
#            self.active_ax = self.fig.add_subplot(
#                nrows=self.subplot_shape[0], ncols=self.subplot_shape[1], index=m, projection=self.projection(m))
#            self.ax.append(self.active_ax)
#
#            if self.is_2D[m]:
#                if self.dat_type[m] == 'g':
#                    prepare_data(self.datx)
#                    self.active_ax.plot()
#                else:
#                    self.active_ax.imshow()
#            else:
#                pass
#
#    def prepare_data(is_2D, datx, daty=None)::
#        pass
#


def stack2plot(x:np.ndarray, ystack:Union[list,np.ndarray], refs:list=None, title:str=None, xl:list=None, xlabel:str=None, ylabel:str=None, colors:list=None, mmarker:str='', mlinestyle:str='-', mlinewidth:list=None, legend:Union[str,list]=[1, 1.05], legend_col:int=1,xlims:Union[list,np.ndarray]=None, ylims:Union[list,np.ndarray]=None, ax_inside:dict=None, figsize:tuple=(8, 8), show_plot:bool=True, ptight:bool=True, ax=None, nbrs:bool=True, nbrs_color:list=[1, 1, 1], nbrs_size:float=None, nbrs_text:str=None, nbrs_offset:list=[],err_bar:Union[list,np.ndarray]=None, err_capsize:int=3, fonts_labels:list=[None,],fonts_sizes:list=[None,],set_clipon:bool=False) -> tuple:
    '''
    Prints a 1d-"ystack" into 1 plot and assigns legends + titles.

    returns fig,ax

    Example:
    =======
    fig_size=(8,8)
    fonts_size = np.max(fig_size)*2
    fonts_labels = ['title', 'x', 'y', 'xtick', 'ytick']
    fonts_sizes = [fonts_size,fonts_size,fonts_size,[fonts_size*0.75,]*9,[fonts_size*0.75,]*9]
    '''
    # sanity
    if not mlinewidth is None and not type(mlinewidth) in [np.ndarray, list, tuple]:
        mlinewidth = [mlinewidth, ]*len(ystack)
    if not type(mlinestyle) in [np.ndarray, list, tuple]:
        mlinestyle = [mlinestyle, ]*len(ystack)
    if not type(mmarker) in [np.ndarray, list, tuple]:
        mmarker = [mmarker, ]*len(ystack)

    if xl == None and type(x[0]) in [str, np.str_]:
        xl = [np.arange(len(x)), np.array(x)]
        x = np.arange(len(x))

    if not type(x[0]) in [list,tuple,np.ndarray,nip.image]:
        x = [x,]*len(ystack)
    x=np.array(x)

    # get figure
    if ax is None:
        fig1 = plt.figure(figsize=figsize)
        ax = plt.subplot(111)
    else:
        fig1 = ax.get_figure()

    # parameters
    legend_font_size = None

    # plot
    for m in range(len(ystack)):
        label = str(m) if refs is None else refs[m]
        colorse = tuple(np.random.rand(3)) if colors is None else colors[m]
        xlabel = 'Pixel' if xlabel is None else xlabel
        ylabel = 'Pixel' if ylabel is None else ylabel
        title = datetime.now().strftime("%Y%M%D") if title is None else title
        if err_bar is None:
            line, = ax.plot(x[m], ystack[m], label=label, color=colorse,
                            marker=mmarker[m], linestyle=mlinestyle[m])
        else:
            line = ax.errorbar(x[m], ystack[m], err_bar[m], label=label, color=colorse,
                               marker=mmarker[m], linestyle=mlinestyle[m], capsize=err_capsize)
        line.set_clip_on(set_clipon)
        if not mlinewidth is None:
            line.set_linewidth(mlinewidth[m])
        line.set_antialiased(False)

    # set labels and ticks
    ax.set_xlabel(xlabel)
    if not xl is None:
        ax.set_xticks(xl[0])
        ax.set_xticklabels(xl[1])
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # axis limits
    if xlims is None:
        xlims=[[np.min(mx), np.max(mx)] for mx in x]
        xlims=[np.min(xlims),np.max(xlims)]
    else:
        xlims = [ax.get_xlim()[m] if mx==None else mx for m,mx in enumerate(xlims)]
    if ylims is None:
        ylims=[[np.min(my), np.max(my)] for my in ystack]
        ylims=[np.min(ylims),np.max(ylims)]
    else:
        ylims = [ax.get_ylim()[m] if my==None else my for m,my in enumerate(ylims)]
    ylims_dist = ylims[1]-ylims[0]
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    # font sizes
    if not fonts_labels[0] is None:
        if not fonts_sizes[0] is None:
            fonts_dict={'title':ax.title, 'x':ax.xaxis.label, 'y': ax.yaxis.label, 'xtick':
                    ax.get_xticklabels(),'ytick': ax.get_yticklabels(), 'legend':'legend'}
            if len(fonts_sizes)<len(fonts_labels):
                fonts_sizes=[fonts_sizes[0],]*len(fonts_labels)
            for m,item in enumerate(fonts_labels):
                if type(fonts_dict[item])==list:
                    for n,itemitem in enumerate(fonts_dict[item]):
                        if type(fonts_sizes[m])==list:
                            fs = fonts_sizes[m][len(fonts_sizes[m])-1] if len(fonts_sizes[m])< n else fonts_sizes[m][n]
                            itemitem.set_fontsize(fs)
                        else: 
                            itemitem.set_fontsize(fonts_sizes[m]*0.75)
                else:
                    if item == 'legend':
                        legend_font_size=fonts_sizes[m]*0.75
                    else:
                        fonts_dict[item].set_fontsize(fonts_sizes[m])

    if not ax_inside is None:
        # example: ax_inside={"x":-15,"y":"-22"}
        for axkey in ax_inside:
            ax.tick_params(axis=axkey, direction="in", pad=ax_inside[axkey])

    if nbrs:
        text_size_factor = ax.figure.bbox_inches.bounds[-1]*ax.figure.dpi
        nbrs_psize = text_size_factor * 0.05 if nbrs_size is None else text_size_factor*nbrs_size
        nbrs_offset = [np.mean(ax.get_xticks()[:2]),ylims[1]-ylims_dist/5.0] if nbrs_offset == [] else nbrs_offset
        nbrs_text = chr(97+np.random.randint(26))+')' if nbrs_text is None else nbrs_text
        ax.text(nbrs_offset[0], nbrs_offset[1], nbrs_text, fontsize=nbrs_psize,
                color=nbrs_color, fontname='Helvetica', weight='normal')
 
    # legend
    if not legend is None:
        legs=[None,legend] if type(legend) == list else [legend,None]
        ax.legend(loc=legs[0],bbox_to_anchor=legs[1], fontsize=legend_font_size,ncol=legend_col)

    if ptight:
        plt.tight_layout()

    if show_plot:
        plt.show()

    return fig1, ax


def plot_3dstacks(res_noisy_s, defocus_range, noise_stack, myfilters, axis_labels=['', '', ''], show_plot=True, suptitle='', nbrs=False, nbrs_color=[0, 0, 0]):
    #fig3 = plt.figure(figsize=(6, 10))  #
    mcols = 3
    mrows = len(myfilters)//mcols
    fig3, axm3 = plt.subplots(mrows, mcols, figsize=(9, 12),
                              subplot_kw={'projection': '3d'})  # sharey=True,sharex=True,
    axm3 = axm3.flat

    ax_meas = [axm3[0].bbox.width, axm3[0].bbox.height]

    try:
        if len(defocus_range[0]) != 0:
            defocus_use_indiv = True
    except:
        defocus_use_indiv = False
    try:
        if len(noise_stack[0]) != 0:
            noise_use_indiv = True
    except:
        noise_use_indiv = False

    for m, mfilters in enumerate(myfilters):
        ax3 = axm3[m]
        # ax3 = fig3.add_subplot(len(myfilters)//3, 3, m+1, projection='3d')
        xrange = defocus_range[m] if defocus_use_indiv else defocus_range
        yrange = noise_stack[m] if noise_use_indiv else noise_stack
        Y, X = np.meshgrid(xrange, yrange)
        ax3.plot_surface(np.log(X), Y, res_noisy_s[m],
                         cmap=cm.coolwarm, linewidth=0, antialiased=False)
        # ax3.plot_trisurf(np.log(X), Y, res_noisy_s[m],                     triangles=tri.triangles, cmap='viridis', linewidths=0.2)
        ax3.set_xlabel(axis_labels[0])
        ax3.set_ylabel(axis_labels[1])
        ax3.set_zlabel(axis_labels[2])
        if nbrs:
            ax3.text2D(x=0.01, y=0.85, s=chr(97+m)+')', fontsize=int(
                ax_meas[1]/8), color=nbrs_color, fontname='Helvetica', weight='normal', transform=ax3.transAxes)
        else:
            ax3.set_title(mfilters)
        # ax3.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        # ax3.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        if m//3 < (mrows-1):
            ax3.get_xaxis().set_visible(False)
            ax3.xaxis.set_ticklabels("")
            ax3.set_xlabel('')
            ax3.get_yaxis().set_visible(False)
            ax3.yaxis.set_ticklabels("")
            ax3.set_ylabel('')
            # ax.xaxis.set_major_locator(ticker.NullLocator())
        if m % 3 < (mcols-1):
            ax3.get_yaxis().set_visible(False)
            ax3.yaxis.set_ticklabels("")
            ax3.set_ylabel('')
            ax3.get_zaxis().set_visible(False)
            ax3.zaxis.set_ticklabels("")
            ax3.set_zlabel('')
            # ax.yaxis.set_major_locator(ticker.NullLocator())
            # ax.xaxis.ticks
        # ax3.set_xlim(0, np.round(np.max(X)))
        # ax3.set_ylim(0, np.max(Y))
        # ax3.set_zlim(0, 1)
    # fig3.subplots_adjust(hspace=1.0, wspace=1.0)
    # plt.tight_layout()
    fig3.suptitle(suptitle)
    if show_plot:
        plt.show()
    return fig3, axm3


def plot_kradial(imb: np.ndarray, krad_scale_fac: float = 1, nbr_bins: int = 32, obj: np.ndarray = None, res_param: list = [], ref_names: list = [], ref_colors: list = [], imname='h', figsize: tuple = (10, 8), fonts_sizes: list = [10, ], fonts_labels: list = ['x', 'y', 'xtick', 'ytick', 'legend'], do_nn=True, nn_range: int = 0, region_draw: bool = True, region_im2use: list = [1, -1], region_yxlim: list = [1.0, 1.0], region_line_colors: list = [], region_colors: list = [], region_colors_scale: list = [1.7, ], xlims: list = [], ylims: list = [], draw_Abbe: bool = False, movavg_do: bool = False, movavg_len: int = 4, log_do: bool = True):
    '''
    krad_scale_fac = 0.9*np.sqrt(2)
    res_param=[PIXELSIZE,NA,N_IMMERSION,λem,λex,TECHNIQUE]=[pad_snr['pixelsize'][-1],pad_snr['NA'],pad_snr['n_immersion'],pad_snr['lem'],pad_snr['lex'],'confocal']
    '''
    # im_maxfreq = imb.shape[-1]//2

    if nbr_bins is None:
        nbr_bins = int(np.ceil(krad_scale_fac*np.min(imb.shape[-2:])//2))  # 0.9*
    im = np.zeros([imb.shape[0], nbr_bins])
    for m, mIm in enumerate(np.abs(imb)):
        im[m], idx = radial_sum(
            mIm, maxfreq=nbr_bins, return_idx=True, nbr_bins=nbr_bins)
    if not obj is None:
        obj_radsum, oidx = radial_sum(
            np.abs(nip.ft2d(obj*1000)), maxfreq=nbr_bins, return_idx=True, nbr_bins=nbr_bins)
    _, dk_2c = convert_x_to_k(2*nbr_bins, res_param[0]/krad_scale_fac)
    d_abbe = calculate_resolution(obj_na=res_param[1], obj_n=res_param[2], wave_em=res_param[3],
                                       technique=res_param[5], criterium='Abbe', fluorescence=True, wave_ex=res_param[4], printout=True)
    k_abbe_2c, _ = convert_x_to_k(2*nbr_bins, d_abbe)
    xrange2c = np.arange(nbr_bins)*dk_2c/k_abbe_2c[-1]  # *1000
    # im_norm = im/np.sum(np.sqrt(ystack2b), axis=-1, keepdims=True)
    # imshepp_nf_log = np.log10(1+np.mean(imshepp_nf))
    if log_do:
        im = np.log10(1+im)
        ylabel = ['$log_{10}(1+|', '', ')$']
    else:
        ylabel = ['$|', '', '$']

    if movavg_do:
        im = nip.image(np.array([*im, *moving_average_2d(im, movavg_len)]))

    if do_nn:
        if nn_range < 0:
            imbh = np.log10(1+np.abs(imb)) if log_do else imb
            im = im/np.mean(imbh
                            [:, ~(idx+1).astype('bool')], axis=-1, keepdims=True)
            del imbh
            # nbrh=int(np.ceil(np.sqrt(2)*imb.shape[-1]//2))
            # imh = np.zeros([imb.shape[0], nbrh])
            # for m, mIm in enumerate(np.abs(imb)):
            #    imh[m], idxh = mipy.radial_sum(
            #        mIm, maxfreq=nbrh, return_idx=True, nbr_bins=nbrh)

        elif nn_range == 0:
            im = im/np.mean(im[:, -nbr_bins//4:], axis=-1, keepdims=True)
        else:
            im = im/np.mean(im[:, -nn_range:], axis=-1, keepdims=True)
        region_yxlim[0] = 1.0
        ylabel[1] = imname+'|/\\sigma^{('+imname+')}'
    else:
        ylabel[1] = imname+'|'

    ylabel = ''.join(ylabel)

    if region_draw:
        noisecuts = [np.min(np.where((im[mreg] <= region_yxlim[0]) * (xrange2c >=
                            region_yxlim[1])))-1 for mreg in region_im2use]
        if region_line_colors == []:
            region_line_colors = [ref_colors[mreg] for mreg in region_im2use]
        if region_colors == []:
            rcols = convert_colors_name2rgba(region_line_colors)
            region_colors_scale = [region_colors_scale[0], ] * \
                len(rcols) if len(region_colors_scale) < len(rcols) else region_colors_scale
            region_colors = [scale_lightness(mcol, region_colors_scale[m])
                             for m, mcol in enumerate(rcols)]
    else:
        noisecuts = []

    xlims = None if xlims == [] else xlims
    if ylims == []:
        ylims = [np.min(im), np.max(im)]
        ylims[0] = ylims[0]-(ylims[1]-ylims[0])*0.1

    fig2c, ax2c = stack2plot(xrange2c, im, refs=ref_names, title='', xlabel='$k_x/k^{(Abbe)}$',
                                  ylabel=ylabel, colors=ref_colors, legend='upper right', legend_col=2, figsize=figsize, fonts_sizes=fonts_sizes, fonts_labels=fonts_labels, nbrs=False, xlims=xlims, ylims=ylims, set_clipon=True, mmarker='.', mlinestyle='-')  # $k_x/\mu m^{-1}$
    if region_draw:
        ax2c.plot([xrange2c[0], xrange2c[-1]], [region_yxlim[0], ] *
                  2, color='darkviolet', linestyle='--', linewidth=5)
        ncs = np.argsort(noisecuts, axis=-1, kind='quicksort', order=None)
        rlcs = np.array(region_line_colors)[ncs]
        nscs = np.array(noisecuts)[ncs]
        rcs = np.array(region_colors)[ncs]
        for m, mcut in enumerate(nscs):
            ax2c.plot([xrange2c[mcut], ]*2, ylims,
                      color=rlcs[m], linestyle='--', linewidth=2)
            patch_xlim_upper = xrange2c[nscs[m+1]]-xrange2c[mcut] + \
                1 if m+1 < len(nscs) else xrange2c[-1]-xrange2c[mcut]
            patch_color = rcs[m] if m < len(nscs)-1 else 'darkgray'
            ax2c.add_patch(
                Rectangle((xrange2c[mcut], ylims[0]), patch_xlim_upper, ylims[1]-ylims[0], color=patch_color))
    if draw_Abbe:
        ax2c.plot([1, ]*2, ylims, color='gold', linestyle='--',
                  linewidth=5, marker='x')  # k_abbe_2c[0]*1000
    if not obj is None:
        obj_radsum = np.log10(1+obj_radsum) if log_do else obj_radsum
        ax2c.plot(xrange2c, obj_radsum, color='lightgrey',
                  linestyle='solid', linewidth=3, marker='.')  # /200

    ret_dict = {'xrange': xrange2c, 'yrange': im, 'idx': idx, 'k_abbe': k_abbe_2c, 'd_abbe': d_abbe,
                'noisecuts': noisecuts}

    return fig2c, ax2c, ret_dict

def plot_save(ppointer, save_name, save_format='png', dpi=300):
    '''
    Just an easy wrapper.
    '''
    ppointer.savefig(save_name + f".{save_format}",
                     dpi=dpi, bbox_inches='tight', format=save_format)


def crop_pdf(file_path, crop_param=None, file_path_cropped=None):
    print(f"Cropping file: {file_path}")
    call_list = []

    # add cropper
    call_list.append("/usr/bin/pdfcrop")

    # add margin-params
    if not crop_param is None:
        if 'margins' in crop_param:
            call_list.append("--margins '" + " ".join(crop_param['margins']) + "' ")

    # add file_path
    call_list.append(file_path)

    # add name of output or overwrite
    if file_path_cropped is None:
        call_list.append(file_path)
    else:
        call_list.append(file_path_cropped)

    # eval
    subprocess.run(call_list, stdin=None,
                   stdout=None, stderr=None, shell=False)


def save_n_crop(fig, fname:str, dpi:int=300, crop_me:bool=True, crop_param:str=None, file_path_cropped:str=None):
    fig.savefig(fname=fname, format=fname[-3:], dpi=dpi)
    if crop_me:
        crop_pdf(fname, crop_param=crop_param, file_path_cropped=file_path_cropped)

# %% ------------------------------------------------------
# ---                 Color Functions                   ---
# ---------------------------------------------------------
#
def convert_colors_name2rgba(colors: list):
    '''
    Example:
    =======
    mipy.convert_colors_name2rgba(['salmon','goldenrod'])
    '''
    return [matplotlib_colors.to_rgba(mcol) if type(mcol) == str else mcol for mcol in colors]


def scale_lightness(rgb, scale_l):
    '''mainly from: https://stackoverflow.com/a/60562502
    for now only implemented for rgb not rgba! Hence: Conversion is done if rgba is detected.
    '''
    # sanity
    if len(rgb) > 3:
        rgb = rgb[:3]

    # convert rgb to hls
    h, l, s = rgb_to_hls(*rgb)

    # manipulate h, l, s values and return as rgb
    rgb_changed = hls_to_rgb(h, min(1, l * scale_l), s=s)

    return rgb_changed

# %% ------------------------------------------------------
# ---            RESHAPE FOR COOL DISPLAY               ---
# ---------------------------------------------------------
#


def get_tiling(im, form='quadratic'):
    '''
        Gets the 2D-tiling-shape to concate an input image.
    '''
    cols = int(np.ceil(np.sqrt(im.shape[0])))
    rows = int(np.ceil(im.shape[0]/cols))
    return [rows, cols]


def stack2tiles(im, tileshape=None):
    '''
    Converts a 3D-stack into a 2D set of tiles. -> good for displaying. Tries to fill according to tile-shape as long as there is images left.

    :PARAM:
    =======
    im:         3D-image with order [Z,Y,X]
    shape:      list with shape like e.g. [2,3] meaning 2rows and 3 columed output
    OUT:
    '''
    if tileshape == None:
        tileshape = get_tiling(im, form='quadratic')
    ims = im.shape[-2:]
    iml = nip.image(
        np.zeros([ims[0]*tileshape[0], ims[1]*tileshape[-1]], dtype=im.dtype))
    imhh = nip.image(np.zeros(im.shape[-2:]))
    # concatenate list
    for m in range(tileshape[0]):
        for n in range(tileshape[1]):
            if m*tileshape[1]+n < len(im):
                imh = im[m*tileshape[1]+n]
            else:
                imh = imhh
            iml[m*ims[0]:(m+1)*ims[0], n*ims[1]:(n+1)*ims[1]] = imh
    return iml


def concat_list(imlist, cols=4, normal=False, gammal=None, method='np', dims=(-3, -2, -1)):
    # normalize
    if normal:
        imlist = normNoff(imlist, dims=dims, direct=False)

    if gammal is None:
        gammal = np.ones(len(imlist))
    elif type(gammal) in [float, int]:
        gammal = np.ones(len(imlist))*gammal
    imlist = imlist**add_multi_newaxis(gammal, newax_pos=[-1, ]*imlist[0].ndim)

    if method == 'iter':
        # prepare
        col_list = [imlist[0]**gammal[0], ]
        row_list = []

        # add images to list
        for m in range(1, len(imlist)):
            if np.mod(m, cols) == 0:
                row_list.append(nip.cat(col_list, axis=-1))
                col_list = [imlist[m], ]
            else:
                col_list.append(imlist[m])

        # make sure that final row is filled
        if not len(col_list) == cols:
            ims = np.array(imlist[0].shape)
            ims[-1] *= (cols-len(col_list))
            row_list.append(nip.cat(np.zeros(ims, dtype=imlist[0].dtype), axis=-1))

        imlist = nip.cat(row_list, axis=-2)

    else:

        nbr_fills = np.mod(cols-len(imlist), cols)
        imlist = nip.cat((imlist, np.zeros([nbr_fills, ]+list(imlist[0].shape))), axis=0)
        imlist = transpose_arbitrary(
            imlist, idx_startpos=[0, ], idx_endpos=[-2], direction='forward')
        imlist = np.reshape(imlist, list(
            imlist.shape[:-2])+[imlist.shape[-2]//cols, imlist.shape[-1]*cols])
        imlist = transpose_arbitrary(
            imlist, idx_startpos=[0, ], idx_endpos=[-1], direction='forward')
        imlist = np.reshape(imlist, list(imlist.shape[:-2])+[imlist.shape[-2]*imlist.shape[-1]])
        imlist = np.transpose(imlist, np.roll(np.arange(imlist.ndim), -1))

    # done?
    return imlist


def format_list(alist, formatting, as_one=True):
    if as_one:
        format_string = ["{:"+formatting+"}", ]*len(alist)
        format_string = ",".join(format_string)
        format_string = "["+format_string.format(*alist)+"]"
    else:
        format_string = [("{:"+formatting+"}").format(m) for m in alist]
    return format_string

# %% ------------------------------------------------------
# ---                        Time                      ---
# ---------------------------------------------------------
#


def format_time(tsec):
    '''
    Simple time formatter.

    :param:
    ======
    tsec:INT:   seconds
    '''
    th, trest = divmod(tsec, 3600)
    tm, ts = divmod(trest, 60)
    tform = '{:02}h{:02}m{:02}s.'.format(int(th), int(tm), int(ts))
    return tform
