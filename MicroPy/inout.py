import time
import os
import re
import socket
from datetime import datetime
import NanoImagingPack as nip
import numpy as np
import matplotlib.pyplot as plt
from tifffile import imread as tifimread
# mipy imports

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
    '''

    def __init__(self, fname):
        self._fid = open(fname, 'rb')
        self._load_size()

    def _load_size(self):
        self._xdim = np.int64(self.read_at(42, 1, np.int16)[0])
        self._ydim = np.int64(self.read_at(656, 1, np.int16)[0])

    def _load_date_time(self):
        rawdate = self.read_at(20, 9, np.int8)
        rawtime = self.read_at(172, 6, np.int8)
        strdate = ''
        for ch in rawdate:
            strdate += chr(ch)
        for ch in rawtime:
            strdate += chr(ch)
        self._date_time = time.strptime(strdate, "%d%b%Y%H%M%S")

    def get_size(self):
        return (self._xdim, self._ydim)

    def read_at(self, pos, size, ntype):
        self._fid.seek(pos)
        return np.fromfile(self._fid, ntype, size)

    def load_img(self):
        img = self.read_at(4100, self._xdim * self._ydim, np.uint16)
        return img.reshape((self._ydim, self._xdim))

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


def osaka_load_2Dscan(im_path, imd=[128, 128], overscan=[1, 1.25], nbr_det=[16, 16], reader=2):
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
    #glob(load_experiment + '*.' + extension)
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


def print_stack2subplot(imstack, plt_raster=[4, 4], plt_format=[8, 6], title=None, titlestack=None, colorbar=True, axislabel=False, laytight=True):
    '''
    Plots an 3D-Image-stack as set of subplots
    Based on this: https://stackoverflow.com/a/46616645
    '''
    if type(imstack) == list:
        imstack_len = len(imstack)
    # needs NanoImagingPack imported as nip
    elif type(imstack) == nip.image or type(imstack) == np.array:
        imstack_len = imstack.shape[0]
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
    return fig


def stack2plot(x, ystack, refs=None, title=None, xlabel=None, ylabel=None, colors=None):
    '''
    Prints a 1d-"ystack" into 1 plot and assigns legends + titles.
    '''
    fig1 = plt.figure()
    for m in range(len(ystack)):
        label = str(m) if refs is None else refs[m]
        colorse = tuple(np.random.rand(3)) if colors is None else colors[m]
        xlabel = 'Pixel' if xlabel is None else xlabel
        ylabel = 'Pixel' if ylabel is None else ylabel
        title = datetime.now().strftime("%Y%M%D") if title is None else title
        line, = plt.plot(x, ystack[m], label=label, color=colorse)
        line.set_antialiased(False)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc="best")

    plt.show()

    return fig1


def plot_save(ppointer, save_name, save_format='png', dpi=300):
    '''
    Just an easy wrapper.
    '''
    ppointer.savefig(save_name + f".{save_format}",
                     dpi=dpi, bbox_inches='tight', format=save_format)


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
