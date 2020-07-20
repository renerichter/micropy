from .inout import get_filelist, get_batch_numbers, loadStack, loadStackfast, dir_test_existance, add_logging, rename_files, fill_zeros, format_time
from .basicTools import getIterationProperties, get_range
from .utility import channel_getshift, image_getshift, add_multi_newaxis, create_value_on_dimpos, subtract_from_max
from .filters import stf_basic, diff_tenengrad
import numpy as np
import NanoImagingPack as nip
import cv2
import os
import PIL
import psutil
import json
from time import time
import logging
from datetime import datetime
from time import time, sleep
import matplotlib.pyplot as plt
from socket import gethostname
import yaml

# %%
# -------------------------------- VIDEO TOOLS -------------------------------


def convert_stk2vid(save_file, file_list=None, batch_size=None, mode=0, vid_param={}):
    '''
    This function converts a stack of images into a movie. On standard configuration stores frames into split-videos.

    :param:
    =======
    save_file: Filename (with path) of output
    save_format: chooses file-format(compressor) used -> 0: AVI(XVID), 1: MP4(X264)
    file_list: file-list of images to be loaded and put together
    batch_size: max-size of load batch (to enable to run on weaker systems)
    mode: which channels to use -> 0: red, 1: green, 2: blue, 3: rgb, 4: split and displayed in 1 frame with roi (a 3rd), 5: split storage (default)
    frame_size: resolution of the output-image
    frame_rate: Frames per second
    frame_rescale: 1: rescales the image to the output frame_size; 0: cuts out roi from Roi position
    roi: coordinates of ROI in (xstart,xend,ystart,yend) -> (default:) maximum ROI from center that fits frame_size

    :out:
    =====
    none

    '''
    # implement basic functionality for just storing video with certain properties
    if file_list == None:
        raise ValueError('No filelist given.')
    else:
        # get parameters
        [fl_len, fl_iter, fl_lastiter] = get_batch_numbers(
            filelist=file_list, batch_size=batch_size)
        # check sanity of data
        save_file, vid_param = sanitycheck_save2vid(save_file, vid_param)
        # start out-container
        out = save2vid_initcontainer(save_file[0] + save_file[1], vid_param)
        # iterate over batches
        for cla in range(fl_iter):
            # get right iteration values
            ids = cla * batch_size
            ide = ids+batch_size-1
            if cla == fl_iter:
                batch_size = fl_lastiter
            # load batch
            data_stack, data_stack_rf = loadStack(
                file_list=file_list, ids=ids, ide=ide, channel=3)
            # limit to bit_depth and reshape for writing out
            dss = data_stack.shape
            im = np.reshape(data_stack, newshape=(
                dss[:-3]+(dss[-2],)+(dss[-1],)+(dss[-3],)))
            data_stack = limit_bitdepth(
                im, vid_param['bitformat'], imin=None, imax=None, inorm=True, hascolor=True)
            save2vid(save_file, vid_param, out)
    out.release()
    print("Thanks for calling: Convert_stk2vid. ")
    return True


def save2vid(im, save_file=None, vid_param={}, out=False):
    '''
    Saves an image or image-stack to a given path. All necessary parameters are hidden in vid_param.
    Needs opencv3 (py3) and FFMPEG installed (has to be installed explicitely in windows).

    :param:
    =======
    :im:        image or image-stack
    :save_file: absolute path and filename (without filetype ending) as two-element list: [PATH,FILENAME]
    :vid_param: Dictionary containing all the necessary parameters. Check 'sanitycheck_vid_param'-function desrciption for further explanation.
    :out:       opencv-videoWriter-object -> can be used to not close a video and continue to append images, e.g. for stack processing

    Complete Example:
    =================
    vid_param= {'vformat':'H264','vcontainer':'mp4',
        'vaspectratio':[16,9],'vscale':'h','vfps':12}
    save_file = [
        'D:/Data/01_Fluidi/data/processed/Inkubator-Setup02-UC2_Inku_450nm/20190815/expt_017/','20190815-01']
    '''
    im, hasChannels, isstack = save2vid_assure_stack_shape(im)
    if type(out) == bool:
        save_path, vid_param = sanitycheck_save2vid(save_file, vid_param)
        vid = save2vid_initcontainer(save_path, vid_param)
    else:
        vid = out
    if not im.dtype == np.dtype(vid_param['bitformat']):
        im = limit_bitdepth(
            im, vid_param['bitformat'], imin=None, imax=None, inorm=False, hascolor=hasChannels)
    # save stack or image
    if isstack:
        imh = np.transpose(im, [0, 2, 3, 1]) if im.shape[-3] == 3 else im
        for m in range(im.shape[0]):
            vid.write(imh[m])
    else:
        if hasChannels:
            imh = np.transpose(im, [1, 2, 0])
        else:
            imh = np.repeat(im[:, :, np.newaxis], axis=-1, repeats=3)
        vid.write(imh)
    # close stack
    if type(out) == bool:
        if out == False:  # to save release
            save2vid_closecontainer(vid)
            vid = out
    # return
    return vid, vid_param, hasChannels, isstack


def limit_bitdepth(im, iformat='uint8', imin=None, imax=None, inorm=True, hascolor=False):
    '''
    Rescales the image into uint8. immin and immax can be explicitely given (e.g. in term of stackprocessing).
    Asssumes stack-shape [ndim,y,x] or [ndim,Y,X,COLOR] and takes imin and imax of whole stack if not given otherwise.
    :param:
    ======
    :vformat:   e.g. 'uint8' (...16,32,64), 'float16' (32,64), 'complex64' (128)   -> possible formats of np.dtype -> https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.dtypes.html

    Example:
    ========
    a = nip.readim('orka')
    a1 = limit_bitdepth(a,'uint32')
    <intern: im = nip.readim('orka');iformat='uint8';imin=None;imax=None;inorm=True;  >
    '''
    np.array
    iformatd = np.dtype(iformat)
    imdh = im.shape
    if im.ndim < 2:
        raise ValueError(
            "Image dimension too small. Only for 2D and higher implemented.")
    else:
        if inorm == True:
            # normalize to 0 only if minimum value of aimed datatype is zero (=avoid clipping)
            if np.iinfo(iformatd).min == 0:
                if imin == None:
                    if hascolor:
                        if im.ndim > 3:
                            imin = np.min(
                                im, axis=[-3, -2])[..., np.newaxis, np.newaxis, :]
                        else:
                            imin = np.min(
                                im, axis=[-3, -2])[np.newaxis, np.newaxis, :]
                    else:
                        imin = np.min(im)
                # if not imin.shape == ():
                #    imin = imin[...,np.newaxis,np.newaxis]
                im = im - imin
            if imax == None:
                if hascolor:
                    if im.ndim > 3:
                        imax = abs(
                            np.max(abs(im), axis=[-3, -2]))[..., np.newaxis, np.newaxis, :]
                    else:
                        imax = abs(
                            np.max(abs(im), axis=[-3, -2]))[..., np.newaxis, np.newaxis, :]
                # maximum could be a negative value (inner abs) AND do not want to have change of sign (outer abs)
                imax = abs(np.max(abs(im)))
            # if not imax.shape == ():
            #    imin = imin[...,np.newaxis,np.newaxis]
            im = im / imax
        else:  # just compress range to fit into int
            im_min = np.min(im)
            im_max = np.max(im)
            im_vr = im_max - im_min  # value range
            try:
                dtm = np.iinfo(iformatd).max
            except:
                dtm = np.finfo(iformatd).max
            if im_vr > dtm:
                im -= im_min
                im = im / np.max(im) * dtm
            else:
                if np.max(im) > dtm:  # assume that even though the range fits into uint8 the image somehow (due to processing) got shifted above max-value of value-range that the original format had -> hence: shifting back to max instead of resetting by min-offset
                    im -= (np.max(im) - dtm)
                elif np.min(im) < 0:  # same as above, just for min
                    im -= np.min(im)
        # if np.issubdtype(iformatd, np.integer):
        #    im = im*np.iinfo(iformatd).max
        im = np.array(im, dtype=iformatd)
    return im


def save2vid_assure_stack_shape(im):
    '''
    Tests the stack for the appropriate format and converts if necessary for conversion.
    Input-Image needs to have X,Y-dimensions at last position, hence for 3D-stack with 2 channels e.g. [stackdim,channeldim,Y,X].

    :param:
    =======
    :im:        numpy or nip-array (image)
    '''
    isstack = False
    hasChannels = False
    # convert stack to correct color_space
    if im.ndim < 2:
        raise ValueError("Dimension of input-image is too small.")
    if im.ndim == 2:  # only 2D-image
        pass
    else:  # nD-image
        hasChannels = True
        if im.shape[-3] > 3 and im.ndim > 3:  # -3.dim is not channel, but stack-dimension
            im = np.reshape(im, newshape=[np.prod(
                im.shape[:-2], im.shape[-2], im.shape[-1])])
            isstack = True
        else:  # -3.dim is channel dimension
            if im.shape[-3] == 1:
                im = np.repeat(im, repeats=3, axis=-3)
            elif im.shape[-3] == 2:
                ims = im.shape
                ims[-3] = 1
                im_zeros = np.zeros(ims)
                im = np.concatenate([im, im_zeros], axis=-3)
            else:  # no problem
                pass
    # return image and whether it is a stack
    return im, hasChannels, isstack
    # if channel < 3:
    #        frame = cv2.cvtColor(
    #            np.array(a_pil, 'uint8'), cv2.COLOR_GRAY2BGR)
    #    else:
    #        frame = cv2.cvtColor(
    #           np.array(a_pil, 'uint8'), cv2.COLOR_RGB2BGR)


def rearrange_array(im, idx=None, axis=-3, assureIndices=True):
    '''
    TODO: NOT FINISHED!
    Rearranges an axis according to an index-list. Could lead to dropouts etc as well. Only positive index numbers allowed.
    '''
    # sanity for axis
    if abs(axis) > im.ndim:
        axis = im.ndim
    # sanity for idx
    if not type(idx) == list:
        if not idx == None:
            idx = [idx, ]
        else:  # nothing to be done here
            pass
    else:
        if idx == []:
            pass
    # sanity assuring indices are non-neg and smaller max-val
    if assureIndices:
        idx = abs(np.array(idx))
        idxm = im.shape[axis]
        idx = np.delete(idx, idx[idx >= idxm])
    idx_forward, idx_backward = array_switchpos_idx(
        listdim=im.ndim, pstart=axis, pend=0)
    im.transpose(idx_forward)
    # TODO: WHAT HERE?
    im.transpose(idx_backward)
    return im


def array_switchpos_idx(listdim=3, pstart=-2, pend=0):
    '''
    # TODO: NOT FINISHED!
    Calculates the indeces neccessary to switch two elements of a list (e.g. dimensions of an image)
    '''
    axis = pend
    idx1 = list(range(listdim))
    idx2 = list(range(listdim))
    idx_forward = [idx1.pop(axis), ] + idx1
    idx2.pop(0)
    idf_backward = idx2[:axis] + [0, ] + idx2[axis:]
    return idx_forward, idf_backward


def save2vid_initcontainer(save_name, vid_param={}):
    '''
    Creates an OpenCV-VideoWriter-object with the given parameters.
    Note: OpenCV switches height and width on creation/storage. Hence, formats have to be switched. (Input is: Height x Width).
    '''

    out = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc(
        *vid_param['vformat']), vid_param['vfps'], (vid_param['vpixels'][1], vid_param['vpixels'][0]))
    return out


def save2vid_closecontainer(out):
    '''
    Safely closes the container.
    '''
    sleep(0.05)
    out.release()
    sleep(0.5)


def sanitycheck_save2vid(save_file, vid_param):
    '''
    Tests the necessary entries in vid_param and adds things if missing. First values of param mark default values.

    :param:
    =======
    :vformat:STRING:            'XVID','H264','H265'            -> tested video formats
    :vcontainer:STRING:         'avi','mp4','mkv'               -> tested containers
    :vaspectratio: INT_ARRAY:   [16,9],[22,9], [4,3], [1,1]     -> tested ratios, but all should work
    :vpixels:INT_ARRAY:         [1920,1080]                     -> final resolution of the output video
    :vscale:STRING:             None,'h','w'                    -> rescale final video size using vaspectratio and the height or width of the input image(stack)
    :vfps:INT:                  14                              -> frames per second
    '''
    # Set standard-configuration
    from sys import platform as sysplatform

    # if sysplatform == 'linux': #somehow it is not working on FA8_TITANX_UBU system... -.-'
    #    std_conf = {'vformat': 'H264', 'vcontainer': 'mp4', 'vaspectratio':[16, 9], 'vscale': None, 'vfps': 12,'vpixels':[1920, 1080],'bitformat': 'uint8'} #X264+mp4 does not work on windows
    # else:
    std_conf = {'vformat': 'XVID', 'vcontainer': 'avi', 'vaspectratio': [16, 9], 'vscale': None, 'vfps': 12, 'vpixels': [
        1920, 1080], 'bitformat': 'uint8'}  # X264+mp4 does not work on windows
    # check param dict
    if not type(vid_param) == dict:
        vid_param = {}
    for m in std_conf:
        if m not in vid_param:
            vid_param[m] = std_conf[m]
    # check save_file path etc
    if save_file == None:
        save_file = [os.getcwd(), 'save2vid.' + vid_param['vcontainer']]
    else:
        save_file = save_file + '.' + vid_param['vcontainer']
    # sanity check that path exists and acessible
    save_path = check_path(save_file)
    # return
    return save_file, vid_param


def check_path(file_path):
    '''
    Checks whether file_path exists and if not, creates it.
    '''
    if type(file_path) == str:
        if file_path[-1] not in ["/", "\\"]:
            import re
            regs = re.compile(r'([\w:]+)*/((\w+)*/)*')
            regss = regs.search(file_path)
            if regss == None:
                regs = re.compile(r'([\w:]+)*\\((\w+)*\\)*')
                regss = regs.search(file_path)
            if regss == None:
                raise ValueError(
                    'Could not match directory path in file_path.')
            else:
                file_path = regss.group()
        if not os.path.isdir(file_path):
            os.makedirs(file_path)
            print('File path' + file_path + ' freshly created')
    else:
        raise ValueError("File_Path not of string-type.")
    return file_path


def convert_time2str(nbr=1,nbr2Time=[0,0,1,0,0]):
    '''
    Converts a given timepoint (in ms) to strangi
    '''
    tdict = ['ms','s','m','h','d']
    fdict = [1000,60,60,24,100000]
    timestr = ""
    start_found = False

    # skip empty tails and convert into 
    tt = [nbr*m for m in nbr2Time]
    for m in range(len(tt)):
        if nbr2Time[m] > 0 and not start_found:
            start_found = True
        if start_found:
            if m == 4:
                assert(tt[m] < fdict[m],"Unbelieveable, you made    measurements longer than 273years?!")
            a = tt[m] // fdict[m]
            if a > 0:
                tt[m+1] += a
                tt[m] = np.mod(tt[m],fdict[m])
            timestr = "{:02}{}".format(tt[m],tdict[m]) + timestr
        else:
            pass
    
    # done?
    return timestr

def label_image(im,nbr=0,nbr2Time=[0,0,1,0,0],pixelsize=None,font_size=8):
    '''
    Atomic version of deprecated vid_addContent_and_format.
    Assumes 2D-grey images for now.

    PARAMS:
    =======
    :im:        (IMAGE) File to work on
    :nbr:       (INT) number of image
    :nbr2Time:  (ARRAY) List of distance between equally spaced time-events -> units are: ['ms','s','m','h','d'] -> on default: [0,0,1,0,0] means 1 minute per image
    :pixelsize: (FLOAT) size of a pixel assumed in µm

    OUTPUT:
    =======
    :im:    labeled_image
    '''
    # parameters
    scalebar_ratio = 5
    scalebar_size = 0.1

    # create image -> assumed 2D-grey-images for now
    a_pil = PIL.Image.fromarray(im.astype('uint8'), 'L')
    a_fill = 255
    draw = PIL.ImageDraw.Draw(a_pil)
    font = PIL.ImageFont.truetype(font='complex_.ttf', size=font_size)

    # get and draw time
    timestr = convert_time2str(nbr=nbr,nbr2Time=nbr2Time)
    draw.text((0, a_pil.height-font_size), timestr, fill=a_fill, font=font)

    # get scalebar properties and draw
    if pixelsize is not None:
        # scalebar
        scalebar_height = scalebar_height//scalebar_ratio
        scalebar_width = int(im.shape[-1]*scalebar_size)
        scalebar_pos = [a_pil.height - 2*scalebar_height,a_pil.width - int(1.5*scalebar_width)]
        draw.rectangle((scalebar_pos[0],  scalebar_pos[1] , scalebar_height, scalebar_width), fill=a_fill)

        # text
        
        draw.text((vid_prop[1]-font_size//2*5, 0),
                    text_channel, fill=a_fill, font=font)
        


def vid_addContent_and_format(im1=[], im_sel=[], path_dict={}, out='', image_timer=[], channel=1, vid_prop=[], font_size=12, text_channel=1):
    '''
    HAS TO BE CHANGED! EXTRACTED VIDEO-WRITING FUNCTION.

    Adds Time and Experiment-date + number to each frame. Finally writes it to a cv2-file

    Param:
        im1:        image stack to be written
        im_sel:     subset out of image-stack
        path_dict:  dictionary containing path and naming content
        out:
        image_timer:containing days,hours,minutas
        channel:    selected channel

    out:
        image_timer: although inline changed!


    '''
    for myc in range(len(im1)):
        if (image_timer[1] >= 23) and (image_timer[2] >= 60):
            image_timer[0] += 1
            image_timer[1] = 0
            image_timer[2] = 0
        elif (image_timer[2] >= 60):
            image_timer[1] += 1
            image_timer[2] = 0

        if im_sel[myc]:
            # print("myc = {0}, im1_sl={1} ".format(myc,im1_sl[myc]))
            if channel < 3:
                a_pil = PIL.Image.fromarray(
                    im1[myc, :, :].astype('uint8'), 'L')
                a_fill = 255
            else:
                a_pil = PIL.Image.fromarray(
                    im1[myc, :, :, :].astype('uint8'), 'RGB')
                a_fill = (255, 255, 255)
            draw = PIL.ImageDraw.Draw(a_pil)
            # font = ImageFont.truetype(<font-file>, <font-size>)
            font = PIL.ImageFont.truetype(font='complex_.ttf', size=font_size)
            draw.text((0, a_pil.height-font_size), 't='+str(image_timer[0])+'d'+str(image_timer[1])+'h'+str(
                image_timer[2])+'min', fill=a_fill, font=font)  # if RGB: fill=(255,255,255)
            draw.text((vid_prop[1]-font_size//2*19, a_pil.height-font_size), path_dict['load_path'][-11:-1].replace(
                '-', '') + ' e' + path_dict['experiment'][5:-1], fill=a_fill, font=font)  # if RGB: fill=(255,255,255)
            # draw.text((x, y),"Sample Text",(r,g,b))
            draw.text((vid_prop[1]-font_size//2*5, 0),
                      text_channel, fill=a_fill, font=font)

        image_timer[2] += 1
    return image_timer


def makeVid(path_dict, bg=[], im_mean=[], batch_size=100, file_limit=0, vid_channel='RGB', vid_format='XVID', vid_prop=[1080, 1920], vid_roi=True, vid_fps=14, vid_process=False, vid_binning=0, add_data=False,  vid_scale=None, vid_ratio=[16, 9]):
    '''
    Saving input Data as video.
    Optionally processing input-data.

    :param:
    ~~~~~~~~~~~~~~~~~~~
    :vid_channel:   Channel to be used for video creation -> 'R', 'G', 'B', 'RGB'


    :outputs:
    ~~~~~~~~~~~~~~~~~~~

    '''
    #
    # ----------------- get path and filenames --------------------------
    load_experiment = path_dict['load_path'] + path_dict['experiment']
    save_experiment = path_dict['save_path'] + \
        path_dict['experiment'] + 'cleaned/'
    #
    # ----------------- define video text and get channel--------------------------
    text_channel = ('p' if vid_process else 'raw') + vid_channel
    vidc = {'R': 0, 'G': 1, 'B': 2, 'RGB': 3}[vid_channel]
    save_name = save_experiment + \
        path_dict['experiment'][:-1] + path_dict['name_addon'] + text_channel
    save_video_name = save_name + {'XVID': '.avi', 'H264': '.mp4'}[vid_format]
    #
    # ----------------- get file-list and batch size --------------------------
    file_list, file_total, batch_iterations, batch_last_iter = getIterationProperties(
        load_experiment=load_experiment, offset=0, file_limit=file_limit, batch_size=batch_size)
    print('Total amount of files to be used: {}.'.format(file_total))
    #
    # ----------------- prepare counters and sizes ------------------------
    # test open to get sizes
    im_help = np.squeeze(nip.readim(load_experiment + file_list[0]))
    if not vid_roi:
        vid_prop = im_help.shape[0:2]
    if add_data:
        font_size = vid_prop[0]//12  # was 9
    if vid_binning:
        pass
    if vid_scale == 'short':
        pass
    range_l = get_range(im_help.shape, vid_prop)
    myt1 = nip.timer(units='s')
    myadd = 1 if batch_last_iter > 0 else 0
    computation_follower1 = {}
    image_timer = [0, 0, 0]  # [days,hours,minutes] to write into video
    #
    # ----------------- if processing activated --------------------------
    # if not existent
    if vid_process:
            # squeeze to correct for wrong dimensionalities due to changes NIP-toolbox
        im_mean = np.squeeze(im_mean)
        bg = np.squeeze(bg)
        if len(im_mean.shape) > 2:
            im_mean = im_mean[:, range_l[0]:range_l[1], range_l[2]:range_l[3]]
            im_meang = im_mean[vidc, :, :] if vidc < 3 else im_mean
        else:
            im_meang = im_mean[range_l[0]:range_l[1], range_l[2]:range_l[3]]
        if len(bg.shape) > 2:
            bg = bg[:, range_l[0]:range_l[1], range_l[2]:range_l[3]]
            bg = bg[vidc, :, :] if vidc < 3 else bg
        else:
            bg = bg[range_l[0]:range_l[1], range_l[2]:range_l[3]]
    #
    # ----------------- open movie-obj --------------------------------
    # (size_a_x_to_keep_video_dimensions*3+2*h_blackline.shape[1],im_help.shape[1])
    out = cv2.VideoWriter(save_video_name, cv2.VideoWriter_fourcc(
        *vid_format), vid_fps, (vid_prop[1], vid_prop[0]))
    #
    # ----------------- run through batches ---------------------------
    if file_limit:
        file_total = file_limit
    else:
        file_total = len(file_list)
    # get parameters from test-file
    myt1 = nip.timer(units='s')  # "(the) mighty ONE" ^^
    myadd = 1 if batch_last_iter > 0 else 0
    computation_follower1 = {}
    # ----------------- run through batches ---------------------------
    print("My iteration_limit={}".format(batch_iterations+myadd))
    for myb in range(0, batch_iterations+myadd):
        computation_follower1[str(myb)] = {}
        computation_follower1[str(myb)]['mem'] = dict(
            psutil.virtual_memory()._asdict())
        computation_follower1[str(myb)]['cpu'] = [psutil.cpu_percent(), '%']
        #
        # ----------------------- set limits ----------------------------------
        myoffset = myb * batch_size
        if myb == batch_iterations:
            c_limit = myoffset+batch_last_iter - 1
        else:
            c_limit = myoffset+batch_size - 1
        #
        # ------------------ read in data --------------------------------------
        im1, read_list = loadStack(
            file_list=file_list, channel=vidc, ids=myoffset, ide=c_limit)
        # format to correct size
        if vid_binning:
            print("Put binning in here! Fourier-based.")
        im1 = im1[:, range_l[0]:range_l[1], range_l[2]:range_l[3]]
        myt1.add('Loaded batch {0}/{1} -> Shape: {2}'.format(myb,
                                                             batch_iterations+myadd-1, im1.shape))
        #
        # ----------------------- process further if selected ---------------------
        # shift mean if necessary
        # xy_limit=0.05 # 1/20 pixel-preciseness -> tooo precise?
        # im1,shift_list = getShiftNMove(im_mean,im1,xy_limit)
        if vid_process:
            im1 = (im1-bg[np.newaxis])/im_meang[np.newaxis]
            im1 -= np.min(im1)
            im1 *= 255/np.max(im1)
            # [im1,im1min,im1max,im_sel] = processStack(im1,myb,im1min,im1max)
            # for testing
            im1h = np.array(im1)
            batch_iterations+myadd
            print("Normalization distribution of processed stack: {} from batch {} / {}".format(
                [np.min(im1h), np.mean(im1h), np.median(im1h), np.max(im1h)], myb+1, batch_iterations+myadd))
        im_sel = [1, ]*len(im1)
        #
        # -------------------------------------- add text to image and write to file ------------------------------
        if add_data:
            image_timer = addTextNWrite(im1=im1, im_sel=im_sel, path_dict=path_dict, out=out, image_timer=image_timer,
                                        channel=vidc, vid_prop=vid_prop, font_size=font_size, text_channel=text_channel)
        myt1.add()
        time_needed = np.round(myt1.times[myb+1]-myt1.times[myb], 2)
        print("Iteration: {0} of {1} done in {2}s!".format(
            myb+1, batch_iterations+1, time_needed))
        computation_follower1[str(myb)]['batch_size'] = [
            c_limit-myoffset, 'images']
        computation_follower1[str(myb)]['time'] = [time_needed, 's']
        # save to file
    sleep(0.5)  # let system settle for closing file
    out.release()
    sleep(0.5)  # let system settle for closing file
    computation_follower1['total_time'] = [myt1.times[myb+1], 's']
    with open(save_name + '_cb.txt', 'w') as file:
        file.write(json.dumps(computation_follower1))


# %% ------------------------------------------------------
# ---          Legacy Processing Functions              ---
# ---------------------------------------------------------
#
def processStack(im1, myb, im1min, im1max):
    if len(im1.shape) < 4:
        # variance select
        im1v = np.var(np.reshape(
            im1, [im1.shape[0], np.prod(im1.shape[1:3])]), axis=1)
        im1vm = np.mean(im1v, axis=0)
        im1_sl = np.equal(im1v > 2 * im1vm, im1v < 0.5*im1vm)
        if myb == 0:
            im1min = np.min(im1[im1_sl, :, :])
            im1max = np.max(im1[im1_sl, :, :])
    else:
        im1v = np.var(np.reshape(
            im1, [im1.shape[0], np.prod(im1.shape[1:3]), im1.shape[3]]), axis=1)
        im1vm = np.mean(im1v, axis=0)
        im1_sl = np.equal(im1v > 2 * im1vm, im1v < 0.5*im1vm)
        im1_sla = np.logical_and(im1_sl[:, 0], im1_sl[:, 1], im1_sl[:, 2])
        # im1mean = np.mean(im1[im1_sla,:,:,:],axis=0) #np.reshape(im1,[np.prod(im1.shape[0:3]),im1.shape[3]])
        # im1s = np.reshape(im1[im1_sla,:,:,:],[np.prod(im1[im1_sla,:,:,:].shape[0:3]),im1[im1_sla,:,:,:].shape[3]])
        if myb == 0:
            im1min = np.min(im1[im1_sl, :, :, :])
            im1max = np.max(im1[im1_sl, :, :, :])
    im1 = im1 - im1min
    im1 = im1 / im1max
    im1u = np.uint8(np.round(im1 * 255))
    im1u[im1u < 0] = 0
    im1u[im1u > 255] = 255
    # raise gamma
    gamma = 0.9
    im1a = np.uint8(np.round(((im1u**gamma)/np.max(im1u**gamma))*255))
    return [im1a, im1min, im1max, im1_sl]


def addTextNWrite(im1, im_sel, path_dict=None, out=False, image_timer=None, channel=3, vid_prop=None, font_size=12, text_channel='G'):
    '''
    Not implemented ....where did I loose thi?
    '''
    print("NOT IMPLEMENTED")
    return True


# %% ------------------------------------------------------
# --- Final UC2-Inkubator Processing Functions          ---
# ---------------------------------------------------------
#

def uc2_eval_shifts(res, kl=[], cla=0, ids=0, ide=0, channel=1, threshhold=[], logger=None):
    '''
    Evaluate non-excluded shifts from the analysed ROIs. Rule: if majority of ROIs is shifted, accept shift.
    Only for 1 channel!
    For now: only shift for the least amount of changes:

    :param:
    ======
    res:        dictionary containing the shift-data from various rois
    kl:         list of files that will be excluded
    ids:        start-index
    ide:        ending-index
    channel:    to be used

    :out:
    =====
    shift_list:     shifts
    sellist:        index-list of images to be shifted
    '''
    # old
    '''
    myshifts = np.squeeze(np.array(res['roi_shifts']))[:, :, channel, :]
    mysa = np.abs(np.copy(myshifts))
    shift_list = []
    sellist = []
    # eval-shift for all ROIs
    collx = []
    colly = []
    colll = []
    for roi in mysa:
        resx = np.squeeze(np.array(np.where(roi[:, 0] >= 0.9)))
        resy = np.squeeze(np.array(np.where(roi[:, 1] >= 0.9)))
        collx.append(resx)
        colly.append(resy)
        colll.append([resx.size, resy.size])
    colll = np.array(colll)
    selx, sely = [np.argmin(colll[:, 0]), np.argmin(colll[:, 1])]
    if not (colll[selx, 0] + colll[sely, 1] == 0):
        if colll[selx, 0] < colll[sely, 1]:
            sellist = colly[sely]
            selidx = sely
        else:
            sellist = collx[selx]
            selidx = selx
        sellist = [m for m in sellist if not m in kl]
        shift_list = [myshifts[selidx, m]  for m in sellist]
    '''
    # format shift_list to have [analysedROI,image-nbr,calculated x-y-shifts]
    myshifts = np.transpose(np.squeeze(
        np.array(res['roi_shifts']))[cla], [1, 0, 2])

    # propare helpful arrays -> for reconstruction: shift_list will just contain zeros where there was no shifts applied
    shift_list = np.zeros((ide-ids, 2))
    sel_shift_list = []
    x_shifts_accepted = []
    y_shifts_accepted = []
    size_of_accepted_lists = []
    if threshhold == []:
        threshhold = [0.9, 0.9]

    # only accept shifts that are bigger than threshhold and store accepted shifts into list
    for roi in myshifts:
        x_shifts_greater_thresh = np.squeeze(
            np.array(np.where(np.abs(roi[:, 0]) >= threshhold[0])))
        y_shifts_greater_thresh = np.squeeze(
            np.array(np.where(np.abs(roi[:, 1]) >= threshhold[1])))
        x_shifts_accepted.append(x_shifts_greater_thresh)
        y_shifts_accepted.append(y_shifts_greater_thresh)
        size_of_accepted_lists.append(
            [x_shifts_greater_thresh.size, y_shifts_greater_thresh.size])
    size_of_accepted_lists = np.array(size_of_accepted_lists)

    # select ROI with the least amount of shifts for x and y individually
    selx, sely = [np.argmin(size_of_accepted_lists[:, 0]),
                  np.argmin(size_of_accepted_lists[:, 1])]

    # if shifts have to be applied -> if length of shifts of accepted_x ROI is shorter than accepted_y -> work on longer list, hence y -> and vice-versa
    if not (size_of_accepted_lists[selx, 0] + size_of_accepted_lists[sely, 1] == 0):
        if size_of_accepted_lists[selx, 0] < size_of_accepted_lists[sely, 1]:
            sel_shift_list = y_shifts_accepted[sely]
            sel_ROI = sely
        else:
            sel_shift_list = x_shifts_accepted[selx]
            sel_ROI = selx
        try:
            sel_shift_listn = []
            for m in range(ide-ids):
                if not m in kl:
                    if m in sel_shift_list:
                        sel_shift_listn.append(m)
                        shift_list[m] = myshifts[sel_ROI, m]
            sel_shift_list = sel_shift_listn
        except Exception as e:
            if not logger == None:
                logger.warning(e)
                logger.warning(
                    'To continue proper processing, empty lists where generated.')
            sel_shift_list = []
            shift_list = []
    return shift_list, sel_shift_list


def uc2_preprocessing_refuseImages(data_stack, res, ids=0, ide=0, channel=1, criteria='mean'):
    '''
    Using a given criteria to refuse empty/insufficient images

    :param:
    =======
    data_stack: image stack in
    res:        super-functions dictionary to store processing info
    criteria:   Only 'mean' yet

    :out:
    =====
    data_stack: manipulated data_stack
    res:        manipulated dictionary
    kl:         index-list of excluded elements
    '''
    im_dat = np.array([np.array(res['image_min'])[ids:ide, channel], np.array(
        res['image_mean'])[ids:ide, channel], np.array(res['image_max'])[ids:ide, channel]])
    if criteria == 'mean':
        # use mean of active stack to have a more stable measure
        tm = np.mean(im_dat[1])
        kl = np.array(np.where(im_dat[1] < 0.15*tm))
        if kl.size:
            for m in kl:
                m = int(m)
                res['image_skipped_filename'].append(
                    res['image_filename_list'][ids+m])
                res['image_skipped_index'].append(ids + m)
                data_stack = np.delete(data_stack, obj=m, axis=0)
            # del
    return data_stack, res, kl


def uc2_preprocessing_metrics(data_stack, mean_ref=[], pred=[], ROIs=None, res=None, ids=0, ide=0, cla=0, name_metrics=None, name_stacks=None, prec=10):
    '''
    Calculates a set of predefined metrics to populate the dictionary for further analysis

    :param:
    ======
    ids:    Start index
    ide:    End-index
    cla:    counter for global (superior/calling functions) iteration
    '''

    # make sure mean_ref is non-empty
    if len(mean_ref) == 0:
        raise ValueError(
            'Reference for calculation of relative shifts not provided.')

    # generate ROIs for analysis -> assumes stack to be 4D ([series,color,y,x]) and ROI to be [ymin,ymax,xmin,xmax]
    data_stack_ROIs = []
    for m in ROIs:
        dat = data_stack[:, :, m[0]:m[1], m[2]:m[3]]
        data_stack_ROIs.append(dat)

    # 1st -> interchannel shifts; 2nd -> interimage changes
    if cla == 0:
        res['interchannel_shift_offset'] = channel_getshift(
            np.array(nip.DampEdge(data_stack[0], rwidth=0.1)))
        res['interchannel_shift_offset_roi'] = [channel_getshift(
            np.array(nip.DampEdge(dat, rwidth=0.1))) for dat in data_stack_ROIs]
    elif cla > 0:
        res['image_shifts'] = res['image_shifts'] + \
            list(image_getshift(
                data_stack[0], mean_ref, prec=prec)[np.newaxis])
        res['image_diff_var'] = res['image_diff_var'] + \
            list(np.var(data_stack[0]-pred, axis=(-2, -1))[np.newaxis])
    if ide-ids > 0:
        roi_shifts = []
        roi_diff_var = []
        for m in range(len(ROIs)):
            # nip.v5(nip.cat((data_stack_ROIs[m][1][1], mean_ref[1, ROIs[m][0]: ROIs[m][1], ROIs[m][2]: ROIs[m][3]])))
            roi_shifts.append(image_getshift(
                data_stack_ROIs[m], mean_ref[:, ROIs[m][0]: ROIs[m][1], ROIs[m][2]: ROIs[m][3]], prec=prec))
        res['roi_shifts'] = res['roi_shifts'] + roi_shifts
        res['image_shifts'] = res['image_shifts'] + \
            list(image_getshift(
                data_stack[1:], data_stack[:-1], prec=prec))
        # res['roi_diff_var'] = res['roi_diff_var'] + list(np.var(data_stack_roi[1:] - data_stack_roi[:-1], axis=(-2, -1)))
        res['image_diff_var'] = res['image_diff_var'] + \
            list(np.var(data_stack[1:]-data_stack[:-1], axis=(-2, -1)))
    datal = [data_stack]  # , data_stack_roi
    namel = [name_stacks[0]]  # , name_roi
    for m in range(len(datal)):
        ima = datal[m]
        res_stf_basics = stf_basic(ima)
        for n in range(4):
            res[namel[m] + name_metrics[n]] = res[namel[m] +
                                                  name_metrics[n]] + list(res_stf_basics[n])
        res[namel[m] + name_metrics[4]] = res[namel[m] +
                                              name_metrics[4]] + list(diff_tenengrad(ima))
    # store for next round
    pred = data_stack[-1].copy()
    # pred_roi = data_stack_roi[-1].copy()

    return res, pred


def get_mean_from_stack(load_path, save_path='', load_fn_proto='jpg', mean_range=[], channel=None, batch_size=50, binning=None, colorful=1, inverse_intensity=False, save_every=False, clear_path=False):
    '''
    Calcutes the mean from a range of readin-images from a folder. Idea: Create a reference, e.g. to do shift analysis, but have a cleaner stack with better SNR and less influenced by noise.

    :param:
    load_path:  path to be used
    mean_range: Array denoting start and end of filelist (hence files) to be used -> e.g.: [75,110]
    channel:    Color-channel to be used

    :out:
    ref_mean
    '''
    # variable
    stack_meanh = []

    # read in
    os.chdir(load_path)
    fl = get_filelist(load_path=load_path, fn_proto=load_fn_proto)
    if not mean_range == []:
        fl = fl[mean_range[0]:mean_range[1]]
    [fl_len, fl_iter, fl_lastiter] = get_batch_numbers(
        filelist=fl, batch_size=batch_size)

    # start iteration
    for cla in range(fl_iter):
        tstart = time()

        # get right iteration values
        ids = cla * batch_size
        ide = (ids + batch_size) if cla < (fl_iter -
                                           1) else (ids + fl_lastiter)
        if cla == fl_iter-1:
            batch_size = fl_lastiter

        # read in
        data_stack, _ = loadStackfast(fl[ids:ide], colorful=colorful)

        # inverse
        if inverse_intensity:
            data_stack = subtract_from_max(data_stack)

        # if not data_stack.shape
        if not channel == None:
            data_stack = data_stack[:, channel, :, :]
        if binning:
            data_stack = np.array(nip.resample(data_stack, create_value_on_dimpos(
                data_stack.ndim, axes=[-2, -1], scaling=[1.0/binning[0], 1.0/binning[1]]))) / (binning[0] * binning[1])

        # calculate mean
        stack_meanh1 = np.mean(data_stack, axis=0)
        stack_meanh.append(stack_meanh1)

        # save every single mean if selected
        if save_every:
            if not stack_meanh1.dtype == np.dtype('float16'):
                stack_meanh1 = nip.image(limit_bitdepth(nip.image(
                    stack_meanh1), 'float16', imin=None, imax=None, inorm=False, hascolor=True))
            save_name = 'mean_cla_' + fill_zeros(cla, fl_iter)
            nip.imsave(stack_meanh1, save_path + save_name,
                       form='tif', BitDepth='auto')

    # calculate global mean
    stack_meanh = np.squeeze(np.array(stack_meanh))
    if fl_iter > 1:
        stack_mean = np.mean(stack_meanh, axis=0)
    else:
        stack_mean = stack_meanh
    stack_mean = nip.image(stack_mean)

    # save if wanted
    if not stack_mean.dtype == np.dtype('float16'):
        stack_mean = nip.image(limit_bitdepth(
            nip.image(stack_mean), 'float16', imin=None, imax=None, inorm=False, hascolor=True))
    if not save_path == '':
        if not mean_range == []:
            save_name = 'meanref_f{}-{}'.format(mean_range[0], mean_range[1]) if (
                mean_range[1]-mean_range[0] <= fl_len) else 'stack_mean'
        else:
            save_name = 'stack_mean'
        nip.imsave(stack_mean, save_path + save_name,
                   form='tif', BitDepth='auto')

    # clear load_path if necessary
    if clear_path:
        delete_files_in_path(load_path)

    return stack_mean


def uc2_preprocessing(load_path, save_path, binning=[2, 2], batch_size=50, preview=False, interleave=100, ROIs=[], channel=1, mean_ref=False, load_fn_proto='jpg', vid_param={}, proc_range=None, prec=10, inverse_intensity=False, delete_means=True, threshhold=[]):
    '''
    UC2-preprocessing.
    if preview=True:
        > Takes every n-th image from a stack
        > selects 1 channel
        > BINS
        > stores results (Video + selected-channel) in pre-view folder
    else:

    '''
    # flags
    check_filenames = True

    # supporting definitions
    chan_dict = {0: 'red', 1: 'green', 2: 'blue'}
    name_metrics = ['max', 'min', 'mean', 'median', 'tenengrad']

    # parameters
    tbegin = time()
    tnow = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    criteria = 'mean'

    # paths
    dir_test_existance(save_path)
    name_fullim = 'image_'
    name_roi = 'roi_'
    mean_path_pre = save_path + 'mean' + os.sep
    if 'vname' in vid_param:
        save_vid = save_path + vid_param['vname'] + \
            '-ORIG--BIN-{}x_{}y--{}fps--AllChan'.format(
                binning[0], binning[1], vid_param['vfps'])
        save_meas = save_vid + '-res_calculation.yaml'
    else:
        save_vid = save_path + 'BIN_'
        save_meas = save_path + 'res.yaml'
    logpref = 'preview-' if preview else 'preproc-'
    logger_filepath = save_path + logpref + 'log-' + tnow + '.uc2log'
    os.chdir(load_path)

    # set-Logger and add system parameters
    if not 'logger' in locals():
        logger_root, logger = add_logging(
            logger_filepath, start_logger='PREVIEWprocessor')
    logger.debug('Processing running on DEVICE = {}'.format(
        gethostname()))
    logger.debug(
        'Logger created and started. Saving to: {}'.format(logger_filepath))

    #  creating result containers
    if preview == True:
        res = {'image_number': [], 'image_filepath': [], 'data_load_path': load_path,
               'data_save_path': save_path, 'image_filename_list': [], 'date_eval_start': '', 'date_eval_end': '', 'save_vid': ''}
    else:
        res = {'interchannel_offsets': [], 'image_shifts': [], 'image_number': [], 'image_filepath': [], 'data_load_path': load_path,
               'data_save_path': save_path, 'image_skipped_filename': [], 'image_skipped_index': [], 'res_stf_basics': [], 't_times': [], 't_comments': [], 'image_filename_list': [], 'roi_shifts': [], 'roi_diff_var': [], 'image_shifts': [], 'image_diff_var': [], 'shift_list': [], 'date_eval_start': '', 'date_eval_end': '', 'save_vid': ''}
        for n in [name_fullim, name_roi]:
            for m in name_metrics:
                res[n + m] = []
    res['date_eval_start'] = tnow

    # Filenames and Filelist
    if check_filenames:
        tdelta, brename = rename_files(load_path, version=2)
        if brename:
            logger.debug('Renaming of files took {}s.'.format(tdelta))
        else:
            logger.debug('No renaming necessary.')
    logger.debug('Get Filelist for load_path={}. Results will be stored in save_path={}.'.format(
        load_path, save_path))
    fl = get_filelist(load_path=load_path, fn_proto=load_fn_proto)
    if preview:
        fl = fl[::interleave]
    elif delete_means:
        delete_files_in_path(mean_path_pre)
    if not proc_range == None:
        fl = fl[proc_range[0]:proc_range[1]]
    [fl_len, fl_iter, fl_lastiter] = get_batch_numbers(
        filelist=fl, batch_size=batch_size)  # fl[2500:2510]

    # do iterations over all files
    for cla in range(fl_iter):
        tstart = time()
        if cla > 0:
            tform = format_time(np.round(tend * (fl_iter-cla-1), 0))
            logger.debug(
                'Estimated time needed until processing is completed: {}'.format(tform))

        # get right iteration values
        ids = cla * batch_size
        ide = (ids + batch_size) if cla < (fl_iter -
                                           1) else (ids + fl_lastiter)
        if cla == fl_iter-1:
            batch_size = fl_lastiter
        # Fast read data-stack  -> format = [time,color,y,x]
        logger.debug('Load stack {} of {}.'.format(cla+1, fl_iter))
        data_stack, data_stack_rf = loadStackfast(
            file_list=fl[ids:ide], logger=logger.warning)
        data_stack_rf = [m+ids for m in data_stack_rf]
        #if preview == True:
            #logger.debug('Only work on {} channel'.format(chan_dict[channel]))
            #data_stack = data_stack[:, channel, :, :]
        res['image_number'] = res['image_number'] + data_stack_rf
        res['image_filename_list'] = res['image_filename_list'] + [fl[m]
                                                                   for m in data_stack_rf]

        # work on inverse as transmission image taken
        if inverse_intensity:
            data_stack = subtract_from_max(data_stack)

        # subtract min to take "readout offset" into account
        logger.debug('Subtract Minima.')
        data_stack = data_stack - np.min(data_stack, keepdims=True)

        # Do binning and set video-pixel-param
        if binning:
            logger.debug('Bin image-stack.')
            data_stack_dtype = data_stack.dtype
            data_stack = np.array(nip.resample(data_stack, create_value_on_dimpos(
                data_stack.ndim, axes=[-2, -1], scaling=[1.0/binning[0], 1.0/binning[1]]))) / (binning[0] * binning[1])
        if cla == 0:
            vid_param['vpixels'] = list(data_stack.shape[-2:])

        if preview:
            kl = []
            sellist = []
        else:
            # Calculate chosen metrics --> all channels
            logger.debug('Calculate Metrics for Stack and ROI.')
            if cla == 0:
                pred = []
            res, pred = uc2_preprocessing_metrics(
                data_stack, mean_ref=mean_ref, ROIs=ROIs, pred=pred, res=res, ids=ids, ide=ide, cla=cla, name_metrics=name_metrics, name_stacks=[name_fullim, name_roi])

            # get rid of empty-images --> only 1 channel
            logger.debug(
                'Refuse images by using criteria {}.'.format(criteria))
            data_stack, res, kl = uc2_preprocessing_refuseImages(
                data_stack, res, ids=ids, ide=ide, channel=channel, criteria=criteria)
            mean_path = mean_path_pre + \
                'mean_cla_' + fill_zeros(cla, fl_iter)

            # get shifts (previously calculated in uc2_proprocessing_metrics)
            shift_list, sellist = uc2_eval_shifts(
                res, kl=kl, cla=cla, ids=ids, ide=ide, channel=channel, threshhold=threshhold, logger=logger)
            res['shift_list'] = res['shift_list'] + \
                list(shift_list[np.newaxis])

        # store video
        logger.debug('Write images into video and single TIFF-files.')
        if not data_stack.dtype == np.dtype('uint8'):
            data_stack = nip.image(limit_bitdepth(
                data_stack, 'uint8', imin=None, imax=None, inorm=False, hascolor=True))
        for udi in range(len(data_stack)):
            out = True if (cla == 0 and udi == 0) else out
            out, vid_param, hasChannels, isstack = save2vid(
                data_stack[udi], save_file=save_vid, vid_param=vid_param, out=out)

        # shift and store 1 channel of stack
        data_stack = np.squeeze(data_stack[:, channel])
        nbr_shifts = []
        for udi in range(batch_size):
            if udi not in kl:
                if udi in sellist:
                    # shift-values can be used directly -> negative values mean "image has to be shifted back  towards bigger pixel-pos-values"
                    try:
                        data_stack[udi] = nip.shift2Dby(
                            data_stack[udi], shift_list[udi])
                        nbr_shifts.append(m)
                    except Exception as e:
                        logger.warning(e)
                        logger.warning(
                            'Skipped step {}of{}, because shift_list has no such entry. Shift_list='.format(udi, batch_size))
                        logger.warning(shift_list)
                try:
                    if not data_stack.dtype == np.dtype('uint8'):
                        data_stack[udi] = nip.image(limit_bitdepth(
                            data_stack[udi], 'uint8', imin=None, imax=None, inorm=False, hascolor=True))
                    nip.imsave(data_stack[udi], save_path + 'images' + os.sep +
                               res['image_filename_list'][ids+udi][:-4], form='tif', BitDepth='auto')
                except Exception as e:
                    logger.warning(
                        'Tried to access out of bound index {} of data_stack. Still continued safely.'.format(udi))
        logger.debug('Applied {} shifts.'.format(len(nbr_shifts)))
        # calculate mean of active & shifted stack and store --> only 1 channel
        mean_format = 'tif'
        if not preview:
            logger.debug(
                'Calculate mean of active stack and store into {}.'.format(mean_path + mean_format))
            ds_mean = np.array(np.mean(data_stack, axis=0), dtype=np.float16)
            nip.imsave(nip.image(ds_mean), mean_path,
                    form=mean_format, BitDepth='auto')
        tend = time() - tstart
        del data_stack
        logger.debug('Iteration {} took: {}s.'.format(
            cla+1, np.round(tend, 2)))
        logger.debug(
            '--- > Succesfully written stack {} of {}'.format(cla+1, fl_iter))
    save2vid_closecontainer(out)
    res['date_eval_end'] = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    res['save_vid'] = save_vid
    with open(save_meas, 'w') as file:
        yaml.dump(res, file)
    logger.debug('<<<< FINISHED AFTER {} >>>>'.format(
        format_time(np.round(time() - tbegin))))
    return res


def uc2_processing(load_path, save_path, res_old=None, batch_size=50, stack_mean=None, vid_param=None, load_fn_proto='tif', channel=None, colorful=0, inverse_intensity=False, correction_method='mean', draw_frameProperties=False):
    '''
    Final clean-up of data
    '''
    # parameters
    tbegin = time()
    tnow = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")

    # paths
    save_vid = save_path + vid_param['vname'] + \
        '-processed-{}fps'.format(vid_param['vfps'])
    logger_filepath = save_path + 'processing-log-' + tnow + '.uc2log'
    os.chdir(load_path)
    save_path_images = save_path + 'images-proc' + os.sep
    save_meas = save_path_images + 'res-proc.yaml'
    check_path(save_path_images)

    # set-Logger and add system parameters
    if not 'logger' in locals():
        logger_root, logger = add_logging(
            logger_filepath, start_logger='LASTprocessor')
    logger.debug('Processing running on DEVICE = {}'.format(
        gethostname()))
    logger.debug(
        'Logger created and started. Saving to: {}'.format(logger_filepath))

    #  creating result containers
    res = {'data_load_path': load_path,
           'date_eval_start': '', 'date_eval_end': '', 'image_number': [], 'image_filename_list': []}
    res['date_eval_start'] = tnow
    logger.debug('Get Filelist for load_path={}. Results will be stored in save_path={}.'.format(
        load_path, save_path))

    logger.debug('Get Filelist for load_path={}. Results will be stored in save_path={}.'.format(
        load_path, save_path))
    fl = get_filelist(load_path=load_path, fn_proto=load_fn_proto)
    [fl_len, fl_iter, fl_lastiter] = get_batch_numbers(
        filelist=fl, batch_size=batch_size)

    # Parameters
    global_max = np.max(np.array(res_old['image_max'])[:, 1])
    norm_factor1 = 255/global_max
    if len(res_old['shift_list']) == 0:
        shifts_max = [0, 0]
    else:
        if not type(res_old['shift_list']) == np.ndarray:
            res_help = np.array(res_old['shift_list'][:-1])
            res_helpr = res_old['shift_list'][-1]
            shifts_maxh = np.max(np.abs(res_help),axis=(0,1))
            shifts_maxhr = np.max(np.abs(res_helpr),axis=0)
            shifts_max = np.array(np.round(np.max(np.vstack((shifts_maxh,shifts_maxhr)),axis=0)),dtype=np.uint16)
        else: 
            shifts_max = np.array(np.round(np.max(np.abs(res_old['shift_list']), axis=((0, 1)))), dtype=np.uint16)
        if (shifts_max == [0, 0]).all():
            shifts_max = [4, 4]

    # do iteration
    for cla in range(fl_iter):
        tstart = time()
        if cla > 0:
            tform = format_time(np.round(tend * (fl_iter-cla), 0))
            logger.debug(
                'Estimated time needed until processing is completed: {}'.format(tform))

        # get right iteration values
        ids = cla * batch_size
        ide = (ids + batch_size) if cla < (fl_iter -
                                           1) else (ids + fl_lastiter)
        if cla == fl_iter-1:
            batch_size = fl_lastiter
        # Fast read data-stack  -> format = [time,color,y,x]
        logger.debug('Load stack {} of {}.'.format(cla+1, fl_iter))
        data_stack, data_stack_rf = loadStackfast(
            file_list=fl[ids:ide], logger=logger.warning, colorful=0)
        data_stack_rf = [m+ids for m in data_stack_rf]
        res['image_number'] = res['image_number'] + data_stack_rf
        res['image_filename_list'] = res['image_filename_list'] + [fl[m]
                                                                   for m in data_stack_rf]

        # small error in divison/correction, but avoids division by zero problems
        if stack_mean.min() < 1:
            delta = stack_mean.min()
            stack_mean += (1-delta)

        # correct by mean and normalize to stack-full range
        if correction_method == 'max':
            corr_factor = data_stack.max(axis=(-2, -1))
            data_stack = (data_stack / stack_mean[np.newaxis])
            data_stack = data_stack * dsm[:, np.newaxis, np.newaxis]/np.max(
                data_stack, axis=(-2, -1))[:, np.newaxis, np.newaxis]  # *norm_factor1
        elif correction_method == 'mean':
            data_stack = (data_stack / stack_mean[np.newaxis])
            # corr_factor = np.iinfo(np.uint8).max / data_stack.mean()
            corr_factor = 100
            data_stack = data_stack / data_stack.mean(
                (-2, -1))[:, np.newaxis, np.newaxis] * corr_factor
            data_stack[data_stack > np.iinfo(
                np.uint8).max] = np.iinfo(np.uint8).max
            if inverse_intensity:
                data_stack = np.iinfo(np.uint8).max - data_stack
        else:
            pass
        # crop image
        data_stack = data_stack[:, shifts_max[0]:data_stack.shape[1] -
                                shifts_max[0], shifts_max[1]:data_stack.shape[2]-shifts_max[1]]

        # set video dimensions
        if cla == 0:
            vid_param['vpixels'] = list(data_stack.shape[-2:])

        # save images
        data_stack = nip.image(data_stack)
        logger.debug('Store images in TIFF-files.')
        if not data_stack.dtype == np.dtype('uint8'):  # float16
            data_stack = nip.image(limit_bitdepth(
                data_stack, 'uint8', imin=None, imax=None, inorm=False, hascolor=True))
        for udi in range(len(data_stack)):
            nip.imsave(data_stack[udi], save_path_images +
                       res['image_filename_list'][ids+udi][:-4], form='tif', BitDepth='auto')
        # store video
        logger.debug('Create Video')
        if not data_stack.dtype == np.dtype('uint8'):
            data_stack = limit_bitdepth(
                data_stack, 'uint8', imin=None, imax=None, inorm=False, hascolor=True)
        data_stack = nip.image(
            np.repeat(data_stack[:, np.newaxis, :, :], axis=1, repeats=3))
        for udi in range(len(data_stack)):
            out = True if (cla == 0 and udi == 0) else out
            if draw_frameProperties:
                data_stack[udi] = label_image(im=data_stack[udi],nbr=ids+udi,nbr2Time=[0,0,1,0,0],pixelsize=None)
            out, vid_param, hasChannels, isstack = save2vid(
                data_stack[udi], save_file=save_vid, vid_param=vid_param, out=out)
        tend = time() - tstart
        del data_stack
        logger.debug('Iteration {} took: {}s.'.format(
            cla+1, np.round(tend, 2)))
        logger.debug(
            '--- > Succesfully written stack {} of {}'.format(cla+1, fl_iter))
    save2vid_closecontainer(out)
    res['date_eval_end'] = datetime.strftime(datetime.now(), "%Y%m%d_%H%M%S")
    res['save_vid'] = save_vid
    with open(save_meas, 'w') as file:
        yaml.dump(res, file)
    logger.debug('<<<< FINISHED AFTER {} >>>>'.format(
        format_time(np.round(time() - tbegin))))
    return res
