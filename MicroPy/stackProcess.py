from .inout import get_filelist, get_batch_numbers, loadStack
from .basicTools import getIterationProperties, get_range
from .utility import channel_getshift
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
from time import time
import matplotlib.pyplot as plt

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
        #check sanity of data
        save_file, vid_param = sanitycheck_save2vid(save_file, vid_param)
        # start out-container
        out = save2vid_initcontainer(save_file[0] + save_file[1], vid_param)
        #iterate over batches
        for cla in range(fl_iter):
            # get right iteration values
            ids = cla * batch_size
            ide = ids+batch_size-1
            if cla == fl_iter:
                batch_size = fl_lastiter
            #load batch 
            data_stack, data_stack_rf = loadStack(file_list=file_list, ids=ids, ide=ide, channel=3)
            #limit to bit_depth and reshape for writing out
            dss = data_stack.shape
            im = np.reshape(data_stack,newshape=(dss[:-3]+(dss[-2],)+(dss[-1],)+(dss[-3],)))
            data_stack= limit_bitdepth(im,vid_param['bitformat'],imin=None,imax=None,inorm=True,hascolor=True)
            save2vid(save_file,vid_param,out)
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
        im = limit_bitdepth(im,vid_param['bitformat'],imin=None,imax=None,inorm=False,hascolor=hasChannels)
    # save stack or image
    if isstack:
        imh = np.transpose(im,[0,2,3,1]) if im.shape[-3] == 3 else im
        for m in range(im.shape[0]):
            vid.write(imh[m])
    else:
        imh = np.transpose(im,[1,2,0]) if im.shape[-3] == 3 else im
        vid.write(imh)
    # close stack
    if type(out) == bool:
        if out == False:  # to save release
            save2vid_closecontainer(vid)
            vid = out
    # return
    return vid, vid_param, hasChannels, isstack

def limit_bitdepth(im,iformat='uint8',imin=None,imax=None,inorm=True,hascolor=False):
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
    iformatd = np.dtype(iformat)
    imdh = im.shape
    if im.ndim <2:
        raise ValueError("Image dimension too small. Only for 2D and higher implemented.")
    else:
        if inorm== True:
            # normalize to 0 only if minimum value of aimed datatype is zero (=avoid clipping)
            if np.iinfo(iformatd).min == 0:
                if imin == None: 
                    if hascolor:
                        if im.ndim>3:
                            imin = np.min(im,axis=[-3,-2])[...,np.newaxis,np.newaxis,:]
                        else: 
                            imin = np.min(im,axis=[-3,-2])[np.newaxis,np.newaxis,:]
                    else: 
                        imin = np.min(im)
                #if not imin.shape == ():
                #    imin = imin[...,np.newaxis,np.newaxis]
                im = im - imin
            if imax == None: 
                if hascolor: 
                    if im.ndim>3:
                        imax = abs(np.max(abs(im),axis=[-3,-2]))[...,np.newaxis,np.newaxis,:]
                    else: 
                        imax = abs(np.max(abs(im),axis=[-3,-2]))[...,np.newaxis,np.newaxis,:]
                imax = abs(np.max(abs(im))) #maximum could be a negative value (inner abs) AND do not want to have change of sign (outer abs)
            #if not imax.shape == ():
            #    imin = imin[...,np.newaxis,np.newaxis]
            im = im / imax
        else: #just compress range to fit into int
            im_vr = np.max(im) - np.min(im) # value range
            if im_vr > np.iinfo(iformatd).max:
                im_vr -= np.min(im_vr)
                im_vr = im_vr /np.max(im_vr) * np.iinfo(iformatd).max
            else: 
                if np.max(im) > np.iinfo(iformatd).max: # assume that even though the range fits into uint8 the image somehow (due to processing) got shifted above max-value of value-range that the original format had -> hence: shifting back to max instead of resetting by min-offset
                    im -= (np.max(im) - np.iinfo(iformatd).max)
                elif np.min(im) < 0: # same as above, just for min
                    im -= np.min(im)
        #if np.issubdtype(iformatd, np.integer):
        #    im = im*np.iinfo(iformatd).max
        im = np.array(im,dtype=iformatd)        
    return im

def save2vid_assure_stack_shape(im):
    '''
    Tests the stack for the appropriate format and converts if necessary for conversion.
    Input-Image needs to have X,Y-dimensions at last position, hence for 3D-stack with 2 channels e.g. [stackdim,channeldim,Y,X].

    :param:
    =======
    :im:        numpy or nip-array (image)
    '''
    isstack=False
    hasChannels=False
    # convert stack to correct color_space
    if im.ndim < 2:
        raise ValueError("Dimension of input-image is too small.")
    if im.ndim == 2:  # only 2D-image
        pass
    else:  # nD-image
        hasChannels=True
        if im.shape[-3] > 3 and im.ndim>3:  # -3.dim is not channel, but stack-dimension
            im = np.reshape(im,newshape=[np.prod(im.shape[:-2],im.shape[-2],im.shape[-1])])
            isstack = True
        else:  # -3.dim is channel dimension
            if im.shape[-3] == 1:
                im = np.repeat(im, repeats=3, axis=-3)
            elif im.shape[-3] == 2:
                ims = im.shape
                ims[-3] = 1
                im_zeros = np.zeros(ims)
                im = np.concatenate([im, im_zeros], axis=-3)
            else: # no problem
                pass  
    #return image and whether it is a stack
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
    time.sleep(0.05)
    out.release()
    time.sleep(0.5)

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
    std_conf = {'vformat': 'XVID', 'vcontainer': 'avi', 'vaspectratio':[16, 9], 'vscale': None, 'vfps': 12,'vpixels':[1920, 1080],'bitformat': 'uint8'} #X264+mp4 does not work on windows
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
        if file_path[-1] not in ["/","\\"]:
            import re
            regs = re.compile(r'([\w:]+)*/((\w+)*/)*')
            regss = regs.search(file_path)
            if regss == None: 
                regs = re.compile(r'([\w:]+)*\\((\w+)*\\)*')
                regss = regs.search(file_path)
            if regss == None: 
                raise ValueError('Could not match directory path in file_path.')
            else: 
                file_path = regss.group()
        if not os.path.isdir(file_path):
            os.mkdir(file_path)
            print('File path' + file_path + ' freshly created')
    else:
        raise ValueError("File_Path not of string-type.")
    return file_path


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
    save_video_name = save_name + {'XVID': '.avi','H264': '.mp4'}[vid_format]
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
        im1, read_list = loadStack(file_list=file_list,channel=vidc, ids=myoffset, ide=c_limit)
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
    time.sleep(0.5)  # let system settle for closing file
    out.release()
    time.sleep(0.5)  # let system settle for closing file
    computation_follower1['total_time'] = [myt1.times[myb+1], 's']
    with open(save_name + '_cb.txt', 'w') as file:
        file.write(json.dumps(computation_follower1))


# %%
# --------------------------------------------------------------------------------------------------
#                                   CLEAN STACK

def processStack(im1,myb,im1min,im1max):
    if len(im1.shape) < 4:
        # variance select
        im1v = np.var(np.reshape(im1,[im1.shape[0],np.prod(im1.shape[1:3])]),axis=1)
        im1vm = np.mean(im1v,axis=0)
        im1_sl = np.equal(im1v > 2* im1vm,im1v < 0.5*im1vm)
        if myb ==0: 
            im1min = np.min(im1[im1_sl,:,:])
            im1max = np.max(im1[im1_sl,:,:])
    else: 
        im1v = np.var(np.reshape(im1,[im1.shape[0],np.prod(im1.shape[1:3]),im1.shape[3]]),axis=1)
        im1vm = np.mean(im1v,axis=0)
        im1_sl = np.equal(im1v > 2* im1vm,im1v < 0.5*im1vm)
        im1_sla = np.logical_and(im1_sl[:,0],im1_sl[:,1],im1_sl[:,2])
        # im1mean = np.mean(im1[im1_sla,:,:,:],axis=0) #np.reshape(im1,[np.prod(im1.shape[0:3]),im1.shape[3]])
        # im1s = np.reshape(im1[im1_sla,:,:,:],[np.prod(im1[im1_sla,:,:,:].shape[0:3]),im1[im1_sla,:,:,:].shape[3]])
        if myb == 0:
            im1min = np.min(im1[im1_sl,:,:,:])
            im1max = np.max(im1[im1_sl,:,:,:])            
    im1 = im1 - im1min
    im1 = im1 / im1max    
    im1u = np.uint8( np.round(im1 * 255) )
    im1u[im1u<0] = 0
    im1u[im1u>255]=255
    # raise gamma    
    gamma = 0.9
    im1a = np.uint8(np.round(((im1u**gamma)/np.max(im1u**gamma))*255))
    return [im1a,im1min,im1max,im1_sl]
    
def addTextNWrite(im1, im_sel, path_dict=None, out=False,image_timer=None,channel=3, vid_prop=None, font_size=12, text_channel='G'):
    '''
    Not implemented ....where did I loose thi?
    '''
    print("NOT IMPLEMENTED")
    return True

# %% Plot to Graphs
def convert2graph(res, save_fpath):
    '''
    Directly fitting to the shape 
    '''
    print("Here the functions will be converted to proper graphs")