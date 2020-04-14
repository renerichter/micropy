import numpy as np
import NanoImagingPack as nip
import psutil
import sys
from ..stackProcess import loadStack, getIterationProperties
from ..filters import *

# %%
# -------------------------------------------------------------------------------------------------
#                                        IMAGE SHIFTING




# %%
# ----------------------------------------------------
#          MEAN/SUM/... Operations
# ----------------------------------------------------

def getMean(load_path, save_path, bg, ignore_bad=1, offset=0, file_limit=0, batch_size=100, channel=1, frame_interleave=0):
    '''
    Calculates the mean of a Set of images in a folder. If ignore_bad FLAG is set, images with to far deviance from median will not be taken into account. (E.g. images where light did not turn on)

    Param: 
        load_path:  load path
        save_path:  save path
        bg:         background image for offset-correction
        ignore_bad: ignores images with too big variances as compared to previous image
        file_limit: maximum number of images to be used for calculation
        batch_size: maximum size of batch per processing step
        channel:    channel to be used

    Return:
        final_mean: The resulting mean of the used batch
        rlf:        List of the used images

    Example:
        path_dict = {'load_path':'E:/Data/01_Fluidi/data/raw/2018-12-13/', 'experiment':'expt_003/','save_path': 'E:/Data/01_Fluidi/data/processed/2018-12-13/','name_addon': '_24fps_'}
        getMean(load_experiment=pd,file_limit=0,batch_size=100)
    '''
    file_list, file_total, batch_iterations, batch_last_iter = getIterationProperties(
        load_path, offset=offset, file_limit=file_limit, batch_size=batch_size, frame_interleave=frame_interleave)
    # print(file_list)
    # print(file_list)
    if file_limit:
        file_total = file_limit
    else:
        file_total = len(file_list)
    # get parameters from test-file
    myt1 = nip.timer(units='s')  # "(the) mighty ONE" ^^
    myadd = 1 if batch_last_iter > 0 else 0
    computation_follower1 = {}
    im1mf = []
    rlf = []
    # ----------------- run through batches ---------------------------
    print("My iteration_limit={0}".format(batch_iterations+myadd))
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
        # myoffsetc = myoffset+1 if myb > 0 else myoffset
        # print("getMean-->c_limit={0}".format(c_limit))
        # time.sleep(1)
        im1, read_list = loadStack(file_list=file_list,
                                   ids=myoffset, ide=c_limit, channel=channel)  # exp_path=load_path,prepent_path=True
        # print("im_stack1.shape={0}".format(im1.shape))
        # print("myoffset={0}, c_limit={1}".format(myoffsetc,c_limit))
        myt1.add('Loadad batch {}/{}'.format(myb, batch_iterations+myadd-1))
        #
        # ------------------ get mean ------------------------------------------
        im1mf, rlf = seqMean(im1=im1, im1mf=im1mf, rlf=rlf, myt1=myt1, ignore_bad=ignore_bad, myadd=myadd,
                             read_list=read_list, myb=myb, batch_iterations=batch_iterations)
        print('Loading batch {}/{} and calculating mean took: {}s.'.format(myb +
                                                                           1, batch_iterations+myadd, myt1.times[myb+1]))
        sys.stdout.flush()
    # ------- calculate total mean
    if not rlf == []:
        # weight mean of bigger subsets more
        im_nbr = np.array([len(x) for x in rlf])
        print("im_nbr={0}, sum(im_nbr)={1}".format(im_nbr, np.sum(im_nbr)))
        nshape = [im1mf.shape[0], ] + [1, ]*(im1mf.ndim-1)
        im_nbr = np.reshape(im_nbr / np.sum(im_nbr), nshape)
        print("after norm: sum(im_nbr)={0}".format(np.sum(im_nbr)))
        final_mean = np.mean(im1mf*im_nbr, axis=0) - bg
        final_mean[final_mean < 0] = 0  # inherent cut --> avoid?
        try:
            np.save(save_path+'im_mean.npy', final_mean)
            np.save(save_path+'im_mean--used_files.npy', rlf)
        except Exception as err:
            print(err)
        rlf = [list(np.array(x) + offset) for x in rlf]
        return final_mean, rlf, myt1
    else:
        errMsg = 'No mean calculated as no images could be successfully read-in.'
        if ignore_bad:
            errMsg += '\nTry again with setting ignore_bad=False.'
        raise ValueError(errMsg)


def seqMean(im1, im1mf, rlf, myt1, ignore_bad, myadd, read_list, myb, batch_iterations):
    '''
    Calculates the mean on a sequence
    No Error-Correction yet!
    '''
    im1m = np.mean(im1, axis=0)
    # im1med = np.median(im1,axis=0)
    myt1.add(
        'Calculated Mean for batch {0}/{1}'.format(myb, batch_iterations-1))
    if ignore_bad:
        print('Ignoring bad')
        im1m = np.mean(killVarDiff(
            im_stack=im1, im_mean=im1m, lim_low=0.1, lim_high=2))
        myt1.add(
            'Killed differing images for batch {0}/{1}'.format(myb, batch_iterations+myadd-1))
    if not len(read_list) == 0:
        if myb == 0:
            im1mf = im1m[np.newaxis]
            rlf = [read_list, ]
        else:
            im1mf = np.concatenate((im1mf, im1m[np.newaxis]), axis=0)
            rlf.append(read_list)
    return im1mf, rlf

# %%
# --------------------------------------
#                DATA-CLEANING
# --------------------------------------

def killVarDiff(im_stack, im_mean=[], lim_low=0.1, lim_high=2):
    '''
    Cancels out images whose variance differs too much from the mean, given a user-input threshhold.
    Caution: Could tend to cancel out the good images if the bad images are dominant! 
    Caution2: Implemented for 3d
    Idea to improve: Use rois for and region growing to distinguish and find machine limits OR use global thresholding like e.g. OTSU

    Param:
        im_stack: stack of images
        im_mean: mean of image along 0th dimension
        lim_low: lower variance limit
        lim_high: upper variance limit

    '''
    if im_mean == []:
        im_mean = np.mean(im_stack, axis=0)
    imh = im_stack/im_mean[np.newaxis]
    imv = np.var(np.reshape(
        imh, [imh.shape[0], np.prod(imh.shape[1:])]), axis=1)
    imvm = np.mean(imv)
    # was: np.equal(a2v > 2*a2vm, a2v < 0.1*a2vm) #TODO: correct? Delete if not
    im_sel_list = np.equal(imh > 2*imvm, imh < 0.1*imvm)
    return im_stack[im_sel_list]
