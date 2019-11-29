from imports import *

# %%
# -------------------------------------------------------------------------------------------------
#                                        IMAGE SHIFTING


def getShiftNMove(im_base, im_shift, xy_limit=0.05, shift_list=[]):
    '''
    gets shift between a base and a stack. Moves base if difference to stack is bigger than xy_limit.

    param:
        im_base:    base image that will be used for comparison
        im_shift:   image that will be shifted
        xy_limit:  min_shift threshhold -> if shift is below this limit, nothing will be done to frame
        shift_list: adds to a shift_list or creates empty
    out:
        im_moved:
        shift_list: list of shifts -> 0 if lower xy_limit

    '''

    shift_calc = np.array(get_shift(im_base, im_shift, method='centroid', center=im_shift.mid()[
                          1:], edge_length=[100]*(im_shift.ndim-1)))
    if any(shift_calc > xy_limit):
        shift_image = nip.shiftby(im_shift, shift_calc)
    return shift_image


def get_shift(im1, im2, method='maximum',center = [0,0], edge_length = [100,100],store_correl=False):
    from NanoImagingPack import correl, extract, centroid;
    from NanoImagingPack.util import max_coord;
    from NanoImagingPack.mask import create_circle_mask;
    if (np.mod(edge_length[0],2) == 1): edge_length[0] = edge_length[0] + 1;
    if (np.mod(edge_length[1],2) == 1): edge_length[1] = edge_length[1] + 1;
    
    # max_rad = 2
    correls = []
    correl_max = []   
    width = 40
    
    if method == 'maximum':
        cc = correl(extract(im1, ROIsize = tuple(edge_length),centerpos = tuple(center)),extract(im2, ROIsize = tuple(edge_length),centerpos = tuple(center)));
        if store_correl:
            correls.append(cc);        
        correl_max.append((max_coord(cc)[0],max_coord(cc)[1]));
        # return((max_coord(cc[0])[0]-edge_length[0]//2,max_coord(cc[0])[1]-edge_length[1]//2))
        
        # shift of image 2 with respect to image 1 at the selected coordinate and around edge_length
        return((max_coord(cc)[0]-edge_length[0]//2,max_coord(cc)[1]-edge_length[1]//2))   # new code 30.11.17
    elif method == 'centroid': 
        max_rad = 2
        mr_old = max_rad;
        def __get_cc__(edge_length, center):
            if (np.mod(edge_length[0],2) == 1): edge_length[0] = edge_length[0] + 1;
            if (np.mod(edge_length[1],2) == 1): edge_length[1] = edge_length[1] + 1;
            cc = correl(extract(im1, ROIsize = tuple(edge_length),centerpos = tuple(center)),extract(im2, ROIsize = tuple(edge_length),centerpos = tuple(center)));
            # mc = max_coord(cc[0]);
            mc = max_coord(cc);
            try: 
                max_rad in locals()
            except:
                max_rad = 2
            max_rad = min([max_rad, abs(edge_length[0]-mc[0]), mc[0], mc[1], abs(edge_length[1]-mc[1]) ]);
            return(mc, cc, edge_length)
        #
        mc, cc, edge_length = __get_cc__(edge_length, center);
        if max_rad == 0:
            print('Maximum of cross correlation at boarder -> Increasing cc_edge_length by 10');
            max_rad = mr_old;                
            edge_length[0] += 10;
            edge_length[1] += 10;
            mc, cc, max_rad, edge_length = __get_cc__(edge_length, max_rad, center);
        if store_correl:
            correls.append(cc);  
        centr_coord = centroid((cc-np.min(cc))*create_circle_mask(mysize =edge_length,maskpos = mc ,radius=max_rad, zero = 'image')); # in create_circle_mask (module: mask.py) -> change xx(...mode=...) to xx(...placement=...) !!
        return((edge_length[0]//2-centr_coord[0],edge_length[1]//2-centr_coord[1]))
    
    elif method == 'fit_gauss':
        from NanoImagingPack.fitting import fit_gauss2D;
        cc = correl(extract(im1, ROIsize = tuple(edge_length),centerpos = tuple(center)),extract(im2, ROIsize = tuple(edge_length),centerpos = tuple(center)));
        if store_correl:
            correls.append(cc);        
        clip = extract(cc, ROIsize = (width,width),centerpos=max_coord(cc));   # clip more or less symetrical around maximum of correlation
        f= fit_gauss2D(clip, False);               # changed l.10 in fitting.py -> from scipy import optimize as opt; l.42 -> p, success = opt.leastsq(deviation, guess)
        pos = (f[0][1]+max_coord(cc)[0]-width//2, f[0][2]+max_coord(cc)[1]-width//2); # position of new maximum in non-clip correlation 
        correl_max.append(pos);
        return((pos[0]-edge_length[0]//2,pos[1]-edge_length[1]//2))

# %%
# -------------------------------------------------------------------------------------------------
#                                        MEAN/SUM/... Operations


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
        im1, read_list = loadStack(exp_path=load_path, file_list=file_list,
                                   ids=myoffset, ide=c_limit, channel=channel, prepent_path=True)
        # print("im_stack1.shape={0}".format(im1.shape))
        # print("myoffset={0}, c_limit={1}".format(myoffsetc,c_limit))
        myt1.add('Loadad batch {}/{}'.format(myb, batch_iterations+myadd-1))
        #
        # ------------------ get mean ------------------------------------------
        im1mf, rlf = seqMean(im1=im1, im1mf=im1mf, rlf=rlf, myt1=myt1, ignore_bad=ignore_bad,
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


def seqMean(im1, im1mf, rlf, myt1, ignore_bad, read_list, myb, batch_iterations):
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
# -------------------------------------------------------------------------------------------------
#                                        DATA-CLEANING


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
        im1h, [imh.shape[0], np.prod(imh.shape[1:])]), axis=1)
    imvm = np.mean(imv)
    im_sel_list = np.equal(a2v > 2*a2vm, a2v < 0.1*a2vm)
    return im_stack[im_sel_list]
