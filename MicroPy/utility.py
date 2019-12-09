# %% imports
import numpy as np
import NanoImagingPack as nip

# %%
# -------------------------------------------------------------------------
# NOISE-Functions
# -------------------------------------------------------------------------
#
def noise_stack(im,mode='Poisson',param=[1,10,100,1000]):
    '''
    Generates a stack from given image along the (new) 0th dimension. 

    Example: 
    im = nip.readim();
    mode='Poisson';param=[1,10,100,1000];
    noise_stack(im,mode=mode,param=param)

    mode='Gaussian';param=[[0],[10,20,50,100,200]];
    mode='PoissonANDGauss';param=[[0],[10,10,40,40],[1,10,1,100]];

    '''
    # prepare modes
    if mode=='Poisson':
        noisy = lambda param: noise_poisson(im,param,norm='mean')
        length = len(param)
    elif mode == 'Gaussian':
        noisy = lambda param: noise_gaussian(im,mu=param[0],sigma=param[1])
        mu,sigma = gauss_testparam(param)
        param = list(zip(mu,sigma))
        length = len(sigma)
    elif mode == 'PoissonANDGauss':
        noisy = lambda param: noise_gaussian(noise_poisson(im,param[2],norm='mean'),mu=param[0],sigma=param[1])
        mu,sigma = gauss_testparam(param[:2])
        phot = param[2]
        param = list(zip(mu,sigma,phot))
        length=len(param)
    else: 
        raise ValueError("Mode not implemented yet.")
    # run iteration
    for m in range(length):
        resim = noisy(param[m])[np.newaxis]
        if m== 0: 
            res = resim
        else: 
            res = nip.cat((res,resim),axis=0)
    #output
    return res

def noise_poisson(im, phot=None,norm='mean'):
    '''
    Calculates the Poisson-noise. If parameter phot is given, Image is mean-normalized to phot.
    '''
    if phot:
        im = noise_normalize(im, phot, norm='mean')
    # each pixel means λ and hence the mean value
    return nip.image(np.random.poisson(im))


def noise_gaussian(im, mu=None, sigma=2):
    '''
    Adds gaussian (additive) noise.
    '''
    if mu == None:  # normalize noise to 10th of mean of image
        mu = 0.1 * np.mean(im,axis=(-2,-1))
    im = im + np.random.normal(mu, sigma, im.shape)
    return im


def noise_poisson_gauss(im, phot, mu, sigma):
    '''
    Applies Poisson noise and then adds Gaussian noise (interpreted as read-noise) to it.
    '''
    im = noise_poisson(im, phot=phot)
    im = im + noise_gaussian(im, mu=mu, sigma=2)
    return im


def noise_normalize(im, phot, norm='mean'):
    '''
    Normalizes the image so that proper noise can be applied.
    '''
    if norm == 'mean':
        norm = np.mean(im,axis=(-2,-1))/ phot
    elif norm == 'max':
        norm = np.max(im,axis=(-2,-1))/ phot
    elif norm == 'min':
        norm = np.min(im,axis=(-2,-1))/ phot
    else:
        raise AssertionError("So what? Didn't expect such a indecisive move.")
    # assure that dimensions are correct
    if im.ndim>2:
        norm = norm[...,np.newaxis,np.newaxis]
    im = im/norm
    return im


# %% TO SORT!

def process_image(im1, im2, operations=[]):
    '''
    Operations on channels between two images
    '''
    res = []
    if operations == []:
        print('No operation had to be done, so did nothing')
    for m in range(im1.shape[0]):
        for y in range(operations.shape[0]):
            res[y, m] = operations[y](im1[m], im2[m])
    return res


def channel_getshift(im):
    '''
    Operations on 1 image within channels
    '''
    shift = []
    myshift = 0
    id1, id2 = shift_indexlist(im)
    for m in range(len(id1)):
        myshift, _, _, _ = findshift(im[id1[m]], im[id2[m]], 100)
        shift.append(myshift)
    return shift


def image_getshift(im1, im2):
    shift = []
    myshift = 0
    for m in range(im1.shape[0]):
        myshift, _, _, _ = findshift(im1[m], im2[m], 100)
        shift.append(myshift)
    return shift


def shift_indexlist(im):
    '''
    Creates a two list that complete cover all permutations that are possible within the channel dimensionality -> n*(n-1)/2 possibilities

    just an idea -> not finished
    bl=list(range(im.shape[0]))
    id1 = []
    id2 = []
    while len(bl) > 0:
        for c in range(bl)
            id1.append()
            id2.append()
    id2=id1
    id2=id2.append(id2.pop(0))
    '''
    # now just fast for 3-channels
    id1 = [0, 0, 1]
    id2 = [1, 2, 2]
    # id1=[0,0,0,1,1,2]
    # id2=[1,2,3,2,3,3]
    return id1, id2


def image_sharpness(im, im_filters=['Tenengrad']):
    '''
    Calculating the image sharpness with different filters. Only for 2D inputs! For all neighboring pixel-using techniques the outer-most pixel-borders of the image are not used.

    :param:
    =======
    :filters:LIST: List of possible filters. Options are:

    :out:
    =====
    res:LIST: List of all sharpness values calculated.
    '''
    #
    from numpy import mean
    if 'Tenengrad' in im_filters:
        res.append(tenengrad(im))
    elif 'VollathF4' in im_filters:
        res.append(vollathF4(im))


def findshift(im1, im2, prec=100):
    '''
    Just a wrapper for the Skimage function using sub-pixel shifts, but with nice info-printout.
    link: https://scikit-image.org/docs/dev/auto_examples/transform/plot_register_translation.html
    :param:
    =======
    :im1: reference image
    :im2: shifted image
    :prec: upsample-factor = number of digits used for sub-pix-precision (for sub-sampling)

    :out:
    =====
    :shift: calculated shift-vector
    :error: translation invariant normalized RMS error between images
    :diffphase: global phase between images -> should be 0
    :tend: time needed for processing
    '''
    from time import time
    from skimage.feature import register_translation
    tstart = time()
    # 'real' marks that input-data still has to be fft-ed
    shift, error, diffphase = register_translation(
        im1, im2, prec, space='real', return_error=True)
    tend = np.round(time() - tstart, 2)
    print("Found shifts={} with upsampling={}, error={} and diffphase={} in {}s.".format(
        shift, prec, np.round(error, 4), diffphase, tend))
    return shift, error, diffphase, tend




def image_binning(im, bin_size=2, mode='real_sum', normalize='old'):
    '''
    Does image binning. Only implemented for even-sized input-images. If input image-size is not a multiple of the output-image size, then image will be cut to the next even image-size per dimension.

    FOR FUTURE: allow binning in all directions by concatenating a set of 1D binning functions. Easiest: get param "binning_directions=[0,3,8]", sort range(ndim)-array (im_in) to bring bin-dimensions to front (imh)  -> bin from first to last and always sort binned direction at the end of the sort-range of imh -> at the end: have the same dim-distribution as in the beginning with imh -> sort-back to im -> output

    :param:
    ======
    :im:IMAGE:      Input image
    :bin_size:INT:  size of a bin
    :mode:STRING:   Variant to be used, e.g.: 'real','conv','fourier'
    '''
    # for now:
    if im.ndim > 2:
        id1 = list(np.arange(im.ndim))
        id2 = id1[-2:] + id1[:-2]
        id3 = id1[2:] + id1[:2]
        imh = np.transpose(im, id2)
    else: 
        imh = im
    res = nip.image(np.zeros(imh[::bin_size, ::bin_size].shape))
    # only do 2d binning
    if mode == 'real_sum':
        offset = list(np.arange(bin_size))
        for m in range(bin_size):
            for n in range(bin_size):
                res += imh[offset[m]:imh.shape[0]-bin_size+1+offset[m]:bin_size,
                           offset[n]:imh.shape[1]-bin_size+1+offset[n]:bin_size]
            # not finished
    elif mode == 'real_pix': #only selects the pixel-value at the binning-position
        res = imh[::bin_size,::bin_size]
    elif mode == 'real_median': # takes median of regions -> reshape to split, e.g. 800x800 ->(bin2)-> 400x400x2x2 -> median along dim-2&3
        pass
    elif mode == 'conv':
        # NOT FINISHED
        ims = np.array(im.shape)
        imh = nip.extract(im,)
        nip.convolve()
        # EXTRACT for pauding (to avoid wrap-around)
        # create kernel with ones
        # extract result
        pass
    elif mode == 'fourier':
        # use function of nip-toolbox -> does not introduce aliasing even though pixel-reduction as cutting (extraction) is done in fourier-space = "just cutting away high frequencies"
        nip.resample(imh,factors=[1.0/bin_size,1.0/bin_size]+[1,]*(imh.ndim-2)])
    else:
        raise ValueError('Mode not existing.')

    if normalize == 'old':  # normalize to old stack-maximum -> rather: per-slice?
        res = res/np.max(res) * np.max(im)
    elif normalize == 'mean':
        res/=bin_size
    elif normalize == 'none': # leave res as is
        pass
    else:
        raise ValueError('Normalization not existing.')

    if im.ndim > 2:
        res = np.transpose(res, id3)
    return res


def filter_pass(im, filter_size=(10,), mode='circle', kind='low', filter_out=True):
    '''
    Does a fourier-based low-pass filtering of the desired shape. Filter-values: 0=no throughput, 1=max-throughput. 
    For now only works on 2D-images. 
    :param:
    =======
    :im:IMAGE:          Input image to wrok on. 
    :filter_size:LIST:  LIST of INT-sizes of the filter -> for circle it's e.g. just 1 value
    :mode:STRING:       Defines which lowpass_filtering shall be used -> 'Circle' (hard edges)
    '''
    #create filter-shape 
    if mode == 'circle':
        # make sure that result is not bool but int-value for further calculation
        pass_filter = (nip.rr(im) < filter_size[0]) * 1
    else: # just leave object unchanged
        pass_filter = 1
    #decide which part to use
    if kind == 'high':
        pass_filter = 1 - pass_filter
        pass
    elif kind == 'band':
        print("Not implemented yet.")
        pass
    else: # kind == 'low' -> as filters are designed to be low-pass to start with, no change to filters
        pass
    #apply
    res_filtered = nip.ft(im, axes=(-2, -1)) * pass_filter
    res = nip.ift(res_filtered).real
    return res, res_filtered, pass_filter

def harmonic_sum(a,b):
    ''' 
    calculates the harmonic sum of two inputs. 
    '''
    return a * b / (a + b)


def calculate_na_fincorr(obj_M_new = 5,obj_M=10,obj_NA=0.25,obj_N=1,obj_di=160,roundout=0,printout=False):
    '''
    Calculates the NA of a finite-corrected objective given the standard and the new (calibration) measurements. 

    :param:
    =======
    :obj_M:FLOAT: magnification of objective
    :obj_NA:FLOAT: 0.25 
    :obj_N:FLOAT: refractive index -> air
    :obj_di:FLOAT: distance to image 

    :output:
    ========
    NA:FLOAT:   new, calculated NA

    Innline Testing: 
    obj_M_new = 3.3;obj_M=10;obj_NA=0.25;obj_N=1;obj_di=160;printout=True;roundout=3
    '''

    # here: finite corrected, hence: 
    obj_ds = obj_di / obj_M
    # now use thin-lense approximation for objective to get a guess -> assume objective as "thin-lens"
    obj_f  = harmonic_sum(obj_ds,obj_di)#focal-plane distance
    obj_alpha = np.arcsin(obj_NA/obj_N)
    obj_D2 = obj_ds * np.tan(obj_alpha)# half Diameter of "lens"
    # calc new NA and distances
    #f *M1 / (M1+1) = g1 -> given f is constant
    obj_ds_new = obj_f *  (obj_M_new + 1) / obj_M_new
    obj_di_new = obj_M_new * obj_ds_new
    obj_alpha_new = np.arctan(obj_D2 / obj_ds_new)
    obj_na_new = obj_N * np.sin(obj_alpha_new)
    res_list=[obj_na_new,obj_alpha_new,obj_ds_new,obj_di_new]
    if roundout>0: 
        res_list = [round(res_list[m],roundout) for m in range(len(res_list))]
    if printout==True:
        print("NA_new=\t\t{}\nOBJ_alpha_new=\t{}\nOBJ_DS_new=\t{}\nOBJ_DI_new=\t{}".format(res_list[0],res_list[1],res_list[2],res_list[3]))
    return res_list


def calculate_magnification(pixel_range=1000,counted_periods=10,pixel_size=1.4,period_length_real=100,printout=False):
    '''
    Calculates the magnification using a grid-sample. 
    
    :param:
    =======
    :pixel_range:       Length of the used set
    :counted_periods:   Number of periods in measured length
    :period_length_real: Real period length
    :pixel_size:        size of pixel -> ideally in same dimension as real period length

    :out:
    ====
    :magnification:   

    '''
    period_length_pixel = pixel_range/counted_periods
    magnification = period_length_pixel * pixel_size/ period_length_real
    if printout==True: 
        print("New Magnification is: M={}".format(np.round(magnification,5)))
    return magnification

def calculate_maxbinning(res_lateral=100,obj_M=10,pixel_size=6.5,printout=True):
    '''
    Calculates the maximum bin-size possible to ensure (at least) correct Sampling, ignoring sub-pixel sampling.
    '''
    dmax_sampling_detector = res_lateral/2 * obj_M #to be correctly nyquist sampled on the detector
    max_binning = np.floor(dmax_sampling_detector / pixel_size)
    if printout==True: 
        print("Maximum sampling steps on detector are dmax={}.\nHence Maximum binning is b={}.".format(np.round(dmax_sampling_detector,2),max_binning))
    return max_binning,dmax_sampling_detector

def calculate_resolution(obj_na=0.25,obj_n=1,wave_em=525,technique='brightfield',criterium='Abbe', cond_na=0, fluorescence=False,wave_ex=488, printout=False):
    '''
    Calculates the resolution for the selected technique with the given criteria in lateral xy and axial z.  

    For inline testing: obj_na=0.25;obj_n=1;wave_em=525;technique='brightfield';criterium='Abbe'; cond_na=0; fluorescence=False; wave_ex=488; printout=False
    '''
    res = np.zeros(3)
    # get right na
    if fluorescence: 
        na = 2*obj_na
    else:
        if cond_na:
            na = cond_na + obj_na
        else: 
            na = obj_na
    #calculate Abbe-support-limit
    if technique == 'brightfield': 
        res[0]  = wave_em/na
        res[1] = res[0]
        alpha = np.arcsin(obj_na/obj_n)
        res[2] = wave_em/(obj_n*(1-np.cos(alpha))) #check this again!
    else: 
        raise ValueError("Selected technique not implemented yet.")
    # multiply factors for criteria
    if criterium == 'Abbe':
        res = res
    else: 
        raise ValueError("Selected criterium not implemented yet.")
    # print out
    if printout==True: 
        print("The calculated resolution is: x={}, y={}, z={}".format(res[0],res[1],res[2]))
    # finally, return result
    return res

# %% -----------------------------------------------------
# ----                  PROPAGATORS
# --------------------------------------------------------


# %% -----------------------------------------------------
# ----                  DEFOCUS
# --------------------------------------------------------
def defocus(im,mode='Gauss',param=[[0,0],[10,20]],axes=-1):
    '''
    Calculates the defocus of an image for a given 

    Example: 
    mode='Gauss';param=[[0,10],[0,20]];axes=[-2,-1]
    defocus(im,mode=mode,param=param,axes=axes)
    '''
    if not (type(axes) == tuple or type(axes) == list): 
        axes = (axes,)
    res = im
    if mode=='Gauss':
        mu,sigma = gauss_testparam(param=param,length=len(axes))
        #calculate
        for m in range(len(axes)):
            res = nip.convolve(res,gaussian1D(size=im.shape[axes[m]],mu=mu[m],sigma=sigma[m],axis=axes[m]),axes=axes[m])
    return res

def gauss_testparam(param,length=None):
    '''
    Preparers parameters for defocus-calculation
    '''
    #read param
    mu = param[0]
    sigma = param[1]
    if length==None: 
        length = len(sigma)
    #easy sanity 
    if not type(mu) == list:
        mu = [mu,] * length
    if len(mu) <= 1:
        mu = [mu[0],] * length
    if not type(sigma) == list: 
        sigma = [sigma, ] * length
    return mu,sigma

def defocus_stack(im,mode='symmGauss',param=[0,[1,10]],start='center'):
    '''
    Calculates a defocus-stack. Assumes a sigma longer than 1. Does not catch errors with mu. 
    For now, only start from 'center' and 'symmGauss' as mode are implemented.

    Example: 
    mode='symmGauss';param=[0,[1,10]];start='center';
    param=[0,list(np.arange(1,20,1))];
    '''
    if start=='center':
        res = im[np.newaxis] 
    else: 
        raise ValueError("Start-type not implemented yet.")
    if mode=='symmGauss':
        #read param
        mu,sigma = gauss_testparam(param=param,length=len(param[1]))
        #eval defocus to create stack
        for m in range(len(sigma)):
            res = nip.cat((res,defocus(im,mode='Gauss',param=[mu[m],sigma[m]],axes=[-2,-1])[np.newaxis]),axis=0)
    else:
        raise ValueError("This mode is not implemented yet.")
    return  res
    
def gaussian1D(size=10,mu=0,sigma=20,axis=-1,norm='sum'):
    '''
    Calculates a 1D-gaussian.

    For testing: size=100;mu=0;sigma=20;
    '''
    xcoords = nip.ramp1D(mysize=size,placement='center',ramp_dim=axis)
    gaussian1D = 1.0 / np.sqrt(2*np.pi*sigma) * np.exp(- (xcoords - mu)**2 / (2 * sigma**2))
    if norm=='sum':
        gaussian1D /= np.sum(gaussian1D)
    return gaussian1D

def gaussian2D(size=[],mu=[],sigma=[]):
    '''
    Calculates a 2D gaussian. 
    Note that mu and sigma can be different for the two directions and hence have to be input explicilty.

    Example: 
    size=[100,100];mu=[0,5];sigma=[2,30];
    '''
    # just to make sure
    if not type(size) == tuple: 
        size = [size,size]
    if not type(mu) == tuple:
        mu = [mu,mu]
    if not sigma == tuple:
        sigma = [sigma,sigma]
    gaussian2D = gaussian1D(size=size[0],mu=mu[0],sigma=sigma[0],axis=-1)* gaussian1D(size=size[1],mu=mu[1],sigma=sigma[1],axis=-2)
    return gaussian2D



# size=im.shape[-2:];mu=0;sigma=20