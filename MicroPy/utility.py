# %% imports
import numpy as np
import NanoImagingPack as nip
# mipy imports
from .simulation import gen_shift_loop

# %%
# -------------------------------------------------------------------------
# NOISE-Functions
# -------------------------------------------------------------------------
#


def noise_stack(im, mode='Poisson', param=[1, 10, 100, 1000]):
    '''
    Generates a stack from given image along the (new) 0th dimension. 

    :PARAMS:
    ========
    :im:    (IMAGE) Ndim input image
    :mode:  (STRING) noise to be added -> 'Poisson', 'Gaussian', 'PoissonANDGauss'
    :param: (list)  necessary parameters per noise type. Note, that Gaussian needs list of [mu,sigma] pairs while poisson only needs simple [mu1,m2,..] list.

    :OUT:
    =====
    :res:   (IMAGE) N+1 dim noisy image

    :EXAMPLE: 
    =========
    im = nip.readim();
    mode='Poisson';param=[1,10,100,1000];
    noise_stack(im,mode=mode,param=param)

    mode='Gaussian';param=[[0],[10,20,50,100,200]];
    mode='PoissonANDGauss';param=[[0],[10,10,40,40],[1,10,1,100]];

    '''
    # prepare modes
    if mode == 'Poisson':
        def noisy(param): return noise_poisson(im, param, norm='mean')
        length = len(param)
    elif mode == 'Gaussian':
        def noisy(param): return noise_gaussian(
            im, mu=param[0], sigma=param[1])
        mu, sigma = gauss_testparam(param)
        param = list(zip(mu, sigma))
        length = len(sigma)
    elif mode == 'PoissonANDGauss':
        def noisy(param): return noise_gaussian(noise_poisson(
            im, param[2], norm='mean'), mu=param[0], sigma=param[1])
        mu, sigma = gauss_testparam(param[:2])
        phot = param[2]
        param = list(zip(mu, sigma, phot))
        length = len(param)
    else:
        raise ValueError("Mode not implemented yet.")
    # run iteration
    for m in range(length):
        resim = noisy(param[m])[np.newaxis]
        if m == 0:
            res = resim
        else:
            res = nip.cat((res, resim), axis=0)
    # output
    return res


def noise_poisson(im, phot=None, norm='mean'):
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
        mu = 0.1 * np.mean(im, axis=(-2, -1))
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
        norm = np.mean(im, axis=(-2, -1)) / phot
    elif norm == 'max':
        norm = np.max(im, axis=(-2, -1)) / phot
    elif norm == 'min':
        norm = np.min(im, axis=(-2, -1)) / phot
    else:
        raise AssertionError("So what? Didn't expect such a indecisive move.")
    # assure that dimensions are correct
    if im.ndim > 2:
        norm = norm[..., np.newaxis, np.newaxis]
    im = im/norm
    return im

# %%

# %%
# ------------------------------------------------------------------
#                       IMAGE-SHIFT OPERATIONS
# ------------------------------------------------------------------


def channel_getshift(im):
    '''
    Operations on 1 image within channels
    '''
    shift = []
    myshift = 0
    id1, id2 = generate_combinations(3, combined_entries=2)
    for m in range(len(id1)):
        myshift, _, _, _ = findshift(im[id1[m]], im[id2[m]], 100)
        shift.append(myshift)
    return np.array(shift)


def image_getshift(im, im_ref, prec=100):
    '''
    Wrapper for findshift to enable shifts between stacks or with respect to a reference. Made for nD images, that will be compared on a 2D level.
    TODO: Does not work properly for e.g. ims=[10,3,120,120] and im_refs=[3,120,120]. ETC!
    :PARAM:
    ======
    im:        im-stack or images to compare
    im_ref:     reference image (2d) or comparison stack of same sace as im

    :OUT:
    =====
    shift:      the calculated shifts
    '''
    # param
    shift = []
    myshift = 0
    imd = im.ndim
    im_refd = im_ref.ndim
    ims = im.shape
    im_refs = im_ref.shape
    # sanity checks
    if imd > 3:
        im = np.reshape(im, [np.prod(ims[:-2]), ims[-2], ims[-1]])
    if im_refd > 3:
        im_ref = np.reshape(im_ref, [np.prod(ims[:-2]), ims[-2], ims[-1]])
    # execute different dimensional cases
    if imd == 2 and im_refd == 2:
        myshift, _, _, _ = findshift(im, im_ref, prec)
        shift.append(myshift)
    elif imd > im_refd and im_refd == 2:
        for m in range(ims[0]):
            myshift, _, _, _ = findshift(im[m], im_ref, prec)
            shift.append(myshift)
    elif imd > 2 and im_refd > 2:
        for m in range(im.shape[0]):
            myshift, _, _, _ = findshift(im[m], im_ref[m % im_refs[0]], prec)
            shift.append(myshift)
    else:
        raise ValueError('Wrong dimensions of input arrays.')
    if imd > 3:
        shift = np.reshape(np.array(shift), tuple(list(ims[:-2]) + [2, ]))
    else:
        shift = np.array(shift)
    return shift


def findshift(im1, im2, prec=100, printout=False):
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
    if printout:
        print("Found shifts={} with upsampling={}, error={} and diffphase={} in {}s.".format(
            shift, prec, np.round(error, 4), diffphase, tend))
    return shift, error, diffphase, tend


def shiftby_list(psf, shifts=[], shift_offset=[1, 1], nbr_det=[3, 3]):
    '''
    Shifts an image by a list of shift-vectors. 
    If shifts not given, calculates an equally spaced rect-2D-array for nbr_det (array) of   with shift_offset spacing in pixels between them.
    TODO: Implement for 2D-PSF. Then force: "The shifts have to have the same shape as the image. "

    :PARAM:
    =======
    :psf:           3D-(Detection)PSF
    :shifts:        shifts to be applied
    :shift_offset:  (OPTIONAL) distance between 2D-array
    :nbr_det:       (OPTIONAL) number of detector elements to be generated

    :OUTPUT:
    ======= 
    :psf_res:       list of N shifted 3D-(Detection)PSF

    Example:
    ========
    mipy.shiftby_list(nip.readim(),shifts=[[1,1],[2,2],[5,5]])
    '''
    if shifts == []:
        shifts = gen_shift_loop(shift_offset, nbr_det)
    phase_mapx = nip.xx(psf.shape[-2:], freq='ftfreq')
    phase_mapy = nip.yy(psf.shape[-2:], freq='ftfreq')
    # phase_ramp_list = mipy.add_multi_newaxis(imstack, newax_pos=[-1,-1,-1]
    phase_ramp_list = np.exp(-1j*2*np.pi*(shifts[:, 0][:, np.newaxis, np.newaxis, np.newaxis]*phase_mapx[np.newaxis,
                                                                                                         np.newaxis, :, :] + shifts[:, 1][:, np.newaxis, np.newaxis, np.newaxis]*phase_mapy[np.newaxis, np.newaxis, :, :]))
    psf_res = np.real(
        nip.ift3d(nip.ft(psf)[np.newaxis, :, :, :]*phase_ramp_list))
    return psf_res

# %%
# ------------------------------------------------------------------
#                       SHAPE-MANIPULATION
# ------------------------------------------------------------------


def transpose_arbitrary(imstack, idx_startpos=[-2, -1], idx_endpos=[0, 1]):
    '''
    creates the forward- and backward transpose-list to change stride-order for easy access on elements at particular positions. 

    TODO: add security/safety checks
    '''
    # some sanity
    if type(idx_startpos) == int:
        idx_startpos = [idx_startpos, ]
    if type(idx_endpos) == int:
        idx_endpos = [idx_endpos, ]
    # create transpose list
    trlist = list(range(imstack.ndim))
    for m in range(len(idx_startpos)):
        idxh = trlist[idx_startpos[m]]
        trlist[idx_startpos[m]] = trlist[idx_endpos[m]]
        trlist[idx_endpos[m]] = idxh
    return trlist


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
    elif mode == 'real_pix':  # only selects the pixel-value at the binning-position
        res = imh[::bin_size, ::bin_size]
    # takes median of regions -> reshape to split, e.g. 800x800 ->(bin2)-> 400x400x2x2 -> median along dim-2&3
    elif mode == 'real_median':
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
        nip.resample(imh, factors=[1.0/bin_size,
                                   1.0/bin_size]+[1, ]*(imh.ndim-2))
    else:
        raise ValueError('Mode not existing.')

    if normalize == 'old':  # normalize to old stack-maximum -> rather: per-slice?
        res = res/np.max(res) * np.max(im)
    elif normalize == 'mean':
        res /= bin_size
    elif normalize == 'none':  # leave res as is
        pass
    else:
        raise ValueError('Normalization not existing.')

    if im.ndim > 2:
        res = np.transpose(res, id3)
    return res


def bin_norm(im, bins):
    im = np.array(nip.resample(im, create_value_on_dimpos(
        im.ndim, axes=[-2, -1], scaling=[1.0/bins[0], 1.0/bins[1]]))) / (bins[0] * bins[1])
    return im


def norm_back(imstack, normstack, normtype):
    '''
    implemented for ndim>3.
    TODO: implement for all dimensions.
    '''
    ndimdiff = imstack.ndim - normstack.ndim
    for m in range(ndimdiff):
        normstack = normstack[..., np.newaxis]
    imstack_changed = np.round(imstack / np.mean(imstack, axis=(-1, -2))
                               [..., np.newaxis, np.newaxis] * normstack, 0).astype(normtype)
    return imstack_changed


def add_multi_newaxis(imstack, newax_pos=[-1, ]):
    '''
    Adds new-axis at the mentioned position with respect to the final shape per step. So hence, if ndim = 3 and e.g. (3,256,256) and newax_pos = (1,2,-1) it will lead to:(3,1,1,256,256,1), where negative indexes are added first (postpending) and positive indexes are inserted last (prepend).

    TODO: 
        1) no error-prevention included
        2) axes not always appended as thought

    :param:
    =======
    :imstack:   n-dimensional data_stack
    :newax_pos: positions of the newaxis

    :example: 
    ========
    import NanoImagingPack as nip
    a = np.repeat(nip.readim('orka')[np.newaxis],3,0)
    b = add_multi_newaxis(a,[1,2,-1])
    print("dim of a="+str(a.shape)+"\ndim of b="+str(b.shape)+".")

    '''
    # problem: list().insert interprets -1 as first element before last element of list -> recalc dimensions
    newax_pos.sort(reverse=False)
    imshape = list(np.array(imstack).shape)  # im_shape_start
    for m in range(len(newax_pos)):
        if newax_pos[m] == -1:
            newax_pos[m] = len(imshape)
        elif newax_pos[m] < -1:
            newax_pos[m] += 1
        imshape.insert(newax_pos[m], 1)
    return np.reshape(imstack, imshape)


def create_value_on_dimpos(dims_total, axes=[-2, -1], scaling=[0.5, 0.5]):
    '''
    Creates dim-vectors for e.g. scaling.

    :param:
    =======
    dims_total:     e.g. via np.ndim
    :axes:          axes to change
    :scaling:       values to change to 

    :Example: 
    =========
    a = np.repeat(nip.readim('orka')[np.newaxis],3,0)
    b = add_multi_newaxis(a,[1,2,-1])
    print("dim of a="+str(a.shape)+"\ndim of b="+str(b.shape)+".") 
    atrans = create_value_on_dimpos(b.ndim,axes=[1,-2,-1],scaling=[2,3,0.25])
    print("scaling factors for array="+str(atrans)+".")
    '''
    res = [1, ] * dims_total
    for m in range(len(axes)):
        res[axes[m]] = scaling[m]
    return res


def get_nbrpixel(im, dim=[0, 1]):
    '''
    just calculates the number of the pixels for the given dimensions.
    '''
    hl = [m for c, m in enumerate(im.shape) if c in dim]
    return np.prod(hl)


def get_immax(im):
    '''
    Gets max value of an image.
    '''
    try:
        dmax = np.iinfo(im.dtype).max
    except:
        dmax = np.finfo(im.dtype).max
    return dmax


def subtract_from_max(im):
    '''
    Inverse intensity values of an image by subtracting the image values from the image-max.
    '''
    im = get_immax(im) - im
    return im


# %% -----------------------------------------------------
# ----                  PROPAGATORS
# --------------------------------------------------------


# %% -----------------------------------------------------
# ----                  DEFOCUS
# --------------------------------------------------------
def defocus(im, mode='Gauss', param=[[0, 0], [10, 20]], axes=-1):
    '''
    Calculates the defocus of an image for a given 

    Example: 
    mode='Gauss';param=[[0,10],[0,20]];axes=[-2,-1]
    defocus(im,mode=mode,param=param,axes=axes)
    '''
    if not (type(axes) == tuple or type(axes) == list):
        axes = (axes,)
    res = im
    if mode == 'Gauss':
        mu, sigma = gauss_testparam(param=param, length=len(axes))
        # calculate
        for m in range(len(axes)):
            res = nip.convolve(res, gaussian1D(
                size=im.shape[axes[m]], mu=mu[m], sigma=sigma[m], axis=axes[m]), axes=axes[m])
    return res


def gauss_testparam(param, length=None):
    '''
    Preparers parameters for defocus-calculation
    '''
    # read param
    mu = param[0]
    sigma = param[1]
    if length == None:
        length = len(sigma)
    # easy sanity
    if not type(mu) == list:
        mu = [mu, ] * length
    if len(mu) <= 1:
        mu = [mu[0], ] * length
    if not type(sigma) == list:
        sigma = [sigma, ] * length
    return mu, sigma


def defocus_stack(im, mode='symmGauss', param=[0, [1, 10]], start='center'):
    '''
    Calculates a defocus-stack. Assumes a sigma longer than 1. Does not catch errors with mu. 
    For now, only start from 'center' and 'symmGauss' as mode are implemented.

    Example: 
    mode='symmGauss';param=[0,[1,10]];start='center';
    param=[0,list(np.arange(1,20,1))];
    '''
    if start == 'center':
        res = im[np.newaxis]
    else:
        raise ValueError("Start-type not implemented yet.")
    if mode == 'symmGauss':
        # read param
        mu, sigma = gauss_testparam(param=param, length=len(param[1]))
        # eval defocus to create stack
        for m in range(len(sigma)):
            res = nip.cat((res, defocus(im, mode='Gauss', param=[
                          mu[m], sigma[m]], axes=[-2, -1])[np.newaxis]), axis=0)
    else:
        raise ValueError("This mode is not implemented yet.")
    return res
