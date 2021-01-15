# %% imports
import numpy as np
import NanoImagingPack as nip
# mipy imports
from .functions import gaussian1D
from .numbers import generate_combinations

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
        def noisy(param): return noise_poisson(im, param, norm='max')
        length = len(param)
    elif mode == 'Gaussian':
        def noisy(param): return noise_gaussian(
            im, mu=param[0], sigma=param[1])
        mu, sigma = gauss_testparam(param)
        param = list(zip(mu, sigma))
        length = len(sigma)
    elif mode == 'PoissonANDGauss':
        def noisy(param): return noise_gaussian(noise_poisson(
            im, param[2], norm='max'), mu=param[0], sigma=param[1])
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
        raise AssertionError("So what? Didn't expect such an indecisive move.")
    # assure that dimensions are correct
    if im.ndim > 2:
        norm = norm[..., np.newaxis, np.newaxis]
    im /= norm
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

    Check "image_getshift" to use for a stack or with a reference.

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


def shiftby_list(psf, shifts=None, shift_offset=[1, 1], shift_axes=[-2, -1], nbr_det=[3, 3], retreal=True):
    """Shifts an image by a list of shift-vectors.
    If shifts not given, calculates an equally spaced rect-2D-array for nbr_det (array) of  with shift_offset spacing in pixels between them.
    If shifts given, uses the shape (nbr_det) to calculate the distances between detectors. 
    The shifts are applied from the right, hence if e.g. a list of 2D-shift-vectors is provided they are applied to [-2,-1].

    Parameters
    ----------
    psf : image
        3D-(Detection)PSF
    shifts : list, optional
        shifts to be applied, by default []
    shift_axes : list, optional
        Axes to be used for applying the shift (and hence the Fourier-Transform)
    shift_offset : list, optional
        distance between 2D-array, by default [1, 1]
    nbr_det : list, optional
        , by default [3, 3]
    retreal : bool, optional
        whether result should be real or complex, by default True


    Returns
    -------
    psf_res : nip.image
        list of N+1 DIM of shifted PSF

    Example
    -------
    >>> mipy.shiftby_list(nip.readim(),shifts=[[1,1],[2,2],[5,5]])
    """
    # calculate shifts if not provided
    if shifts is None:
        shift_offset = [1, 1] if shift_offset is None else shift_offset
        nbr_det = [1, 1] if nbr_det is None else nbr_det
        shifts = gen_shift_loop(shift_offset, nbr_det)

    # assure use of positive numbers
    shift_axes = np.array(shift_axes)
    shift_axes[shift_axes < 0] += psf.ndim
    dimdiff = psf.ndim - len(shifts[0])

    # function shortening
    maxis = add_multi_newaxis

    def r1d(psfshape, ramp_dim):
        '''directly adds another dimension (=psf.ndim + 1) to fit with stacking dimension for pinhole'''
        ramped = maxis(nip.ramp1D(psfshape[ramp_dim], ramp_dim=ramp_dim, placement='center',
                                  freq='ftfreq', pixelsize=psf.pixelsize), [-1, ]*(len(psfshape)-1-ramp_dim))
        dimdiff = len(psfshape) - ramped.ndim
        return maxis(ramped, [0, ]*(dimdiff+1))

    # pre-allocate for speed-up
    phase_ramp_list = np.ones((len(shifts),)+psf.shape, dtype=np.complex_)

    # calculate shifts
    for m in shift_axes:
        phase_ramp_list *= np.exp(-1j*2*np.pi * maxis(shifts[:, m-dimdiff], [-1, ]
                                                      * psf.ndim) * r1d(psf.shape, m))

    # apply shifts in FT-space and transform back
    # psf_res = nip.ift(maxis(nip.ft(psf, axes=shift_axes), [0])*phase_ramp_list, axes=shift_axes)
    psf_res = nip.ift(nip.ft(psf, axes=shift_axes) *
                      phase_ramp_list, axes=shift_axes+1)

    if retreal:
        psf_res = psf_res.real

    # correct pixelsizes
    psf_res.pixelsize[1:] = psf.pixelsize if not psf_res.pixelsize[1:
                                                                   ] == psf.pixelsize else psf_res.pixelsize[1:]

    return psf_res


def find_shiftperiod(shift_stack, thresh=0.25):
    '''
    Finds the shift-period for an input-stack.

    TODO:
    1) fix for small stacks. 
    2) find soft thresholds instead of arbitrarily chosen value.

    :PARAMS:
    ========
    :shift_stack:   (ARRAY) Array of estimated shifts per image
    :thresh:        (FLOAT) relative (range: [0,1]) value to use for period cut

    :OUTPUTS:
    =========
    :pmean:         (FLOAT) calculated mean period-length
    :pmedian:       (FLOAT) calculated median period-length
    :period_lengths:(LIST) found period_lengths

    :EXAMPLE:
    ========
    c = mipy.shiftby_list(nip.readim(),shift_offset=[3.3,5.6],nbr_det=[1,10])
    c = np.reshape(np.transpose(nip.repmat(c,replicationFactors=[7,1,1,1]),[1,0,2,3]),[70,c.shape[-1],c.shape[-2]])
    shifts = mipy.image_getshift(c,c[0])
    pmean,pmedian, period_lengths = find_shiftperiod(shifts,thresh=0.05)
    print("pmean={}\npmedian={}\nperiod_lengths={}".format(pmean,pmedian,period_lengths))     
    '''
    periods = []
    bias = 0

    # get difference-values
    for m, val in enumerate(shift_stack[:-1]):
        if np.any(abs(shift_stack[m+1]-val) > abs(val)*thresh):
            periods.append(m)

    # get first period length (take "0" into account)
    period_lengths = [periods[0]+1, ]

    # get other lengths
    for m, val in enumerate(periods[:-1]):
        period_lengths.append(periods[m+1] - val)

    # average over period-lengths
    pmean = np.mean(period_lengths)
    pmedian = np.median(period_lengths)

    # done?
    return pmean, pmedian, period_lengths


def gen_shift_npfunc(soff, pix):
    '''
    Calculates coordinates for an equally spaced rect-2D-array. Use with e.g. shiftby_list to generate a shifted set of images.

    NOTE: Slower than gen_shift_loop

    :PARAMS:
    ========
    :soff:      (LIST) offset between pixel-array (2D)
    :pix:       (LIST) number of pixels per direction (2D)

    :OUT:
    =====
    :c:         (NP.NDARRAY) calculated array spacing

    Test with ipython:
    ==================
    %timeit c1 = gen_shift_npfunc([1,1],[5,5])
    %timeit c2 = gen_shift_loop([1,1],[5,5])
    # 29.7 µs ± 883 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    # 18.7 µs ± 1.09 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    '''
    a = np.repeat(
        np.arange(-int(pix[0]/2.0), int(pix[0]/2.0)+1)[np.newaxis, :], pix[1], 0).flatten()
    b = np.repeat(
        np.arange(-int(pix[1]/2.0), int(pix[1]/2.0)+1)[:, np.newaxis], pix[0], 1).flatten()
    c = [[soff[0]*m, soff[1]*n] for m, n in zip(b, a)]
    return c


def gen_shift_loop(soff, pix):
    """Calculates coordinates for an equally spaced rect-2D-array. Use with e.g. shiftby_list to generate a shifted set of images.

    TODO:
        1) generalize for different dimensions (e.g. give unit-cell and generate pattern)
        2) generalize for nD-input

    Parameters
    ----------
    soff : list
        offset between pixel-array (2D)
    pix : list
        number of pixels per direction (2D)

    Returns
    -------
    shiftarr : array of shifts
        calculated array spacing

    Example
    -------
    Tries to always return a centered shiftlist, hence it should be expected that
    >>> gen_shift_loop([2,3],[1,1])
    gen_shift_loop([2,3],[1,1])
    >>> gen_shift_loop([2,3],[1,3])
    array([[ 0, -3],[ 0,  0],[ 0,  3]])
    >>> gen_shift_loop([2,3],[0,3])
    array([], dtype=float64)

    """
    shiftarr = np.ones([pix[0]*pix[1], 2])
    xo = -int(pix[1]/2.0)
    yo = -int(pix[0]/2.0)
    for jj in range(pix[0]):
        for kk in range(pix[1]):
            shiftarr[jj*pix[1]+kk] = [soff[0]*(yo+jj), soff[1]*(xo+kk)]
    return shiftarr
# %%
# ------------------------------------------------------------------
#                       PINHOLE-MANIPULATIONS
# ------------------------------------------------------------------


def pinhole_getcenter(im, method='sum', saxis=None):
    """Uses argmax to find pinhole center assuming axis=(0,1)=pinhole_axes and axis=(-2,-1)=sample-axis. Calculates shift-mask.

    Parameters
    ----------
    im : image
        input image (at least 4dim)
    method : str, optional
        method used to calculate center position, implemented: 'sum', 'max', 'min', by default 'sum'
    saxis : tuple, optional
        Axis to be used for searching for the center pinhole, by default None  

    Returns
    -------
    pinhc : list
        coordinates of pinhole-center
    smask : list
        shift-coordinates necessary to be used with nip.extract() to shift image center to new center position
    maskshape : list
        shape of pinhole-dimension of input image

    Raises
    ------
    ValueError
        In case of "method"-variable given that was not implemented
    """
    # parameter
    if saxis == None:
        saxis = tuple(np.arange(1, im.ndim))

    # project scan into detector plane
    if method == 'sum':
        im_detproj = np.sum(im, axis=saxis)
    elif method == 'max':
        im_detproj = np.max(im, axis=saxis)
    elif method == 'min':
        im_detproj = np.min(im, axis=saxis)
    else:
        raise ValueError("Chosen method not implemented")

    # find center via maximum-point
    pincenter = np.argmax(im_detproj)

    # done?
    return pincenter, im_detproj


def pinhole_shift(pinhole, pincenter):
    '''
    Gets pinhole mask and shift.

    :PARAMS:
    ========
    :pinhole:   (IMAGE)     pinhole mask/image
    :pincenter: (INT)       Center position of pinhole

    :OUT:
    =====
    :pinshift:  (LIST)      calculated shifts
    :pinhole:   (IMAGE)     shifted pinhole
    '''

    # pincenter to 2d
    if pinhole.ndim > 1:
        pincentern = [pincenter//pinhole.shape[-1],
                      np.mod(pincenter, pinhole.shape[-1])]
        pinshift = np.array(np.array(pinhole.shape) //
                            2-pincentern, dtype=np.int8)
        pinholen = nip.extract(pinhole, pinhole.shape, pinshift)
    else:
        pinshift = np.array(len(pinhole)//2-pincenter, dtype=np.int8)
        pinholen = np.roll(pinhole, shift=pinshift, axis=0)

    # done?
    return pinholen, pinshift

# %%
# ------------------------------------------------------------------
#                       SHAPE-MANIPULATION
# ------------------------------------------------------------------


def transpose_arbitrary(imstack, idx_startpos=[-2, -1], idx_endpos=[0, 1], direction='forward'):
    '''
    Exchange-based successive array transposition to exchange given start_pos with according endpos. 

    EXAMPLE: 
    =======
    a = np.reshape(np.arange(2*3*4*5*6),[2,3,4,5,6])
    b = transpose_arbitrary(a,[-2,-1],[0,1],direction='forward')
    b1 = transpose_arbitrary(b,[-2,-1],[0,1],direction='backward')
    print(f"a.shape={a.shape}\nb.shape={b.shape}\nb1.shape={b1.shape}")
    '''
    # some sanity
    if type(idx_startpos) == int:
        idx_startpos = [idx_startpos, ]
    if type(idx_endpos) == int:
        idx_endpos = [idx_endpos, ]

    # assert correct dimensionality
    if not (len(idx_startpos) == len(idx_endpos)):
        raise Exception(
            'idx_startpos and idx_endpos do not have the same size.')

    # work on view of imstack
    im = imstack

    # check direction
    if not(direction == 'forward'):
        idx_startpos = idx_startpos[::-1]
        idx_endpos = idx_endpos[::-1]

    for k, start in enumerate(idx_startpos):
        im = np.swapaxes(im, axis1=start, axis2=idx_endpos[k])

    # done?
    return im


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
        id2 = id1[-2:] + id1[: -2]
        id3 = id1[2:] + id1[: 2]
        imh = np.transpose(im, id2)
    else:
        imh = im
    res = nip.image(np.zeros(imh[:: bin_size, :: bin_size].shape))
    # only do 2d binning
    if mode == 'real_sum':
        offset = list(np.arange(bin_size))
        for m in range(bin_size):
            for n in range(bin_size):
                res += imh[offset[m]: imh.shape[0]-bin_size+1+offset[m]: bin_size,
                           offset[n]: imh.shape[1]-bin_size+1+offset[n]: bin_size]
            # not finished
    elif mode == 'real_pix':  # only selects the pixel-value at the binning-position
        res = imh[:: bin_size, :: bin_size]
    # takes median of regions -> reshape to split, e.g. 800x800 ->(bin2)-> 400x400x2x2 -> median along dim-2&3
    elif mode == 'real_median':
        pass
    elif mode == 'conv':
        # NOT FINISHED
        # ims = np.array(im.shape)
        # imh = nip.extract(im,)
        # nip.convolve()
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


def normto(im, dims=(), method='max', direct=True):
    '''
    Norms input image to chosen method. Default: max and using u_func (=in-place normalization).

    :PARAM:
    =======
    :im:        (ARRAY)     input image
    :dims:      (TUPLE)     dimensions to be used for normalization
    :method:    (STRING)    method for normalization -> 'max','min','mean'
    :direct:    (BOOL)      apply in-place?

    OUTPUT:
    =======
    :im:        normalized image

    EXAMPLE:
    ========
    nip.v5(normto(nip.readim(),'min'))
    '''
    # keep sane
    if dims == ():
        dims = tuple(np.arange(im.ndim))
    if direct:
        if method == 'max':
            im /= im.max(dims, keepdims=True)
        elif method == 'min':
            im /= im.min(dims, keepdims=True)
        elif method == 'mean':
            im /= im.mean(dims, keepdims=True)
        else:
            raise ValueError(
                "Normalization method not existent or not chosen properly.")
    else:
        if method == 'max':
            im = im / im.max(dims, keepdims=True)
        elif method == 'min':
            im = im / im.min(dims, keepdims=True)
        elif method == 'mean':
            im = im / im.mean(dims, keepdims=True)
        else:
            raise ValueError(
                "Normalization method not existent or not chosen properly.")
    return im


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


def midVallist(iml, dims, method=1, keepdims=False):
    '''
    Gets middle-value or list of Middle-values according to volume (dimensions) provided by dims.
    Note:
        -> keepdims implemented, but numpy seems to ignore it... :(

    TODO:
        1) catch User-errors
        2) implement slice-method

    :PARAM:
    =======
    :iml:           (NDARRAY) nD-image
    :dims:          (LIST) of dimensions

    :OUT:
    =====
    :midl:          (LIST) of midVals

    :EXAMPLE:
    ========
    midl = mipy.midVallist(nip.readim(),dims=(1,))

    '''
    # get parameters
    orish = list(iml.shape)
    imds = len(orish)

    # get rid of double entries and reverse order
    dimss = np.unique(np.mod(dims, imds)).tolist()
    dimss.sort(reverse=True)

    # center always lies at floor-div. by 2
    imlsc = np.array(iml.shape)//2

    # get centers
    if method == 1:
        midl = iml
        for dim in dimss:
            midl = midl.take(indices=imlsc[dim], axis=dim)
            orish[dim] = 1
    else:
        # nip.slice()
        pass

    if keepdims == True:
        midl = np.reshape(midl, orish)

    return midl


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


def ismR_generateRings(det_geo=[0, 0], ring=0, aslist=True):
    '''
    Generates binary mask for selection of rings for deconvolution. Assumes 2D detector distribution geometry.
    '''
    sel = np.zeros(det_geo)
    selc = np.array(sel.shape)//2

    if ring == 0:
        sel[selc[0], selc[1]] = 1
    else:
        sel[selc[-2]-ring:selc[-2]+ring+1, selc[-1]-ring:selc[-1]+ring+1] = 1
        sel[selc[-2]-ring+1:selc[-2]+ring, selc[-1]-ring+1:selc[-1]+ring] = 0
    sel = np.array(sel, dtype=bool)

    if aslist == True:
        sel = np.reshape(sel, np.prod(sel.shape))

    return sel


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

# %% -----------------------------------------------------
# ----                  EXTREMUM-ANALYSIS
# --------------------------------------------------------


def getPoints(im, viewer=None, compare=False):
    '''
    Function to collect and reformat points from an image using nip.v5 viewer. Inspired from "nip->transformations->findTransformFromMarkers" function.
    Comparison of up to two images implemented for now. Easily enhanceable for n-images (=stack).

    :PARAMS:
    ========
    :im:            (IMAGE) input image
    :viewer:        (VIEWER) already open v5 instance
    :compare:       (BOOL) whether to use comparison or 1-image mode

    :OUTPUT:
    ========
    :picklist:      (ARRAY) of selected points

    :EXAMPLE:
    =========
    picklist = mipy.getPoints([nip.readim(),nip.readim()],viewer=None,compare=True)
    print(picklist)
    '''

    if viewer is None:
        viewer = nip.v5(im)
    input('Please position markers (press "m") in alternating elements (toggle with "e") or simply alternating positions.\nUse "0" or "9" to step through markers and "M" to delete a marker.\n "A" and "a" to Zoom.\nTo turn off the automatic maximum search for marker positioning use the menu "n".')
    mm = viewer.getMarkers()
    if compare:
        if len(mm) < 6:
            raise ValueError('At least 3 markers are needed for both images!')
        src = np.array(mm[::2])[:, 4:2:-1]
        dst = np.array(mm[1::2])[:, 4:2:-1]
        picklist = [src, dst]
    else:
        picklist = np.array(mm)[:, 4:2:-1]

    return picklist


def find_extrema_1D(im, visres=True):
    '''
    Searches extrema based on 1D function argrelextrema from scipy.signal.
    '''
    # imports
    from scipy.signal import argrelextrema

    # parameters
    output_shape = im.shape
    flatim = im.flatten()

    # find local maxima
    a = argrelextrema(flatim, np.greater)

    # display
    if visres:
        b = visualize_extrema_1D(
            flatim=flatim, localmax=a, value=10, output_shape=output_shape)

    # done?
    return a, b


def visualize_extrema_1D(flatim, localmax, value, output_shape):
    '''
    Calculates and Visualizes extrema from Scipy-signal (1D) toolbox. 
    '''
    b = np.zeros(flatim.shape)
    for m in localmax[0]:
        b[m] = 10
    return b.reshape(output_shape)
