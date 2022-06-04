"""
---------------------------------------------------------------------------------------------------

	@author René Lachmann
	@email herr.rene.richter@gmail.com
	@create date 2022-05-27 14:32:52
	@modify date 2022-06-04 10:29:31
	@desc Collection of deprecated functions ...just in case

---------------------------------------------------------------------------------------------------
"""
from deprecated import deprecated

# mipy imports
from .general_imports import *
from .utility import  findshift, midVallist,pinhole_getcenter

# %% ---------------------------------------------------------------
# ---                         from processingISM.py                         ---
# ------------------------------------------------------------------

@ deprecated(version='0.1.3', reason='Updated general interface to 1D-list. See recon_sheppardSUMming for new operation.')
def ismR_sheppardSUMming(im, mask, sum_method='all'):
    '''
    Ways for summing the shifted parts together. Idea behind:
    1) Pair different regions before shifting (ismR_pairRegions), then find regional-shifts (ismR_getShiftmap) and finally sheppardSum them.
    2) sheppardSUM only until/from a limit to do different reconstruction on other part (eg Deconvolve within deconvMASK and apply sheppardSUM on outside)

    TODO: --------------------------------------------
        1) TEST!
        2) fix dimensionality! (avoid np.newaxis)
        3) implement 'ring'
        4) fix to work with deconvolution
    --------------------------------------------------

    :PARAM:
    =======
    :im:            input image
    :shift_map:          shift-map
    :sum_method:    Method for applied summing -> 'all', 'ring'

    :OUT:
    =====
    :ismR:       sheppardsummed ISM_image

    '''

    if sum_method == 'all':
        ismR = np.sum(im * mask[..., np.newaxis, np.newaxis], axis=(0, 1))
    elif sum_method == 'ring':
        pass
    else:
        raise ValueError("Sum_method not implemented.")

    return ismR


@ deprecated(version='0.1.3', reason='Interface was updated to 1D-list. Check recon_genShiftmap.')
def ismR_genShiftmap(im, mask, pincenter, shift_method='nearest'):
    '''
    Generates Shiftmap for ISM SheppardSUM-reconstruction. Assumes shape: [LIST,pinhole_y,pinhole_x]

    :PARAM:
    =======
    :im:            input nD-image (first two dimensions are detector plane)
    :mask:    should be within the first two image dimensions
    :pincenter:        center of mask-pinhole

    :OUT:
    =====
    :shift_map:          shift-map for all pinhole-pixels
    '''
    shift_map = np.array([[[0, 0], ]*mask.shape[1]
                          for m in range(mask.shape[0])])
    if shift_method == 'nearest':

        # checks for nearest only if dimension is bigger than 0
        if mask.shape[1] > 1:
            xshift, _, _, _ = findshift(
                im[pincenter[0], pincenter[1]+1], im[pincenter[0], pincenter[1]], 100)
        else:
            xshift = np.zeros(2)
        if mask.shape[0] > 1:
            yshift, _, _, _ = findshift(
                im[pincenter[0]+1, pincenter[1]], im[pincenter[0], pincenter[1]], 100)
        else:
            yshift = np.zeros(2)

        # build map per pinhole
        for k in range(mask.shape[0]):
            for l in range(mask.shape[1]):
                shift_map[k][l] = (k-pincenter[0])*yshift + \
                    (l-pincenter[1])*xshift

    elif shift_method == 'mask':
        for k in range(mask.shape[0]):
            for l in range(mask.shape[1]):
                if mask[k, l] > 0:
                    shift_map[k][l], _, _, _ = findshift(
                        im[k, l], im[pincenter[0], pincenter[1]], 100)
    elif shift_method == 'complete':
        for k in range(mask.shape[0]):
            for l in range(mask.shape[1]):
                shift_map[k, l], _, _, _ = findshift(
                    im[k, l], im[pincenter[0], pincenter[1]], 100)
    else:
        raise ValueError("Shift-method not implemented")

    figS, axS = ismR_drawshift(shift_map)

    return shift_map, figS, axS


@ deprecated(version='0.1.3', reason='Interface was updated to 1D-list. Check recon_sheppardShift.')
def ismR_sheppardShift(im, shift_map, method='iter', use_copy=False):
    '''
    Does shifting for ISM-SheppardSum.

    :PARAMS:
    ========
    :im:            image to be shifted (first two dimensions should match first two dim of shift_map)
    :shift_map:          shift map
    :method:        Available methods for shifting -> 'iter','parallel'

    :OUT:
    =====
    :im:            shifted_image
    :

    '''
    # work on copy to keep original?
    if use_copy:
        imh = nip.image(np.copy(im))
    else:
        imh = im

    if method == 'iter':
        for k in range(imh.shape[0]):
            for l in range(imh.shape[1]):
                imh[k, l] = nip.shift2Dby(im[k, l], shift_map[k][l])
    elif method == 'parallel':
        raise Warning("Method 'parallel' is not implemented yet.")
    else:
        raise ValueError("Chosen method not existent.")

    return imh


@ deprecated(version='0.1.3', reason='General Interface was changed to 1D-list-operations. Check recon_confocal.')
def ismR_confocal(im, pinsize=None, pinshape='circle', pincenter=None, store_masked=False):
    '''
    **!!THIS FUNCTION IS DEPRECATED!!**

    Confocal reconstruction of ISM-data. For now: only implemented for 2D-pinhole/detector plane (assumed as (0,1) position).

    TODO:
        1) Add for arbitrary (node) structure.
        2) arbitrary detector-axes

    PARAM:
    =====
    :im:            input nD-image
    :detaxes:       (TUPLE) axes of detector (for summing)
    :pinsize:        (LIST) pinhole-size
    :pincenter:        (LIST) pinhole-center
    :pinshape:      (STRING) pinhole-shape -> 'circle', 'rect'
    :store_masked:  store non-summed confocal image (=selection)?

    OUTPUT:
    =======
    :imconf:    confocal image OR list of masked_image and confocal image

    EXAMPLE:
    =======
    im = mipy.shiftby_list(nip.readim(),shifts=np.array([[1,1],[2,2],[5,5]]))
    # imconf = mipy.ismR_confocal(im,axes=(0),pinsize=None,pinshape='circle',pincenter=None)

    '''
    if pincenter == None:
        pincenter, detproj = pinhole_getcenter(im, method='max')

    if pinsize == None:
        pinsize = np.array(mask_shape//8, dtype=np.uint)+1

    # closed pinhole case only selects central pinhole
    if pinsize[0] == 0 and len(pinsize) == 1:
        imconfs = np.squeeze(im[pincenter[0], pincenter[1], :, :])

    else:
        shift_mask = ismR_shiftmask2D(
            im, pinsize=pinsize, mask_shape=mask_shape, pincenter=pincenter, pinshape=pinshape)

        # do confocal summing
        imconf = im * shift_mask[:, :, np.newaxis, np.newaxis]
        imconfs = np.sum(imconf, axis=(0, 1))

        # stack mask Confocal and result if intended
        if store_masked == True:
            imconfs = [imconf, imconfs]

    return imconfs


@ deprecated(version='0.1.3', reason='General Interface was changed to 1D-list-operations. Check recon_sheppardSUM.')
def ismR_sheppardSUM(im, shift_map=[], shift_method='nearest', pincenter=None, pinsize=None, pindim=None, pinshape=None):
    '''
    Calculates sheppardSUM on image (for arbitrary detector arrangement), meaning:
    1) find center of pinhole for all scans using max-image-peak (on center should have brightest signal) if not particularly given
    2) define mask/pinhole
    3) find shifts between all different detectors within mask, result in sample coordinates, shift back
    4) further processing?

    TODO:
    1) catch User-input-error!

    :PARAM:
    =======
    :im:            (IMAGE)     Input image; assume nD for now, but structure should be like(pinholeDim=pd) [pd,...(n-3)-extraDim...,Y,X]
    :shift_map:     (LIST)      list for shifts to be applied on channels -> needs to have same structure as pd
    :shift_method:  (STRING)    Method to be used to find the shifts between the single detector and the center pixel -> 1) 'nearest': compare center pinhole together with 1 pix-below and 1 pix-to-the-right pinhole 2) 'mask': all pinholes that reside within mask (created by pinsize)  3) 'complete': calculate shifts for all detectors
    :pincenter:     (INT)       center-pinhole position (e.g. if pindim=[5,6], then eg pincenter=14)
    :pinsize:       (LIST)      Diameter of the pinhole-mask to be used for calculations, eg "[5,]" to have a circular pinhole or "[3,8]" for rect
    :pindim:        (LIST)      Dimensions of pinhole, eg [9,9] if rectangular
    :pinshape:      (STRING)    shape the pinhole should be generated with, options:
                                'circ' : circular pinhole
                                'rect' : rectangular pinhole

    OUT:
    ====
    :ismR:          reassinged ISM-image (within mask)
    :shift_map:     shift_map
    :mask:          applied shift_mask
    :pincenter:        center-point of mask/pinhole
    '''
    # start clean and straight

    # find shift-list -> Note: 'nearest' is standard
    if shift_map == []:
        shift_map, figS, axS = ismR_genShiftmap(
            im=im, mask=mask, pincenter=pincenter, shift_method=shift_method)

    # apply shifts -> assumes that 1st two dimensions are of detector-shape ----> for now via loop, later parallel
    ims = ismR_sheppardShift(im, shift_map, method='iter', use_copy=True)
    # import matplotlib.pyplot as plt
    # different summing methods
    ismR = ismR_sheppardSUMming(im=ims, mask=mask, sum_method='all')

    # return created results
    return ismR, shift_map, mask, pincenter


@ deprecated(version='0.1.3', reason='General Interface was changed to 1D-list-operations. Check recon_sheppardSUM.')
def ismR_drawshift(shift_map):
    '''
    Vector-drawing of applied shifts.
    '''
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    V = np.array(shift_map)[:, :, 0]
    U = np.array(shift_map)[:, :, 1]
    # U, V = np.meshgrid(X, Y)
    q = ax.quiver(U, V, cmap='inferno')  # X, Y,
    ax.quiverkey(q, X=0.3, Y=1.1, U=1,
                 label='Quiver key, length = 1', labelpos='E')
    plt.draw()
    plt.plot()
    return fig, ax


@ deprecated(version='0.1.3', reason='General ISM interface changed to 1D and naming convention changed. Check recon_weightedAveraging for more info.')
def ismR_weightedAveraging(imfl, otfl, noise_norm=True, wmode='leave', fmode='fft', fshape=None, closing=2, suppcomp=False):
    '''
    Weighted Averaging for multi-view reconstruction. List implementation so that it can be applied to multiple Data-sets. Needs list of PSFs (list) for different images (=views).
    Note, make sure that:
        -> applied FT is the same in both cases
        -> sum of all PSFs (views) was normalized to 1
        -> symmetric Fourier-transform [normalization 1/sqrt(N_image)] is used

    TODO:
        1) check normalization of IFT
        2) Catch user-errors
        3) generalize for higher dimensions
        4) add covariance-terms

    :PARAM:
    =======
    :imfl:          (NDARRAY) of Fourier-transformed images to work on -> Shape: [VIEWDIM,otherDIMS]
    :otfl:          (NDARRAY) of OTFs
    :noise_norm:    (BOOL)
    :wmode:         (STRING) Which mode to use for OTF in weights -> options: 'real', 'conj', 'abs', 'leave' (DEFAULT)
    :fmode:         (STRING) Transformation used for transformed input images
    :fshape:        (LIST)  Original Shape of Fourier-Transformed image to do right back-transformation in case of noise-normalization
    :suppcomp:      (BOOL) gives back an image with the comparison of the noise-level vs support size

    :OUT:
    =====
    :ismWA:      (IMAGE) recombined image
    :weights:    (LIST) weights calculated for scaling

    '''
    # parameter
    dims = list(range(1, otfl.ndim, 1))
    validmask, _, _, _, _ = otf_get_mask(
        otfl, mode='rft', eps=1e-5, bool_mask=True, closing=closing)

    # In approximation of Poisson-Noise the Variance in Fourier-Space is the sum of the Mean-Values in Real-Space -> hence: MidVal(OTF); norm-OTF by sigma**2 = normalizing OTF to 1 and hence each PSF to individual sum=1
    sigma2_otfl = midVallist(otfl, dims, keepdims=True).real
    weights = otfl / sigma2_otfl
    # weightsn[~validmask] = 0

    # norm max of OTF to 1 = norm sumPSF to 1;
    # sigma2_imfl = mipy.midVallist(imfl,dims,keepdims=True).real
    if wmode == 'real':
        weights = weights.real
    elif wmode == 'conj':
        weights = np.conj(weights)
    elif wmode == 'abs':
        weights = np.abs(weights)
    else:
        pass
    # weights = otfl / sigma2_imfl
    weightsn = nip.image(np.copy(weights))
    weightsn[~np.repeat(validmask[np.newaxis],
                        repeats=otfl.shape[0], axis=-otfl.ndim)] = 0
    # 1/OTF might strongly diverge outside OTF-support -> put Mask
    eps = 0.01
    wsum = np.array(weightsn[0])
    wsum = np.divide(np.ones(weightsn[0].shape, dtype=weightsn.dtype), np.sum(
        weightsn+eps, axis=0), where=validmask, out=wsum)
    # wsum = 1.0/np.sum(weightsn+eps,axis=0)
    # wsum[~validmask]=0

    # apply weights
    ismWA = wsum * np.sum(imfl * weightsn, axis=0)

    # noise normalize
    if noise_norm:
        # noise-normalize, set zero outside of OTF-support
        sigman = np.array(weightsn[0])
        sigman = np.divide(np.ones(weightsn[0].shape, dtype=weightsn.dtype), np.sqrt(
            np.sum(weightsn * weights, axis=0)), where=validmask, out=wsum)
        ismWAN = np.sum(imfl * weightsn, axis=0) * sigman

        # get Poisson-noise for Frequencies > k_cutoff right
        ismWANh = np.real(nip.ift(ismWAN))
        ismWANh = nip.poisson(ismWANh - ismWANh.min(), NPhot=None)
        ismWAN = ismWAN + nip.ft(ismWANh)*(1-validmask)
        ismWAN = nip.ift(ismWAN).real

    # return in real-space
    ismWA = nip.ift(ismWA).real

    # print
    if suppcomp:
        import matplotlib.pyplot as plt
        a = plt.figure()
        plt.plot(x, y,)

    # done?
    return ismWA, ismWAN


@deprecated(version='0.1.3', reason='Interface not updated to 1D-list yet.')
def ismR_shiftmask2D(im, pinsize, mask_shape, pincenter, pinshape):
    '''
    ISM-Reconstruction Toolbox.
    Calculates shift-mask for 2D-detector aray. Assumes that 1st two dimensions are detector-dimensions.

    TODO: catch error by boundary-hit

    PARAM:
    ======
    ...

    OUTPUT:
    ======
    shiftmask

    '''
    #
    if pinshape == 'circle':
        pins = np.min(pinsize)
        # nip.extract((nip.rr((mask_shape)) <= pins)*1,mask_shape,mask_shift)
        shiftmask = (nip.rr(mask_shape) <= pins)*1
        shiftmaskc = np.array(mask_shape//2, np.int)
    elif pinshape == 'rect':
        pinsize = pinsize*2 if not(len(pinsize) == 2) else pinsize
        shiftmask = ((nip.yy(mask_shape, placement='positive') <
                      pinsize[0]) * (nip.xx(mask_shape, placement='positive') < pinsize[1]))*1
        shiftmaskc = np.array(np.array(pinsize)//2, np.int)
        pincenter = [pincenter[m]-shiftmaskc[m] for m in range(len(pincenter))]
    else:
        raise ValueError("Given pinshape not implemented yet.")

    # shift mask ->needs negative shift-vectors for shift2Dby function
    pincenter = [-m for m in pincenter]
    shiftmask = nip.shift2Dby(shiftmask, pincenter)

    # clean mask
    shiftmask = np.abs(shiftmask)
    shiftmask[shiftmask > 0.5] = 1
    shiftmask[shiftmask < 0.5] = 0

    return shiftmask


# %% ---------------------------------------------------------------
# ---                         FROM utility.py                       ---
# ------------------------------------------------------------------


@deprecated(version='0.1.5', reason='Name_scope resolution and passing of arrays etc not working as planned. Changed to call by relay-function.')
def time_me_loop(call_str, myfuncs=None, repeats=1000, averages=100, name_scope=None, **kwargs):
    """Loop based timing of input-function. Helpful to get more concise data for averaging. 

    Parameters
    ----------
    call_str : str
        input function call to be evaluated
    myfuncs : list, optional
        functions called in call_str, by default None
    repeats : int, optional
        number of repitions, by default 1000
    averages : int, optional
        number of evaluations of myfunc per 1 repetition, by default 100
    name_scope : str, optional
        module to import from, by default None


    Returns
    -------
    time_nbrsl : array
        list if timings
    time_stats : array
        list of parameters: min, max, median, mean, variance

    Example
    -------
    >>> def mysq(x):
    >>>     return x*x*x
    >>> def mysum(x,y):
    >>>     return 2*x+y
    >>> def myminus(x,y):
    >>>     return 2*x-np.min([x,y])
    >>> print(mipy.time_me_loop(call_str="mysq(mysum(10,myminus(5,4)))",myfuncs=[mysq,mysum,myminus],repeats=1000,averages=1000,name_scope=__name__)[1][0])
    9.433002560399473e-06
    See Also
    --------
    time_me
    """
    if name_scope is None:
        name_scope = get_caller_function()

    setup_str = time_me_setup_string(myfuncs, name_scope)
    time_nbrsl = []
    for m in range(repeats):
        time_nbrsl.append(timeit.timeit(
            call_str, setup=setup_str, number=averages, globals=kwargs)/averages)

    time_stats = time_me_stats(time_nbrsl)
    return time_nbrsl, time_stats


@deprecated(version='0.1.3', reason='General Change of ISM-interface lead to rather use distances instead of rectangular geometry to allow for arbitrary shapes. See mask_from_dist for more.')
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
