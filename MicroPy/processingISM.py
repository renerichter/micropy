'''
The ISM processing toolbox.
'''
# %% imports

from copy import deepcopy
import numpy as np
import NanoImagingPack as nip
from pandas import DataFrame
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage import binary_closing
from typing import Union

# mipy imports
from .transformations import irft3dz, ft_correct, get_cross_correlations, lp_norm
from .utility import avoid_division_by_zero, findshift, midVallist, pinhole_getcenter, add_multi_newaxis, shiftby_list, subslice_arbitrary, get_slices_from_shiftmap, normNoff
from .inout import stack2tiles, format_list, load_data, fix_dict_pixelsize
from .filters import savgol_filter_nd, stf_basic
from .fitting import extract_multiPSF
from .deconvolution import recon_deconv_list, deconv_test_param_on_list, deconv_test_param, default_dict_tiling


# %%
# ---------------------------------------------------------------
#                   ISM-Reconstruction
# ---------------------------------------------------------------
def dist_from_detnbr(nbr_det, pincenter):
    mgrid = np.meshgrid(np.arange(nbr_det[-2]), np.arange(nbr_det[-1]))
    detpos = np.array([item for item in zip(mgrid[1].flat, mgrid[0].flat)]) - \
        np.unravel_index(pincenter, nbr_det)
    detdist = lp_norm(detpos, p=2, normaxis=(-1,))

    return detdist, detpos

def mask_from_dist(detdist, radius_outer, radius_inner=0.0):
    """Generates 1D-representation of a mask within Lp-distance representation. Use in conjunction with 'mipy.lp_norm' (to calculate distance of detectors (detdist)).

    Parameters
    ----------
    detdist : list
        distances of detector elements w.r.t. to a central pixel
    radius_outer : float
        size of outer radius
    radius_inner : float, optional
        size of inner radius, by default 0 (=smallest)

    Returns
    -------
    sel_mask : bool-list
        selection mask

    See Also
    --------
    mipy.lp_norm, mipy.pinhole_getcenter
    """
    # select
    sel_mask = (detdist >= radius_inner) * (detdist <= radius_outer)
    return sel_mask

def get_ring_radiis(detdist):
    '''
    See Also:
    ---------
    select_pinhole_radius
    '''
    ring_radii = np.unique(detdist)
    radii_all_comb = list(iterprod(ring_radii, ring_radii))
    radii_unique_comb = [m for m in radii_all_comb if m[0] <= m[1]]
    return radii_unique_comb

def get_pincenter(im, nbr_det, imfunc=np.max, rfunc=np.round, im_axes=(-2, -1), com_axes=(-2, -1)):
    '''only 2D for now; im.shape=[arb_dims,DETYX,Y,X]'''
    im_axesh = np.mod(im_axes, im.ndim)
    rshape = np.array([(mshape if not m in im_axesh else 0) for m, mshape in enumerate(im.shape)])
    rshape = rshape[rshape > 0]
    rshape = list(rshape[:-1])+list(nbr_det)
    im_cpsearch = np.reshape(imfunc(im, axis=im_axes), rshape)
    pincenter_CoM_raw = center_of_mass(im_cpsearch, com_axes=com_axes, im_axes=im_axes)
    pincenter_CoM = rfunc(pincenter_CoM_raw).astype('int')
    return pincenter_CoM[0]*nbr_det[1]+pincenter_CoM[1]

def recon_genShiftmap(im, pincenter, im_ref=None, nbr_det=None, pinmask=None, shift_method='nearest', shiftval_theory=None, roi=None, shift_axes=None, cross_mask_len=[4,4], period=[],factors=[], sg_para=[13,2,0,1,'wrap'],subpix_lim=2,printmap=False):
    """Generates Shiftmap for ISM SheppardSUM-reconstruction.
    Assumes shape: [Pinhole-Dimension , M] where M stands for arbitrary mD-image.
    Fills shift_map from right-most dimension to left, hence errors could be evoked in case of only shifts along dim=[1,2] applied, but array has spatial dim=[1,2,3]. Hence shifts will be applied along [2,3] with their respective period factors.

    Parameters
    ----------
    im : image
        Input Image which has the pinhole-dimension as its 0th dimension
    pincenter : int
        central detector
    nbr_det : array, optional
        Number and Shape of pinhole dimension, eg [2,3] for a 2x3. Needed for shift_method 'nearest', by default None
    pinmask: array, optional
        Pinhole-mask needed for mask-based shift calculation, by default None
    shift_method : str, optional
        Method to be used for shift-calculation. , by default 'nearest'
            'nearest': compare center pinhole together with 1 pix-below and 1 pix-to-the-right pinhole; assumes rect detector-grid for now and needs nbr_det
            'mask': all pinholes that reside within mask ; needs pinmask
            'complete': calculate shifts for all detectors
            'theory': generates shifts from shiftval_theory provided
    shiftval_theory: list, optional
        if shift_method 'theory' is active, generates shiftmap from these factors. In pixel-dimensions und unit-vector notation,  hence eg [[-0.5,0],[0,-0.5]] generates "back-shift" for pixel-reassignments by half the distance to the pincenter, by default None
    roi : list, optional
        subslice of image (spatial coordinates) to be used for shift-finding -> see subslice_arbitrary for more info, by default None
    printmap : bool, optional
        print shift-vectors as quiver map, by default False

    Returns
    -------
    shift_map : array
         shift-map for all pinhole-pixels

    Raises
    ------
    ValueError
        Raises ValueError if chosen shift_method is not implemented

    See Also
    --------
    recon_sheppardSUM, subslice_arbitrary

    TODO
    ----
    1) fix dimensionality problem of factors generation (eg by taking in a 'shift_axis' parameter)
    2) 'cross' method not implemented for arbitrary shape
    3) add general detector-position calculation for arbitrary period -> for now: airyDetector shape is hardcoded
    """
    # sanity
    if shift_axes is None:
        shift_axes = np.arange(1, im.ndim)
    if not type(shift_axes) == np.ndarray:
        shift_axes = np.mod(np.array(shift_axes), im.ndim)
    calc_factors = True if factors == [] else False

    if im_ref is None:
        im_ref = savgol_filter_nd(im[pincenter], sg_axis=-im.ndim+shift_axes, sg_para=sg_para, direct=False)

    if roi is not None:
        im = subslice_arbitrary(im, roi, axes=np.arange(im.ndim))

    if shift_method in ['nearest', 'cross','theory']:
        # convert pincenter to det-space
        pcu = np.unravel_index(pincenter, nbr_det)

        # find shift per period-direction
        shiftvec = []
        for m in range(len(nbr_det)):
            # calculate shift, but be aware to not leave the array
            perioda = int(np.prod(np.array(nbr_det[(m+1):]))) if period == [] else period[m]
            if shift_method == 'nearest':
                shifth, _, _, _ = findshift(im_ref,
                                            im[np.mod(pincenter+perioda, im.shape[0])],  10**subpix_lim)
            elif shift_method == 'cross':
                cmask=np.zeros(im.shape[0],dtype='bool')
                direct_pixel_dist=np.arange(-cross_mask_len[m],cross_mask_len[m]+1)#+cmask.shape[m]
                pixel_offset=np.prod(nbr_det[-(m+1):])//2
                cmask[np.mod(pixel_offset+(direct_pixel_dist+nbr_det[m]//2)*perioda,im.shape[0])]=True
                shifth,_,_=recon_genShiftmap(im, pincenter, im_ref=im_ref, nbr_det=nbr_det, pinmask=cmask, shift_method='mask', shiftval_theory=None, roi=None, shift_axes=None, printmap=False)
                shifth=np.mean(avoid_division_by_zero(shifth[cmask],direct_pixel_dist[:,np.newaxis]),axis=0)
            else:
                shifth = shiftval_theory[m]
            shiftvec.append(np.round(shifth,subpix_lim))

            # create factors
            if calc_factors:
                factors.append(nip.ramp(nbr_det, ramp_dim=m,
                                    placement='corner').flatten()-pcu[m])

        # generate shiftvec -> eg [eZ,eY,eX] where eZ=[eZ1,eZ2,eZ3]
        shiftvec = np.array(shiftvec)
        if calc_factors:
            shiftvec = np.array([shiftvec[:, m] for m in shift_axes-1]).T
            factors = np.array(factors).T

            # sanity for arbitrary dimensionality -> find non-shifted dimensions and add to factors --> need to check logic again
            #shiftsums = np.sum(abs(shiftvec), axis=0)
            #shiftfree_dim = shift_axes[list(np.where(shiftsums == 0)[0])]
            #zero_shifts = np.arange(1,im.ndim)
            # zero_shifts[shift_axes-1]
            # if len(zero_shifts) > 0:
            #    factors = add_multi_newaxis(factors, zero_shifts)

            # generate shifts for whole array (elementwise-multiplication)
            shift_map = np.matmul(factors, shiftvec)
            #if not pinmask is None:
            #    shift_map = shift_map[pinmask]
        else:
            shift_map=np.array([m[0]*shiftvec[0]+m[1]*shiftvec[1] for m in factors])
    elif shift_method == 'mask':
        imh = im[pinmask]
        # shift_map = np.array([[0, ]*im.ndim]*im.shape[0])
        shift_maph = []
        for m in range(imh.shape[0]):
            shifth, _, _, _ = findshift(im_ref,
                                        imh[m], 100)
            shift_maph.append(shifth)
        shift_maph = np.array(shift_maph)
        shift_map = np.zeros([im.shape[0], ]+[shift_maph.shape[-1], ])
        shift_map[pinmask] = np.array(shift_maph)

    elif shift_method == 'complete':
        shift_map = []
        for m in range(im.shape[0]):
            shift_maph, _, _, _ = findshift(im_ref,
                                            im[m], 100)
            shift_map.append(shift_maph)
        shift_map = np.array(shift_map)
    else:
        raise ValueError("Shift-method not implemented")

    # print shift-vectors
    rdict = {'im_ref':im_ref}
    if printmap:
        sm = np.reshape(shift_map, nbr_det + list(shift_map[0].shape))
        if sm.shape[-1] > 2:
            useaxes = [-2, -1]
        rdict['figS'], rdict['axS'] = recon_drawshift(sm, useaxes=useaxes)

    # done?
    return shift_map, rdict


def recon_sheppardShift(im, shift_map, method='parallel', use_copy=False, shift_pad=True):
    """Does shifting for ISM-SheppardSum.

    Parameters
    ----------
    im : image
        image to be shifted
    shift_map : array
        list of nD shifts to be applied along the 0th dimension of im
    method : str, optional
        Methods to be used for shifting, by default 'parallel'
            'iter': shifts by iteration
            'parallel': shifts all in parallel
    use_copy : bool, optional
        whether to us copy of im, by default False

    Returns
    -------
    imshifted : image
        shifted image

    See Also
    --------
    recon_sheppardSUM

    TODO
    ----
    1) implement iterative shifting for nD-images by assuring shiftmap has same dimensionality as image.
    """
    # work on copy to keep original?
    im_origshape=np.copy(im.shape)
    imshifted = nip.image(np.copy(im)) if use_copy else im

    # method to be used
    if method == 'parallel':
        imshifted, _ = shiftby_list(im, shifts=shift_map, listaxis=0)       
    else:
        #prepare arrays
        shift_axes=tuple(np.arange(-shift_map.shape[1],0))
        im=np.squeeze(im)
        imshifted=np.squeeze(imshifted)
        dimdiff = im.ndim-1 - shift_map.shape[-1]
        if dimdiff > 0:
            shifts = np.zeros(list(shift_map.shape[:-1])+[shift_map.shape[-1]+dimdiff,])
            shifts[:,-shift_map.shape[-1]:]=shift_map
        else:
            shifts=shift_map

        #shift
        for m in range(shift_map.shape[0]):
            imshifted[m] = nip.shift(im=im[m],delta=shifts[m],dampOutside=True)#shifts,axes=shift_axes#shift_map

        # fixup shape
        imshifted = np.reshape(imshifted,im_origshape)
        imshifted.pixelsize = im.pixelsize

    # done
    return imshifted


def recon_ism_unitshifts_fromline(im, nbr_det, pincenter, mask_range2):
    pinc = np.unravel_index(pincenter, nbr_det)
    _, detpos = dist_from_detnbr(nbr_det, pincenter)
    hline = np.zeros(nbr_det, dtype='bool')
    vline = np.copy(hline)
    vline[pinc[0]-mask_range2[0]:pinc[0]+mask_range2[0]+1, pinc[1]] = True
    hline[pinc[0], pinc[1]-mask_range2[1]:pinc[1]+mask_range2[1]+1] = True
    vshift, _ = recon_genShiftmap(
        im=im, im_ref=im[pincenter], pincenter=pincenter, nbr_det=nbr_det, pinmask=vline.flatten(), shift_method='mask', shiftval_theory=None, roi=None, shift_axes=None)
    hshift, _ = recon_genShiftmap(
        im=im, im_ref=im[pincenter], pincenter=pincenter, nbr_det=nbr_det, pinmask=hline.flatten(), shift_method='mask', shiftval_theory=None, roi=None, shift_axes=None)
    basic_shifts = [np.mean(avoid_division_by_zero(vshift, detpos)[vshift[:, 0] != 0], axis=0), np.mean(
        avoid_division_by_zero(hshift, detpos)[hshift[:, 1] != 0], axis=0)]

    retdict = {'hline': hline, 'vline': vline, 'hshift': hshift, 'vshift': vshift}

    return basic_shifts, retdict

def recon_sheppardSUMming(im, pinmask=None, sum_method='normal'):
    """Does the actual summing operation for the sheppard sum routine.

    Parameters
    ----------
    im : image
        pre-(sheppard) shifted image
    pinmask : bool-list, optional
        mask to be used for summing operation, by default None
    sum_method : str, optional
        method to be used for sheppard summing, by default 'normal'
            'normal': sum without any weighting
            'invvar': inverse variance weighted summing

    Returns
    -------
    imshepp: image
        sheppard shifted imag
    weights : image
        weights used for summing

    See Also
    --------
    recon_sheppardSUM
    """
    # sum over all pinholes if different pinmask not provided
    if pinmask is None:
        pinmask = [True, ]*im.shape[0]
    weights = []

    # su
    if sum_method == 'invvar':
        imvar = np.var(im[pinmask], axis=tuple(np.arange(im.ndim-1)+1), keepdims=True)
        eps = 1e-5
        eps = (1+1j)*eps if imvar.dtype == np.complex else eps
        weights = 1 / (imvar+eps)
        weights /= np.sum(weights)
        # print(np.sum(weights))
        # nip.v5(weights)
        imshepp = np.sum(im[pinmask] * weights, axis=0)
    else:
        imshepp = np.sum(im[pinmask], axis=0)

    # done?
    return imshepp, weights

def airy_factors(indiv_rings=False):
    '''Assuming to be multiplied with unit vectors of order ey,ex and shape [Y,X]'''
    ring_dict = {'R0': [[0, 0], ],
                 'R1': [[0,  1], [-1,  1], [-1,  0], [0, -1], [1, -1], [1,  0]],
                 'R2': [[1,  1], [0,  2], [-1,  2], [-2,  2], [-2,  1], [-2,  0],
                        [-1, -1], [0, -2], [1, -2], [2, -2], [2, -1], [2,  0]],
                 'R4': [[2,  1], [1,  2], [-1,  3], [-2,  3], [-3,  2], [-3,  1],
                        [-2, -1], [-1, -2], [1, -3], [2, -3], [3, -2], [3, -1], [3,  0]], }
    ring_dict['Rlen'] = [len(ring_dict['R0']), len(ring_dict['R1']),
                         len(ring_dict['R2']), len(ring_dict['R3'])]
    rings_complete = np.array([*ring_dict['R0'], *ring_dict['R1'],
                              *ring_dict['R2'], * ring_dict['R3']])
    return rings_complete, ring_dict

def recon_widefield(im, detaxes):
    '''
    Scanned widefield reconstruction of ISM-data.

    PARAM:
    =====
    :im:        input nD-image
    :detaxes:   (TUPLE) axes of detector (for summing)

    OUTPUT:
    =======
    :out:

    EXAMPLE:
    =======
    im = mipy.shiftby_list(nip.readim(),shifts=np.array([[1,1],[2,2],[5,5]]))
    imwf = mipy.ismR_widefield(im,(0))

    '''
    return np.sum(im, axis=detaxes)


def recon_confocal(im, detdist, pinsize=0, pincenter=None, pinmask=None):
    """Confocal Processing of ISM-Data. Assumes ISM-Direction as 0th-dimension and nD-afterwards for arbitrary image data.

    Ways for further change in the implementation:
    1) Pair different regions before shifting (ismR_pairRegions), then find regional-shifts (ismR_getShiftmap) and finally sheppardSum them.
    2) sheppardSUM only until/from a limit to do different reconstruction on other part (eg Deconvolve within deconvMASK and apply sheppardSUM on outside)


    Parameters
    ----------
    im : image
        List of nD input images
    detdist : list
        Detector distance with respect to a center detector (assumes to be precalculated)
    pinsize : int, optional
        Size of pinhole (if pinmask not provided), by default 0
    pincenter : int, optional
        central detector, by default None
    pinmask : list, optional
        1D-list of pinholes to be used (should have the same 0th-dimension as input im), by default None

    Returns
    -------
    im_conf : image
        Calculated Confocal image
    pinmask : list
        Used pinhole-mask

    Example
    -------
    >>> obj = nip.readim()
    >>> psf = mipy.calculatePSF(obj)
    >>> psfr, _, _, shiftlist = mipy.calculatePSF_ism(
    psf, psf, shift_offset=[[10, 0], [0, 5]], shift_axes=[-2, -1], nbr_det=[2,3], fmodel='fft', faxes=[-2, -1], pinhole=None)
    >>> im_conf, pinmask = recon_confocal(psfr,10)

    """
    # sanity
    if pincenter is None:
        pincenter = np.squeeze(np.where(detdist == 0))

    # get pinmask
    if pinmask is None:
        pinmask = mask_from_dist(detdist=detdist, radius_outer=pinsize, radius_inner=0.0)

    # calculate confocal image
    im_conf = np.squeeze(np.sum(im[pinmask], axis=0))

    # done?
    return im_conf, pinmask


def ism_recon(im, method='wf', **kwargs):
    '''
    General ISM-reconstruction wrapper. Takes care of necesary preprocessing for the different techniques.

    :PARAM:
    =======
    :im:            (IMAGE)     Input image; assume nD for now, but structure should be like(pinholeDim=pd) [pd,...(n-3)-extraDim...,Y,X]
    :method:        (STRING)    method to be used for reconstruction, options are:
                                'wf':       reconstructs as laser-widefield
                                'conf':     confocal reconstruction
                                'shepp':    sheppardSUM
                                'wAVG':     weighted averaging

    TODO
    ----
    1) bring up-to-date for 1D-interface!!

    '''
    # sanity check on kwargs to find the necessary pinhole parameters
    pincenterFIND = 'sum' if kwargs.get(
        'pincenterFIND') is None else kwargs.get('pincenterFIND')
    pinsize = im.shape[0] if kwargs.get(
        'pinsize') is None else kwargs.get('pinsize')
    pinshape = 'circ' if kwargs.get(
        'pinshape') is None else kwargs.get('pinshape')
    pindim = kwargs.get('pindim')

    if kwargs.get('pincenter') is None:
        pincenter, im_detproj = pinhole_getcenter(im, method=pincenterFIND)

    if kwargs.get('pinmask') is None:
        pass

    # get pinhole center
    if pincenter == [] or pincenter is None:
        pincenter, mask_shift, mask_shape = pinhole_getcenter(
            im, pincenterFIND)

    # get pinhole mask
    if pinsize == [] or pincenter is None:
        mask = nip.image(np.ones(im.shape[:2]))
    else:
        # test whether circular pinhole
        pinsize = pinsize*2 if len(pinsize) == 1 else pinsize
        pinshape = 'circ' if pinsize[-1] == pinsize[-2] else 'rect'

        # generate pinhole
        if pinshape == 'rect':
            mask_x = (abs(nip.xx(pindim, placement='center'))
                      <= pinsize[-1]//2)*1
            mask_y = (abs(nip.yy(pindim, placement='center'))
                      <= pinsize[-2]//2)*1
            mask = mask_x + mask_y
            mask[mask < (np.max(mask_x) + np.max(mask_y))] = 0
        else:
            mask = (nip.rr(pindim, placement='center') <= pinsize[-1]//2)*1

    # call routines

    # done?
    return pinsize


def recon_sheppardSUM(im, nbr_det, pincenter, im_ref=None, shift_method='nearest', shift_map=[], shift_roi=None, shiftval_theory=None, shift_axes=(-2,-1), shift_extract=False, pinmask=None, pinfo=False, sum_method='normal', shift_style='parallel', shift_use_copy=True, ret_nonSUM=False,cross_mask_len=[4,4], period=[],factors=[],subpix_lim=2):
    """Calculates sheppardSUM on image (for arbitrary detector arrangement). The 0th-Dimension is assumed as detector-dimension and hence allows for arbitrary, but flattened detector geometries.

    The algorithm does:
    1) find center of pinhole for all scans using max-image-peak (on center should have brightest signal) if not particularly given
    2) define mask/pinhole
    3) find shifts between all different detectors within mask ->  result used in sample-coordinates
    4) shift back

    Parameters
    ----------
    im : image
        image with dimensions [N , M] where N is the 1D-pinhole dimension of N elements and M stands for arbitrary mD-image
    nbr_det : list
        number and shape of detectors, eg [3,4]
    pincenter : int
        center-pinhole position (e.g. if nbr_det=[5,6], then eg pincenter=14)
    shift_method : str, optional
        Method to be used to find the shifts between the single detector and the center pixel -> see recon_genShiftmap, by default 'nearest'
    shift_map : list, optional
        list for shifts to be applied on channels -> needs to have same length as im.shape[0], by default []
    shiftval_theory : list, optional
        if shift_method 'theory' is active, generates shiftmap from these factors, eg [[0.5,0],[0,0.5]] --> see recon_genShiftmap, by default None
    pinmask : bool-list, optional
        selection/shape of pinholes to be used for calculation, by default None
    pinfo : bool, optional
        print info, by default False
    sum_method : str, optional
        method to be used for sheppard summing, by default 'normal'
            'normal': sum without any weighting
            'invvar': inverse variance weighted summing

    Returns
    -------
    ismR : image
        reassinged ISM-image (within mask)
    shift_map : array
        used shift_map
    pinmask : list
        list of bools (=elements of 0th-dim) which where used
    pincenter : int
        position of center-point of mask/pinhole in 0th-dim
    weights : image
        weights used for summing (empty, if sum_method=='normal')

    See Also
    --------
    recon_genShiftmap, recon_sheppardShift, recon_sheppardSUMming,
    """
    # find shift-list -> Note: 'nearest' is standard
    if shift_map == []:
        shift_map, rdict = recon_genShiftmap(
            im=im, im_ref=im_ref, pincenter=pincenter, nbr_det=nbr_det, pinmask=pinmask, shift_method=shift_method, shiftval_theory=shiftval_theory, roi=shift_roi, shift_axes=shift_axes,cross_mask_len=cross_mask_len,period=period,factors=factors,subpix_lim=subpix_lim)
    else:
        rdict={'im_ref':[]}

    # apply shifts
    if not pinmask is None:
        im = im[pinmask]
        shift_map = shift_map[pinmask] if not len(shift_map) == np.sum(pinmask) else shift_map
    ims = recon_sheppardShift(im, shift_map, method=shift_style, use_copy=shift_use_copy)

    # different summing methods
    imshepp, weights = recon_sheppardSUMming(im=ims, pinmask=None, sum_method=sum_method)
    
    shift_slices=get_slices_from_shiftmap(im=imshepp,shift_map=shift_map, shift_axes=shift_axes)
    if shift_extract:
        imshepp = imshepp[shift_slices]

    # info for check of energy conservation
    if pinfo:
        print("~~~~~\t Results of SheppardSUM-routine:\t~~~~~")
        print(
            f"SUM(im)={np.sum(im)}\nim.shape={im.shape}\nSUM(imshifted)-SUM(im)={np.sum(ims)-np.sum(im)}\nSUM(imshepp)-SUM(im[pinmask])={np.sum(imshepp)-np.sum(im)}\nShift-Operations={np.sum(pinmask)}")
        print("~~~~~\t\t\t\t\t\t~~~~~")

    res_dict = {'shift_map': shift_map, 'pinmask': pinmask,
                'pincenter': pincenter, 'weights': weights, 'im_ref':rdict['im_ref'], 'shift_slices':shift_slices}

    res_dict['ims'] = ims if ret_nonSUM else None

    # return created results
    return imshepp, res_dict

def sheppardSUM_analytical_factors(λ1=488, λ2=515, na=1.3, mphot=1, printout=False):
    # position and insecurity of reassign value
    mu = gaussProd_mu(λ1/np.sqrt(mphot), λ2)
    sigma2 = gaussProd_sigma(0.21*λ1/(na*np.sqrt(mphot)), 0.21*λ2/na)

    if printout:
        print(
            f"For the linear product of beams with λ(Iex)={λ1} and λ(Idet)={λ2} the reassignment factor is:\nd={mu}\ninsecurity σ={np.sqrt(sigma2)}")

    return mu, sigma2

def recon_sheppardSUM_list(ims,im_ref=None,nbr_det=[3,3],pincenter=0,shift_method='nearest',pinmask=None,**kwargs):
    '''Wrapper for recon_sheppardSUM on list-dimension 0
    
    TODO:
    1) Update docstring
    2) add arbitrary list dimension
    '''    
    im_shepp = nip.image(
        np.zeros([ims.shape[0], ]+list(ims.shape[-2:]), dtype=ims.dtype))
    im_shepp_dict = [{}, ]*ims.shape[0]
    for m, mIm in enumerate(ims):
        im_shepp[m], im_shepp_dict[m] = recon_sheppardSUM(mIm, im_ref=im_ref, nbr_det=nbr_det,pincenter=pincenter, shift_method=shift_method, pinmask=pinmask, **kwargs)
    
    # done?
    return im_shepp, im_shepp_dict


def recon_weightedAveraging_testmodes(**kwargs):
    """Wrapper to test all options of weighted Averaging routine.

    Returns
    -------
    ismWAl, weightsnl, ismWANl: lists
        concatenated lists of output of recon_weightedAveraging

    See Also
    --------
    recon_weightedAveraging
    """
    # Sanity
    if 'wmode' in kwargs:
        del kwargs['wmode']

    # preparation
    wmodes = ['real', 'imag', 'conj', 'abs', 'leave']
    ismWAl, weightsnl, ismWANl = [], [], []

    # loop over entries
    for m, wmode in enumerate(wmodes):
        ismWA, weightsn, ismWAN = recon_weightedAveraging(wmode=wmode, **kwargs)
        ismWAl.append(ismWA), weightsnl.append(weightsn), ismWANl.append(ismWAN)

    ismWAl = np.squeeze(nip.image(np.array(ismWAl)[:, np.newaxis]))
    ismWANl = np.squeeze(nip.image(np.array(ismWANl)[:, np.newaxis]))

    # done?
    return ismWAl, weightsnl, ismWANl


def recon_weightedAveraging(imfl, otfl, pincenter, noise_norm=True, wmode='conj', fmode='fft', fshape=None, faxes=(-2, -1), dtype_im=np.float32, closing=2, use_mask=True, mask_eps=1e-4, reg_reps=0, reg_aeps=0, add_ext_noise=False, pixelsize=None, backtransform=True, norm='ortho'):
    """Weighted Averaging for multi-view reconstruction. List implementation so that it can be applied to multiple Data-sets. Needs list of PSFs (list) for different images (=views).
    Note, make sure that:
        -> applied FT is the same in both cases
        -> sum of all PSFs (views) is 1
        -> symmetric Fourier-transform [normalization 1/sqrt(N_image)] is used

    Parameters
    ----------
    imfl : image
         Fourier-transformed images to work on -> Shape: [VIEWDIM,otherDIMS]
    otfl : image
        Array of OTFs -> Shape: [VIEWDIM,otherDIMS]
    pincenter : int
        central pinhole (or image) -> used if OTF-support is calculated for all OTFs from the central one
    noise_norm : bool, optional
        True if noise normalized weighting should be calculated as well, by default True
    wmode : str, optional
         mode to use for OTF in weights, by default 'conj'
            'real': take real-part of weights for calculation
            'imag': take imaginary-part of weights for calculation
            'conj': take conjugate of weights for calculation
            'abs': take abs of weights for calculation
            'leave': does not alter anything
    fmode : str, optional
        Transformation used for transformed input images, by default 'fft'
            'fft': Fourier-Trafo
            'rft': Real fourier-trafo
    fshape : list, optional
        Original Shape of Fourier-Transformed image to do right back-transformation in case of noise-normalization, by default None
    closing : int, optional
        Closing to be used for closed OTF-support -> see otf_get_mask for more infos, by default 2
    suppcomp : bool, optional
        gives back an image with the comparison of the noise-level vs support size, by default False
    mask_eps : float, optional
        relative intensity used to calculate OTF support
    div_eps : float, optional
        relative value to add to weightsn to avoid division by zero, by default 1e-5

    Returns
    -------
    ismWA : image
        Reconstructed weighted averaged Image
    weights: image
        calculated weights per spatial position
    ismWAN : image
        noise normalized image (if noise_norm was set to true)

    Example
    -------
    obj=nip.extract(mipy.generate_spokes_target([124,124],pixelsize=[40,40]),[128,128])
    ashift=mipy.gen_shift(method='uvec', uvec=[[2,0.5],[0.5,2]],nbr=[1,3])
    psf=nip.psf(obj)
    psfs,_=mipy.shiftby_list(psf,ashift)
    im=nip.poisson(nip.convolve(obj,psfs),NPhot=10)
    ismWA, weights, ismWAN=mipy.recon_weightedAveraging(nip.ft2d(im),nip.ft2d(psfs),pincenter=1)

    See Also
    --------
    recon_sheppardSUM

    TODO
    ----
    1) add flag for support calculation -> for in-center OTF (and use for others) vs individually
    2) generalize for higher dimensions
    3) add covariance-terms

    """
    # In approximation of Poisson-Noise the Variance in Fourier-Space is the sum of the Mean-Values in Real-Space -> hence: MidVal(OTF); norm-OTF by sigma**2 = normalizing OTF to 1 and hence each PSF to individual sum=1
    dims = list(range(1, otfl.ndim, 1))
    #sigma2_otfl = midVallist(otfl, dims, keepdims=True).real
    sigma2_imfl = midVallist(imfl, dims, keepdims=True).real

    # noise-normalize otfs
    #otfl = otfl / np.sqrt(sigma2_otfl)
    weights = otfl / sigma2_imfl  # / np.sqrt(sigma2_otfl)

    # get OTF-support(=validmask)
    if use_mask:
        _, validmask, _, _, _ = otf_get_mask(
            weights, center_pinhole=pincenter, mode='fmode', eps=mask_eps, bool_mask=True, closing=closing, pixelsize=pixelsize)
    else:
        validmask = np.ones(weights.shape, dtype=bool)

    if wmode == 'real':
        weights = weights.real
    elif wmode == 'imag':
        weights = weights.imag
    elif wmode == 'conj':
        weights = np.conj(weights)
    elif wmode == 'abs':
        weights = np.abs(weights)
    else:
        pass

    # set weights outside of support to 0
    if not validmask.shape == weights.shape:
        validmask = add_multi_newaxis(validmask, newax_pos=[0, ]*(weights.ndim-validmask.ndim))
        rep_factors = np.array(weights.shape)//np.array(validmask.shape)
        validmask = nip.repmat(validmask, rep_factors)
    weights[~validmask] = 0

    # apply weights
    ismWA = np.sum(imfl * weights, axis=0)

    # noise normalize
    if noise_norm:
        # 1/OTF might strongly diverge outside OTF-support -> put Mask
        sigma_nn = np.sqrt(np.sum(weights * np.conj(weights), axis=0))
        if reg_aeps == 0:
            reg_aeps = np.max(sigma_nn)*reg_reps
            reg_aeps = reg_aeps * (1+1j) if weights.dtype == np.complex else reg_aeps
        # wsum = np.ones(weights[0].shape, dtype=weights.dtype)
        # wsum = np.divide(wsum, np.sum(weights, axis=0) + reg_aeps, where=validmask[0], out=wsum)

        # noise normalize
        ismWAN = np.zeros(ismWA.shape, dtype=ismWA.dtype)
        ismWAN = np.divide(ismWA, sigma_nn + reg_aeps, where=validmask[0], out=ismWAN)

        otfWAN = np.sqrt(np.sum(otfl*np.conj(otfl)/sigma2_imfl, axis=0))  # /sigma2_otfl
        otfWAN[~validmask[0]] = 0.0

        # nip.v5(nip.catE(ismWA, ismWAN))
        # ismWAN_rs=np.abs(nip.ift(ismWAN))

        # get poisson noise for freqencies outside of support -> not working!
        # if add_ext_noise:
        #    ismWAN_syn=nip.poisson(ismWAN_rs.astype('int')).astype('float')
        #    ismWAN_syn=nip.ft(ismWAN_syn)
        #    ismWAN_syn /= midVallist(ismWAN_syn, np.arange(ismWAN_syn.ndim), keepdims=True).real
        #    ismWAN_syn1=np.zeros(ismWAN.shape, dtype=ismWAN.dtype)
        #    ismWAN_syn1[validmask[0]]=ismWAN[validmask[0]]
        #    ismWAN_syn1[~validmask[0]]=ismWAN_syn[~validmask[0]]
        #    ismWAN=np.abs(nip.ift(ismWAN_syn1))
        # else:
        #    ismWAN=ismWAN_rs
        if backtransform:

            ismWAN = ft_correct(ismWAN, faxes, im_shape=fshape, mode=fmode,
                                dir='bwd', dtype=dtype_im, norm=norm, use_abs=True)
            psfWAN = ft_correct(otfWAN, faxes, im_shape=fshape, mode=fmode,
                                dir='bwd', dtype=dtype_im, norm=norm, use_abs=True)
        else:
            psfWAN = otfWAN
    else:
        ismWAN = None
        psfWAN = None

    # return in real-space
    if backtransform:
        ismWA = ft_correct(ismWA, faxes, im_shape=fshape, mode=fmode,
                           dir='bwd', dtype=dtype_im, norm=norm, use_abs=True)

    # done?
    return ismWA, weights, ismWAN, psfWAN


def recon_wiener(imFT: nip.image, otf: nip.image, use_generalized: bool = True, use_mask: bool = False, eps_mask: float = 1e-8, closing: Union[int, np.ndarray] = 2, reg_aeps: float = 0.0, reg_reps: float = 1e-5, faxes: tuple = (-2, -1), pincenter: int = None, multiview_dim: int = 0):
    # use_mask
    if use_mask:
        _, validmask, _, _, _ = otf_get_mask(
            otf, center_pinhole=pincenter, mode='fmode', eps=eps_mask, bool_mask=True, closing=closing, pixelsize=imFT.pixelsize)
    else:
        validmask = np.ones(otf.shape, dtype=bool)

    # get filter size
    otf2 = np.abs(otf*np.conj(otf))

    # get regularization
    if use_generalized:
        regval = np.max(otf2)*reg_reps if reg_aeps == 0 else reg_aeps
    else:
        regval = 1
        print("NOT IMPLEMENTED YET!")

    # calculate filter
    filter_func = np.zeros(otf.shape, dtype=otf.dtype)
    filter_func = np.divide(np.conj(otf), otf2 + regval, where=validmask, out=filter_func)

    # apply
    im_filtered = np.abs(nip.ift(filter_func*imFT, axes=faxes))
    res_dict = {'filter_func': filter_func, 'validmask': validmask,
                'reg_aeps': reg_aeps, 'reg_reps': reg_reps, 'mask_eps': eps_mask}

    if not pincenter is None:
        im_filtered = np.sum(im_filtered, axis=multiview_dim)

    # done?
    return im_filtered, res_dict


def recon_drawshift(shift_map, useaxes=[-2, -1]):
    '''
    Vector-drawing of applied shifts.
    '''
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    V = np.squeeze(np.array(shift_map)[:, :, -2, useaxes[0]:useaxes[1]])
    U = np.squeeze(np.array(shift_map)[:, :, -1, useaxes[0]:useaxes[1]])
    # U, V = np.meshgrid(X, Y)
    q = ax.quiver(U, V, cmap='inferno')  # X, Y,
    ax.quiverkey(q, X=0.3, Y=1.1, U=1,
                 label='Quiver key, length = 1', labelpos='E')
    plt.draw()
    plt.plot()
    return fig, ax

def calculate_reference(im, avg_mode=np.median, sg_axis=[-1, -2], sg_para=[13, 2, 0, 1, 'wrap']):
    '''
    sg_axis=None to ignore usage
    avg_mode=np.median, np.mean, ... -> if None should be applied, use: lambda x,axis:x
    '''
    imm = avg_mode(im, axis=0)

    if not sg_axis is None:
        imm = savgol_filter_nd(imm, sg_axis=sg_axis, sg_para=sg_para)

    return imm

# %%
# ---------------------------------------------------------------
#                   THICKSLICE with LINEAR UNMIXING
# ---------------------------------------------------------------


def otf_fill(otf, fill_method='rot'):
    '''
    Fills the 3D-otf of an nD-Stack by copying from a calculated region. For now, assumes that lower z-half (full-xy) is calulated. Further, the input image has the shape: [...n-dim...,Z,Y,X].

    :PARAMS:
    ========
    :otf:           3D OTF of shape [Z,Y,X]
    :fill_method:   method used to fill upper support. 'rot', 'ax' are implemented.

    OUT:
    ====
    :otfh:          filled OTF
    '''
    # get shape paremeters
    otfdim = otf.ndim
    if otf.ndim > 3:
        otf = np.reshape(otf, [np.prod(otf.shape[:-3]),
                               otf.shape[-3], otf.shape[-2], otf.shape[-1]])
    else:
        otf = otf[np.newaxis]
    otfh = nip.image(np.zeros(otf.shape, dtype=np.complex_))
    otfh_c = np.array(np.array(otf.shape)/2.0, dtype=np.uint)
    otfhm = np.mod(np.array(otf.shape), 2)
    # copy lower half of original otf
    otfh[:, otfh_c[1]+1:] = otf[:, otfh_c[1]+1:]
    # copy data into upper half by chosen method
    if fill_method == 'rot':
        # pi-rotate around origin and conjugate
        # take side-sizes into account -> if even:
        # np.rot90(np.conj(otfh[:,otfh_c[-3]+1:]),2)
        otfh = np.conj(otfh[:, ::-1, ::-1, ::-1])
    if np.mod(otfh.shape[-1], 2) == 0:
        otfh[:, :, :, 1:otfh_c[-1]] = otfh[:, :, :, 0:otfh_c[-1]-1]
        otfh[:, :, :, 0] = 0
    if np.mod(otfh.shape[-2], 2) == 0:
        otfh[:, :, 1:otfh_c[-2]] = otfh[:, :, 0:otfh_c[-2]-1]
        otfh[:, :, 0] = 0
    elif fill_method == 'ax':
        # reflect around kx/ky axis and conjugate
        otfh[:, 0:otfh_c[-3]-1] = np.conj(otfh[:, otfh_c[-3]+1:])[::-1]
    else:
        raise ValueError('Chosen method not implemented yet.')
    # shift for all
    if np.mod(otfh.shape[-3], 2) == 0:
        otfh[:, 1:otfh_c[-3]] = otfh[:, 0:otfh_c[-3]-1]
        otfh[:, 0] = 0

    otfh[:, otfh_c[1]:] = otf[:, otfh_c[1]:]

    # reduce dimensions
    if otfdim < otfh.ndim:
        otfh = nip.image(np.squeeze(otfh))

    return otfh


def otf_get_mask(otf, center_pinhole, mode='rft', eps=1e-5, bool_mask=False, closing=None, pixelsize=None, multiview=True):
    '''
    Calculate necessary mask for unmixing.
    In principal better to construct mash with geometrical shapes, but for now just guessed it -> maybe only in noise-free case possible.

    :PARAM:
    =======
    :otf:       (IMAGE) known 3D-otf of shape [pinhole_dim,Z,Y,X] -> basically works on any dimensionality, but assumes first index to be list-dimension and uses center as reference
    :mode:      (STRING) mode used to calculate the given OTF (to properly calculate mask) -> 'old', 'rft', 'fft' (DEFAULT)
    :eps:       (FLOAT) limit (multiplied by center_pinhole) for calculating OTF-support shape
    :bool_mask: (BOOL) decide whether to return mask as image or a boolean-map
    :closing:   (INT/LIST) whether binary closing operation should be applied onto map
                if INT -> choose with 0,1,..,4 from list [np.ones((2, 2)), np.ones((2, 2)), nip.rr([3,3]) <= 1,nip.rr([5,5]) <= 2, nip.rr([7,7]) <= 3]
                if LIST -> custom-shape to be used

    TODO:   1) compare scipy.ndimage.binary_closing with scipy.ndimage.morphology.binary_fillholes

    :OUT:
    =====
    :my_mask:           calculated Mask = used OTF Support; Note: If closing used, new (closed) my_mask is concatenated onto existing my_mask!
    :proj_mask:         z-projection of mask to get range for inversion calculation
    :zoff:              z-position of 0 frequency in fourier-space
    :center_pinhole:    central pinhole from input pinhole stack (by numbers, not by correlation)
    '''
    # otfref
    otf_ref = otf[center_pinhole] if multiview else otf

    # get parameters
    center_max = np.max(np.abs(otf_ref))
    epsabs = center_max * eps

    # calculate mask
    my_mask = (np.abs(otf_ref) > epsabs).astype(np.float32)

    # close using the chosen structuring element
    if closing is not None:

        # create closing
        if type(closing) == int:
            closing = ([np.ones((2, 2)), np.ones((2, 2)), nip.rr(
                [3, 3]) <= 1, nip.rr([5, 5]) <= 2, nip.rr([7, 7]) <= 3][closing])*1

        # fill mask -> only 2D operation
        if my_mask.ndim > 2:
            mms = my_mask.shape
            my_mask = np.reshape(
                my_mask, (int(np.prod(mms[:-2])), mms[-2], mms[-1]))
            my_mask_filled = np.copy(my_mask)
            for m in range(my_mask.shape[0]):
                my_mask_filled[m] = binary_closing(
                    my_mask[m], structure=closing).astype(np.int)
            my_mask_filled = np.reshape(my_mask_filled, mms)
            my_mask = np.reshape(my_mask, mms)
        else:
            my_mask_filled = binary_closing(
                my_mask, structure=closing).astype(np.int)
    else:
        my_mask_filled = my_mask

    # make image
    my_mask = nip.image(my_mask)
    my_mask_filled = nip.image(my_mask_filled)

    zoff = np.zeros(otf.shape[-2:], dtype=int)
    proj_mask = my_mask_filled.sum(axis=0)
    # if mode == 'rft' else nip.catE((my_mask_filled.shape[0]//2-my_mask_filled[:my_mask_filled.shape[0]//2].sum(axis=0), my_mask_filled.sum(axis=0)))

    if bool_mask:
        my_mask_filled = my_mask_filled.astype('bool')
        my_mask = my_mask.astype('bool')

    # add pixelsize
    my_mask.pixelsize = pixelsize if not pixelsize is None else otf.pixelsize[-my_mask.ndim:]
    my_mask_filled.pixelsize = pixelsize if not pixelsize is None else otf.pixelsize[-my_mask.ndim:]

    # done?
    return my_mask, my_mask_filled, proj_mask, zoff, center_pinhole


def test_otf_symmetry(otf):
    '''
    Tests the symmetry properties (within numerical tolerances) of the OTF.

    :PARAM:
    =======
    otf:       OTF of shape [Z,Y,X]

    :OUT:
    =====
    results
    '''
    # select center pinhole and prepare array
    otf_c = int(otf.shape[0]/2.0)
    otfc = nip.image(otf[otf_c])

    # conjugation-symmetr -> pi-rotation and conjugate
    otfh2 = otf_fill(otfc, fill_method='rot')

    # kx,ky-axis symmetry -> conjugation and copy
    otfh3 = otf_fill(otfc, fill_method='ax')

    # comparison
    nip.v5(nip.cat((otfh3, otfc, otfh2)))
    psfc = np.real(nip.ft3d(otfc))
    psfh2 = np.real(nip.ft3d(otfh2))
    psfh3 = np.real(nip.ft3d(otfh2))
    nip.v5(nip.cat((psfh3, psfc, psfh2)))


def pinv_unmix(a, svdlim=1e-15, svdnum=None, eps_reg=0, eps_reg_rel=0, use_own=True):
    """
    Functions as a wrapper and a regularized version of the np.linalg.pinv.

    :PARAM:
    =======
    :a:         (ARRAY/IMAGE) Array to invert
    :svdlim:     (FLOAT) relative cutoff
    :svdnum:    (INT) maximum nbr of SVD-values to keep
    :eps:       (FLOAT) regularizer

    """
    # use original pinv?
    if not use_own:
        res = np.linalg.pinv(a=a, rcond=svdlim)
        outdict = {}

    else:
        a = a.conjugate()
        u, s, vt = np.linalg.svd(a, full_matrices=False)
        s_full = np.copy(s)

        # discard small singular values; regularize singular-values if wanted -> note: eps_reg is 0 on default
        if svdnum is None:
            if svdlim is None:
                large = np.ones(s.shape, dtype=bool)
            else:
                cutoff = np.array(svdlim)[..., np.newaxis] * \
                    np.amax(s, axis=-1, keepdims=True)
                large = s > cutoff

        else:
            # cutoff = s[:svdnum] if svdnum < len(s) else s
            large = s > s[svdnum] if svdnum < len(s) else np.ones(s.shape, dtype=bool)

        if eps_reg == 0:
            eps_reg = s[0]*eps_reg_rel

        # Tikhonov regularization analogue to idea 1/(OTF+eps) where eps becomes dominant when OTF<eps, especially reduces to 1/eps when OTF<<eps
        s = np.divide(s, s*s+eps_reg, where=large,
                      out=s) if eps_reg else np.divide(1, s, where=large, out=s)
        s[~large] = 0

        # res = np.matmul(np.transpose(vt), np.multiply(s[..., np.newaxis], np.transpose(u)))
        res = np.transpose(np.dot(u*s, vt))

        # return results
        outdict = {'u': u, 's_full': s_full, 's': s, 'vt': vt, 'eps_reg': eps_reg}

    # done?
    return res, outdict


def unmix_svd_stat(svd_range, sing_vals, eps_reg, kk, jj):
    s = svd_range['biggest_ratio']
    if not len(sing_vals) == 0 and (sing_vals[0]/sing_vals[-1]) > s['ratio']:
        s['ratio'] = sing_vals[0]/sing_vals[-1]
        s['max'] = sing_vals[0]
        s['min'] = sing_vals[-1]
        s['s_list'] = sing_vals
        s['eps_reg'] = eps_reg
        s['kk'] = kk
        s['jj'] = jj
    s = svd_range['smallest_sv']
    if not len(sing_vals) == 0 and (sing_vals[-1] < s['sv']):
        s['sv'] = sing_vals[-1]
        s['s_list'] = sing_vals
        s['eps_reg'] = eps_reg
        s['kk'] = kk
        s['jj'] = jj
    s = svd_range['biggest_sv']
    if not len(sing_vals) == 0 and (sing_vals[0] > s['sv']):
        s['sv'] = sing_vals[0]
        s['s_list'] = sing_vals
        s['eps_reg'] = eps_reg
        s['kk'] = kk
        s['jj'] = jj
    return svd_range


def unmix_svdstat_pp(svd_range):
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\tSVD-Statistics\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    a = svd_range['biggest_ratio']
    print(
        f"Biggest Ratio:\n >>>> ratio={a['ratio']:.2e}, max={a['max']:.2e}, min={a['min']:.2e}, eps_reg={a['eps_reg']:.2e}, len(s)={len(a['s_list'])}, [kx,ky]=[{a['kk']},{a['jj']}].\n >>>> sv_list={format_list(a['s_list'],'.3e')}.")
    a = svd_range['biggest_sv']
    print(
        f"Biggest Singular Value:\n >>>> λ={a['sv']:.2e}, eps_reg={a['eps_reg']:.2e}, len(s)={len(a['s_list'])}, [kx,ky]=[{a['kk']},{a['jj']}].\n >>>> sv_list={format_list(a['s_list'],'.3e')}.")
    a = svd_range['smallest_sv']
    print(
        f"Smallest Singular Value:\n >>>> λ={a['sv']:.2e}, eps_reg={a['eps_reg']:.2e}, len(s)={len(a['s_list'])}, [kx,ky]=[{a['kk']},{a['jj']}].\n >>>> sv_list={format_list(a['s_list'],'.3e')}.")


def unmix_matrix(otf, mode='rft', eps_mask=5e-4, eps_reg=1e-17, eps_reg_rel=0, svdlim=1e-15, svdnum=None, hermitian=False, use_own=True, closing=None, center_pinhole=None, verbose=True, svd_stat=False):
    '''
    Calculates the unmixing matrix. Assums pi-rotational-symmetry around origin (=conjugate), because PSF is real and only shifted laterally. No aberrations etc. Hence, only half of the OTF-support along z is calculated.

    :PARAMS:
    ========
    :PSF_eff:   (IMAGE)     Multi-Pinhole PSF of shape [pinhole_dim,Z,Y,X]
    :eps:       (FLOAT)     thresh-value for calculating the non-zero support (and hence the used kx,ky frequencies)
    :mode:      (STRING)    mode of used FTs -> 'old': manual stitching; 'rft': RFT-based
    :svdrel:    (FlOAT)     minimum relative size of the smallest SVD-value to the biggest for the matrix inversion (limits the non-empty SVD-matrix-size)
    :svdlim:    (INT)       maximum number of included singular values -> note: if svdlim is set, svdrel is ignored
    :hermitian: (BOOL)      input hermitian? (typically: False, because of (at least) numerical differences)
    :use_own:   (BOOL)        whether to use own inversion-implementation
    :closing:   (INT/ARRAY) closing mask (chosen by int) or shape (ARRAY) to be used
    :center_pinhole: (INT)  position of central pinhole to be used for calculation of OTF-support

    :OUT:
    ====
    :otf_unmix:         (IMAGE) the (inverted=) unmix OTF of shape [z/2,pinhole_dim,Y,X]
    :otf_unmix_full:    (IMAGE) <DEPRECATED>!! -> for old reconstruction: unmix was done on half-space using FFT, hence full OTF had to be constructed manually -> not needed anymore because reconstruction is now done via RFT
    :my_mask:           (IMAGE) mask used for defining limits of OTF-support
    :proj_mask:         (IMAGE) sum of mask along z-axis (used for inversion-range)
    '''
    if verbose:
        print("Calculating unmixing matrix.")
    # parameters/preparation
    otf_unmix = np.transpose(np.zeros(otf.shape, dtype=otf.dtype), [1, 0, 2, 3])
    svd_counter = np.zeros(otf_unmix.shape, dtype=np.int16)
    svd_lim_counter = np.zeros(otf_unmix.shape, dtype=np.int16)
    svd_range = {'biggest_ratio': {'ratio': 0.0, 'max': 0.0, 'min': 0.0, 's_list': [], 'eps_reg': eps_reg, 'kk': 0, 'jj': 0, },
                 'smallest_sv': {'sv': 1e7, 's_list': [], 'eps_reg': eps_reg, 'kk': 0, 'jj': 0},
                 'biggest_sv': {'sv': 0, 's_list': [], 'eps_reg': eps_reg, 'kk': 0, 'jj': 0}}

    if center_pinhole is None:
        center_pinhole = otf.shape[0]//2

    # calculate mask
    _, my_mask, proj_mask, zoff, _ = otf_get_mask(
        otf, mode=mode, eps=eps_mask, bool_mask=False, closing=closing, center_pinhole=center_pinhole)

    proj_mask = proj_mask.astype(np.int16)

    # calculate SVs at a central pixel and generate svdnum for further processing
    if svdnum is None:
        max_sv_pos = np.unravel_index(np.argmax(proj_mask.flatten()), proj_mask.shape)
        _, outdict_pre = pinv_unmix(otf[:, :, max_sv_pos[0], max_sv_pos[1]],
                                    svdlim=svdlim, svdnum=None, eps_reg=eps_reg, eps_reg_rel=eps_reg_rel, use_own=use_own)
        svdnum = len(outdict_pre['s_full'][outdict_pre['s'] > 0])
        svd_range = unmix_svd_stat(
            svd_range, outdict_pre['s_full'][outdict_pre['s'] > 0], outdict_pre['eps_reg'], max_sv_pos[0], max_sv_pos[1])

        # loop over all kx,ky
    for kk in range(otf_unmix.shape[-2]):
        for jj in range(otf_unmix.shape[-1]):
            if my_mask[:, kk, jj].any():
                otf_unmix[zoff[kk, jj]:zoff[kk, jj]+proj_mask[kk, jj], :, kk, jj], outdict = pinv_unmix(
                    otf[:, zoff[kk, jj]:zoff[kk, jj]+proj_mask[kk, jj], kk, jj], svdlim=svdlim, svdnum=svdnum, eps_reg=eps_reg, eps_reg_rel=eps_reg_rel, use_own=use_own)
                s_lim = outdict['s'][outdict['s'] > 0]
                svd_lim_counter[zoff[kk, jj]:zoff[kk, jj]+proj_mask[kk, jj],
                                :, kk, jj] = len(s_lim)
                svd_counter[zoff[kk, jj]:zoff[kk, jj]+proj_mask[kk, jj],
                            :, kk, jj] = len(outdict['s_full'])

                # gather some unmixing statistics
                if svd_stat:
                    svd_range = unmix_svd_stat(
                        svd_range, outdict['s_full'][outdict['s'] > 0], outdict['eps_reg'], kk, jj)

                # otf[:, zoff[kk, jj]:zoff[kk, jj]+proj_mask[kk, jj], kk, jj] #otf[:, :,90,90]
    otf_unmix = nip.image(otf_unmix)

    # create full otf_unmix if necessary
    # if mode == 'old':
    #    otf_unmix = otf_unmix[zoff:]
    #    otf_unmix = nip.image(otf_unmix)
    #    otf_unmix_full = np.transpose(
    #        otf_fill(np.transpose(otf_unmix, [1, 0, 2, 3])), [1, 0, 2, 3])
    # else:
    #    otf_unmix_full = otf_unmix
    svd_counter_dict = {'svd_counter': svd_counter,
                        'svd_lim_counter': svd_lim_counter, 'svd_range': svd_range}

    if verbose and svd_stat:
        unmix_svdstat_pp(svd_range)

    # done? otf_unmix_full
    return otf_unmix, svd_counter_dict, my_mask, proj_mask


def unmix_image_ft(im_unmix_ft, recon_shape=None, mode='rft', show_phases=False, verbose=True):
    '''
    Fouriertransforms along -2,-1 and applies RFT along -3 (=kz), as only half space was used due to positivity and rotation symmetry.
    '''
    if verbose:
        print("Backtransform unmixed image.")
    # sanity
    if recon_shape is None:
        recon_shape = np.array(im_unmix_ft.shape)
        recon_shape[-3] = (recon_shape[-3]*2)-1

    # inverse transformation depending on mode
    if mode == 'old':
        # normal ft along kx,ky as infos are symmetric
        unmix_im = nip.ift(im_unmix_ft, axes=(-2, -1), ret='complex')

        # show phases before and after two FTs
        if show_phases:
            nip.v5(stack2tiles(im_unmix_ft), showPhases=True)
            nip.v5(stack2tiles(unmix_im), showPhases=True)

        # per definition does a RFT transform on the last axis given and normal fft along the others
        # unmix_im = np.fft.irfft(unmix_im,n=recon_shape,axes=0)
        unmix_im = nip.image(np.fft.fftshift(np.fft.irfft(
            unmix_im, n=recon_shape[-3], axis=-3), axes=-3), im_unmix_ft.pixelsize)
    elif mode == 'rft':
        unmix_im = irft3dz(im_unmix_ft, recon_shape)
    elif mode == 'fft':
        unmix_im = nip.ift3d(im_unmix_ft)
    else:
        raise ValueError("No proper mode chosen!")

    unmix_im = np.abs(unmix_im)
    # done?
    return unmix_im, recon_shape


def unmix_recover_thickslice(unmixer, im, verbose=True, dtype=None):
    '''
    Does the recovery multiplication step. Need to apply unmix matrix for every kx and ky individually.

    :PARAM:
    =======
    :otf_unmix:     4D unmix OTF -> [kZ, pinhole_dim, kx,ky]
    :ism_im:        N * 2D ft-images (resembling the pinholes)

    :OUT:
    ====
    :im_unmix:      3D unmixed images of dim [kz,kx,ky]

    '''
    if verbose:
        print("Recovering Thickslice using Einstein-Summation.")
    # the real thing
    im_unmix = nip.image(np.einsum('ijkl,jkl->ikl', unmixer, im, dtype=dtype))

    # done?
    return im_unmix

def thickslice_normalize(im: np.ndarray, axes: tuple = None, direct: bool = True, min_fac: float = 1.01) -> np.ndarray:
    im_min = np.min(im, axis=axes)
    im_std = np.std(im, axis=axes)

    if direct:
        im -= (im_min*min_fac)
        im /= im_std
    else:
        im = ((im-im_min*min_fac)/im_std).astype(im.dtype)
    return im


def thickslice_get_diag_shift(diag_shape, zshift_list):
    '''
    eg
    diag_shape = [30,1,1]
    zshift_list = np.arange(-15.15)
    '''
    # preallocate
    dshift = np.zeros([len(zshift_list), ]+list(diag_shape), dtype='complex')

    # calculate shift-matrices
    for m, zshift in enumerate(zshift_list):
        dshift[m] = np.reshape(np.exp(1j*2*np.pi*nip.ramp1D(diag_shape[0], ramp_dim=0, placement='center',
                                                            freq='ftfreq',)*zshift), diag_shape)

    return dshift


def thickslice_unmix(ims_ft: nip.image, otfs: nip.image, pad: dict, full_ret: bool = True, ret_real: bool = True, zslices: list = [], verbose=True, pixelsize=None, t1: nip.timer = None) -> Union[nip.image, dict]:
    '''


    '''
    t1 = nip.timer('s') if t1 is None else t1

    # sanity
    if pixelsize is None:
        pixelsize = ims_ft.pixelsize
    if not np.iscomplexobj(ims_ft):
        if not 'imsize' in pad or pad['imsize'] is None:
            pad['imsize'] = ims_ft.shape
        ims_ft = ft_correct(ims_ft, (-2, -1), im_shape=None,
                            mode=pad['fmodel'], dir='fwd', dtype=pad['dtype_complex'])
    if not np.iscomplexobj(otfs):
        otfs = ft_correct(otfs, (-3, -2, -1), im_shape=None,
                          mode=pad['fmodel'], dir='fwd', dtype=pad['dtype_complex'])
    if ims_ft.ndim > 3:
        ims_ft = np.reshape(ims_ft, [np.prod(ims_ft.shape[:-2]), ]+list(ims_ft.shape[-2:]))
        otfs=nip.repmat(otfs,[int(ims_ft.shape[0]/otfs.shape[0]),1,1,1])
    ims_ft.pixelsize = pixelsize if ims_ft.pixelsize is None else ims_ft.pixelsize
    otfs.pixelsize = pixelsize if otfs.pixelsize is None else otfs.pixelsize

    # note: dimensions need to at max reduce to singletons, but not vanish for the svd-unmixing to work
    if not zslices == []:
        #ims_ft = np.squeeze(ims_ft.swapaxes(0, 1)[zslices].swapaxes(0, 1))
        otfs = otfs.swapaxes(0, 1)[zslices].swapaxes(0, 1)
        #ims_ft.pixelsize = pixelsize
        otfs.pixelsize = pixelsize

    # prepare container
    ret_dict = {}

    # calculate unmix-matrix
    t1.add('Start Unmixing.')
    otf_unmix, otf_unmix_svd_counter_dict, otf_unmix_mask, otf_unmix_projmask = unmix_matrix(
        otfs, mode=pad['fmodel'], eps_mask=pad['eps_mask'], eps_reg=pad['eps_reg'], eps_reg_rel=pad['eps_reg_rel'], svdlim=pad['svdlim'], svdnum=pad['svdnum'], use_own=pad['use_own'], closing=pad['closing'], verbose=verbose, svd_stat=pad['svd_stat'])

    # unmix
    t1.add('Start Recovering.')
    im_unmix_ft = unmix_recover_thickslice(
        unmixer=otf_unmix, im=ims_ft, verbose=verbose, dtype=pad['dtype_complex'])

    # back-transform
    t1.add('Transform back.')
    im_unmix = ft_correct(im_unmix_ft, pad['faxes'],
                          im_shape=pad['imsize'], mode=pad['fmodel'], dir='bwd', dtype=pad['dtype_real'])

    # return real-valued?
    if ret_real:
        im_unmix = np.abs(im_unmix)
    t1.add('Recovering done.')

    # return complete?
    if full_ret:
        ret_dict = {'otf_unmix': otf_unmix, 'otf_unmix_svd_counter_dict': otf_unmix_svd_counter_dict, 'otf_unmix_mask': otf_unmix_mask,
                    'otf_unmix_projmask': otf_unmix_projmask, 'im_unmix_ft': im_unmix_ft, 'timer': t1}

    if verbose:
        t1.get()

    # done?
    return im_unmix, ret_dict

# %%
# ---------------------------------------------------------------
#                   DSAX-ISM
# ---------------------------------------------------------------
def dsax_isolateOrder_prefactors(Iex, Isat, tau, order=3,):
    Gamma = tau/(1+Isat*tau)
    Iex = np.array(Iex)
    orderl = np.arange(order)+1
    Iexn = ((Gamma*(np.e/orderl)*(Iex-Isat))**orderl)/(tau*np.sqrt(2*np.pi*orderl))
    return Iexn


def dsax_isolateOrder(ims: Union[nip.image, np.ndarray], Iex: list, order: int = 2, shield: bool = False, ret_summands: bool = False):
    '''
    Calculate dsax non-linear order reconstruction.
    If ret_summands=True returns:
        order==2: [sum0,NL1]
        order==3: [sum0,NL1,sum1,sum2,im_NL2]
    Note: shielding not working correctly yet!
    '''

    if order == 1:
        return ims[0]
    elif order == 2:
        sum0 = Iex[1]/Iex[0]*ims[0]
        im_NL1 = sum0 - ims[1]
        if ret_summands:
            im_NL1 = nip.cat((sum0[np.newaxis], im_NL1[np.newaxis]), axis=0)
        im_NL1 = np.max(im_NL1)-im_NL1 if shield and (np.sum(ims[1]) > np.sum(sum0)) else im_NL1
        return im_NL1
    elif order == 3:
        im_NL1 = dsax_isolateOrder(ims, Iex, order=2, ret_summands=ret_summands)
        I10 = Iex[1]/Iex[0]
        I20 = Iex[2]/Iex[0]
        I21 = Iex[2]/Iex[1]
        # (I10*ims[0]-ims[1])  # I20*(1-I21)*ims[0]
        sum1 = I21*I21*(im_NL1[-1] if ret_summands else im_NL1)
        sum2 = -I20*ims[0]  # I21*I21*ims[1]
        sum12 = sum1+sum2
        im_NL2 = sum12 + ims[2]  # sum12 - ims[2]
        # im_NL = sum
        im_NL2 = np.max(im_NL2)-im_NL2 if shield and (np.sum(ims[2]) > np.sum(sum12)) else im_NL2

        if ret_summands:
            im_NL = nip.image(np.zeros([5, ]+list(ims.shape[1:]), dtype=ims.dtype))
            im_NL[0] = im_NL1[0]
            im_NL[1] = im_NL1[1]
            im_NL[2] = sum1
            im_NL[3] = sum2
            im_NL[4] = im_NL2
        else:
            im_NL = nip.image(np.array([im_NL1, im_NL2]))

        return im_NL

def dsax_complete_recon_functest(ims, psfs, pad, rparams, ddh, do_test_tiling=False, do_load_data=False):
    '''Fast test within dsax_complete_recon to find proper tiling parameters and behaviour'''
    resdt = {}
    if do_test_tiling:
        im_rest = ims[:, :, 20:40, 20:40]
        psfs_rest = psfs[:, :, 20:40, 20:40]
        test_tdh = default_dict_tiling(imshape=im_rest.shape, basic_shape=list(im_rest.shape[:-3])+list(
            [1, 15, 15]), basic_roverlap=[0, ]*(im_rest.ndim-3)+list(pad['td']['overlap_rel'][-3:])) if rparams['do_tiling'] else None
        test_tdh['tiling_low_thresh'] = 1
        resdt = deepcopy(ddh)
        resdt['param'] = 'lambdal'
        resdt['NIter'] = 200
        resdt['lambdal_range'] = [[10.0**(-m), ] for m in np.arange(9, 11, 0.5)]
        a = deconv_test_param(im=im_rest, psf=psfs_rest, tiling_dict=test_tdh,
                              deconv_dict=resdt, param=resdt['param'], param_range=resdt['lambdal_range'])
        b = deconv_test_param(im=im_rest, psf=psfs_rest, tiling_dict=test_tdh,
                              deconv_dict=resdt, param=resdt['param'], param_range=resdt['lambdal_range'])
        c = deconv_test_param(im=im_rest, psf=psfs_rest, tiling_dict=test_tdh,
                              deconv_dict=resdt, param=resdt['param'], param_range=resdt['lambdal_range'])

    if do_load_data:
        resdt, _ = load_data(
            {'save_path': pad['save_path'], 'save_name': pad['save_name_base']+'_resd'})
        fix_dict_pixelsize(resdt, pad['pixelsize'], [
            'im_raw', 'im_drift', 'imshepp', 'imshepp_ref', 'conf_closed', 'conf', 'conf_mask', 'imshepp_masked', 'im_shepp', 'psf_shepp', 'ismWA', 'ismWAN', 'ismWAN_psf', 'wiener', 'dec_2D_allview_indivim', 'dec_2D_conf', 'dec_2D_shepp', 'im_wf', 'psf_wf', 'im_conf_closed', 'psf_conf_closed', 'im_conf', 'psf_conf', 'dec_2D_allview_allim', 'wiener_rtest', 'ismWA_rtest', 'ismWAN_rtest', 'ismWAN_psf_rtest', 'ismWA_weights', 'dec2D_av_aim_rtest', 'dec2D_av_indivim_rtest', 'dec2D_conf_rtest', 'dec2D_shepp_rtest', 'ims'])

    return resdt

def dsax_complete_recon(pad, ims, psfs, rparams={}):
    '''Alias for complete_recon'''
    rparams['ism_method'] == 'dsax'
    return  complete_recon(pad=pad, ims=ims, psfs=psfs, rparams=rparams)

def thickslice_complete_recon(pad, ims, psfs, rparams={}):
    '''Alias for complete_recon'''
    rparams['ism_method'] == 'dsax'
    return complete_recon(pad=pad, ims=ims, psfs=psfs, rparams=rparams)

def complete_recon(pad, ims, psfs, rparams={}):
    '''
    ims.shape=[DETPIXELS,IEX,Z,Y,X]
    rparams={'wiener_reg_reps':1e-8,'reg_reps':1e-7,'lambdal':1e-12,'lambdal_complete':[1e-11.5,]}

    To use this function make sure that photon-statistics of IM and PSF are the same as used to find the optimal reconstruction parameters placed in rparams.
    '''
    resd = {}

    # sanity
    pad = cr_sanity_checks(ims=ims, pad=pad, rparams=rparams)

    # normalize input data
    ims, psfs, pad, resd = cr_input_normalization(
        ims=ims, psfs=psfs, pad=pad, resd=resd, rparams=rparams)

    # do simple WF, Confocal +-1AU and SheppSUM
    pad, resd, rparams = cr_basic_methods(ims=ims, psfs=psfs, pad=pad, resd=resd, rparams=rparams)

    # prepare data for deconv
    im_conf, psf_conf, im_shepp, psf_shepp, ims_ft, otfs, ims_rs, psfs_rs = cr_prepare_basic_data_for_deconv(
        ims=ims, psfs=psfs, pad=pad, resd=resd, rparams=rparams)

    if not rparams['find_optimal_parameters']:
        fix_dict_pixelsize(resd, pad['pixelsize'], ['im_shepp', 'psf_shepp', 'wiener',
                                                    'ismWA', 'ismWAN', 'ismWAN_psf', 'dec_2D_allview_indivim'])

        # wavg recon
        resd = recon_wavg_list(ims_ft=ims_ft, otfs=otfs, pad=pad,
                               rparams=rparams, resd=resd, ldim=1)

        # wiener recon
        resd = recon_wiener_list(ims_ft=ims_ft, otfs=otfs, pad=pad,
                                 rparams=rparams, resd=resd, ldim=1)

        # deconv --> multiview, all-views and 3 images together
        if rparams['ism_method'] == 'dsax':
            pad['dd']['BorderRegion'][1] = 0
            pad['dd']['lambdal'] = rparams['lambdal_complete']
            resd['dec_2D_allview_allim_dict'] = dict(pad['dd'])
            tdh_rs = default_dict_tiling(imshape=ims_rs.shape, basic_shape=list(ims_rs.shape[:-3])+list(
                pad['td']['tile_shape'][-3:]), basic_roverlap=[0, ]*(ims_rs.ndim-3)+list(pad['td']['overlap_rel'][-3:])) if rparams['do_tiling'] else None
            resd['dec_2D_allview_allim'], resd['dec_2D_allview_allim_stats'] = deconv_switcher(
                im=ims_rs, psf=psfs_rs, tiling_dict=tdh_rs, deconv_dict=pad['dd'], do_tiling=rparams['do_tiling'])
        elif rparams['method'] == 'thickslice':
            # oof
            resd['thickslice'] = thickslice_unmix(
                ims_ft, otfs, pad, full_ret=pad['full_ret'], ret_real=pad['ret_real'], pixelsize=pad['pixelsize'], verbose=pad['verbose'])

            # 2D to 3D

            # 3D

        # confocal deconvolution
        resd = recon_deconv_list(ims=im_conf, psfs=psf_conf, pad=pad, rparams=rparams, resd=resd, sname=[
                                 'dec_2D', 'allview_indivim'], ldim=0, ddh=pad['dd'], do_tiling=False, td=None)

        # sheppard deconvolution
        resd = recon_deconv_list(ims=im_shepp, psfs=psf_shepp, pad=pad, rparams=rparams, resd=resd, sname=[
                                 'dec_2D', 'allview_indivim'], ldim=0, ddh=pad['dd'], do_tiling=rparams['do_tiling'], td=None)

        # multiview deconvolution
        resd = recon_deconv_list(ims=ims[rparams['pinmask_mdec'][1]], psfs=psfs[rparams['pinmask_mdec'][1]], pad=pad, rparams=rparams, resd=resd, sname=[
                                 'dec_2D', 'allview_indivim'], ldim=0, ddh=pad['dd'], do_tiling=rparams['do_tiling'], td=None)

    else:
        # test
        resdt = dsax_complete_recon_functest(
            ims=ims_rs, psfs=psfs_rs, pad=pad, rparams=rparams, ddh=pad['dd'], do_test_tiling=False, do_load_data=False)

        # Weighted Averaging in Fourierspace
        resd = wavg_testparam(ims_ft=ims_ft, otfs=otfs, pad=pad, resd=resd)

        # wiener
        resd = wiener_testparam(ims_ft=ims_ft, otfs=otfs, pad=pad, resd=resd)

        # deconv --> multiview, all-views and 3 images together
        if rparams['method'] == 'dsax':
            resd = deconv_test_param_on_list(ims=ims_rs[np.newaxis], psfs=psfs_rs[np.newaxis], pad=pad,
                                             resd=resd, rparams=rparams, ddh=pad['dd'], sname=['dec2D', 'av_aim'], ldim=0, do_tiling=rparams['do_tiling'])

        # deconv --> multiview, all-views for each raw (and calculated) image
        resd = deconv_test_param_on_list(ims=im_conf, psfs=psf_conf, pad=pad,
                                         resd=resd, rparams=rparams, ddh=pad['dd'], sname=['dec2D', 'conf'], ldim=0, do_tiling=False)
        resd = deconv_test_param_on_list(ims=im_shepp, psfs=psf_shepp, pad=pad,
                                         resd=resd, rparams=rparams, ddh=pad['dd'], sname=['dec2D', 'shepp'], ldim=0, do_tiling=False)
        resd = deconv_test_param_on_list(ims=ims[rparams['pinmask_mdec'][1]], psfs=psfs[[rparams['pinmask_mdec'][1]]], pad=pad,
                                         resd=resd, rparams=rparams, ddh=pad['dd'], sname=['dec2D', 'av_indivim'], ldim=0, do_tiling=rparams['do_tiling'])

        # select comparison data
        compare_list, obj = dsax_create_deconv_complist(resd=resd, rparams=rparams,)
        deconv_test_param_find_best_combination(compare_list=compare_list, obj=obj, resd=resd)

        # find best combinations via NCC
        resd, index_list = deconv_test_param_find_best_combination(
            compare_list=compare_list, obj=obj, resd=resd, index_list=[])

        # store some stats about the data
        ax_stats = (0, -3, -2, -1)
        resd['im_stats'], _ = stf_basic(
            ims, faxes=ax_stats, cols=index_list, printout=True)
        resd['psf_stats'], _ = stf_basic(
            psfs, faxes=ax_stats, cols=index_list, printout=True)

    return resd


def dsax_vary_Iex(im, markers, Iex=[1, ], order=2, Iex_sr=[2.6, 4.7, 0.05], critlim=[5e-1, 5e-1, 5e-4], im_axes=(-2, -1), bead_roi=[20, 20], use_criteria=[1, 1, 0, 0, 1], shield=False):
    '''
    Vary dsax-factors and test change of FWHM of psf.
    Iex_sr should be the range of the ratio between the varied Iex and one non-varied Iex'. In case of 2nd order: Iex2/Iex1=Iex_sr.
    '''

    # allocate space
    Iex2tr = np.arange(Iex_sr[0], Iex_sr[1], Iex_sr[2])
    im_NL = nip.image(np.zeros([len(Iex2tr), ]+list(im.shape[-2:])))
    im_NL_psf = nip.image(np.zeros([len(Iex2tr), ]+bead_roi))
    im_NL_psf_dict = [{}, ]*len(Iex2tr)
    fwhm = np.zeros([len(Iex2tr), 2])
    im_NLc = np.array(im_NL_psf.shape)//2

    # calculate comparison
    for m, Iratio in enumerate(Iex2tr):
        Iext = [Iex[0], Iratio*Iex[0]] if order == 2 else [Iex[0], Iex[1], Iratio*Iex[1]]
        a = dsax_isolateOrder(im, Iext, order=order, shield=shield)
        im_NL[m] = a[-1] if order == 3 else a
        if use_criteria[0]:
            im_NL_psf[m], im_NL_psf_dict[m] = extract_multiPSF(
                im_NL[m], markers=markers, im_axes=(-2, -1), bead_roi=bead_roi)
            fwhm[m] = im_NL_psf_dict[m]['para'][0][-2:]

    # find closest fwhm to minimum
    crit_sel = [True, ]*im_NL.shape[0]
    # select smallest fwhms within a range -> use mean as processing is symmetric and hence average fwhm is sufficient
    if use_criteria[0]:
        fwhm = np.abs(fwhm)
        fwhm_m = np.mean(fwhm, axis=-1)
        fwhm_ratio = np.abs(fwhm_m/np.min(fwhm_m)-1)
        fwhm_s0 = fwhm_ratio < critlim[0]
        crit_sel *= fwhm_s0
    # select only combinations where a choice of surrounding pixels is bigger than center
    if use_criteria[1]:
        fwhm_s1 = (im_NL_psf[:, im_NLc[-2], im_NLc[-1]] > im_NL_psf[:, im_NLc[-2]+3, im_NLc[-1]]) * \
            (im_NL_psf[:, im_NLc[-2], im_NLc[-1]] > im_NL_psf[:, im_NLc[-2], im_NLc[-1]+3])
        crit_sel *= fwhm_s1
    # select only elements that have a max bigger than the max-median
    if use_criteria[2]:
        im_max = np.max(im_NL_psf, axis=(-2, -1)
                        ) if any(use_criteria[:2]) else np.max(im_NL, axis=(-2, -1))
        fwhm_s2 = im_max > np.median(im_max)
        crit_sel *= fwhm_s2
        # crit_choice = np.argmax(crit_sel)
    # select smallest std-deviation within a region
    if use_criteria[3]:
        im_use = im_NL_psf if any(use_criteria[:2]) else im_NL
        im_use_std = np.std(normNoff(im_use, dims=im_axes), axis=im_axes)
        fwhm_s3 = np.array((im_use_std/np.min(im_use_std)-1) < critlim[1])
        crit_sel *= fwhm_s3
        # crit_choice = np.argmin(im_use_std)
    # contrast
    if use_criteria[4]:
        im_use = im_NL_psf if any(use_criteria[:2]) else im_NL
        im_contrast = (np.max(im_use, axis=im_axes)-np.min(im_use, axis=im_axes)) / \
            (np.max(im_use, axis=im_axes)+np.min(im_use, axis=im_axes))
        fwhm_s4 = 1-im_contrast/np.max(im_contrast) < critlim[2]
        crit_sel *= fwhm_s4
    fwhm_s = np.array(np.where(crit_sel)).flatten()

    if fwhm_s.size == 0:
        Iratio_sel = Iex[order-1]/Iex[order-2]
    else:
        # select smallest fwhm_ratio from selection
        crit_choice = np.min(fwhm_s) if not any(
            use_criteria[:2]) else fwhm_s[np.argmin(fwhm_ratio[fwhm_s])]
        Iratio_sel = float(Iex_sr[0]+crit_choice*Iex_sr[-1])  # crit_sel[crit_choice]

    # display result and prepare return dict
    print(
        f"First best selection for FWHM at Iex2/Iex1={Iratio_sel} using relative-devation thresholds={critlim}.")
    ret_dict = {'im_NL': im_NL, 'im_NL_psf': im_NL_psf,
                'im_NL_psf_dict': im_NL_psf_dict, 'fwhm': fwhm, 'fwhm_s': fwhm_s}

    return Iratio_sel, ret_dict

def dsax_extract_higher_order_curves(curve_Iex, curve_fluo, lims_Iex, order=3):

    # sanity
    lims_Iex = np.array(lims_Iex).astype('int')

    dsax_iex = curve_Iex[lims_Iex+np.array([0, lims_Iex[0], lims_Iex[0]+lims_Iex[1]])]
    Iex1 = curve_Iex[lims_Iex[0]]
    curve_nl1 = (curve_Iex[lims_Iex[0]+1:]/lims_Iex[0]) * \
        curve_fluo[lims_Iex[0]+1:] - curve_fluo[lims_Iex[0]]

    Iex2 = curve_Iex[lims_Iex[0]+lims_Iex[1]]
    Iex3curve = curve_Iex[lims_Iex[0]+lims_Iex[1]+2:]
    sum1 = Iex3curve/Iex1*(1-Iex2/Iex1)*curve_fluo[lims_Iex[0]]
    sum2 = (Iex3curve/Iex2)**2*curve_nl1[lims_Iex[1]]
    curve_nl2 = sum1+sum2 - \
        curve_fluo[lims_Iex[0]+lims_Iex[1]+2]

    curve_nl1_range = curve_Iex[lims_Iex[0]+1:]
    curve_nl2_range = curve_Iex[lims_Iex[0]+lims_Iex[1]+2:]

    return dsax_iex, curve_nl1_range, curve_nl2_range, curve_nl1, curve_nl2

def dsax_find_Iex_combination(im, Iex, Iex_test_range=[[1.5, 8.1, 0.1], ], critlim=[[5e-1, 5e-1, 5e-4], [1e-4, 5e-4, 5e-4]], bead_roi=[16, 16], markers=[[], ], use_criteria=[1, 1, 0, 0, 1], shield=False, verbose=True):
    # sanity
    if len(Iex_test_range) < 2:
        Iex_test_range = list(Iex_test_range)*2

    I21_sel, I21_d = dsax_vary_Iex(im=im, markers=markers,
                                   Iex=Iex, order=2, Iex_sr=Iex_test_range[0], critlim=critlim[0], im_axes=(-2, -1), bead_roi=bead_roi, use_criteria=use_criteria, shield=shield)
    # fix factor for 3rd order
    res_Iex = np.array([Iex[0], Iex[0]*I21_sel, Iex[2]])
    I32_sel, I32_d = dsax_vary_Iex(im=im, markers=markers,
                                   Iex=res_Iex, order=3, Iex_sr=Iex_test_range[1], critlim=critlim[1], im_axes=(-2, -1), bead_roi=bead_roi, use_criteria=use_criteria, shield=shield)
    res_Iex[2] = I21_sel*I32_sel*Iex[0]

    if verbose:
        print(
            f"Difference in intensities:\nOrig={Iex}\nCalc={res_Iex}\nRatio={np.round(Iex/res_Iex,2)}%")

    # cleanup
    del I21_d, I32_d

    # done?
    return res_Iex

def cr_sanity_checks(ims, pad, rparams):
    '''Sanity Checks for input data of dsax_complete_recon.'''
    allowed_ds = [list, np.ndarray, tuple]

    if not 'detdist' in pad:
        pad['detdist'], pad['detpos'] = dist_from_detnbr(
            pad['nbr_det'], pad['pincenter'])
    if not 'conf_pinsize' in pad:
        pad['conf_pinsize'] = 1.1*np.max(pad['shift_offset'])  # [-1, -1]
    if not rparams['find_optimal_parameters']:
        if not type(rparams['wiener_reg_reps']) in allowed_ds:
            rparams['wiener_reg_reps'] = [rparams['wiener_reg_reps'], ]*ims.shape[1]
        if not type(rparams['reg_reps']) in allowed_ds:
            rparams['reg_reps'] = [rparams['reg_reps'], ]*ims.shape[1]
        if len(rparams['lambdal']) < 2:
            ah = rparams['lambdal'] if type(rparams['lambdal'][0]) in allowed_ds else [
                rparams['lambdal'], ]
            rparams['lambdal'] = ah*ims.shape[1]

    # done?
    return pad


def cr_input_normalization(ims, psfs, pad, resd, rparams):
    '''Input Normalization routine of dsax_complete_recon.'''
    pad['norm_ax'] = tuple([0, ]+list(pad['faxes']))

    # make sure ims is =>0
    if rparams['im_norm']:
        ims -= np.min(ims, pad['norm_ax'], keepdims=True)
        resd['ims_sum'] = np.sum(ims, axis=pad['norm_ax'], keepdims=True)
        ims *= (np.squeeze(resd['ims_sum'])[0]/resd['ims_sum'])*rparams['ims_NPhot_fact']
    if rparams['psf_norm']:
        psfs = normNoff(psfs, dims=pad['norm_ax'], method='sum', atol=0, direct=False)
    resd['im_conf_mask'] = rparams['pinmask_conf'][1] if rparams['pinmask_conf'][0] else None

    return ims, psfs, pad, resd


def cr_basic_methods(ims, psfs, pad, resd, rparams):
    '''Set of basic reconstructions of dsax_complete_recon.'''

    # widefield
    resd['im_wf'] = np.sum(ims, axis=0)
    resd['psf_wf'] = np.sum(psfs, axis=0)

    # confocal closed pinhole
    resd['im_conf_closed'] = ims[pad['pincenter']]
    resd['psf_conf_closed'] = psfs[pad['pincenter']]

    # confocal open pinhole
    resd['im_conf'], resd['im_conf_mask'] = recon_confocal(
        ims, detdist=pad['detdist'], pinsize=pad['conf_pinsize'], pincenter=pad['pincenter'], pinmask=resd['im_conf_mask'])
    resd['psf_conf'], _ = recon_confocal(
        psfs, detdist=pad['detdist'], pinsize=pad['conf_pinsize'], pincenter=pad['pincenter'], pinmask=resd['im_conf_mask'])

    # get masks
    rparams['pinmask_shepp'][1] = (resd['im_conf_mask'] if rparams['pinmask_shepp'][1]
                                   is None else rparams['pinmask_shepp'][1]) if rparams['pinmask_shepp'][0] else None
    rparams['pinmask_mdec'][1] = (rparams['pinmask_shepp'][1] if rparams['pinmask_mdec'][1] is None else rparams['pinmask_mdec']
                                  [1]) if rparams['pinmask_mdec'][0] else np.ones(ims.shape[0], dtype='bool')
    rparams['pinmask_amdec'][1] = (rparams['pinmask_shepp'][1] if rparams['pinmask_amdec'][1] is None else rparams['pinmask_amdec']
                                   [1]) if rparams['pinmask_amdec'][0] else np.ones(ims.shape[0], dtype='bool')

    # calculate SheppSum
    resd['im_shepp'] = nip.image(np.zeros(ims.shape[1:]))
    resd['im_shepp_resd'] = [{} for m in range(ims.shape[1])]
    resd['psf_shepp'] = nip.image(np.zeros(resd['im_shepp'].shape))
    fix_dict_pixelsize(resd, pad['pixelsize'], ['im_shepp', 'psf_shepp'])
    for m in range(ims.shape[1]):
        imshepph, resd['im_shepp_resd'][m] = recon_sheppardSUM(
            ims[:, m], nbr_det=pad['nbr_det'], pincenter=pad['pincenter'], shift_axes=pad['faxes'], shift_style='iter', pinmask=rparams['pinmask_shepp'][1])
        psfshepph, _ = recon_sheppardSUM(
            psfs[:, m], nbr_det=pad['nbr_det'], pincenter=pad['pincenter'], shift_axes=pad['faxes'], shift_style='iter', shift_map=resd['im_shepp_resd'][m]['shift_map'], pinmask=rparams['pinmask_shepp'][1])
        resd['im_shepp'][m] = imshepph.astype(pad['dtype_real'])
        resd['psf_shepp'][m] = psfshepph.astype(pad['dtype_real'])

    return pad, resd, rparams


def cr_prepare_basic_data_for_deconv(ims, psfs, pad, resd, rparams):
    '''Preparation of data for deconv in dsax_complete_recon.'''
    # use sliced result?
    im_shepp = resd['im_shepp'][[slice(resd['im_shepp'].shape[0]), ]+resd['im_shepp_resd']
                                [2]['shift_slices']] if rparams['im_shepp_use_slices'] else resd['im_shepp']
    psf_shepp = resd['psf_shepp'][[slice(resd['im_shepp'].shape[0]), ]+resd['im_shepp_resd']
                                  [2]['shift_slices']] if rparams['im_shepp_use_slices'] else resd['psf_shepp']
    # normalize results
    im_conf = (resd['im_conf']-np.min(resd['im_conf'],
               axis=tuple(pad['faxes']), keepdims=True))[:, np.newaxis]
    psf_conf = normNoff(resd['psf_conf'], dims=tuple(
        pad['faxes']), atol=0, method='sum')[:, np.newaxis]
    im_shepp = im_shepp-np.min(im_shepp, axis=tuple(pad['faxes']), keepdims=True)
    psf_shepp = normNoff(psf_shepp, dims=tuple(pad['faxes']), method='sum', atol=0)

    im_shepp, psf_shepp = im_shepp.astype(pad['dtype_real']), psf_shepp.astype(pad['dtype_real'])

    # factor?
    if 'im_fact' in rparams:
        im_conf *= rparams['im_fact']['conf']
        im_shepp *= rparams['im_fact']['shepp']

    # helpful variables
    ims_ft = nip.ft2d(ims).astype(pad['dtype_complex'])
    otfs = nip.ft2d(psfs).astype(pad['dtype_complex'])
    ims_rs = np.reshape(ims[rparams['pinmask_amdec'][1], :3], [np.prod(ims[rparams['pinmask_amdec'][1], :3].shape[:2]),
                                                               ]+list(ims.shape[2:]))
    psfs_rs = np.reshape(psfs[rparams['pinmask_amdec'][1], :3], [np.prod(
        psfs[rparams['pinmask_amdec'][1], :3].shape[:2]), ]+list(psfs.shape[2:]))

    return im_conf, psf_conf, im_shepp, psf_shepp, ims_ft, otfs, ims_rs, psfs_rs


def wavg_testparam(ims_ft, otfs, pad, resd):
    '''Search for optimal parameters of recon_weightedAveraging.'''
    resd['ismWA_reps_testrange'] = 10**(-np.arange(0, 9, 0.5))
    resd['ismWA_rtest'] = nip.image(
        np.zeros([len(resd['ismWA_reps_testrange']), ]+list(ims_ft.shape[-4:]), dtype=pad['dtype_real']))
    resd['ismWAN_rtest'] = nip.image(
        np.zeros([len(resd['ismWA_reps_testrange']), ]+list(ims_ft.shape[-4:]), dtype=pad['dtype_real']))
    resd['ismWAN_psf_rtest'] = nip.image(
        np.zeros([len(resd['ismWA_reps_testrange']), ]+list(ims_ft.shape[-4:]), dtype=pad['dtype_real']))
    for m, reps in enumerate(resd['ismWA_reps_testrange']):
        resd['ismWA_rtest'][m], resd['ismWA_weights'], resd['ismWAN_rtest'][m], resd['ismWAN_psf_rtest'][m] = recon_weightedAveraging(
            ims_ft, otfs, pincenter=pad['pincenter'], mask_eps=pad['eps_mask'], noise_norm=pad['noise_norm'], use_mask=pad['use_mask'], reg_reps=reps, reg_aeps=pad['reg_aeps'])
    # done?
    return resd


def wiener_testparam(ims_ft, otfs, pad, resd):
    '''Search for optimal parameters of recon_weightedAveraging.'''
    resd['wiener_reg_eps_testrange'] = 10**(-np.arange(0, 9, 0.5))
    resd['wiener_rtest'] = nip.image(
        np.zeros([len(resd['wiener_reg_eps_testrange']), ]+list(ims.shape[-4:]), dtype=ims.dtype))

    for m, reps in enumerate(resd['wiener_reg_eps_testrange']):
        resd['wiener_rtest'][m], resd['wiener_dict_rtest'] = recon_wiener(ims_ft, otfs, use_generalized=pad['wiener_use_generalized'], pincenter=pad['pincenter'], eps_mask=pad[
            'eps_mask'], use_mask=pad['use_mask'], reg_reps=reps, reg_aeps=pad['wiener_reg_aeps'], faxes=pad['faxes'], multiview_dim=pad['multiview_dim'])

        # done?
        return resd

def recon_wiener_list(ims_ft, otfs, pad, rparams, resd, ldim=0,):
    ''''''
    # prepare data
    if not ldim == 0:
        ims_ft = np.swapaxes(ims_ft, 0, ldim)
        otfs = np.swapaxes(otfs, 0, ldim)
    resd['wiener'] = nip.image(np.zeros([ims_ft.shape[0], ]+list(ims_ft.shape[2:])))
    resd['wiener_dict'] = [{} for m in range(ims_ft.shape[0])]

    # calculate wiener-filter
    for m, mIm in enumerate(ims_ft):
        resd['wiener'][m], resd['wiener_dict'][m] = recon_wiener(mIm, otfs[m], use_generalized=pad['wiener_use_generalized'], pincenter=pad['pincenter'], eps_mask=pad[
            'eps_mask'], use_mask=pad['use_mask'], reg_reps=rparams['wiener_reg_reps'][m], reg_aeps=pad['wiener_reg_aeps'], faxes=pad['faxes'], multiview_dim=pad['multiview_dim'])
    # if not ldim == 0:
    #    resd['wiener'] = np.swapaxes(resd['wiener'], 0, ldim)

    return resd


def recon_wavg_list(ims_ft, otfs, pad, rparams, resd, ldim=0,):
    ''''''
    # prepare data
    if not ldim == 0:
        ims_ft = np.swapaxes(ims_ft, 0, ldim)
        otfs = np.swapaxes(otfs, 0, ldim)
    resd['ismWA'] = nip.image(np.zeros([ims_ft.shape[0], ]+list(ims_ft.shape[2:])))
    resd['ismWA_weights'] = [{} for m in range(ims_ft.shape[0])]
    resd['ismWAN'] = nip.image(np.zeros(resd['ismWA'].shape))
    resd['ismWAN_psf'] = nip.image(np.zeros(resd['ismWA'].shape))

    # calculate wiener-filter
    for m, mIm in enumerate(ims_ft):
        resd['ismWA'][m], resd['ismWA_weights'][m], resd['ismWAN'][m], resd['ismWAN_psf'][m] = recon_weightedAveraging(
            ims_ft[:, m], otfs[:, m], pincenter=pad['pincenter'], mask_eps=pad['eps_mask'], noise_norm=pad['noise_norm'], use_mask=pad['use_mask'], reg_reps=rparams['reg_reps'][m], reg_aeps=pad['reg_aeps'])

    # if not ldim == 0:
    #    resd['wiener'] = np.swapaxes(resd['wiener'], 0, ldim)

    # done?
    return resd

def dsax_create_deconv_complist(resd, rparams,):
    '''Create Comparison list for finding NCC-best deconv results for dsax_complete_recon.'''
    compare_list = np.array([resd['ismWA_rtest'], resd['ismWAN_rtest'], resd['wiener_rtest'], resd['dec2D_conf_rtest'], nip.extract(resd['dec2D_shepp_rtest'], resd['ismWA_rtest'].shape), resd['dec2D_av_indivim_rtest'], nip.repmat(
        resd['dec2D_av_aim_rtest'][:, np.newaxis], [1, resd['wiener_rtest'].shape[1], 1, 1, 1])])
    obj = resd['dec2D_av_indivim_rtest'][resd['dec2D_av_indivim_rtest'].shape[0] //
                                         2, 2] if rparams['obj'] is None else rparams['obj']

    if rparams['im_shepp_use_slices']:
        slic = resd['im_shepp_resd'][2]['shift_slices']
        compare_list = compare_list[:, :, :, :, slic[-2], slic[-1]]
        obj = obj[slic]

    # swap axes such that order is: [Iex,ReconMETHOD,param_range,Z,Y,X]
    compare_list = np.transpose(compare_list, [2, 0, 1, 3, 4, 5])
    compare_list = normNoff(np.squeeze(compare_list), dims=(-2, -1))
    return compare_list, obj


def deconv_test_param_find_best_combination(compare_list, obj, resd, index_list=[]):
    '''Find NCC-best combinations of deconv parameters for dsax_recon_complete.'''

    resd['ncc_find_reps'] = []
    resd['ncc_find_reps_latex'] = []
    ncc_full_cols = np.arange(compare_list.shape[2])
    ncc_full_rows = ['WA', 'WAN', 'Wiener', 'CmDEC', 'PRmDEC', 'mDEC', 'amDEC']
    for m, cval in enumerate(compare_list):
        ncc_full, ncc_full_latex = get_cross_correlations(
            [np.squeeze(obj), ]*cval.shape[0], cval, rows=ncc_full_rows, cols=ncc_full_cols)
        resd['ncc_find_reps'].append(ncc_full)
        resd['ncc_find_reps_latex'].append(ncc_full_latex)

    # order(best regs)=[Iex,ReconMETHOD]
    if index_list == []:
        index_list = ['Iex1', 'Iex2', 'Iex3', 'NL1', 'NL2']
    best_regs = DataFrame(
        np.argmax(np.array(resd['ncc_find_reps']), axis=-1), columns=ncc_full_rows, index=index_list)
    resd['best_regs'] = DataFrame(best_regs)

    # translate to reg-values for methods
    resd['best_regs_sel'] = DataFrame(np.swapaxes(np.array([np.array(resd['ismWA_reps_testrange'])[best_regs.values[:, 0]], np.array(resd['ismWA_reps_testrange'])[
        best_regs.values[:, 1]], np.array(resd['wiener_reg_eps_testrange'])[best_regs.values[:, 2]], np.array(resd['dec2D_conf_rtest_dict']['lambdal_range'])[best_regs.values[:, 3]][:, 0], np.array(resd['dec2D_shepp_rtest_dict']['lambdal_range'])[best_regs.values[:, 4]][:, 0], np.array(resd['dec2D_av_indivim_rtest_dict']['lambdal_range'])[best_regs.values[:, 5]][:, 0], np.array(resd['dec2D_av_aim_rtest_dict']['lambdal_range'])[best_regs.values[:, 6]][:, 0]]), 0, 1), columns=ncc_full_rows, index=index_list)

    return resd, index_list