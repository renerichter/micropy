'''
All functions relating to fitting come in here. 
'''
# %% imports
# extern
import numpy as np
import NanoImagingPack as nip

# intern
from .utility import get_coords_from_markers, center_of_mass, findshift, normNoff
from .transformations import lp_norm, dampEdge
from .inout import stack2tiles

# %%
# ------------------------------------------------------------------
#                       PSF-RECONSTRUCTION
# ------------------------------------------------------------------


def extract_PSFlist(im, ref=None, markers=None, im_axes=(-2, -1), bead_roi=[16,16], **kwargs):
    """Extract PSFs for a list of images, eg a set of Pinholes (ISM).
    For now: only implented in [LIST,Y,X] dimension.

    Parameters
    ----------
    im : list
        List of images (ndarray)
    ref : image (ndarray), optional
        reference image to be used for finding marker positions (eg mean image along stack-dimension), by default None
    markers : list, optional
        see 'extract_multiPSF', by default None
    list_dim : int, optional
        dimension to be understood as stacking-dimension, by default 0
    im_axes : tuple, optional
        see 'extract_multiPSF', by default (-2, -1)
    bead_roi : list, optional
        see 'extract_multiPSF', by default [16, 16]

    Returns
    -------
    gaussfits, residuums, beadls, paras, markers --> see 'extract_multiPSF'

    Example
    -------
    will follow

    See Also
    --------
    extract_multiPSF

    TODO
    ----
    1) add example!
    """
    # get markers from reference
    if markers is None:
        ref = im[0] if ref is None else ref
        markers, viewer = get_coords_from_markers(ref, viewer=None, cdim=len(im_axes))

    # allocate arrays
    beadf_list=nip.image(np.zeros([im.shape[0],]+bead_roi,dtype=im.dtype))
    res_dict_list = [{},]*im.shape[0]

    # loop through all list-entries
    for m,mIm in enumerate(im):
        beadf_list[m], res_dict_list[m] = extract_multiPSF(mIm, markers=markers, im_axes=im_axes, bead_roi=bead_roi, **kwargs)

    # done?
    return beadf_list,res_dict_list


def extract_multiPSF(im, markers:list=[], im_axes:tuple=(-2, -1), bead_roi:list=[16, 16], bead_tol:float=-1.0,compare=False, pre_damp:dict={}, post_damp:dict={},shift_subpix:bool=True):
    """ Extracts PSF from image. 
    Extracts ROI from Marker-Positions, aligns selections and fits Gauss to them.
    Implemented for 2D and 3D.

    Parameters
    ----------
    im : image (ndarray)
        input image
    markers : list, optional
        Positions of ROI centers to be used for evaluation, by default None
    im_axes : tuple, optional
        axes of im to be used as image-dimensions for analysis, by default (-2, -1)
    bead_roi : list, optional
        region of interest for bead extraction, by default [16, 16]
    compare : bool, optional
        whether a small comparison of the results shall be displayed, by default False
    damp: dict, optional
        params for mipy.dampEdge function to apply damping to beads

    Returns
    -------
    gaussfit : image
        (Gauss-fitted) PSF
    residuum : image
        Difference between shifted and mean-summed bead-image and gaussian-fit
    beads : 
        extracted and shifted individual beads used for calculation
    para : list
        Gauss-fit parameters, shape: (amplitude, center_x, center_y, sigma_x, sigma_y, rotation, offset); only for 2D: additionally [FWHM_x,FWHM_y] are appended on the list
    markers : list
        Center Positions of selected beads used for PSF calculation
    bead_comp : list
        reference to viewers that display bead-comparison if compare==True, by default []

    Example
    -------
    >>> obj = mipy.generate_obj_beads([128,128], [2,5], 20, 10).astype('float32')
    >>> beadf, res_dict = mipy.extract_multiPSF(obj, markers=None, im_axes=(-2, -1), bead_roi=[16, 16], compare=False)

    See Also
    --------
    extract_PSFlist, get_coords_from_markers, center_of_mass, findshift, nip.shift, nip.fit_gauss2D, nip.multiROIExtract

    TODO
    ----
    1) implement residuum
    2) catch user-errors
    """
    # parameter
    roi_center = np.array(bead_roi)//2

    # select beads
    if markers == []:
        markers, viewer = get_coords_from_markers(im, viewer=None, cdim=len(im_axes))

    # extract ROI
    beads = nip.multiROIExtract(im=im-np.min(im), centers=np.array(
        markers, dtype=int), ROIsize=bead_roi, origin='corner')

    # damp Selections
    if not pre_damp == {}:
        beads = dampEdge(beads, **pre_damp)

    # get rid of NaNs
    beads -= np.min(beads, keepdims=True)
    beads[np.isnan(beads)] = 0

    # get center of mass and distances
    com = center_of_mass(beads, com_axes=im_axes, im_axes=im_axes, placement='corner')
    com_mean = np.mean(com, axis=-1, keepdims=True)
    com_dist = lp_norm(com-com_mean, p=2, normaxis=(-2,))
    # print(f"l2-distances={com}")
    bead_ref = np.argmin(com_dist)
    # print(f"referenceBead={bead_ref}")

    # correlate and find respective centers
    for m, bead in enumerate(beads):
        if not m == bead_ref:
            myshift, _, _, _ = findshift(beads[bead_ref], bead)
            #nip.v5(nip.catE(bead,nip.shift2Dby(bead, myshift),beads[bead_ref]))
            myshift = myshift if shift_subpix else np.round(myshift, 0)
            beads[m] = nip.shift(bead, myshift, axes=im_axes, dampOutside=True)

    # sum to mean
    bead_sum = np.mean(beads, axis=0)

    # shift mean to center -> cut away offset by borders and bounding effects to find proper center pos
    #bead_sum_com = np.mean(center_of_mass(bead_sum, com_axes=im_axes,im_axes=im_axes, placement='corner'), axis=-1, keepdims=True)
    bss=np.array(bead_sum.shape,dtype='int')
    bead_sum_com = center_of_mass(nip.extract(nip.extract(bead_sum,bss//2),bss), com_axes=im_axes, im_axes=im_axes, placement='corner')

    beadf_shift = roi_center-bead_sum_com if shift_subpix else np.round(roi_center-bead_sum_com, 0).astype('int')
    beadf = nip.shift(bead_sum, beadf_shift, axes=im_axes, dampOutside=True)

    # damp shift-artefacts and border problems
    if not post_damp == {}:
        #beads = dampEdge(beads, **pre_damp)
        beadf = nip.DampEdge(beadf,**post_damp)

    # sum-normalize to 1
    beadf-=np.min(beadf,keepdims=True)
    beadf_sum=np.sum(beadf)
    if bead_tol<0:
        bead_tol=float(np.median(beadf)/100)
    beadf=normNoff(beadf, method='sum', atol=bead_tol, direct=False)

    # fit 2D-Gauss -> para =(amplitude, center_x, center_y, sigma_x, sigma_y, rotation, offset)
    if beadf.ndim == 2:
        para, gaussfit = nip.fit_gauss2D(beadf,startPos=np.array(beadf.shape)//2)
        para = [para, ]
    else:
        # simple work-around for 3D --> need proper fit with rotations etc!
        para = []
        gaussfit = []
        slicing = [slice(m) for m in beadf.shape]
        for m in range(beadf.ndim):
            slice_copy = slicing[m]
            slicing[m] = beadf.shape[m]//2
            parah, gaussfith = nip.fit_gauss2D(beadf[slicing])
            #make sure sigmas are positive
            parah[3]=np.abs(parah[3])
            parah[4]=np.abs(parah[4])
            para.append(parah)
            gaussfit.append(gaussfith)
            slicing[m] = slice_copy
        gaussfit = gaussfit[0][np.newaxis]*gaussfit[1][:, np.newaxis]*gaussfit[2][:, :, np.newaxis]
        gaussfit *= (np.max(beadf)/np.max(gaussfit))
    residuum = np.abs(np.array(gaussfit) - beadf)

    # convert to two-2D
    for m, mpara in enumerate(para):
        FWHM_x = 2*(mpara[3] * np.sqrt(-np.log(0.5)*2))
        FWHM_y = 2*(mpara[4] * np.sqrt(-np.log(0.5)*2))
        para[m] = list(mpara)+[FWHM_x, FWHM_y]
    para = np.array(para)
    para_names=['amplitude', 'center_x', 'center_y', 'sigma_x', 'sigma_y', 'rotation', 'offset','FWHMx','FWHMy']

    # do a small comparison
    bead_comp = []
    if compare:
        compare_beads = nip.cat((beads, beadf, gaussfit, residuum), axis=0, destdims=3)
        v1 = nip.v5(stack2tiles(compare_beads))
        v2 = nip.v5(compare_beads[:, np.newaxis])
        bead_comp = [v1, v2]

    # combine to result
    res_dict = {'gaussfit': gaussfit, 'residuum': residuum, 'beads': beads,'beadf_sum':beadf_sum,
                'para': para, 'para_names':para_names,'markers': markers, 'bead_comp': bead_comp}

    # done?
    return beadf, res_dict
