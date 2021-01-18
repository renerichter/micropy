'''
The ISM processing toolbox. 
'''
# %% imports
import numpy as np
import NanoImagingPack as nip
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage import binary_closing

# mipy imports
from .transformations import irft3dz
from .utility import findshift, midVallist, pinhole_shift, pinhole_getcenter
from .inout import stack2tiles


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


def otf_get_mask(otf, mode='rft', eps=1e-5, bool_mask=False, closing=None):
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

    # get parameters
    center_pinhole = int(np.floor(otf.shape[0]/2.0))
    center_max = np.max(np.abs(otf[center_pinhole]))
    eps = center_max * eps

    # calculate mask
    my_mask = (np.abs(otf[center_pinhole]) > eps)*1

    # close using the cosen structuring element
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

    # old mode???? WHAT IS HAPPENING HERE???
    if mode == 'old':
        zoff = otf.shape[1]//2  # z-offset
        proj_mask = my_mask_filled[zoff:].sum(axis=0)
    else:
        zoff = np.zeros(otf.shape[-2:], dtype=int)
        proj_mask = my_mask_filled.sum(axis=0) if mode == 'rft' else nip.catE(
            (int(my_mask_filled.shape[0]/2.0)-my_mask_filled[:int(my_mask_filled.shape[0]/2.0)].sum(axis=0), my_mask_filled.sum(axis=0)))

    if bool_mask:
        my_mask_filled = np.array(my_mask_filled, dtype=bool)
        my_mask = np.array(my_mask, dtype=bool)

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


def pinv_unmix(a, rcond=1e-15, svdnum=None, eps_reg=0, use_own=False):
    """
    Functions as a wrapper and a regularized version of the np.linalg.pinv.

    :PARAM:
    =======
    :a:         (ARRAY/IMAGE) Array to invert
    :rcond:     (FLOAT) relative cutoff
    :svdlim:    (INT) maximum nbr of SVD-values to keep
    :eps:       (FLOAT) regularizer

    """
    # use original pinv?
    if not use_own:
        res = np.linalg.pinv(a=a, rcond=rcond)

    else:
        a = a.conjugate()
        u, s, vt = np.linalg.svd(a, full_matrices=False)

        # discard small singular values; regularize singular-values if wanted -> note: eps_reg is 0 on default
        if svdnum == None:
            cutoff = np.array(rcond)[..., np.newaxis] * \
                np.amax(s, axis=-1, keepdims=True)
            large = s > cutoff

            # Tikhonov regularization analogue to idea 1/(OTF+eps) where eps becomes dominant when OTF<eps, especially reduces to 1/eps when OTF<<eps
            s = np.divide(s, s*s+eps_reg, where=large,
                          out=s) if eps_reg else np.divide(1, s, where=large, out=s)
            s[~large] = 0
        else:
            cutoff = s[:svdnum+1] if svdnum < len(s) else s
            cutoff = cutoff/(cutoff*cutoff+eps_reg) if eps_reg else 1/cutoff

            s = np.zeros(s.shape)
            s[:len(cutoff)] = cutoff

        res = np.matmul(np.transpose(vt), np.multiply(
            s[..., np.newaxis], np.transpose(u)))

    # done?
    return res


def unmix_matrix(otf, mode='rft', eps_mask=5e-4, eps_reg=1e-3, svdlim=1e-8, svdnum=None, hermitian=False, use_own=False, closing=None):
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

    :OUT:
    ====
    :otf_unmix:         (IMAGE) the (inverted=) unmix OTF of shape [z/2,pinhole_dim,Y,X]
    :otf_unmix_full:    (IMAGE) <DEPRECATED>!! -> for old reconstruction: unmix was done on half-space using FFT, hence full OTF had to be constructed manually -> not needed anymore because reconstruction is now done via RFT       
    :my_mask:           (IMAGE) mask used for defining limits of OTF-support
    :proj_mask:         (IMAGE) sum of mask along z-axis (used for inversion-range)
    '''
    # parameters/preparation
    otf_unmix = np.transpose(
        np.zeros(otf.shape, dtype=np.complex_), [1, 0, 2, 3])

    # calculate mask
    _, my_mask, proj_mask, zoff, _ = otf_get_mask(
        otf, mode='rft', eps=eps_mask, bool_mask=False, closing=closing)

    # loop over all kx,ky
    for kk in range(otf_unmix.shape[-2]):
        for jj in range(otf_unmix.shape[-1]):
            if my_mask[:, kk, jj].any():
                otf_unmix[zoff[kk, jj]:zoff[kk, jj]+proj_mask[kk, jj], :, kk, jj] = pinv_unmix(
                    otf[:, zoff[kk, jj]:zoff[kk, jj]+proj_mask[kk, jj], kk, jj], rcond=svdlim, svdnum=svdnum, eps_reg=eps_reg, use_own=use_own)  # otf[:, zoff[kk, jj]:zoff[kk, jj]+proj_mask[kk, jj], kk, jj] #otf[:, :,90,90]
    otf_unmix = nip.image(otf_unmix)

    # create full otf_unmix if necessary
    if mode == 'old':
        otf_unmix = otf_unmix[zoff:]
        otf_unmix = nip.image(otf_unmix)
        otf_unmix_full = np.transpose(
            otf_fill(np.transpose(otf_unmix, [1, 0, 2, 3])), [1, 0, 2, 3])
    else:
        otf_unmix_full = otf_unmix

    # done?
    return otf_unmix, otf_unmix_full, my_mask, proj_mask


def unmix_image_ft(im_unmix_ft, recon_shape=None, mode='rft', show_phases=False):
    '''
    Fouriertransforms along -2,-1 and applies RFT along -3 (=kz), as only half space was used due to positivity and rotation symmetry.
    '''
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
        #unmix_im = np.fft.irfft(unmix_im,n=recon_shape,axes=0)
        unmix_im = nip.image(np.fft.fftshift(np.fft.irfft(
            unmix_im, n=recon_shape[-3], axis=-3), axes=-3), im_unmix_ft.pixelsize)
    elif mode == 'rft':
        unmix_im = irft3dz(im_unmix_ft, recon_shape)
    elif mode == 'fft':
        unmix_im = nip.ift3d(im_unmix_ft)
    else:
        raise ValueError("No proper mode chosen!")

    # done?
    return unmix_im, recon_shape


def unmix_recover_thickslice(unmixer, im, unmixer_full=None):
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
    # the real thing
    im_unmix = nip.image(np.einsum('ijkl,jkl->ikl', unmixer, im))

    # only for backward-compatibility of reconstructed full-PSF
    if unmixer_full is not None:
        im_unmix_full = nip.image(
            np.einsum('ijkl,jkl->ikl', unmixer_full, im))
        im_unmix = [im_unmix, im_unmix_full]

    # done?
    return im_unmix


# %%
# ---------------------------------------------------------------
#                   ISM-Reconstruction
# ---------------------------------------------------------------


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


def ismR_widefield(im, detaxes):
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


# get distances
def vec2dist(avec, center, metric):
    paa


def ismR_confocal(im, detpos=None, pinsize=0, pinmetric=None, pincenter=None):
    '''

    '''
    # sanity
    if pincenter == None:
        pincenter, detproj = pinhole_getcenter(im, method='max')

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


def ismR_confocal_deprecated(im, pinsize=None, pinshape='circle', pincenter=None, store_masked=False):
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
    #imconf = mipy.ismR_confocal(im,axes=(0),pinsize=None,pinshape='circle',pincenter=None)

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
    #weightsn[~validmask] = 0

    # norm max of OTF to 1 = norm sumPSF to 1;
    #sigma2_imfl = mipy.midVallist(imfl,dims,keepdims=True).real
    if wmode == 'real':
        weights = weights.real
    elif wmode == 'conj':
        weights = np.conj(weights)
    elif wmode == 'abs':
        weights = np.abs(weights)
    else:
        pass
    #weights = otfl / sigma2_imfl
    weightsn = nip.image(np.copy(weights))
    weightsn[~np.repeat(validmask[np.newaxis],
                        repeats=otfl.shape[0], axis=-otfl.ndim)] = 0
    # 1/OTF might strongly diverge outside OTF-support -> put Mask
    eps = 0.01
    wsum = np.array(weightsn[0])
    wsum = np.divide(np.ones(weightsn[0].shape, dtype=weightsn.dtype), np.sum(
        weightsn+eps, axis=0), where=validmask, out=wsum)
    #wsum = 1.0/np.sum(weightsn+eps,axis=0)
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


def ismR_drawshift(shift_map):
    '''
    Vector-drawing of applied shifts.
    '''
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    V = np.array(shift_map)[:, :, 0]
    U = np.array(shift_map)[:, :, 1]
    #U, V = np.meshgrid(X, Y)
    q = ax.quiver(U, V, cmap='inferno')  # X, Y,
    ax.quiverkey(q, X=0.3, Y=1.1, U=1,
                 label='Quiver key, length = 1', labelpos='E')
    plt.draw()
    plt.plot()
    return fig, ax
