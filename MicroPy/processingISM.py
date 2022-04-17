'''
The ISM processing toolbox.
'''
# %% imports
import numpy as np
import NanoImagingPack as nip
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage import binary_closing
from deprecated import deprecated

# mipy imports
from .transformations import irft3dz, ft_correct
from .utility import findshift, midVallist, pinhole_shift, pinhole_getcenter, add_multi_newaxis, shiftby_list, subslice_arbitrary, mask_from_dist
from .inout import stack2tiles, format_list
from typing import Union


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


# %%
# ---------------------------------------------------------------
#                   ISM-Reconstruction
# ---------------------------------------------------------------

def recon_genShiftmap(im, pincenter, im_ref=None, nbr_det=None, pinmask=None, shift_method='nearest', shiftval_theory=None, roi=None, shift_axes=None, printmap=False):
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
    """
    # sanity
    if shift_axes is None:
        shift_axes = np.arange(1, im.ndim)
    if not type(shift_axes) == np.ndarray:
        shift_axes = np.mod(np.array(shift_axes), im.ndim)

    im_ref = im[pincenter] if im_ref is None else im_ref

    if roi is not None:
        im = subslice_arbitrary(im, roi, axes=np.arange(im.ndim))

    if shift_method in ['nearest', 'theory']:
        # convert pincenter to det-space
        pcu = np.unravel_index(pincenter, nbr_det)

        # find shift per period-direction
        factors = []
        shiftvec = []
        for m in range(len(nbr_det)):

            # calculate shift, but be aware to not leave the array
            if shift_method == 'nearest':
                period = int(np.prod(np.array(nbr_det[(m+1):])))
                shifth, _, _, _ = findshift(im_ref,
                                            im[np.mod(pincenter+period, im.shape[0])],  100)
            else:
                shifth = shiftval_theory[m]
            shiftvec.append(shifth)

            # create factors
            factors.append(nip.ramp(nbr_det, ramp_dim=m,
                                    placement='corner').flatten()-pcu[m])

        # generate shiftvec -> eg [eZ,eY,eX] where eZ=[eZ1,eZ2,eZ3]
        shiftvec = np.array(shiftvec)
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
        if not pinmask is None:
            shift_map = shift_map[pinmask]

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
    if printmap:
        sm = np.reshape(shift_map, nbr_det + list(shift_map[0].shape))
        if sm.shape[-1] > 2:
            useaxes = [-2, -1]
        figS, axS = recon_drawshift(sm, useaxes=useaxes)
    else:
        figS, axS = [], []

    # done?
    return shift_map, figS, axS


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

    # pad image before shifting
    if shift_pad:
        pass

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
            imshifted[m] = nip.shift(im=im[m],delta=shifts[m])#shifts,axes=shift_axes#shift_map

        # fixup shape
        imshifted = np.reshape(imshifted,im_origshape)
        imshifted.pixelsize = im.pixelsize
    # unpad
    if shift_pad:
        pass

    # done
    return imshifted


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


def recon_sheppardSUM(im, nbr_det, pincenter, im_ref=None, shift_method='nearest', shift_map=[], shift_roi=None, shiftval_theory=None, shift_axes=None, pinmask=None, pinfo=False, sum_method='normal', shift_style='parallel', shift_use_copy=True, ret_nonSUM=False):
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
        shift_map, figS, axS = recon_genShiftmap(
            im=im, im_ref=im_ref, pincenter=pincenter, nbr_det=nbr_det, pinmask=pinmask, shift_method=shift_method, shiftval_theory=shiftval_theory, roi=shift_roi, shift_axes=shift_axes)

    # apply shifts
    if not pinmask is None:
        im = im[pinmask]
        shift_map = shift_map[pinmask]
    ims = recon_sheppardShift(im, shift_map, method=shift_style, use_copy=shift_use_copy)

    # different summing methods
    imshepp, weights = recon_sheppardSUMming(im=ims, pinmask=None, sum_method=sum_method)

    # info for check of energy conservation
    if pinfo:
        print("~~~~~\t Results of SheppardSUM-routine:\t~~~~~")
        print(
            f"SUM(im)={np.sum(im)}\nim.shape={im.shape}\nSUM(imshifted)-SUM(im)={np.sum(ims)-np.sum(im)}\nSUM(imshepp)-SUM(im[pinmask])={np.sum(imshepp)-np.sum(im)}\nShift-Operations={np.sum(pinmask)}")
        print("~~~~~\t\t\t\t\t\t~~~~~")

    res_dict = {'shift_map': shift_map, 'pinmask': pinmask,
                'pincenter': pincenter, 'weights': weights}

    if ret_nonSUM:
        res_dict['ims'] = ims

    # return created results
    return imshepp, res_dict


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


# %% ---------------------------------------------------------------
# ---                         DEPRECATED                         ---
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
