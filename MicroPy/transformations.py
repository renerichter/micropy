'''
Transformations that were helpful will be collected here. No matter whether Hough, Radon, Fourier, Affine, ...
'''
from scipy.fftpack import dct
from pandas import DataFrame

from .utility import midVallist
from .general_imports import *

# %%
# ------------------------------------------------------------------
#               FOURIER (-like) TRAFOS
# ------------------------------------------------------------------


def ft_correct(im, faxes, im_shape=None, mode='fft', dir='fwd', dtype=None, norm='ortho', use_abs=False):
    '''
    In case of RFT: real axis is automatically the first of faxes.
    '''
    # create functions for calls
    #nip.irft(im=x, s=s, axes=faxes[::-1], norm='ortho')
    #nip.rft(im=x, axes=faxes[::-1], norm='ortho')
    func_dict = {'bwd': {
        'rft': lambda x, s: irftnd(im=x, s=s, raxis=faxes[0], faxes=faxes[1:], norm=norm),
        'fft': lambda x, s: nip.ift(im=x, axes=faxes, norm=norm)},
        'fwd': {
        'rft': lambda x, s: rftnd(im=x, raxis=faxes[0], faxes=faxes[1:], norm=norm),
        'fft': lambda x, s: nip.ft(im=x, axes=faxes, norm=norm)}, }

    # call
    im_ft = func_dict[dir][mode](x=im, s=im_shape)

    # abs?
    if use_abs:
        im_ft = np.abs(im_ft)

    # dtype
    if not dtype is None:
        im_ft = im_ft.astype(dtype)

    # done?
    return im_ft


def rft_getshape(im, raxis=None, faxes=None):
    '''
    Returns RFT-and FFT dimensions from input.

    TODO: exclude double dimensions and catch user errors.

    :PARAMS:
    ========
    :im:        (IMAGE) input image
    :raxis:     (INT) dimension of RFT
    :faxes:     (TUPLE) dimensions for FFT

    OUTPUT:
    =======
    :rax:       (TUPLE) real-axis
    :fax:       (TUPLE) FFT-axes
    '''
    # sanity-check for fourier-axes
    if faxes == None:
        fax = np.arange(im.ndim)
    else:
        fax = np.mod(faxes, im.ndim)

    # sanity-check for RFT-axis
    if raxis == None:
        rax = fax[-1]
        fax = fax[:-1]
    else:
        if raxis < 0:
            raxis = np.mod(raxis, im.ndim)
        rax = raxis
        if faxes == None:
            fax = np.delete(fax, raxis)
    return (rax,), tuple(fax)


def rft_check_axes_shape(raxis, faxes, s=None):
    if type(faxes) == int:
        faxes = [faxes, ]
    if type(raxis) == int:
        raxis = [raxis, ]
    axes = list(faxes)+list(raxis)

    # for irft axes and shape must be in the same order for np.fft.irfftn
    if not s is None:
        s = np.array(s)
        s = list(s[faxes])+list(s[raxis])

    return axes, s


def rftnd(im, raxis=None, faxes=None, norm='ortho'):
    '''
    Performs a nD-RFT forward on given or all axes axes. Especially, RFT can only be applied once (half-room selection). Hence, apply RFT first and then resulting FT. If no further axis is given, RFT will be applied along last dimension. 

    :PARAMS:
    ========
    :im:        (IMAGE) image in
    :raxis:     (INT) Axis along which the RFT should be performed
    :faxes:     (TUPLE) Axes along which the FFT should be performed -> if empty: selects all left-over dimensions

    :OUT:
    =====
    Fourier-transformed image.
    '''
    #rax, ax = rft_getshape(im, raxis=raxis, faxes=faxes)
    axes, _ = rft_check_axes_shape(raxis, faxes)

    return nip.rft(im=im,  axes=axes, norm=norm)  # nip.ft(nip.rft(im, axes=rax), axes=ax)


def irftnd(im, s, raxis=None, faxes=None, norm='ortho'):
    '''
    Performs a nD-RFT backward (=irft) on the given (or all) axes. Especially, RFT can only be applied once (half-room selection). Hence, apply FT first and the RFT to reverse the application process of rftnd. 

    :PARAMS:
    ========
    :im:        (IMAGE) image in
    :s:         (TUPLE) shape of output image (needed to shift highest frequency to according position in case of even/uneven image sizes)
    :raxis:     (INT) Axis along which the RFT should be performed
    :faxes:     (TUPLE) Axes along which the FFT should be performed -> if empty: selects all left-over dimensions

    :OUT:
    =====
    Inverse Fourier-Transformed image.
    '''
    #rax, ax = rft_getshape(im, raxis=raxis, faxes=faxes)
    # return nip.irft(nip.ift(im, axes=ax), shift_after=True, axes=rax, s=s)
    # due to change in NIP interface
    axes, s = rft_check_axes_shape(raxis, faxes, s)

    return nip.irft(im=im, s=s, axes=axes, norm=norm)  # ,axes=raxis)


def rft3dz(im):
    '''
    Performs a 3D-RFT forward on the last three axes. Especially, RFT can only be applied once (half-room selection). Hence, apply RFT first and then FT2D. 

    :PARAMS:
    ========
    :im:        image in

    :OUT:
    =====
    Fourier-transformed image.
    '''
    return nip.ft(nip.rft(im, axes=-3), axes=(-2, -1))


def irft3dz(im, s):
    '''
    Performs a 3D-RFT backward (=irft) on the last three axes. Especially, RFT can only be applied once (half-room selection). Hence, apply FT2D first and the RFT to reverse the application process of rft3dz. 

    :PARAMS:
    ========
    :im:        image in
    :s:         shape of output image (needed to shift highest frequency to according position in case of even/uneven image sizes)

    :OUT:
    =====
    Inverse Fourier-Transformed image.
    '''
    return nip.irft(nip.ift(im, axes=(-2, -1)), shift_after=False, axes=-3, s=s)


def dct2(im, forward=True, axes=[-2, -1]):
    '''
    Calculate a 2D discrete cosine transform of symmetric normalization.
    Motivated from here: https://www.reddit.com/r/DSP/comments/1c9mgs/2d_discrete_cosine_transform_calculation/
    '''
    direction = 2 if forward else 3
    return dct(dct(im, type=direction, axis=axes[0], norm='ortho'), type=direction, axis=axes[1], norm='ortho')


# %%
# ------------------------------------------------------------------
#               NORMS
# ------------------------------------------------------------------
def lp_norm(data: np.ndarray, p: float = 2, normaxis: tuple = None, keepdims: bool = False):
    """Calculates the LP-norm.

    Parameters
    ----------
    data : nd.array
        Input nD-data 
    p : int, optional
        norm-dimensionality, by default 2
    normaxis : tuple, optional
        norm-direction , by default None (=all directions)
    keepdims : bool, optional
        whether to keep singular dimensions, by default False

    Returns
    -------
    norm : nd.array
        calculated norm along chosen (or all) directions
    """
    if normaxis is None:
        norm = (np.sum(np.abs(data)**p, keepdims=keepdims))**(1.0/p)
    else:
        norm = (np.sum(np.abs(data)**p, axis=tuple(normaxis), keepdims=keepdims))**(1.0/p)
    return norm


def lp_sparsity(im, p=2):
    '''
    Measures the LP-sparsity having p=2 as default to ensure higher response for sparse images = sharp images.
    '''
    return np.prod(im.shape)**(p-1.0/p)*lp_norm(im, p=1.0/p)/lp_norm(im, p=p)


def mean_norm(im, axes=None, in_place=False):
    if in_place:
        im -= np.mean(im, axis=axes, keepdims=True)
    else:
        im = im-np.mean(im, axis=axes, keepdims=True)
    return im


def std_norm(im, axes=None, in_place=False):
    if in_place:
        im /= np.std(im, axis=axes, keepdims=True, ddof=1)
    else:
        im = im/np.std(im, axis=axes, keepdims=True, ddof=1)
    return im


def stat_norm(im, axes=None, in_place=False):
    im_std = np.std(im, axis=axes, keepdims=True, ddof=1)
    if in_place:
        mean_norm(im, axes=axes, in_place=in_place)
        im /= im_std
    else:
        im = mean_norm(im, axes=axes, in_place=in_place)/im_std
    return im

# %%
# ------------------------------------------------------------------
#               Metrics
# ------------------------------------------------------------------


def normalized_cross_correlation(im1: np.ndarray, im2: np.ndarray, axes: tuple = None, in_place: bool = False, omit_im: int = None):
    # based on: https://xcdskd.readthedocs.io/en/latest/cross_correlation/cross_correlation_coefficient.html
    norm_fac = 1.0/(im1.size-1) if axes is None else 1.0/np.prod([im1.shape[a] for a in axes])

    # accept to compare against pre-normed (or non-variant) objects
    im1_norm = im1 if (omit_im == 1) else stat_norm(im=im1, axes=axes, in_place=in_place)
    im2_norm = im2 if (omit_im == 2) else stat_norm(im=im2, axes=axes, in_place=in_place)

    # fix normalization degrees
    norm_fac = np.sqrt(norm_fac) if omit_im in [1, 2] else norm_fac

    # done?
    return norm_fac*np.sum(im1_norm*im2_norm, axis=axes)


def get_cross_correlations(imlist_rows, imlist_cols, rows=None, cols=None, rlatex=True, triang=False, ncc_pd=None):
    '''Calculates NCC for all combinations of a list'''
    # sanity
    ncc_list_latex = None

    # prepare
    if ncc_pd is None:
        ncc_list = np.zeros([len(imlist_cols), len(imlist_cols[0])])

        # calculate normalized cross-correlation for all images of the list
        for n, imr in enumerate(imlist_rows):
            for m, imc in enumerate(imlist_cols[n]):
                ncc_list[n, m] = normalized_cross_correlation(imr, imc)

        if triang:
            nccsel = (1-np.triu(np.ones(ncc_list.shape))).astype('bool')
            ncc_list[nccsel] = 'NaN'

        ncc_pd = DataFrame(ncc_list, columns=cols, index=rows)

    if rlatex:
        ncc_list_latex = ncc_pd.to_latex(float_format=lambda x: '%0.2f' %
                                         x, column_format='l'+'c'*len(imlist_cols), escape=False, na_rep='')
    # done?
    return ncc_pd, ncc_list_latex


def noise_normalize(im, mode='fft'):
    '''
    Divides Fourier-Image by center frequency. 
    '''
    # find center frequency
    if mode == 'fft':
        im_norm = 1/midVallist(im, dims=np.arange(im.ndim), keepdims=True)
    elif mode == 'rft':
        im_norm = 1/im.flatten()[0]
    else:
        print("Chosen Method unknown hence avaiding normalization.")
        im_norm = 1

    # norm
    im = im*im_norm

    # done?
    return im

def noise_ft_floorval(im_ft:nip.image,axes:tuple=(-2,-1),roi:list=[4,4]):
    '''
    Calculate noise floor from average of region around corners of image.
    Implemented for 2d for now
    '''
    # assure array order
    axes=np.sort(axes)[::-1]
    for m,ax in enumerate(axes):
        im_ft=np.swapaxes(im_ft,-(m+1),ax)
    
    # get shapes
    ishape = np.array(im_ft.shape)[-len(roi):]
    roi = np.array(roi).astype('int')

    #from itertools import combinations
    #[0,0]+list(combinations(([0,ishape[0]],[0,ishape[1]]),2))
    
    # for 2D for now -> extract regions and calculate mean
    roih=roi//2
    floorval = np.mean([nip.extract(im_ft,roi,ishape-roih),
            nip.extract(im_ft,roi,roih),
            nip.extract(im_ft,roi,[roih[0],ishape[1]-roih[1]]),
            nip.extract(im_ft,roi,[ishape[0]-roih[0],roih[1]])],axis=tuple([0,]+list(axes)))
    
    return floorval


def energy_regain(recon, groundtruth, atol=1e-20, snorm=True, use_indiv_abs=False, mode='fft'):
    # make sure distance between two images cannot be bigger than 1
    if snorm:
        recon = noise_normalize(recon, mode=mode)
        groundtruth = noise_normalize(groundtruth, mode=mode)

    # validmask
    nom = np.abs(recon)-np.abs(groundtruth) if use_indiv_abs else np.abs(recon - groundtruth)
    nom *= nom
    denom = np.abs(groundtruth)
    denom *= denom
    frac = np.ones(nom.shape)
    frac = np.divide(nom, denom, where=denom > atol, out=frac)
    return 1 - frac

# %%
# ------------------------------------------------------------------
#               DERIVATIVES
# ------------------------------------------------------------------


def euler_forward_1d(im, dim=0, dx=1):
    '''
    Calculates the forward euler with a stepsize of 1 on default.
    TODO: implement for further dimensions
    '''
    # get dimension and transpose list
    dim_list, dim_list2, dim = deriv_prepdim(im, dim)

    # rotate image
    im = np.transpose(im, dim_list2)

    # do derivation
    im_deriv = (im[1:] - im[:-1]) / dx

    # bring back to normal dimension-order
    return im_deriv.transpose(dim_list)


def deriv_prepdim(im, dim=0):
    '''
    get's the list to shift dimensions and bring intended dimension to frond, e.g. for derivations
    '''
    dim_list = list(range(im.ndim))

    # either deep-copy or new array -> both dim_list point to same obj
    dim_list2 = list(range(im.ndim))
    dim = im.ndim-1 if dim >= im.ndim else dim
    dim_list2 = [dim_list2.pop(dim), ] + dim_list2
    # print('dim_list={},dim_list2={},dim={}'.format(dim_list, dim_list2, dim))

    return dim_list, dim_list2, dim


# %%
# ------------------------------------------------------------------
#                           WRAPPING
# ------------------------------------------------------------------
def fwrap(f, n, T, a):
    '''
    wraps the period onto a function.
    '''
    return a*f(2*np.pi*n/T)


# %%
# ------------------------------------------------------------------
#                     COORDINATE-TRANSFORMATIONS
# ------------------------------------------------------------------


def polar2cartesian(outcoords, inputshape, origin, **kwargs):
    """Function to convert incoming [x,y]-coordinate pair to according polar [ρ,φ]-coordinate pair. 
    Can serve as u_func or for arrays of arbitrary length: 

    Directly copied from: https://stackoverflow.com/a/2418537

    Parameters
    ----------
    outcoords : tuple
        coordinates (x,y) to be used for calculation 
    inputshape : tuple
        shape of input-image -> assumes [ρ,φ] order
    origin : list
        offset points for the origin of the processing

    Returns
    -------
    (r,phi) : tuple
        resulting coordinates

    See Also
    --------
    polar2cartesian

    TODO: 
    -----
    1) fix rotation and offset
    2) not closing the circle, eg
    >>> a = np.zeros([90,360])
    >>> a[45,:] = 1
    >>> b = geometric_transform(a, polar2cartesian, order=0, output_shape=(
    a.shape[0] * 2, a.shape[0] * 2), extra_keywords={'inputshape': a.shape, 'origin': (a.shape[0], a.shape[0])})
    >>> print(b[45,90] - b[135,90]) # problem!
    """
    xindex, yindex = outcoords
    x0, y0 = origin
    x = xindex - x0
    y = yindex - y0

    r = np.sqrt(x*x + y*y)
    phi = np.arctan2(y, x)
    phi_index = np.round((phi + np.pi) * inputshape[1] / (2 * np.pi))

    return (r, phi_index)


def cartesian2polar(outcoords, origin, outshape, printout=False, **kwargs):
    """Inverse function to polar2cartesian.

    Parameters
    ----------
    outcoords : tuple
        coordinates (x,y) to be used for calculation 
    origin : tuple
        offset points for the origin of the processing
    outshape : tuple
        shape of output-image -> assumes [y,x] order

    Returns
    -------
    (r,phi) : tuple
        resulting coordinates

    See Also
    --------
    polar2cartesian

    TODO: 
    -----
    1) fix such that it works with scipy.ndimage.geometric_transform
    >>> asize = [90,90]
    >>> a = ((nip.rr(asize) < 45)*(nip.rr(asize) >=44))*1
    >>> b = geometric_transform(a, cart2polar, order=0, output_shape=outs_, extra_keywords={
                               'outshape': (asize[1]//2,360), 'origin': (0, 0)})
    """
    r, f = outcoords
    r0, f0 = origin

    # subtract offset
    r = r + r0
    f = f + f0

    x = r*np.cos(f/outshape[1]*2*np.pi)
    y = r*np.sin(f/outshape[1]*2*np.pi)
    if printout:
        print(
            f"r={np.round(r,2)}, f={np.round(f,2)}, x={np.round(x,2)}, y={np.round(y,2)}")
    return (x, y)


def radial_projection_sum(im, radius=None, **kwargs):
    """Calculates the Radial projection of an image by means of Converting it from cartesian to polar coordinates and summing over the angle-coordinate phi.
    Note: only works on 2D-arrays OR 3D-arrays, where all higher dimensions are listed into the 3rd dimension, hence eg dim = [X,Y,nz*nt*...]. Center for transformation is always image center (for now).

    Parameters
    ----------
    im : image
        input 2D/3D image
    radius : float, optional
        maximum radius to be used for transformation, by default None

    Returns
    -------
    res : image
        projected image
    """
    from skimage.transform import warp_polar

    # transform to polar coordinates
    is_multichannel = True if im.ndim > 2 else False
    res = warp_polar(im, radius=radius, multichannel=is_multichannel)
    res = np.sum(res, axis=0)

    # sum over phi to get only radial dependence
    return res


def radial_sum(im: np.ndarray, scale: np.ndarray = None, maxfreq: int = None, nbr_bins: int = None, loop: bool = False, return_idx: bool = False) -> np.ndarray:
    """Calculates the radialsum. Use eg to calculate the mean frequency transfer efficiency and to have a quick look at the noise floor when using together with FT-images. Implementation is agnostic to input-datatype and hence needs to get proper (eg modulus of FT) input.

    Parameters
    ----------
    im : np.ndarray
        input image (eg FT)
    scale: np.ndarray, optional
        scaling to be applied and used for mask shape, by default None
    maxfreq : int
        maximum frequency to be used for support calculation (in pixels), by default None
    nbr_bins : int, optional
        number of bins to be used to calculate frequenc, by default None
    loop : bool, optional
        whether to use loop routine, by default False
    return_idx : bool, optional
        whether to return calculated index-map, by default False

    Returns
    -------
    rsum : np.ndarray
        summed projection
    idx : image, optional
        indices of bins used

    Examples
    --------
    >>> im = np.zeros((8,8))
    >>> im[:,3:6] = 1
    >>> imr,idx = radial_sum(im,nbr_bins=None)
    >>> print(f"imr={imr}\nidx={idx}")
    imr=[1.         1.         0.625      0.375      0.16666667     0.  0.        ]
    idx=[[6 5 5 4 4 4 5 5]
    [5 4 4 3 3 3 4 4]
    [5 4 3 2 2 2 3 4]
    [4 3 2 2 1 2 2 3]
    [4 3 2 1 0 1 2 3]
    [4 3 2 2 1 2 2 3]
    [5 4 3 2 2 2 3 4]
    [5 4 4 3 3 3 4 4]]

    See Also
    --------
    lp_norm, radial_projection_sum
    """
    # make sure that non-complex data is used
    if im.dtype == 'complex':
        im = np.abs(im)

    if scale is None:
        scale = np.array([1, ]*im.ndim)

    # calculate maximum number of bins
    if nbr_bins is None:
        nbr_bins = np.ceil(lp_norm(np.array(im.shape[-2:])/2.0)).astype('int32')+1

    # get index-list and bin it
    idx = nip.rr(im.shape[-2:], scale=scale)
    norm_max = np.max(idx)

    if not maxfreq is None:
        idx[idx > maxfreq] = -maxfreq/(nbr_bins-1)
        norm_max = maxfreq
    idx = np.round(idx*(nbr_bins-1)/norm_max).astype('int32')

    # calculate resulting radial sum via loop or directly
    if loop:
        # make sure for single list dimension
        if im.ndim > 3:
            im = reduce_shape(im, -2)

        # allocate space and calculate radial sums
        rsum = np.zeros([im.shape[0], nbr_bins], dtype=im.dtype)
        for m, ima in enumerate(im):
            rsum[m] = np.array([np.mean(ima[idx == m]) for m in range(nbr_bins)])
    else:
        if im.ndim > 2:
            idx = nip.repmat(idx, list(im.shape[:im.ndim-2])+[1, ]*2)
        rsum = np.array([np.mean(im[idx == m]) for m in range(nbr_bins)])

    rsum[np.isnan(rsum)] = 0

    # done?
    if return_idx:
        return rsum, idx
    else:
        return rsum

def reduce_shape(im: np.ndarray, lim_dim: int = -2) -> np.ndarray:
    """Reduces shape of an array until the last dimension lim_dim.

    Parameters
    ----------
    im : np.ndarray
        input array/image
    lim_dim : int, optional
        limiting dimension, by default -2

    Returns
    -------
    np.ndarray
        reduced array

    Examples
    --------
    >>> a = np.ones([2,3,4,5,6])
    >>> a_red = mipy.reduce_shape(a,-3))
    >>> a_red.shape == (6,4,5,6)
    True
    """
    return np.reshape(im, [np.prod(im.shape[:lim_dim]), ]+list(im.shape[lim_dim:]))

# %%
# ------------------------------------------------------------------
#                     DAMPING
# ------------------------------------------------------------------
def dampEdge(im, rwidth, func=nip.cossqr, func_args=None, ret_mask=False, norm_range=False, in_place=False):
    norm_range = True if func == np.power else norm_range

    # get mask
    mask = np.ones(im.shape)
    damp_region = np.mod(np.round((np.array(im.shape)*np.array(rwidth))),im.shape).astype('int')
    for d in range(im.ndim):
        if damp_region[d] > 0:
            rawrange = np.arange(damp_region[d])
            rawrange = rawrange/np.max(rawrange) if norm_range else rawrange
            if func_args is None:
                if func == nip.cossqr:
                    func_args = {'length': damp_region[d], 'x0': damp_region[d]}
                elif func == nip.gaussf:
                    func_args = {'kernelSigma': damp_region[d]}
                elif func == np.power:
                    func_args = [10, ]
            mask.swapaxes(0, d)[:damp_region[d]] *= np.reshape(
                func(rawrange, *func_args), [damp_region[d], ]+[1, ]*(mask.ndim-1))
            mask.swapaxes(0, d)[-damp_region[d]:] *= np.reshape(func(rawrange[::-1],
                                                                     *func_args), [damp_region[d], ]+[1, ]*(mask.ndim-1))

    # apply mask in-place
    if in_place:
        im *= mask
    else:
        im = im*mask

    if ret_mask:
        return im, mask
    else:
        return im
