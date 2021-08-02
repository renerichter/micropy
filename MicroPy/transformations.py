'''
Transformations that were helpful will be collected here. No matter whether Hough, Radon, Fourier, Affine, ...
'''
import NanoImagingPack as nip
from scipy.fftpack import dct
import numpy as np

# %%
# ------------------------------------------------------------------
#               FOURIER (-like) TRAFOS
# ------------------------------------------------------------------


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


def rftnd(im, raxis=None, faxes=None):
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
    rax, ax = rft_getshape(im, raxis=raxis, faxes=faxes)
    return nip.ft(nip.rft(im, axes=rax), axes=ax)


def irftnd(im, s, raxis=None, faxes=None):
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
    rax, ax = rft_getshape(im, raxis=raxis, faxes=faxes)
    return nip.irft(nip.ift(im, axes=ax), shift_after=True, axes=rax, s=s)


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
def lp_norm(data, p=2, normaxis=None, keepdims=False):
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


def radial_sum(im: np.ndarray, maxbins: int = None, loop: bool = False, return_idx: bool = False) -> np.ndarray:
    """Calculates the radialsum. Use eg to calculate the mean frequency transfer efficiency and to have a quick look at the noise floor when using together with FT-images. Implementation is agnostic to input-datatype and hence needs to get proper (eg modulus of FT) input.

    Parameters
    ----------
    im : np.ndarray
        input image (eg FT)
    maxbins : int, optional
        maximum number of bins to be used to calculate frequenc, by default None

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
    >>> imr,idx = radial_sum(im,maxbins=None)
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

    # calculate maximum number of bins
    if maxbins is None:
        maxbins = np.ceil(lp_norm(np.array(im.shape[-2:])/2.0)).astype('int32')+1

    # get index-list and bin it
    idx = nip.rr(im.shape[-2:])
    idx = np.round(idx*(maxbins-1)/np.max(idx)).astype('int32')

    # calculate resulting radial sum via loop or directly
    if loop:
        # make sure for single list dimension
        if im.ndim > 3:
            im = reduce_shape(im, -2)

        # allocate space and calculate radial sums
        rsum = np.zeros([im.shape[0], maxbins], dtype=im.dtype)
        for m, ima in enumerate(im):
            rsum[m] = np.array([np.mean(ima[idx == m]) for m in range(maxbins)])
    else:
        if im.ndim > 2:
            idx = nip.repmat(idx, list(im.shape[:im.ndim-2])+[1, ]*2)
        rsum = np.array([np.mean(im[idx == m]) for m in range(maxbins)])

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
