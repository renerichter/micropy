'''
   ~~~~ Package description ~~~~
    The Filters are build such that they assume to receive an nD-stack, but they only operate in a 2D-manner (meaning: interpreting the stack as a (n-2)D series of 2D-images). Further, they assume that the last two dimensions (-2,-1) are the image-dimensions. The others are just for stacking.



'''

# %%
# -------------------------------------------------------------------------
# AUTHOR-INFO for foobar-module
# -------------------------------------------------------------------------
#
__author__ = "René Lachmann"
__copyright__ = "Copyright 2019"
__credits__ = ["Jan Becker, Sebastian Unger, David McFadden"]
__license__ = "MIT"
__version__ = "0.1a"
__maintainer__ = "René Lachmann"
__status__ = "Creation"
__date__ = "25.11.2019"

# %%
# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------
#
import numpy as np
import NanoImagingPack as nip
from scipy.fftpack import dct
from .utility import transpose_arbitrary

# %%
# -------------------------------------------------------------------------
# Correlative Filters
# -------------------------------------------------------------------------
#


def cf_vollathF4(im, im_out=True):
    '''
    Calculates the Vollath-F4 correllative Sharpness-metric.
    '''
    # res = np.mean(im[:, :-1]*euler_forward_1d(im, dim=1, dx=0))
    im_res = cf_vollathF4_symmetric_corr(im)
    res = np.mean(im_res, axis=(-2, -1))
    if im_out:
        return res, im_res
    else:
        return res


def cf_vollathF4_corr(im):
    '''
    Calculates the Vollath F4-correlation
    '''
    trlist = transpose_arbitrary(im, idx_startpos=[-2, -1], idx_endpos=[0, 1])
    im = np.transpose(im, trlist)
    im_res = im[:, :-2]*(im[:, 1:-1]-im[:, 2:])
    return np.transpose(im_res, trlist)


def cf_vollathF4_symmetric(im, im_out=True):
    '''
    Calculates the symmetric Vollath-F4 correllative Sharpness-metric.
    '''
    impix = 1.0/get_nbrpixel(im, dim=[-2, -1])
    im_res = cf_vollathF4_symmetric_corr(im, keep_size=True)
    res = impix * (np.sum(np.abs(np.sum(im_res, axis=(-2, -1))), axis=0))
    if im_out:
        return res, im_res
    else:
        return res


def cf_vollathF4_symmetric_corr(im, keep_size=True):
    '''
    Calculates the symmetric Vollath F4-correlation
    '''
    trlist = transpose_arbitrary(im, idx_startpos=[-2, -1], idx_endpos=[0, 1])
    im = np.transpose(im, trlist)
    if keep_size == True:
        imh = nip.extract(im, [im.shape[0]+4, im.shape[0]+4])
        im_res = np.repeat(im[np.newaxis, :], repeats=4, axis=0)
        im_res[0] = (imh[:, :-2] * (imh[:, 1:-1] - imh[:, 2:]))[2:-2, 2:]
        im_res[1] = (imh[:, 2:] * (imh[:, 1:-1] - imh[:, :-2]))[2:-2, :-2]
        im_res[2] = (imh[:-2] * (imh[1:-1] - imh[2:]))[2:, 2:-2]
        im_res[3] = (imh[2:] * (imh[1:-1] - imh[:-2]))[:-2, 2:-2]
    else:
        im_res = list()
        im_res.append(im[:, :-2] * (im[:, 1:-1] - im[:, 2:]))
        im_res.append(im[:, 2:] * (im[:, 1:-1] - im[:, :-2]))
        im_res.append(im[:-2] * (im[1:-1] - im[2:]))
        im_res.append(im[2:] * (im[1:-1] - im[:-2]))
    return np.transpose(im_res, trlist)


def cf_vollathF5(im, im_out=True):
    '''
    Calculates the Vollath-F4 correllative Sharpness-metric.
    '''
    # res = np.mean(im[:, :-1]*euler_forward_1d(im, dim=1, dx=0))
    impix = 1.0/np.prod(im.shape[-2:])
    im_res = cf_vollathF5_corr(im)
    res = impix*(np.sum(im_res, axis=(-2, -1)) - 1.0 /
                 impix * np.sum(im_res, axis=(-2, -1))**2)
    if im_out:
        return res, im_res
    else:
        return res


def cf_vollathF5_corr(im):
    '''
    Calculates the Vollath F4-correlation
    '''
    trlist = transpose_arbitrary(im, idx_startpos=[-2, -1], idx_endpos=[0, 1])
    im = np.transpose(im, trlist)
    return np.transpose(im[:-1, :]*im[1:, :], trlist)
#
# -------------------------------------------------------------------------
# Differential Filters
# -------------------------------------------------------------------------
#


def diff_filters(im):
    pass


def diff_tenengrad(im):
    '''
    Calculates Tenengrad-Sharpness Metric.
    '''
    impix = 1.0 / np.sqrt(np.prod(im.shape))
    return impix * np.sum(diff_sobel_horizontal(im)**2 + diff_sobel_vertical(im)**2, axis=(-2, -1))


def diff_sobel_horizontal(im):
    '''
    Calculates the horizontal sobel-filter.
    Filter-shape: [[-1 0 1],[ -2 0 2],[-1 0 1]] -> separabel:  np.outer(np.transpose([1,2,1]),[-1,0,1])
    '''
    # use separability
    trlist = transpose_arbitrary(im, idx_startpos=[-2, -1], idx_endpos=[1, 0])
    im = np.transpose(im, trlist)
    x_res = im[:, 2:] - im[:, :-2]  # only acts on x
    xy_res = x_res[:-2] + 2*x_res[1:-1] + x_res[2:]  # only uses the y-coords
    return np.transpose(xy_res, trlist)


def diff_sobel_vertical(im):
    '''
    Calculates the vertical sobel-filter.
    Filter-shape: [[-1,-2,-1],[0,0,0],[1,2,1]] -> separabel:  np.outer(np.transpose([-1,0,1]),[1,2,1])
    '''
    # use separability
    trlist = transpose_arbitrary(im, idx_startpos=[-2, -1], idx_endpos=[1, 0])
    im = np.transpose(im, trlist)
    x_res = im[:, :-2] + 2*im[:, 1:-1] + im[:, 2:]  # only x coords
    xy_res = x_res[2:] - x_res[:-2]  # further on y coords
    return np.transpose(xy_res)


#
# -------------------------------------------------------------------------
# Spectral Filters
# -------------------------------------------------------------------------
#

def spf_dct_normalized_shannon_entropy(im, r0=100):
    '''
    Calculates the normalized Shannon entropy. Does not catch any division by 0 errors etc.

    :param:
    =======
    :r0:FLOAT:  Radius of DCT-cutoff ?
    '''
    # entropy_term =

    im_res = nip.image(dct2(im, forward=True, axes=[-2, -1]))
    imlp2 = lp_norm(im_res, p=2)
    en_el = np.abs(im_res/imlp2)
    sum_radius = (nip.rr(en_el, placement='positive') < r0) * 1
    res = - 2.0 * r0**(-2) * np.sum((en_el * np.log(en_el)) * sum_radius)
    return res, im_res


#
# -------------------------------------------------------------------------
# Staticstial Filters
# -------------------------------------------------------------------------
#
def stf_basic(im, printout=False):
    '''
    Basic statistical sharpness metrics: MAX,MIN,MEAN,MEDIAN,VAR,NVAR. Reducing the whole dimensionality to 1 value.
    '''
    im_res = list()
    use_axis = (-2, -1)
    im_res.append(np.max(im, axis=use_axis))
    im_res.append(np.min(im, axis=use_axis))
    im_res.append(np.mean(im, axis=use_axis))
    im_res.append(np.median(im, axis=use_axis))
    im_res.append(np.var(im, axis=use_axis))
    im_res.append(im_res[4]/im_res[2]**2)  # normalized variance (NVAR)
    if printout == True:
        print("Basic analysis yields:\nMAX=\t{}\nMIN=\t{}\nMEAN=\t{}\nMEDIAN=\t{}\nVAR=\t{}\nNVAR=\t{}".format(
            im_res[0], im_res[1], im_res[2], im_res[3], im_res[4], im_res[5]))
    return np.array(im_res)


def stf_kurtosis(im, switch_axis=False):
    '''
    Forth moment ( https://en.wikipedia.org/wiki/Kurtosis ).
    Scipy-Kurtosis ->
    '''
    # from scipy.stats import kurtosis
    # res = list()
    # res.append(np.mean((im-np.mean(im))**4) / np.var(im)**2)
    # res.append(np.mean(np.abs(im-np.mean(im))**4) / (np.var(im)**2))
    # res.append(np.mean(np.abs(im-np.mean(im))**4) / np.mean(np.abs(im-np.mean(im))**2))
    # res_comp = kurtosis(im)
    if switch_axis:
        use_axis = (0, 1)
    else:
        use_axis = (-2, -1)
    res = np.mean((im-np.mean(im, axis=use_axis))**4) / \
        np.var(im, axis=use_axis)**2
    return res


def stf_diffim_kurtosis(im):
    '''
    Difference image Kurtosis. Implemented for 2D image.
    '''
    trlist = transpose_arbitrary(im, idx_startpos=[-2, -1], idx_endpos=[0, 1])
    im = np.transpose(im, trlist)
    return stf_kurtosis(im[1:, 1:] - im[:-1, :-1], switch_axis=True)


def stf_histogram_entropy(im, bins=256, im_out=True):
    '''
    Calculates the histogram entropy. Measure is still dependend on image-size and contrast. Improve to make independent?
    TODO: fix for new dimension ordering. Changed from [Y,X] = [0,1] to [-2,-1].
    '''
    if im.ndim > 2:
        # loop over all additional axes
        oldshape = im.shape[:-2]
        imh = np.reshape(
            im, newshape=[np.prod(im.shape[:-2]), im.shape[-2], im.shape[- 1]], order='C')
        im_hist = ['', ]
        for cla in range(imh.shape[-1]):
            im_hist.append(np.histogram(imh[cla, :, :], bins=bins)[1],)
        im_hist.pop(0)
        im_hist = np.array(im_hist)
        im_hist = np.reshape(im_hist, newshape=tuple(
            [len(im_hist[0]), ] + list(oldshape)))
    elif im.ndim == 2:
        im_hist = np.histogram(im, bins=bins)[1]
    else:
        raise ValueError("Wrong input dimensions.")
    # some Photoncases yield NAN -> given empty bins? what to do?
    im_res = im_hist * np.log(im_hist)
    res = np.sum(im_res, axis=(0))
    if im_out:
        return res, im_res, im_hist
    else:
        return res


def stf_lp_sparsity(im, p=2):
    '''
    Measures the LP-sparsity having p=2 as default to ensure higher response for sparse images = sharp images.
    '''
    return np.prod(im.shape)**(p-1.0/p)*lp_norm(im, p=1.0/p)/lp_norm(im, p=p)


# %%
# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------
#
def dct2(im, forward=True, axes=[-2, -1]):
    '''
    Calculate a 2D discrete cosine transform of symmetric normalization.
    Motivated from here: https://www.reddit.com/r/DSP/comments/1c9mgs/2d_discrete_cosine_transform_calculation/
    '''
    direction = 2 if forward else 3
    return dct(dct(im, type=direction, axis=axes[0], norm='ortho'), type=direction, axis=axes[1], norm='ortho')


def lp_norm(im, p=2):
    '''
    Calculates the LP-norm.
    '''
    return (np.sum(np.abs(im)**p))**(1.0/p)


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
