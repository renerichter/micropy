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
__version__ = "0.2a"
__maintainer__ = "René Lachmann"
__status__ = "Creation"
__date__ = "25.11.2019"
__last_update__ = "12.02.2020"

# %%
# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------
#
import numpy as np
import NanoImagingPack as nip

from .utility import transpose_arbitrary, get_nbrpixel
from .transformations import dct2

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
    im_res = cf_vollathF4_corr(im)
    res = np.mean(im_res, axis=(-2, -1))
    if im_out:
        return res, im_res
    else:
        return res


def cf_vollathF4_corr(im):
    '''
    Calculates the Vollath F4-correlation
    '''
    im = transpose_arbitrary(im, idx_startpos=[-2, -1], idx_endpos=[0, 1],direction='forward')
    im_res = im[:, :-2]*(im[:, 1:-1]-im[:, 2:])
    return transpose_arbitrary(im_res, idx_startpos=[-2, -1], idx_endpos=[0, 1],direction='forward')


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
    im = transpose_arbitrary(im, idx_startpos=[-2, -1], idx_endpos=[0, 1],direction='forward')
    if keep_size == True:
        imh = nip.extract(im, [im.shape[0]+4, im.shape[1]+4] + list(im.shape[2:]))
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
        im_res = nip.image(np.array(im_res))

    # insertion of new dimension leads to shift in endpos, but not startpos
    idx_endpos = [1, 2]
    
    return transpose_arbitrary(im_res,idx_startpos=[-2, -1], idx_endpos=idx_endpos,direction='backward')


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
    im = transpose_arbitrary(im, idx_startpos=[-2, -1], idx_endpos=[0, 1])
    return transpose_arbitrary(im[:-1, :]*im[1:, :],idx_startpos=[-2, -1], idx_endpos=[0, 1],direction='backward')
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
    sh = diff_sobel_horizontal(im)
    sv = diff_sobel_vertical(im)
    return impix * np.sum(sh*sh + sv*sv, axis=(-2, -1))


def diff_sobel_horizontal(im):
    '''
    Calculates the horizontal sobel-filter.
    Filter-shape: [[-1 0 1],[ -2 0 2],[-1 0 1]] -> separabel:  np.outer(np.transpose([1,2,1]),[-1,0,1])
    '''
    # use separability
    im = transpose_arbitrary(im, idx_startpos=[-2, -1], idx_endpos=[1, 0],direction='forward')

    x_res = im[:, 2:] - im[:, :-2]  # only acts on x
    xy_res = x_res[:-2] + 2*x_res[1:-1] + x_res[2:]  # only uses the y-coords

    #transpose back
    xy_res = transpose_arbitrary(xy_res, idx_startpos=[-2, -1], idx_endpos=[1, 0],direction='backward')
    return xy_res


def diff_sobel_vertical(im):
    '''
    Calculates the vertical sobel-filter.
    Filter-shape: [[-1,-2,-1],[0,0,0],[1,2,1]] -> separabel:  np.outer(np.transpose([-1,0,1]),[1,2,1])
    '''
    # use separability
    im = transpose_arbitrary(im, idx_startpos=[-2, -1], idx_endpos=[1, 0],direction='forward')
    
    x_res = im[:, :-2] + 2*im[:, 1:-1] + im[:, 2:]  # only x coords
    xy_res = x_res[2:] - x_res[:-2]  # further on y coords

    #transpose back
    xy_res = transpose_arbitrary(xy_res, idx_startpos=[-2, -1], idx_endpos=[1, 0],direction='backward')
    return xy_res


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
    im = transpose_arbitrary(im, idx_startpos=[-2, -1], idx_endpos=[0, 1])
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
# --------------------------------------------------------
#               FREQUENCY FILTERS
# --------------------------------------------------------
#

def filter_pass(im, filter_size=(10,), mode='circle', kind='low', filter_out=True):
    '''
    Does a fourier-based low-pass filtering of the desired shape. Filter-values: 0=no throughput, 1=max-throughput. 
    For now only works on 2D-images. 
    :param:
    =======
    :im:IMAGE:          Input image to wrok on. 
    :filter_size:LIST:  LIST of INT-sizes of the filter -> for circle it's e.g. just 1 value
    :mode:STRING:       Defines which lowpass_filtering shall be used -> 'Circle' (hard edges)
    '''
    # create filter-shape
    if mode == 'circle':
        # make sure that result is not bool but int-value for further calculation
        pass_filter = (nip.rr(im) < filter_size[0]) * 1
    else:  # just leave object unchanged
        pass_filter = 1
    # decide which part to use
    if kind == 'high':
        pass_filter = 1 - pass_filter
        pass
    elif kind == 'band':
        print("Not implemented yet.")
        pass
    else:  # kind == 'low' -> as filters are designed to be low-pass to start with, no change to filters
        pass
    # apply
    res_filtered = nip.ft(im, axes=(-2, -1)) * pass_filter
    res = nip.ift(res_filtered).real
    return res, res_filtered, pass_filter

# %%
# --------------------------------------------------------
#               MEASURES USING THE FILTERS
# --------------------------------------------------------
#


def image_sharpness(im, im_filters=['Tenengrad']):
    '''
    Calculating the image sharpness with different filters. Only for 2D inputs! For all neighboring pixel-using techniques the outer-most pixel-borders of the image are not used.

    TODO: unfinished!

    :param:
    =======
    :filters:LIST: List of possible filters. Options are:

    :out:
    =====
    res:LIST: List of all sharpness values calculated.
    '''
    #
    from numpy import mean
    #if 'Tenengrad' in im_filters:
    #    res.append(tenengrad(im))
    #elif 'VollathF4' in im_filters:
    #    res.append(vollathF4(im))

    #return res
    pass


# %%
# --------------------------------------------------------
#               GENERAL PIXEL-OPERATIONS
# --------------------------------------------------------
#
def local_annealing_atom(im,pos,mode='value',value=0, patch_size=[3,3],iter_anneal=False,iterations=1):
    '''
    Local annealing of a given position. 
    Function does inplace operation and hence technically no return has to be done!

    PARAMS
    =======
    :im:            (IMAGE) used image
    :pos:           (TUPLE) position that will be changed
    :mode:          (STRING) useable modes 
                        'value': overwrites with the given value at pos
                        'mean','max','min': calculates the respective value from the given patch_size around pos
    :value:         (FLOAT) value for overwriting
    :patch_size:    (LIST) size of patch used for calculation
    :iter_anneal:   (BOOL) whether iterative annealing should be used
    :iterations:    (INT) number of iterations for iterative annealing

    OUTPUTS:
    ========
    :im:        (IMAGE) resulting image

    EXAMPLE
    =======
    a = nip.rr([7,7])
    a[3,4] = 100
    b1 = local_annealing_atom(a.copy(),(3,4),mode='value',value=0)
    b2 = local_annealing_atom(a.copy(),(3,4),mode='max')
    b3 = local_annealing_atom(a.copy(),(3,4),mode='mean')
    b4 = local_annealing_atom(a.copy(),(3,4),mode='min')
    b5 = local_annealing_atom(a.copy(),(3,4),mode='mean',iter_anneal=True,iterations=3)
    toshowB = nip.catE((mipy.normto(a),mipy.normto(b1),mipy.normto(b2),mipy.normto(b3),mipy.normto(b4),mipy.normto(b5)))
    nip.v5(mipy.stack2tiles(toshowB))

    '''
    is_complex = True if im.dtype == 'complex' else False

    if mode=='value':
        im[pos] = value
    else:
        iterations = iterations if iter_anneal else 1
        for _ in range(iterations):
            a = nip.extractFt(im,ROIsize=patch_size,mycenter=pos) if is_complex else nip.extract(im,ROIsize=patch_size,centerpos=pos)
            if mode == 'mean':
                a = np.mean(a)
            elif mode == 'max':
                a = np.max(a)
            elif mode == 'min':
                a = np.min(a)
            else: 
                raise ValueError('Not implemented yet, but no problem! What do you wish?')
            im[pos] = a

    # done?
    return im 


def local_annealing(im,pos,mode='value',value=0,patch_size=[3,3],iter_anneal=False,iterations=1):
    '''
    Acts on list of positions. 
    Note: acts on input image in-place and hence out-put should only be used/thought as renaming due to input-data being changed as well. 
    
    :PARAMS:
    ========
    :pos:   (LIST) of TUPLE!
    
    For detailed description on usage-example and further parameters check "local_annealing_atom"-function.

    :EXAMPLE:
    =========
    a=nip.rr([7,7])
    a[1,2] = 100; a[3,4] = 100;
    b6 = local_annealing(a.copy(),[(3,4),(1,2)],mode='mean',patch_size=[4,4],iter_anneal=True,iterations=3)
    toshowA = nip.catE((mipy.normto(a),mipy.normto(b6)))
    nip.v5(mipy.stack2tiles(toshowA))
    
    '''
    for _,apos in enumerate(pos):
        im = local_annealing_atom(im,apos,mode=mode,value=value,patch_size=patch_size,iter_anneal=iter_anneal,iterations=iterations)
    
    # done?
    return im


    
