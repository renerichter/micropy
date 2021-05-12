"""
---------------------------------------------------------------------------------------------------

	@author René Lachmann
	@email herr.rene.richter@gmail.com
	@create date 2019-11-25 10:26:14
	@modify date 2021-05-11 18:16:03
	@desc The Filters are build such that they assume to receive an nD-stack, but they only operate in a 2D-manner (meaning: interpreting the stack as a (n-2)D series of 2D-images). Further, they assume that the last two dimensions (-2,-1) are the image-dimensions. The others are just for stacking.

---------------------------------------------------------------------------------------------------
"""

# %%
# -------------------------------------------------------------------------
# AUTHOR-INFO for foobar-module
# -------------------------------------------------------------------------
#
__author__ = "René Lachmann"
__copyright__ = "Copyright 2019"
__credits__ = ["Jan Becker, Sebastian Unger, David McFadden"]
__license__ = "MIT"
__version__ = "0.3a"
__maintainer__ = "René Lachmann"

# %%
# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------
#
import numpy as np
import NanoImagingPack as nip

from .utility import transpose_arbitrary, get_nbrpixel
from .transformations import dct2, lp_norm
from .simulation import generate_spokes_target

# %%
# -------------------------------------------------------------------------
# Preprocessing Tools
# -------------------------------------------------------------------------
#


def filter_prep_im(im, axes=(-2, -1), direction='forward', pad_shape=None, faxes=(0, 1),):
    """Simple image preparation used for image-filtering

    Parameters
    ----------
    im : image
        input image
    axes : tuple, optional
        data-axes to be used, by default (-2, -1)
    faxes : tuple, optional
        assignment axes for data-axes for further usage, by default (0,1)
    direction : str, optional
        direction of processing, by default 'forward'
            'forward': transpose,count pix, pad
            'backward': transpose-back,add image
    pad_shape : tuple, optional
        shape to be used for padding. If not None, padding is applied, by default None

    Returns
    -------
    im, npix : image and float
        in case of forward direction
    res : list
        in case of backward-direction

    Raises
    ------
    ValueError
        direction has to be chosen properly
    """

    # bring x,y-axes to the front of array for simple notation and count pixels; optionally add padding
    if direction == 'forward':
        im = transpose_arbitrary(im, idx_startpos=list(
            axes), idx_endpos=list(faxes), direction='forward')
        npix = float(np.prod(im.shape[:2]))
        if pad_shape is not None:
            if len(pad_shape) < im.ndim:
                pad_shape = tuple(list(pad_shape) + list(((0, 0),)*(im.ndim-len(pad_shape))))
            im = np.pad(im, pad_shape, mode='constant', constant_values=0)
        return im, npix

    # transform back and add transformed image to output
    elif direction == 'backward':
        return transpose_arbitrary(im, idx_startpos=list(axes), idx_endpos=list(faxes), direction='backward')

    else:
        raise ValueError('Wrong direction used.')


# %%
# -------------------------------------------------------------------------
# Correlative Filters
# -------------------------------------------------------------------------
#

@staticmethod
def cf_vollathF4(im, faxes=(0, 1), **kwargs):
    '''
    Calculates the Vollath-F4 correllative Sharpness-metric.
    '''
    im_filtered = cf_vollathF4_corr(im)
    res = np.mean(im_filtered, axis=faxes)

    # done?
    return res, [im_filtered, ]


def cf_vollathF4_corr(im):
    '''
    Calculates the Vollath F4-correlation
    '''

    im_res = im[:, 2:-2]*(im[:, 3:-1]-im[:, 4:])

    return im_res


@staticmethod
def cf_vollathF4_symmetric(im, faxes=(0, 1), **kwargs):
    '''
    Calculates the symmetric Vollath-F4 correllative Sharpness-metric.
    '''
    im_res = cf_vollathF4_symmetric_corr(im)
    res = np.sum(np.abs(np.mean(im_res, axis=faxes)), axis=0)

    return res, im_res


def cf_vollathF4_symmetric_corr(im):
    '''
    Calculates the symmetric Vollath F4-correlation
    '''
    # imh = nip.extract( im, [im.shape[0]+4, im.shape[1]+4] + list(im.shape[2:]))

    im_res = np.repeat(im[np.newaxis, 2:-2, 2:-2], repeats=4, axis=0)
    im_res[0] = im[2:-2, 3:-1] - im[2:-2, 4:]
    im_res[1] = im[2:-2, 1:-3] - im[2:-2, :-4]
    im_res[2] = im[3:-1, 2:-2] - im[4:, 2:-2]
    im_res[3] = im[1:-3, 2:-2] - im[:-4, 2:-2]
    im_res *= im[np.newaxis, 2:-2, 2:-2]

    return im_res


@staticmethod
def cf_vollathF5(im, faxes=(0, 1), **kwargs):
    '''
    Calculates the Vollath-F4 correllative Sharpness-metric.
    '''
    # res = np.mean(im[:, :-1]*euler_forward_1d(im, dim=1, dx=0))

    im_mean = np.mean(im[:, 1:-1], axis=faxes)
    im_mean *= im_mean
    im_filtered = im[:, 1:-1]*im[:, 2:]
    res = np.mean(im_filtered, axis=faxes) - im_mean

    return res, [im_filtered, ]

#
# -------------------------------------------------------------------------
# Differential Filters
# -------------------------------------------------------------------------
#


@ staticmethod
def diff_tenengrad(im, faxes=(0, 1), **kwargs):
    '''
    Calculates Tenengrad-Sharpness Metric.
    '''
    sh = diff_sobel_horizontal(im)
    sv = diff_sobel_vertical(im)
    res = np.mean(sh*sh + sv*sv, axis=faxes)

    return res, [sh, sv]


def diff_sobel_horizontal(im):
    '''
    Calculates the horizontal sobel-filter.
    Filter-shape: [[-1 0 1],[ -2 0 2],[-1 0 1]] -> separabel:  np.outer(np.transpose([1,2,1]),[-1,0,1])
    '''
    x_res = im[:, 2:] - im[:, :-2]  # only acts on x
    im_filtered = x_res[:-2] + 2*x_res[1:-1] + x_res[2:]  # only uses the y-coords

    return im_filtered


def diff_sobel_vertical(im):
    '''
    Calculates the vertical sobel-filter.
    Filter-shape: [[-1,-2,-1],[0,0,0],[1,2,1]] -> separabel:  np.outer(np.transpose([-1,0,1]),[1,2,1])
    '''
    x_res = im[:, :-2] + 2*im[:, 1:-1] + im[:, 2:]  # only x coords
    im_filtered = x_res[2:] - x_res[:-2]  # further on y coords

    return im_filtered


@ staticmethod
def diff_brenners_measure(im, faxes=(0, 1), **kwargs):
    """Calculates differential brenners measure.
    """
    # calculate measure
    im_filtered = im[:-2]+im[2:]
    res = np.mean(im_filtered, axis=faxes)

    # done?
    return res, [im_filtered, ]


@ staticmethod
def diff_absolute_laplacian(im, faxes=(0, 1), **kwargs):
    """Calculates Absolute Laplacian.
    """
    # calculate measure
    im_filtered = abs(2*im[1:-1, 1:-1]-im[1:-1, :-2]-im[1:-1, 2:]) + \
        abs(2*im[1:-1, 1:-1]-im[:-2, 1:-1]-im[2:, 1:-1])
    res = np.mean(im_filtered, axis=faxes)

    # done?
    return res, [im_filtered, ]


@ staticmethod
def diff_squared_laplacian(im, faxes=(0, 1), **kwargs):
    """Calculates Absolute Laplacian.
    """
    # calculate measure
    im_filtered = 8*im[1:-1, 1:-1]-im[1:-1, :-2]-im[1:-1, 2:] - im[:-2, 1:-1] - \
        im[2:, 1:-1]-im[:-2, :-2]-im[2:, 2:]-im[:-2, 2:]-im[2:, :-2]

    im_filtered *= im_filtered

    res = np.mean(im_filtered, axis=faxes)

    # done?
    return res, [im_filtered, ]


@ staticmethod
def diff_total_variation(im, faxes=(0, 1), **kwargs):
    """Calculates Absolute Laplacian.
    """
    # calculate measure
    tv1 = im[1:-1, 2:]-im[1:-1, :-2]
    tv2 = im[2:, 1:-1]-im[:-2, 1:-1]
    im_filtered = np.sqrt(tv1*tv1 + tv2*tv2)

    res = np.mean(im_filtered, axis=faxes)

    # done?
    return res, [im_filtered, ]

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
# Statistical Filters
# -------------------------------------------------------------------------
#
def stf_basic(im, axes=(-2, -1), printout=False):
    '''
    Basic statistical sharpness metrics: MAX,MIN,MEAN,MEDIAN,VAR,NVAR. Reducing the whole dimensionality to 1 value.
    '''
    im_res = list()
    im_res.append(np.max(im, axis=axes))
    im_res.append(np.min(im, axis=axes))
    im_res.append(np.mean(im, axis=axes))
    im_res.append(np.median(im, axis=axes))
    im_res.append(np.var(im, axis=axes))
    im_res.append(im_res[4]/im_res[2]**2)  # normalized variance (NVAR)
    if printout == True:
        print("Basic analysis yields:\nMAX=\t{}\nMIN=\t{}\nMEAN=\t{}\nMEDIAN=\t{}\nVAR=\t{}\nNVAR=\t{}".format(
            im_res[0], im_res[1], im_res[2], im_res[3], im_res[4], im_res[5]))
    return np.array(im_res)


def stf_kurtosis(im, switch_axis=False, axes=(-2, -1)):
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
        axes = (0, 1)
    res = np.mean((im-np.mean(im, axis=axes))**4) / \
        np.var(im, axis=axes)**2
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


# %%
# --------------------------------------------------------
#               FILTER-INTERFACE
# --------------------------------------------------------
#


class filters():
    """Class that links filters to functions and holds additional information on their necessary padding_shape etc.
    """
    # differential filters
    tenengrad = diff_tenengrad
    brenner = diff_brenners_measure
    abs_laplacian = diff_absolute_laplacian
    squared_laplacian = diff_squared_laplacian
    total_variation = diff_total_variation

    # spectral filters

    # correlative filters
    vollath_f4 = cf_vollathF4
    vollath_f4_symm = cf_vollathF4_symmetric
    vollath_f5 = cf_vollathF5

    # trafo filters

    # dict
    __pad_shape_dict__ = {
        'tenengrad':        ((1, 1), (1, 1)),
        'brenner':          ((1, 1), (0, 0)),
        'abs_laplacian':    ((1, 1), (1, 1)),
        'squared_laplacian': ((1, 1), (1, 1)),
        'total_variation':  ((1, 1), (1, 1)),
        'vollath_f4':       ((0, 0), (1, 1)),
        'vollath_f4_symm':  ((2, 2), (2, 2)),
        'vollath_f5':       ((0, 0), (1, 1)),

    }

    # padding selection
    def _get_padding_(self, filter_chosen='tenengrad'):
        return self.__pad_shape_dict__[filter_chosen]

    def _test_consistance_(self):
        obj = generate_spokes_target()
        for getc in self._get_commands_():
            print(f"Testing for filter: {getc}...", end='')
            _, _ = filter_sharpness(obj, filter=getc)
            print("Done")
        print(f"Works for all filters listed in class {self.__class__}!")

    def _get_commands_(self):
        return [mycmd for mycmd in self.__dir__() if not mycmd.startswith('_')]

    def __str__(self):
        print("Possible commands are:")

        # return ", ".join(cmds)
        return ", ".join(self._get_commands_())


def filter_sharpness(im, filter='tenengrad', **kwargs):
    """Interface to select filters of interest.

    Parameters
    ----------
    im : image
        input nD-image stack
    filter : str, optional
        to check potential filter functions in "filters"-class, by default 'tenengrad'
        print via print(mipy.filters())

    Parameters
    ----------
    im : image
        Input image
    axes : tuple, optional
        axes to be used for 2D-calculation, by default (-2, -1)
    pad_shape : tuple, optional
        if padding (and thereby keeping input-image dimension) shall be used, by default ((1,1),(0,0))
    return_im : bool, optional
        whether the filtered image shall be returned as well, by default False

    Returns
    -------
    res: list
        res[0] calculated metric results
        res[1] list of sub-images created for metric calculation

    See Also
    --------
    filters,
    """
    my_filters = filters()

    # sanity
    if not 'axes' in kwargs:
        kwargs['axes'] = (-2, -1)
    if not 'faxes' in kwargs:
        kwargs['faxes'] = (0, 1)
    if not 'direction' in kwargs:
        kwargs['direction'] = 'forward'
    if not 'return_im' in kwargs:
        kwargs['return_im'] = True
    if not 'pad_shape' in kwargs:
        kwargs['pad_shape'] = my_filters._get_padding_(filter_chosen=filter)

    # get function from filtername
    filter_func = getattr(my_filters, filter)

    # prepare image and put [y,x] to first dimensions
    im, kwargs['npix'] = filter_prep_im(
        im, axes=kwargs['axes'], direction=kwargs['direction'], pad_shape=kwargs['pad_shape'])

    # calculate measure; im_filtered given as list of images
    res, im_filtered = filter_func(im, **kwargs)

    # if return wanted transpose and append
    if kwargs['return_im']:
        kwargs['direction'] = 'backward'
        for m in range(len(im_filtered)):
            im_filtered[m] = filter_prep_im(
                im_filtered[m], axes=kwargs['axes'], direction=kwargs['direction'])
        res = [res, im_filtered]

    # done?
    return res

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
    # if 'Tenengrad' in im_filters:
    #    res.append(tenengrad(im))
    # elif 'VollathF4' in im_filters:
    #    res.append(vollathF4(im))

    # return res
    pass


# %%
# --------------------------------------------------------
#               GENERAL PIXEL-OPERATIONS
# --------------------------------------------------------
#
def local_annealing_atom(im, pos, mode='value', value=0, patch_size=[3, 3], iter_anneal=False, iterations=1):
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
    toshowB = nip.catE((mipy.normto(a),mipy.normto(b1),mipy.normto(b2),
                       mipy.normto(b3),mipy.normto(b4),mipy.normto(b5)))
    nip.v5(mipy.stack2tiles(toshowB))

    '''
    is_complex = True if im.dtype == 'complex' else False

    if mode == 'value':
        im[pos] = value
    else:
        iterations = iterations if iter_anneal else 1
        for _ in range(iterations):
            a = nip.extractFt(im, ROIsize=patch_size, mycenter=pos) if is_complex else nip.extract(
                im, ROIsize=patch_size, centerpos=pos)
            if mode == 'mean':
                a = np.mean(a)
            elif mode == 'max':
                a = np.max(a)
            elif mode == 'min':
                a = np.min(a)
            else:
                raise ValueError(
                    'Not implemented yet, but no problem! What do you wish?')
            im[pos] = a

    # done?
    return im


def local_annealing(im, pos, mode='value', value=0, patch_size=[3, 3], iter_anneal=False, iterations=1):
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
    b6 = local_annealing(a.copy(),[(3,4),(1,2)],mode='mean',
                         patch_size=[4,4],iter_anneal=True,iterations=3)
    toshowA = nip.catE((mipy.normto(a),mipy.normto(b6)))
    nip.v5(mipy.stack2tiles(toshowA))

    '''
    for _, apos in enumerate(pos):
        im = local_annealing_atom(im, apos, mode=mode, value=value,
                                  patch_size=patch_size, iter_anneal=iter_anneal, iterations=iterations)

    # done?
    return im
