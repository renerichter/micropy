"""
---------------------------------------------------------------------------------------------------

	@author René Lachmann
	@email herr.rene.richter@gmail.com
	@create date 2019-11-25 10:26:14
	@modify date 2022-04-23 14:41:28
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
__version__ = "0.7a"
__maintainer__ = "René Lachmann"

# %%
# -------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------
#
import numpy as np
import NanoImagingPack as nip
from scipy.ndimage import binary_closing
from pandas import DataFrame

from .utility import add_multi_newaxis, transpose_arbitrary, split_nd, avoid_division_by_zero, match_dim
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


def diff_brenners_measure(im, faxes=(0, 1), **kwargs):
    """Calculates differential brenners measure.
    """
    # calculate measure
    im_filtered = im[:-2]-im[2:]
    res = np.mean(im_filtered*im_filtered, axis=faxes)

    # done?
    return res, [im_filtered, ]


def diff_absolute_laplacian(im, faxes=(0, 1), **kwargs):
    """Calculates Absolute Laplacian.
    """
    # calculate measure
    im_filtered = abs(2*im[1:-1, 1:-1]-im[1:-1, :-2]-im[1:-1, 2:]) + \
        abs(2*im[1:-1, 1:-1]-im[:-2, 1:-1]-im[2:, 1:-1])

    res = np.mean(im_filtered, axis=faxes)

    # done?
    return res, [im_filtered, ]


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


def diff_total_variation(im, faxes=(0, 1), **kwargs):
    """Calculates Total Variation.
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


def spf_kristans_bayes_spectral_entropy(im, faxes=(-2, -1), tile_exp=3, klim=6, **kwargs):
    '''
    Kristan's Bayes Spectral Entropy filter that works on 2^n x 2^n - tiles and represents only the central region of the image that can be represented by an integer multiple of this tile-size.
    '''
    # get tile_size and extract biggest possible multiple
    tile_sizes = [2**tile_exp, ]*len(faxes)
    im_split, sal = split_nd(im, tile_sizes=tile_sizes,
                             split_axes=np.transpose(np.array(faxes)))

    # calculate DCT along split-axes for the 8x8 tiles
    res = nip.image(dct2(im_split, forward=True, axes=sal))

    # create selection mask for summing
    mask = (nip.rr(tile_sizes, placement='corner') < klim).astype('uint8')
    mask = match_dim(mask, sal, res.ndim)

    # calculate nominator and denominator
    nom = np.sum(res * res * mask, axis=tuple(sal))
    denom = np.sum(res*mask, axis=tuple(sal))
    denom *= denom

    # avoid division by zero
    res = avoid_division_by_zero(nom, denom)
    res = -np.mean(1-res, axis=tuple(np.arange(len(sal))))

    return res, [nom, denom]


def spf_dct_normalized_shannon_entropy(im, faxes=(-2, -1), klim=100, krel=1e-2, klim_max=None, **kwargs):
    """Calculates the normalized shannon entropy.

    Parameters
    ----------
    im : image
        input image
    faxes : tuple, optional
        axes to work on, by default (-2, -1)
    klim : int, optional
        maximum Fourier-frequency in DCT-splace, by default 100
    krel : float, optional
        relativ kvalue to maxval of DCT-image to find limiting support. Used if klim==None, by default 1e-2


    """
    # get DCT and prepare terms
    imDCT = nip.image(dct2(im, forward=True, axes=faxes))
    imlp2 = lp_norm(imDCT, p=2, normaxis=faxes, keepdims=True)
    en_el = np.abs(imDCT/imlp2)

    # make sure for OTF-support
    if klim is None:
        if klim_max is None:
            mask = np.copy(imDCT)
        else:
            mask = np.copy(imDCT[klim_max])
        mask[mask < np.max(mask)*krel] = 0
        mask[mask > np.max(mask)*krel] = 1

        if mask.ndim > 3:
            mask_pad = tuple([(0, 0), ]*(imDCT.ndim-3)+[(4, 4), (4, 4), (4, 4)])
            mask_struct = np.ones([1, ]*(imDCT.ndim-3)+[3, 3, 3])
            sdim = -3
        elif mask.ndim == 3:
            mask_pad = ((0, 0), (4, 4), (4, 4))
            mask_struct = np.ones([1, 3, 3])
            sdim = -3
        else:
            mask_pad = ((4, 4), (4, 4))
            mask_struct = np.ones([3, 3])
            sdim = -2
        mask = np.pad(mask, mask_pad, mode='constant', constant_values=0)
        mask = binary_closing(mask, structure=mask_struct, iterations=3)
        if mask.ndim > 2:
            mask = np.max(mask, axis=tuple(np.arange(0, mask.ndim-2)))
        mask = nip.extract(img=mask, ROIsize=imDCT.shape)
        # assume spherical support and calculate radius from area per slice
        klim_sqr = np.floor((4*np.sum(mask, axis=(-2, -1))/np.pi)).astype('int32')
    else:
        #mask_shape = imDCT.shape[-3:] if imDCT.ndim >= 3 else imDCT.shape[-2:]
        xmask = abs(nip.xx(imDCT.shape, placement='corner'))
        ymask = abs(nip.yy(imDCT.shape, placement='corner'))
        zmask = abs(nip.zz(imDCT.shape, placement='center'))
        mask = (np.sqrt(xmask*xmask+ymask*ymask+zmask*zmask) < klim).astype('bool')
        klim_sqr = klim*klim

    im_res = (en_el * np.log2(en_el, where=en_el != 0, out=np.zeros(en_el.shape)))

    # calculate norm
    res = - 2.0 / klim_sqr * np.sum(im_res*mask, axis=faxes)

    # done?
    return res, [im_res, ]


#
# -------------------------------------------------------------------------
# Statistical Filters
# -------------------------------------------------------------------------
#


def stf_max(im, faxes=(-2, -1), **kwargs):
    '''
    Numpy-Max Wrapper implemented for 2D image and called via `filter_sharpness`-interfacing function to assure proper padding and axis orientation.

    See Also
    --------
    filter_sharpness
    '''
    res = np.max(im, axis=faxes)
    return res, [None, ]


def stf_min(im, faxes=(-2, -1), **kwargs):
    '''
    Numpy-Min Wrapper implemented for 2D image and called via `filter_sharpness`-interfacing function to assure proper padding and axis orientation.

    See Also
    --------
    filter_sharpness
    '''
    res = np.min(im, axis=faxes)
    return res, [None, ]


def stf_mean(im, faxes=(-2, -1), **kwargs):
    '''
    Numpy-Mean wrapper implemented for 2D image and called via `filter_sharpness`-interfacing function to assure proper padding and axis orientation.

    See Also
    --------
    filter_sharpness
    '''
    res = np.mean(im, axis=faxes)
    return res, [None, ]


def stf_median(im, faxes=(-2, -1), **kwargs):
    '''
    Numpy-median wrapper implemented for 2D image and called via `filter_sharpness`-interfacing function to assure proper padding and axis orientation.

    See Also
    --------
    filter_sharpness
    '''
    res = np.median(im, axis=faxes)
    return res, [None, ]


def stf_var(im, faxes=(-2, -1), **kwargs):
    '''
    Numpy Variance-Wrapper implemented for 2D image and called via `filter_sharpness`-interfacing function to assure proper padding and axis orientation.

    See Also
    --------
    filter_sharpness
    '''
    res = np.var(im, axis=faxes)
    return res, [None, ]


def stf_normvar(im, faxes=(-2, -1), **kwargs):
    """Normalized Variance implemented for 2D image and called via `filter_sharpness`-interfacing function to assure proper padding and axis orientation.

    See Also
    --------
    filter_sharpness
    """
    res = stf_var(im, faxes=faxes)[0] / (stf_mean(im, faxes=faxes)[0]**2)
    return res, [None, ]


def stf_basic(im, faxes=(-2, -1), printout=False, cols=[],**kwargs):
    '''
    Collectionfo basic statistical sharpness metrics: MAX,MIN,MEAN,MEDIAN,VAR,NVAR. Reducing the dimensionality of application to 1 value. Implemented for 2D image and called via `filter_sharpness`-interfacing function to assure proper padding and axis orientation.

    See Also
    --------
    filter_sharpness
    '''
    im_res = list()
    im_res.append(stf_max(im, faxes=faxes)[0])
    im_res.append(stf_min(im, faxes=faxes)[0])
    im_res.append(np.sum(im, axis=tuple(faxes)))
    im_res.append(stf_mean(im, faxes=faxes)[0])
    im_res.append(stf_median(im, faxes=faxes)[0])
    im_res.append(stf_var(im, faxes=faxes)[0])
    im_res.append(stf_normvar(im, faxes=faxes)[0])
    
    imstats_index = ['MAX', 'MIN', 'SUM', 'MEAN', 'MEDIAN', 'VAR', 'NVAR']
    if len(cols)!=len(im_res[0]):
        cols = np.arange(im_res[0])
    im_df = DataFrame(im_res, columns=cols, index=imstats_index)

    if printout:
        #print("Basic analysis yields:\nMAX=\t{}\nMIN=\t{}\nSUM=\t{}\nMEAN=\t{}\nMEDIAN=\t{}\nVAR=\t{}\nNVAR=\t{}".format(*im_res))
        print(im_df)
        
    return im_df, [None, ]


def stf_kurtosis(im, faxes=(-2, -1), **kwargs):
    '''
    Forth moment ( https://en.wikipedia.org/wiki/Kurtosis ).
    Scipy-Kurtosis  implemented for 2D image and called via `filter_sharpness`-interfacing function to assure proper padding and axis orientation.

    See Also
    --------
    filter_sharpness
    '''
    res = np.mean((im-np.mean(im, axis=faxes))**4, axis=faxes) / np.var(im, axis=faxes)**2
    return res, [None, ]


def stf_diffim_kurtosis(im, faxes=(0, 1), **kwargs):
    '''
    Difference image Kurtosis implemented for 2D image and called via `filter_sharpness`-interfacing function to assure proper padding and axis orientation.

    See Also
    --------
    filter_sharpness
    '''
    return stf_kurtosis(im[2:, 2:] - im[:-2, :-2], faxes=faxes)


def stf_histogram_entropy(im, bins=256, **kwargs):
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
            im_hist.append(np.histogram(imh[cla], bins=bins)[1],)
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

    return res, [im_res, im_hist]


# %%
# --------------------------------------------------------
#               FILTER-INTERFACE
# --------------------------------------------------------
#


class filters():
    """Class that links filters to functions and holds additional information on their necessary padding_shape etc.
    """
    _filters_dict_ = {
        # differential filters
        'tenengrad':        [diff_tenengrad,            ((1, 1), (1, 1)),   True, [0, 1, 0, 1], [(-2, -1), (0, 1)], 'Tenengrad', 'TEN'],
        'brenner':          [diff_brenners_measure,     ((1, 1), (0, 0)),   True, [0.1, 1, 0.2, 1], [(-2, -1), (0, 1)], 'Brenner', 'BRE'],
        'abs_laplacian':    [diff_absolute_laplacian,   ((1, 1), (1, 1)),   True, [0.2, 0.8, 0.1, 1], [(-2, -1), (0, 1)], 'Absolute Laplacian', 'ALA'],
        'squared_laplacian': [diff_squared_laplacian,   ((1, 1), (1, 1)),   True, [0.1, 0.7, 0.1, 1], [(-2, -1), (0, 1)], 'Squared Laplacian', 'SLA'],
        'total_variation':  [diff_total_variation,      ((1, 1), (1, 1)),   True, [0.2, 0.6, 0.2, 1], [(-2, -1), (0, 1)], 'Total Variation', 'TOV'],
        # spectral filters
        'max':              [stf_max,                   ((0, 0), (0, 0)),   False, [0, 0.85, 1, 1], [(-2, -1), (0, 1)], 'Maximum', 'MAX'],
        'min':              [stf_min,                   ((0, 0), (0, 0)),   False, [0, 0.7, 1, 1], [(-2, -1), (0, 1)], 'Minimum', 'MIN'],
        'mean':             [stf_mean,                  ((0, 0), (0, 0)),   False, [0, 0.4, 1, 1], [(-2, -1), (0, 1)], 'Mean', 'MEA'],
        'median':           [stf_median,                ((0, 0), (0, 0)),   False, [0.4, 0, 1, 1], [(-2, -1), (0, 1)], 'Median', 'MED'],
        'var':              [stf_var,                   ((0, 0), (0, 0)),   False, [0.7, 0, 1, 0.95], [(-2, -1), (0, 1)], 'Variance', 'VAR'],
        'normvar':          [stf_normvar,               ((0, 0), (0, 0)),   False, [0.85, 0, 1, 0.9], [(-2, -1), (0, 1)], 'Normed Variance', 'NVA'],
        'kurtosis':         [stf_kurtosis,              ((0, 0), (0, 0)),   False, [0.3, 0.4, 0.9, 0.85], [(-2, -1), (0, 1)], 'Kurtosis', 'KUR'],
        'diffim_kurtosis':  [stf_diffim_kurtosis,       ((0, 0), (0, 0)),   False, [0.5, 0.4, 0.9, 0.8], [(-2, -1), (0, 1)], 'Difference Kurtosis', 'DKU'],
        # 'hist_entropy':     [stf_histogram_entropy,     ((0, 0), (0, 0)),   False],
        # correlative filters
        'vollath_f4':       [cf_vollathF4,              ((0, 0), (1, 1)),   True, [1, 0, 0, 1], [(-2, -1), (0, 1)], 'Vollath F4', 'VO4'],
        'vollath_f4_symm':  [cf_vollathF4_symmetric,    ((2, 2), (2, 2)),   True, [0.85, 0, 0, 1], [(-2, -1), (0, 1)], 'symmetric Vollath F4', 'VS4'],
        'vollath_f5':       [cf_vollathF5,              ((0, 0), (1, 1)),   True, [0.7, 0, 0, 1], [(-2, -1), (0, 1)], 'Vollath F5', 'VO5'],
        # trafo filters
        'kristans_entropy': [spf_kristans_bayes_spectral_entropy,   ((0, 0), (0, 0)),   True,  [0.8, 0.8, 0.8, 1], [(-2, -1), (-2, -1)], 'Kristans Entropy', 'KEN'],
        'shannon_entropy':  [spf_dct_normalized_shannon_entropy,    ((0, 0), (0, 0)),   True, [0.7, 0.7, 0.7, 1], [(-2, -1), (-2, -1)], 'Shannon Entropy', 'SEN'],
    }
    _filters_special_params_ = {
        'kristans_entropy': {'tile_exp': 4, 'klim': 2, },
        'shannon_entropy':  {'klim': None, 'krel': 1e-2},
    }

    _colors_ = []

    def __init__(self, create_colors=True, cmap='rainbow', myfilters=None):
        if create_colors:
            self.create_filter_colors(cmap, myfilters)
        else:
            self.set_filter_colors()

    # padding selection
    def get_padding(self, filter_chosen):
        return self._filters_dict_[filter_chosen][1]

    def get_axes_faxes(self, filter_chosen):
        return self._filters_dict_[filter_chosen][4]

    def get_pprint_names(self, filter_list_chosen):
        if type(filter_list_chosen) == list:
            res = [self._filters_dict_[m][5] for m in filter_list_chosen]
        else:
            res = self._filters_dict_[filter_list_chosen][5]
        return res

    def get_filter_abbr_names(self, filter_list_chosen):
        if type(filter_list_chosen) == list:
            res = [self._filters_dict_[m][6] for m in filter_list_chosen]
        else:
            res = self._filters_dict_[filter_list_chosen][6]
        return res

    def test_consistance(self):
        obj = generate_spokes_target()
        for getc in self.get_commands():
            print(f"Testing for filter: {getc}...", end='')
            _, _ = filter_sharpness(obj, filter=getc)
            print("Done")
        print(f"Works for all filters listed in class {self.__class__}!")

    def get_commands(self):
        # return [mycmd for mycmd in self.__dir__() if not mycmd.startswith('_') and not type(getattr(self, mycmd)) == type(self.get_padding)]
        return [mycmd for mycmd in self._filters_dict_]

    def get_filter_func(self, filter_chosen):
        return self._filters_dict_[filter_chosen][0]

    def get_return_im(self, filter_chosen):
        return self._filters_dict_[filter_chosen][2]

    def get_special_param(self, filter_chosen):
        if not self._filters_special_params_.get(filter_chosen) is None:
            return self._filters_special_params_[filter_chosen]
        else:
            return False

    def add_special_param(self, kwargs, filter_chosen=filter):
        for sparam in self._filters_special_params_[filter_chosen]:
            kwargs[sparam] = self._filters_special_params_[filter_chosen][sparam]

    def create_filter_colors(self, cmap, myfilters=None):
        collen = len(myfilters) if myfilters is not None else len(self._filters_dict_)
        from matplotlib.pyplot import get_cmap
        self._colors_ = get_cmap(cmap)(np.arange(256))[
            ::int(np.floor(256/collen))]

    def set_filter_colors(self):
        self._colors_ = [[0, 1, 0, 1], [0.1, 1, 0.2, 1], [0.2, 0.8, 0.1, 1],        [0.1, 0.7, 0.1, 1],        [0.2, 0.6, 0.2, 1],         [0, 0.85, 1, 1],         [0, 0.7, 1, 1],         [0, 0.4, 1, 1],         [
            0.4, 0, 1, 1],         [0.7, 0, 1, 0.95],         [0.85, 0, 1, 0.9],         [0.3, 0.4, 0.9, 0.85],         [0.5, 0.4, 0.9, 0.8],        [1, 0, 0, 1],        [0.85, 0, 0, 1],        [0.7, 0, 0, 1],          [0.8, 0.8, 0.8, 1],         [0.7, 0.7, 0.7, 1], ]

    def get_filter_colors(self, m):
        return self._colors_[m]
    
    def get_filter_colors_complete(self):
        return self._colors_

    def __str__(self):
        print("Possible commands are:")

        # return ", ".join(cmds)
        return ", ".join(self.get_commands())


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
        kwargs['axes'] = my_filters.get_axes_faxes(filter_chosen=filter)[0]
    if not 'faxes' in kwargs:
        kwargs['faxes'] = my_filters.get_axes_faxes(filter_chosen=filter)[1]
    if not 'direction' in kwargs:
        kwargs['direction'] = 'forward'
    if not 'return_im' in kwargs:
        kwargs['return_im'] = my_filters.get_return_im(filter_chosen=filter)
    if not 'pad_shape' in kwargs:
        kwargs['pad_shape'] = my_filters.get_padding(filter_chosen=filter)
    if not my_filters.get_special_param(filter_chosen=filter) == False:
        my_filters.add_special_param(kwargs, filter_chosen=filter)
    # get function from filtername
    filter_func = my_filters.get_filter_func(filter)

    # prepare image and put [y,x] to first dimensions
    im, kwargs['npix'] = filter_prep_im(
        im, axes=kwargs['axes'], direction=kwargs['direction'], pad_shape=kwargs['pad_shape'], faxes=kwargs['faxes'])

    # calculate measure; im_filtered given as list of images
    res, im_filtered = filter_func(im, **kwargs)

    # if return wanted transpose and append
    if kwargs['return_im']:
        kwargs['direction'] = 'backward'
        im_filtered = list(im_filtered)
        for m in range(len(im_filtered)):
            im_filtered[m] = filter_prep_im(
                im_filtered[m], axes=kwargs['axes'], direction=kwargs['direction'], faxes=kwargs['faxes'])
        res = [res, im_filtered]
    else:
        res = [res, None]

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
