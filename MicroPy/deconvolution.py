'''
This module shall only be installed/used if really needed as it requires Tensorflow to be working.
'''

# %% imports
import numpy as np
import NanoImagingPack as nip
import InverseModelling as invmod


def ismR_deconvolution(imfl, psfl, method='multi', regl=None, lambdal=None, NIter=100, tflog_name=None):
    '''
    Simple Wrapper for ISM-Deconvolution to be done. Shall especially used for a mixture of sheppard-sum, weighted averaging and deconvolutions.

    :PARAM:
    =======
    :imfl:      (LIST) Image or List of images (depending on method)
    :psfl:      (LIST) OTF or List of OTFs (depending on method)
    :method:    (STRING) Chosing reconstruction -> 'multi', 'combi'
    :regl:      (LIST) of regularizers to be used [in Order]
    :lambdal:   (LIST) of lambdas to be used. Has to fit to regl.
    :NIter:     (DEC) Number of iterations.

    :OUTPUT:
    ========
    :imd:       Deconvolved image

    :EXAMPLE:
    =========
    imd = ismR_deconvolution(imfl,psfl,method='multi')
    '''
    # parameters
    if regl == None:
        regl = ['TV', 'GR', 'GS']
        lambdal = [2e-4, 1e-3, 2e-4]

    if method == 'combi':
        #BorderSize = np.array([30, 30])
        #res = im.Deconvolve(nimg, psfParam, NIter=NIter, regularization=regl, mylambda = rev[m], BorderRegion=0.1, useSeparablePSF=False)
        pass
    elif method == 'multi':
        res = invmod.Deconvolve(imfl, psfl, NIter=NIter, regularization=regl,
                                mylambda=lambdal, BorderRegion=0.1, useSeparablePSF=False)
        # print('test')
    else:
        raise ValueError('Method unknown. Please specify your needs.')
    return res


def default_dict_tiling(im):
    tdd = {'data_shape': im.shape,
           'tile_shape': np.array([im.shape[0], 128, 128]),
           'overlap_rel': np.array([0, 0.2, 0.2]),
           'window': 'hann'}
    tdd['overlap'] = np.asarray(tdd['overlap_rel']*tdd['tile_shape'], 'int')
    return tdd


def default_dict_deconv():

    deconv_dict = {
        # deconv defaults
        'NIter': 12,
        'regl': ['GR', ],  # ['TV', 'GR', 'GS']
        'lambdal': [1e-2, ],  # [2e-4, 1e-3, 2e-4]
        'regEps': 1e-5,
        'BorderRegion': 0.1,
        'optimizer': 'L-BFGS-B',
        'forcePos': 'Model',

        # damping defaults
        'rwdith': 0.1,
        'damp_method': 'damp'}  # 'zero'

    return deconv_dict


def tiled_processing(tile, psf, tile_id, dd, merger):
    # damped_tile = nip.DampEdge(tile, rwidth=dd['rwdith'], method=dd['damp_method'])

    processed_tile = invmod.Deconvolve(nimg=tile, psf=psf, NIter=dd['NIter'], regularization=dd['regl'], mylambda=dd['lambdal'],
                                       regEps=dd['regEps'], BorderRegion=dd['BorderRegion'], optimizer=dd['optimizer'], forcePos=dd['forcePos'])
    merger.add(tile_id, processed_tile)


def tiled_deconv(im, psf, tiling_dict=None, deconv_dict=None, verbose=True):
    # TODO: check proper dimensions!

    # sanity
    td = default_dict_tiling(im) if tiling_dict is None else tiling_dict
    dd = default_dict_deconv() if deconv_dict is None else deconv_dict

    # prepare psf
    psf_tile = nip.DampEdge(nip.extract(psf, td['tile_shape']),
                            rwidth=dd['rwdith'], method=dd['damp_method'])

    # create tiling structure
    tiler = Tiler(data_shape=td['data_shape'], tile_shape=td['tile_shape'],
                  overlap=tuple(td['overlap']), get_padding=True)
    if verbose:
        print(tiler)
    im_padded = tiler.pad_outer(im, tiler.pads)

    # prepare merging strategy and weighting
    merger = Merger(tiler=tiler, window=td['window'])

    # run for each created tile
    for tile_id, tile in tiler(im_padded, progress_bar=verbose):
        tiled_processing(tile, psf_tile, tile_id, dd, merger)

    im_deconv = nip.image(merger.merge(data_orig_shape=td['data_shape'],))
    im_deconv.pixelsize = im.pixelsize

    return im_deconv
