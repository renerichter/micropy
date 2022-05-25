'''
This module shall only be installed/used if really needed as it requires Tensorflow to be working.
'''

# %% imports
import numpy as np
import NanoImagingPack as nip
import InverseModelling as invmod
from tiler import Tiler, Merger
from typing import Union


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
                                mylambda=lambdal, BorderRegion=0.1, useSeparablePSF=False,returnBig=True)
        # print('test')
    else:
        raise ValueError('Method unknown. Please specify your needs.')
    return res


def default_dict_tiling(imshape: Union[tuple, list, np.ndarray],
                        basic_shape: list = [128, 128],
                        basic_roverlap: list = [0.2, 0.2],
                        **kwargs):
    '''Note: tiling_low_thresh defines the lower boundary that the used system is capable of doing a l-bfgs-b based deconvolution with poisson fwd model. If model, gpu, ... is changed, change this value!
    
    possible parameters:
    
    '''
    
    # sanity
    imshape = imshape if type(imshape) in [list, tuple] else imshape.shape

    # generate dict
    tdd = {'data_shape': imshape,
           'tile_shape': np.array(list(imshape[:-(len(basic_shape))]) + list(basic_shape)),
           'overlap_rel': np.array([0.0, ]*(len(imshape)-len(basic_shape))+list(basic_roverlap)),
           'window': 'hann',
           'diffdim_im_psf': None,
           'atol': 1e-10,
           'tiling_low_thresh': 512*128*128*np.dtype(np.float32).itemsize}

    # calculate overlap
    tdd['overlap'] = np.asarray(tdd['overlap_rel']*tdd['tile_shape'], 'int')

    for key in kwargs:
        if key in tdd:
            tdd[key] = kwargs[key]

    # done?
    return tdd


def default_dict_deconv(**kwargs):

    deconv_dict = {
        # deconv defaults
        'NIter': 40,
        'regl': ['TV', ],  # ['TV', 'GR', 'GS']
        'lambdal': [10**(-2.5), ],  # [2e-4, 1e-3, 2e-4]
        'regEps': 1e-5,
        'BorderRegion': 0.1,
        'optimizer': 'L-BFGS-B',
        'forcePos': 'Model',
        'lossFkt': 'Poisson',
        'multiview_dim': -4,
        'log_name': None,
        'retStats': True,
        'oparam': {"learning_rate": 1.5, 'disp': True},
        'validMask': None,
        'anisotropy':[1.0,],
        'returnBig': True,

        # damping defaults
        'rwdith': 0.1,
        'damp_method': 'damp',# 'zero'
        
        # images to evaluate after deconv
        'use_nimgBig':0}  

    # take input into account
    for key in kwargs:
        if key in deconv_dict:
            deconv_dict[key] = kwargs[key]

    return deconv_dict


def deconv_atom(im, psf, dd):
    multiview_dim = dd['multiview_dim'] if 'multiview_dim' in dd else None

    processed_tile = invmod.Deconvolve(nimg=im, psf=psf, NIter=dd['NIter'], lossFkt=dd['lossFkt'], regularization=dd['regl'], mylambda=dd['lambdal'],
                                       regEps=dd['regEps'], BorderRegion=dd['BorderRegion'], optimizer=dd['optimizer'], forcePos=dd['forcePos'], multiview_dim=multiview_dim, tflog_name=dd['log_name'], retStats=dd['retStats'],oparam=dd['oparam'], validMask=dd['validMask'], anisotropy=dd['anisotropy'], returnBig=dd['returnBig'])
    
    return processed_tile


def tiled_processing(tile, psf, tile_id, dd, merger):
    # damped_tile = nip.DampEdge(tile, rwidth=dd['rwdith'], method=dd['damp_method'])
    processed_tile = deconv_atom(tile, psf, dd)
    retStats = None
    if dd['retStats']:
        retStats = processed_tile[1]
        processed_tile = processed_tile[0][dd['use_nimgBig']] if type(processed_tile[0])==list else processed_tile[0]
        if not processed_tile.ndim == len(merger.tiler.tile_shape):
            processed_tile=np.reshape(np.squeeze(processed_tile),tuple(merger.tiler.tile_shape))
    merger.add(tile_id, processed_tile)
    return retStats


def tiled_deconv(im, psf, tiling_dict=None, deconv_dict=None, verbose=True):
    # TODO: check proper dimensions!

    # sanity
    td = default_dict_tiling(im) if tiling_dict is None else tiling_dict
    dd = default_dict_deconv() if deconv_dict is None else deconv_dict
    retStats = []

    # prepare psf
    psf_tile = nip.DampEdge(nip.extract(psf, td['tile_shape']),
                            rwidth=dd['rwdith'], method=dd['damp_method'])

    # create tiling structure
    if not tiling_dict['diffdim_im_psf'] is None:
        repeats = [1, ]*im.ndim
        repeats[tiling_dict['diffdim_im_psf']] = psf_tile.shape[tiling_dict['diffdim_im_psf']]
        im = nip.repmat(im, repeats)
        diffdim_sel = []
        tile_shape_final = psf_tile.shape[1:]
        for m, pshape in enumerate(psf_tile.shape):
            if m == tiling_dict['diffdim_im_psf']:
                diffdim_sel.append(slice(pshape//2, pshape//2+1))
            else:
                diffdim_sel.append(slice(0, pshape))

    tiler = Tiler(data_shape=td['data_shape'], tile_shape=td['tile_shape'],
                  overlap=tuple(td['overlap']))#, get_padding=True
    if verbose:
        print(tiler)
    #im_padded = tiler.pad_outer(im, tiler.pads)

    # prepare merging strategy and weighting
    if tiling_dict['diffdim_im_psf'] is None:
        if len(td['data_shape']) > 3:
            final_shape = im.shape[1:] if all(
                np.array(im.shape[1:]) > np.array(psf.shape[1:])) else psf.shape[1:]
            tiler_final = Tiler(data_shape=final_shape, tile_shape=psf_tile.shape[1:], overlap=tuple(
                td['overlap'])[1:])#get_padding=True
        else:
            tiler_final = tiler
    else:
        tiler_final = Tiler(data_shape=psf.shape[1:], tile_shape=tile_shape_final, overlap=tuple(
            td['overlap'])[1:])#get_padding=True
    if verbose and tiling_dict['diffdim_im_psf']:
        print(tiler_final)
    merger = Merger(tiler=tiler_final, window=td['window'])

    # run for each created tile -> used: im_padded before
    for tile_id, tile in tiler(im, progress_bar=verbose):
        if not tiling_dict['diffdim_im_psf'] is None:
            tile = nip.image(tile[diffdim_sel])
            tile.pixelsize = psf_tile.pixelsize
        retStats.append(tiled_processing(tile, psf_tile, tile_id, dd, merger))

    im_deconv = nip.image(merger.merge())#data_orig_shape=td['data_shape'],
    im_deconv.pixelsize = im.pixelsize

    return im_deconv, retStats
