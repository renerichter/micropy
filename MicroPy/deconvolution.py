'''
This module shall only be installed/used if really needed as it requires Tensorflow to be working.
'''

# %% imports
import numpy as np
import NanoImagingPack as nip
import InverseModelling as invmod
from tiler import Tiler, Merger
from typing import Union

from .inout import store_data


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
                        basic_overlap: list = [],
                        basic_add_overlap2shape: bool = False,
                        **kwargs) -> dict:
    '''Note: tiling_low_thresh defines the lower boundary that the used system is capable of doing a l-bfgs-b based deconvolution with poisson fwd model. If model, gpu, ... is changed, change this value!
    
    possible parameters:
    
    '''
    
    # sanity
    imshape = imshape if type(imshape) in [list, tuple] else imshape.shape

    # generate dict
    tdd = {'data_shape': imshape,
           'tile_shape': list(imshape[:-(len(basic_shape))]) + list(basic_shape),
           'overlap_rel': np.array([0.0, ]*(len(imshape)-len(basic_shape))+list(basic_roverlap)),
           'window': 'hann',
           'diffdim_im_psf': None,
           'atol': 1e-10,
           'tiling_low_thresh': 512*128*128*np.dtype(np.float32).itemsize,
           'tiling_mode':'reflect',}

    # calculate overlap and add to tileshape
    tdd['overlap'] = np.asarray(np.ceil(tdd['overlap_rel']*np.array(tdd['tile_shape'])), 'int') if basic_overlap == [] else np.array([0.0, ]*(len(imshape)-len(basic_shape))+basic_overlap)
    tdd['overlap_rel']= tdd['overlap_rel'] if basic_overlap == [] else np.round(tdd['tile_shape']/tdd['overlap'])
    tdd['tile_shape'] = np.array(tdd['tile_shape'],dtype='int')+tdd['overlap'] if basic_add_overlap2shape else np.array(tdd['tile_shape'],dtype='int')

    for key in kwargs:
        if key in tdd:
            tdd[key] = kwargs[key]

    # done?
    return tdd


def default_dict_deconv(**kwargs) -> dict:

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
        'do_damp_psf': False,
        'damp_rwidth': 0.1,
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

def tiled_processing_deconv(tile, psf, tile_id, dd, merger):
    # damped_tile = nip.DampEdge(tile, rwidth=dd['damp_rwidth'], method=dd['damp_method'])
    processed_tile = deconv_atom(tile, psf, dd)
    retStats = None
    if dd['retStats']:
        retStats = processed_tile[1]
        processed_tile = processed_tile[0][dd['use_nimgBig']] if type(processed_tile[0])==list else processed_tile[0]
        if not processed_tile.ndim == len(merger.tiler.tile_shape):
            processed_tile=np.reshape(np.squeeze(processed_tile),tuple(merger.tiler.tile_shape))
    merger.add(tile_id, processed_tile)
    return retStats

def create_tiling_structure(im,psf,psf_tile,td,verbose):
    diffdim_sel = []
    tile_shape_final = ()
    if not td['diffdim_im_psf'] is None:
        repeats = [1, ]*im.ndim
        repeats[td['diffdim_im_psf']] = psf_tile.shape[td['diffdim_im_psf']]
        im = nip.repmat(im, repeats)
        tile_shape_final = psf_tile.shape[1:]
        for m, pshape in enumerate(psf_tile.shape):
            if m == td['diffdim_im_psf']:
                diffdim_sel.append(slice(pshape//2, pshape//2+1))
            else:
                diffdim_sel.append(slice(0, pshape))

    tiler = Tiler(data_shape=td['data_shape'], tile_shape=td['tile_shape'],
                overlap=tuple(td['overlap']),mode=td['tiling_mode'])#, get_padding=True
    if verbose:
        print(tiler)
    #im_padded = tiler.pad_outer(im, tiler.pads)

    # prepare merging strategy and weighting
    if td['diffdim_im_psf'] is None:
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
    if verbose and td['diffdim_im_psf']:
        print(tiler_final)
    merger = Merger(tiler=tiler_final, window=td['window'])

    return tiler, tiler_final, merger, diffdim_sel


def tiled_deconv(im, psf, tiling_dict=None, deconv_dict=None, verbose=True):
    # TODO: check proper dimensions!

    # sanity
    td = default_dict_tiling(im) if tiling_dict is None else tiling_dict
    dd = default_dict_deconv() if deconv_dict is None else deconv_dict
    retStats = []

    # prepare psf
    psf_tile=nip.extract(psf, td['tile_shape'])
    psf_tile = nip.DampEdge(psf_tile,rwidth=dd['damp_psf_rwidth'], method=dd['damp_psf_method'],) if dd['do_damp_psf'] else psf_tile

    # create tiling structure
    tiler, tiler_final, merger, diffdim_sel = create_tiling_structure(im=im,psf=psf,psf_tile=psf_tile,td=td,verbose=verbose)

    # run for each created tile -> used: im_padded before
    for tile_id, tile in tiler(im, progress_bar=verbose):
        if not td['diffdim_im_psf'] is None:
            tile = nip.image(tile[diffdim_sel])
            tile.pixelsize = psf_tile.pixelsize
        retStats.append(tiled_processing_deconv(tile, psf_tile, tile_id, dd, merger))

    im_deconv = nip.image(merger.merge())#data_orig_shape=td['data_shape'],
    im_deconv.pixelsize = im.pixelsize

    return im_deconv, retStats

def deconv_switcher(im, psf, tiling_dict, deconv_dict, do_tiling):
    tiling_dict = default_dict_tiling(imshape=im.shape, basic_shape=list(
        im.shape[:-2])+[128, 128], basic_roverlap=[0, ]*(im.ndim-2)+[0.2, 0.2]) if tiling_dict is None else tiling_dict
    if do_tiling or any(np.array([im.nbytes, psf.nbytes]) > tiling_dict['tiling_low_thresh']):
        print(
            f"Switched to tiled deconv. Do_tiling={do_tiling}, bigger_tiling_thresh={any(np.array([im.nbytes, psf.nbytes]) > tiling_dict['tiling_low_thresh'])}.")
        return tiled_deconv(im=im, psf=psf, tiling_dict=tiling_dict, deconv_dict=deconv_dict)
    else:
        print("Switched to complete deconv.")
        return deconv_atom(im=im, psf=psf, dd=deconv_dict)

def deconv_test_param(im, psf, tiling_dict, deconv_dict, param, param_range, use_im_big=False, use_tiling_lowthresh=True):
    deconv_list = []
    im_deconv_retStats = []
    if param == 'regl':
        param_val = np.copy(deconv_dict['lambdal'])

    smaller_than_tiling_thresh = True if tiling_dict is None else (all(np.array(
        [im.nbytes, psf.nbytes]) < tiling_dict['tiling_low_thresh']) if use_tiling_lowthresh else False)

    for m, para in enumerate(param_range):
        deconv_dict[param] = para
        if param == 'regl':
            deconv_dict['lambdal'] = param_val[m]
        if tiling_dict is None or smaller_than_tiling_thresh:
            im_deconv = deconv_atom(im, psf, deconv_dict)
        else:
            im_deconv = tiled_deconv(
                im=im, psf=psf, tiling_dict=tiling_dict, deconv_dict=deconv_dict)
        if 'retStats' in deconv_dict and deconv_dict['retStats']:
            im_deconv_retStats.append(im_deconv[1])
            im_deconv = im_deconv[0]
        im_deconv = im_deconv[use_im_big] if type(im_deconv) == list else im_deconv
        deconv_list.append(im_deconv)

    deconv_list = np.squeeze(nip.image(np.array(deconv_list)))
    deconv_list.pixelsize = im.pixelsize

    if param == 'regl':
        deconv_dict['lambdal'] = param_val

    # done?
    if not im_deconv_retStats == []:
        return deconv_list, im_deconv_retStats
    else:
        return deconv_list

def deconv_test_param_on_list(ims, psfs, pad, resd, rparams, ddh, sname=['dec2D', 'conf'], ldim=0, do_tiling=False):
    '''Calculate Deconvolution within a parameterrange for set of images.
    
    TODO:
        1) move ldim to front with transpose instead of swapaxes to keep order of other dimensions, especially if ldim>1
        2) docstrings!
    '''

    # sanity
    snamef = '_'.join(sname)
    if not ldim == 0:
        ims = np.swapaxes(ims, 0, ldim)
        psfs = np.swapaxes(psfs, 0, ldim)

    # prepare deconv parameters
    resd[snamef+'_rtest_dict'] = dict(ddh)
    resd[snamef+'_rtest_dict']['BorderRegion'] = resd[snamef+'_rtest_dict']['BorderRegion'][-(
        ims.ndim-1):]

    # prepare storage
    resd[snamef+'_rtest'] = nip.image(
        np.zeros([ims.shape[0],len(ddh['lambdal_range']), ]+list(ims.shape[-3:])))
    resd[snamef+'_rtest_rstats'] = [{} for m in range(ims.shape[ldim])]
    resd[snamef+'_rtest'].pixelsize = ims.pixelsize

    # normalize
    if rparams['im_norm']:
        resd[sname[1]+'sum'] = np.sum(ims, axis=tuple(pad['faxes']), keepdims=True)
        ims *= np.array(resd['ims_sum'][0])/resd[sname[1]+'sum']

    # tiling dict
    td = default_dict_tiling(imshape=ims.shape[1:], basic_shape=list(ims[0].shape[:-3])+list(
        pad['td']['tile_shape'][-3:]), basic_roverlap=[0, ]*(ims[0].ndim-3)+list(pad['td']['overlap_rel'][-3:])) if do_tiling else None

    # test deconv parameters
    for m in range(ims.shape[0]):
        dech, resd[snamef+'_rtest_rstats'][m] = deconv_test_param(im=ims[m], psf=psfs[m], tiling_dict=td, deconv_dict=resd[
            snamef+'_rtest_dict'], param=resd[snamef+'_rtest_dict']['param'], param_range=resd[snamef+'_rtest_dict']['lambdal_range'])
        resd[snamef+'_rtest'][m] = np.reshape(dech,
                                             resd[snamef+'_rtest'][m].shape)  # [:,m]

    # undo swap:
    swapval= ldim-1 if ldim > 1 else 1
    resd[snamef+'_rtest'] = np.swapaxes(resd[snamef+'_rtest'], 0, swapval)

    # blueprint for saving in case something breaks on the fly
    if 0:
        pad['save_name'] = pad['save_name_base'] + snamef+'_resd'
        store_data(param_dict=pad, proc_dict=resd, data_dict=None)

    # done?
    return resd


def recon_deconv_list(ims, psfs, pad, rparams, resd, sname=['dec_2D', 'allview_indivim'], ldim=0, ddh={}, do_tiling=False, td=None, verbose=True):
    '''allows for different parameters per deconvolution step via rparams['lambdal'].
    TODO:
    1) docstring
    2) add sanity check for ddh and td integrity
    '''
    # verbosity
    if verbose:
        print("Starting Function: recon_deconv_list")

    # sanity
    snamef = '_'.join(sname)
    if not ldim == 0:
        ims = np.swapaxes(ims, 0, ldim)
        psfs = np.swapaxes(psfs, 0, ldim)

    # prepare storage
    resd[snamef] = nip.image(np.zeros([ims.shape[0], ]+list(ims.shape[2:])))
    resd[snamef+'_stats'] = [{} for m in range(ims_ft.shape[0])]
    resd[snamef].pixelsize = ims.pixelsize[2:]

    # prepare deconv parameters
    ddh['lambdal'] = rparams['lambdal'][m]
    resd[snamef+'_dict'] = dict(ddh)
    resd[snamef+'_dict']['BorderRegion'] = resd[snamef+'_dict']['BorderRegion'][-(
        ims.ndim-1):]

    # tiling dict
    td = default_dict_tiling(imshape=ims.shape[1:], basic_shape=list(ims[0].shape[:-3])+list(
        pad['td']['tile_shape'][-3:]), basic_roverlap=[0, ]*(ims[0].ndim-3)+list(pad['td']['overlap_rel'][-3:])) if (do_tiling and td is None) else None

    # normalize
    if rparams['im_norm']:
        resd[sname[1]+'sum'] = np.sum(ims, axis=tuple(pad['faxes']), keepdims=True)
        ims *= resd['ims_sum'][0]/resd[sname[1]+'sum']

    # deconvolve
    ilen = ims.shape[0]
    for m, mIm in enumerate(ims):
        if verbose:
            print((">>> m={:0"+str(ilen)+"d}.").format(m))

        # deconv --> multiview, all-views for each raw (and calculated) image
        ddh['lambdal'] = rparams[snamef+'lambdal'][m]
        resd[snamef+'_dict'][m] = dict(ddh)

        # tried to catch output written by l-bfgs
        if 0:
            from io import StringIO
            from contextlib import redirect_stdout
            f = StringIO()
            with redirect_stdout(f):
                dech, resd[snamef+'_stats'][m] = deconv_atom(
                    im=ims[:, m], psf=psfs[:, m], dd=ddh)
            resd[snamef+''][m] = dech[0][0]
            out = f.getvalue().splitlines()
        else:
            dech, resd[snamef+'_stats'][m] = deconv_switcher(
                im=mIm, psf=psfs[m], tiling_dict=td, deconv_dict=ddh, do_tiling=rparams['do_tiling'])
            resd[snamef][m] = nip.extract(
                dech[0][0], resd['dec_2D_shepp'][m].shape) if rparams['use_slices'] else dech[0][0]

    # done?
    if verbose:
        print("Done.")
    return resd