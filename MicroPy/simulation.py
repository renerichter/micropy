'''
All data generating functions will be found in here.
'''
# mipy imports
from .basicTools import sanityCheck_structure
from .transformations import irft3dz, rft3dz, rftnd
from .utility import shiftby_list, add_multi_newaxis

# external imports
import copy
import NanoImagingPack as nip
import numpy as np

# %%
# ------------------------------------------------------------------
#                       Parameter-generators
# ------------------------------------------------------------------


def PSF_SIM_PARA(para):
    '''
    Set standard-parameters for a PSF-simulation using the nip.PSF_PARAMS() set.

    PARAMS:
    ======
    :para:  (STRUCT) that can be generated by nip.PSF_PARAMS()

    OUTPUT:
    =======
    :para:  (STRUCT) changed param
    '''
    if para == None:
        para = nip.PSF_PARAMS()
    para.NA = 1.4
    para.n = 1.518
    para.lambdaEx = 488
    para.wavelength = 488
    para.pol = para.pols.lin
    para.pol_lin_angle = 0
    para.vectorized = False
    para.aplanar = para.apl.no
    para.aperture_method = 'jinc'

    # done?
    return para


# %%
# ------------------------------------------------------------------
#                       TEST-OBJECT generators
# ------------------------------------------------------------------

def generate_testobj(test_object=3, mypath=''):
    '''
    Generates a 3D testobject depending on choice.
    '''
    if test_object == 0:
        im = nip.readim(mypath + 'images/chromo3d.ics')
    elif test_object == 1:
        a = nip.readim()
        im = nip.image(np.zeros([15, a.shape[0], a.shape[1]]))
        im[7] = a
        im[9] = np.rot90(a)
        im[3] = np.rot90(a, k=3)
    elif test_object == 2:
        # Test-target
        a = nip.readim()

        # get ROI -> first marks MAIN = central slice
        ROIs = [[160, 340, 285, 465], [58, 140, 174, 260],
                [390, 465, 210, 390], [60, 160, 300, 450]]
        b = []
        for m in ROIs:
            b.append(a[m[0]:m[1], m[2]:m[3]])
        b.append(np.rot90(b[1]))
        # for m in range(len(ROIs)):
        #    center = np.array(b[0].shape)- np.array(np.ceil(np.array(b[m].shape)/2),dtype=np.int)
        #    b[m] = nip.extract(b[m],ROIsize=[ROIs[0][1]-ROIs[0][0],ROIs[0][3]-ROIs[0][2]])

        # build image from ROIs
        im = nip.image(
            np.zeros((15, ROIs[0][1]-ROIs[0][0], ROIs[0][3]-ROIs[0][2])))
        imc = int(np.floor(im.shape[0]/2.0))
        im[imc-3] = nip.extract(b[2], ROIsize=im.shape[1:])
        # im[imc-5] = nip.shift2Dby(im[imc-5],[np.floor(im.shape[1]/4.0),0])
        im[imc-2] = nip.extract(b[-2], ROIsize=im.shape[1:])
        im[imc-2] = nip.shift2Dby(im[imc-2], [np.floor(im.shape[1]/6.0), 0])
        im[imc-1, :b[-1].shape[0], :b[-1].shape[1]] = b[-1]
        im[imc] = b[0]
        im[imc+1, :b[1].shape[0], :b[1].shape[1]] = b[1]
        im[imc+1] = im[imc+1, ::-1, ::-1]
        im[imc+2] = nip.extract(b[2], ROIsize=im.shape[1:])
        im[imc+2] = nip.shift2Dby(im[imc+2], [-np.floor(im.shape[1]/4.0), 0])
        im[imc+3] = nip.extract(b[-2], ROIsize=im.shape[1:])

        # max-normalize the planes to the same uint8 max
        immax = im.max((-2, -1))
        immax[immax == 0] = 1
        im = im / immax[:, np.newaxis, np.newaxis]*np.iinfo('uint8').max
        im[im < 0] = 0
        # im[imc+5] = nip.shift2Dby(im[imc+5],[np.floor(im.shape[1]/6.0),0])
    elif test_object == 3:
        # Test-target
        a = nip.readim()

        # get ROI -> first marks MAIN = central slice
        ROIs = [[160, 340, 285, 465], [58, 140, 174, 260],
                [390, 465, 210, 390], [60, 160, 300, 450]]
        b = []
        for m in ROIs:
            b.append(a[m[0]:m[1], m[2]:m[3]])
        b.append(np.rot90(b[1]))

        # build image from ROIs
        im = nip.image(
            np.zeros((15, ROIs[0][1]-ROIs[0][0], ROIs[0][3]-ROIs[0][2])))
        imc = int(np.floor(im.shape[0]/2.0))
        im[imc-3, int(im.shape[-2]/2.0):, :] = nip.xx([int(im.shape[-2]/2.0),
                                                       im.shape[-1]]) + nip.xx([int(im.shape[-2]/2.0), im.shape[-1]]).min()
        im[imc-2, int(im.shape[-2]/10.0):int(im.shape[-2]/10.0) + int(im.shape[-1]/2.0), int(im.shape[-1]/4.0):int(im.shape[-1]/4.0) + int(im.shape[-1]/2.0)
           ] = (nip.rr([int(im.shape[-2]/2.0), int(im.shape[-1]/2.0)]) <= int(im.shape[-2]/4.0))*1 * nip.rr([int(im.shape[-2]/2.0), int(im.shape[-1]/2.0)]) * 4
        im[imc-1, :b[-1].shape[0], :b[-1].shape[1]]
        im[imc-1, :b[-1].shape[0], :b[-1].shape[1]] = b[-1]
        im[imc] = b[0]
        im[imc+1, :b[1].shape[0], :b[1].shape[1]] = b[1]
        im[imc+1] = im[imc+1, ::-1, ::-1]
        im[imc+2] = nip.extract(b[2], ROIsize=im.shape[1:])
        im[imc+2] = nip.shift2Dby(im[imc+2], [-np.floor(im.shape[1]/4.0), 0])
        im[imc+3] = nip.extract(b[-2], ROIsize=im.shape[1:])

        # max-normalize the planes to the same uint8 max
        immax = im.max((-2, -1))
        immax[immax == 0] = 1
        im = im / immax[:, np.newaxis, np.newaxis]*np.iinfo('uint8').max
        im[im < 0] = 0
    elif test_object == 4:
        a = nip.readim()[160:340, 285:465]
        im = nip.image(np.zeros([16, a.shape[0], a.shape[1]]))
        imc = int(np.floor(im.shape[0]/2.0))
        im[imc] = a
    elif test_object == 5:
        a = nip.readim()[160:340, 285:465]
        im = nip.image(np.zeros([16, a.shape[0], a.shape[1]]))
        imc = int(np.floor(im.shape[0]/2.0))
        im[imc+1] = a
    elif test_object == 6:
        a = nip.readim()[160:340, 285:465]
        im = nip.image(np.zeros([16, a.shape[0], a.shape[1]]))
        imc = int(np.floor(im.shape[0]/2.0))
        im[imc-2] = a
    else:
        im = nip.readim('obj3d')[::2, :, :]

    return im


def ismR_defaultIMG(obj, psf, NPhot=None, use2D=True):
    '''
    Calculates Image using default forward model and Poisson-noise.
    '''
    # params
    objc = np.array(np.floor(np.array(obj.shape)/2.0), dtype=np.uint)

    # calculate image
    im_ism = forward_model(obj, psf)

    # select in-focus slice -> assume real-image and PSF
    if use2D:
        im_ism_sel = np.real(im_ism[:, objc[0]])

    # add Poisson-noise
    im_ism_sel /= np.max(im_ism_sel)
    if NPhot is not None:
        im_ism_sel = nip.poisson(im_ism_sel * NPhot)

    # provide transformed image
    im_ism_ft = nip.ft2d(im_ism_sel) if use2D else nip.ft3d(im_ism_sel)

    return im_ism_sel, im_ism_ft, objc
# %%
# ------------------------------------------------------------------
#                       PSF-generators
# ------------------------------------------------------------------


def calculatePSF_confocal(obj, psf_params, psfp=False, psfex=None, psfdet=None, pinhole=None, **kwargs):
    '''
    Calculates confocal PSF. 
    See calculatePSF for explanation on parameters
    '''

    if psfex is None:
        psfex = nip.psf(obj, psf_params[0])
        if psfex.pixelsize == None:
            psfex.pixelsize = obj.pixelsize

    if psfdet is None:
        if not psfp:
            psf_params.append(copy.deepcopy(psf_params[0]))
            psf_params[1].wavelength = 510
            psf_params[1].lambdaEx = 510
            psf_params[1].pinhole = 0.6
        psfdet = nip.psf(obj, psf_params[1])

        # but why is dimension different? What changes in PSF generation?
        if psfex.ndim < psfdet.ndim:
            psfdet = psfdet[0]

        if psfdet.pixelsize is None:
            psfdet.pixelsize = psfex.pixelsize

    if pinhole is None:
        pinhole = generate_pinhole(psf=psfdet, psf_params=psf_params[1])

    psf = psfex * nip.convolve(psfdet, pinhole)

    # done?
    return psf, psfex, psfdet, pinhole


def calculatePSF_sax(psfex, k_fluo):
    '''
    Calculates Saturated excitation PSF from given saturation parameter k_fluo. 
    Assumes PSFex to be normalized to one. (from: Heintzmann, R. Saturated patterned excitation microscopy with two-dimensional excitation patterns. Micron 34, 283–291 (2003).)
    '''
    psf = k_fluo * psfex / (k_fluo+psfex)

    # done?
    return psf


def calculatePSF_ism(psfex, psfdet=None, psfdet_array=None, shifts=None, shift_offset=[2, 3], shift_axes=[-2, -1], shift_method='uvec', nbr_det=[3, 5], fmodel='rft', faxes=[-2, -1], pinhole=None, do_norm=True):
    """ Generates the incoherent intensity ISM-PSF. Takes system excitation and emission PSF and assumes spatial invariance under translation on the detector. 

    Parameters
    ----------
    psfex : image
        excitation PSF
    psfdet : image, optional
        detection PSF; if 'psfdet_array' is provided, than psfdet is not needed
    psfdet_array : list, optional
        1D-list of detection PSFs = detector. Generated from parameters if not given, by default None
    shifts : list, optional
        1D-list of sorted shifts (from biggest negative to biggest positive) to be applied. If not given, shifts (and hence rectangular detector) will be generated. by default None 
    shift_offset : list, optional
        distance in pixels between detector elements, by default [2,2]
    shift_axes : list, optional
        Axes to be used for shifts, by default [-2,-1]
    shift_method : list, optional
        see gen_shift function for more info, by default 'uvec'
    nbr_det : list, optional
        number of detectors of the recording device, by default [3,3]
    fmodel : str, optional
        Fourier-Transformation to be used for Frequency-operations, by default 'rft'
    faxes: list, optional
        Dimensions to be used for the Fourier-Transformation
    pinhole : bool, optional
        switches whether pinhole dimension already exists and is normalized, by default None

    Returns
    -------
    psf_eff : list
        1D list of effective PSFs per pinhole
    otf_eff : list
        1D list of effective OTFs per pinhole
    psfdet_array : list
        1D list of effective PSFs per pinhole
    shifts : list
        1D list of shifts to new detector positions

    Example:
    --------
    >>> psfex = nip.psf2d()
    >>> psf_eff, otf_eff, psfdet_array = mipy.calculatePSF_ism(psfex, psfex)

    TODO: 
    -----
    1) Add shape- and size of pixel. 
    2) Add collection efficiency. 
    """
    # calculate PSF_array for each detector pixel
    # -> normalize array as whole detector can at max detect one of arriving one photon
    if psfdet_array is None:
        if shifts is None:
            psfdet_array, shifts = shiftby_list(
                psfdet, shifts=None, shift_offset=shift_offset, shift_method=shift_method, shift_axes=shift_axes, nbr_det=nbr_det)
        else:
            shifts = np.sort(shifts)
            psfdet_array, _ = shiftby_list(psfdet, shifts=shifts)
        if pinhole is not None:
            nbrnewaxis = psfdet_array.ndim-pinhole.ndim
            psfdet_array = nip.convolve(
                psfdet_array, add_multi_newaxis(pinhole, [0, ]*nbrnewaxis))
        psfdet_array /= np.sum(psfdet_array, keepdims=True)

    # final (confocal) PSF per pixel via PSFex*PSFdet
    psf_eff = psfex[np.newaxis] * psfdet_array

    # assure PSF is proper == sum to 1
    if do_norm:
        psf_eff /= np.sum(psf_eff, keepdims=True)

    # get OTF
    otf_eff = rftnd(psf_eff, raxis=faxes[0], faxes=faxes[1:]
                    ) if fmodel == 'rft' else nip.ft(psf_eff, axes=faxes)

    # done?
    return psf_eff, otf_eff, psfdet_array, shifts


def calculatePSF(obj, psf_params=None, method='brightfield', amplitude=False, **kwargs):
    '''
    Calculates different PSFs according to the selected method. 

    TODO: 1) fix dsax + dsaxISM structure and idea

    :PARAMS:
    ========
    :im:            (IMAGE) object-image to provide shape and pixel-size
    :psf_params:    (LIST) of STRUCTS from nip.PSF_PARAMS-type -> contains all necessary simulation parameters 
    :method:        (STRING) method to be used. Options are:
                'brightfield': standard PSF_em
                'confocal': PSF_ex * CONV(PSF_em,Pinhole)
                '2photon': (PSF_ex*PSF_ex) * PSF
                'ism': CONV(PSF_ex,detector_geometry)*PSF_em
                'sax': Confocal with saturated PSF
                'dsax': multiple saturated PSFs
                'dsaxISM': CONV(PSF^sat_ex,detector_geometry) * PSF_em
                'sim':  to be done
                'light-sheet': to be done
                'ptychography': to  be done
    :amplitude:     (BOOL) if true, returns amplitude PSF

    :OUTPUT:
    ========
    :psf:       (IMAGE) calculated (a)psf
    -> depending on option: 
                'brightfield': psf
                'confocal': [psf, psfex, psfdet, pinhole]
                '2photon': to be done
                'ism':  [psf_eff, otf_eff, psfex,psfdet_array]
                'sax': [psf, psfex, psfex_sat, psfdet, pinhole]
                'dsax': [psf, psfex_sat, psfdet, pinhole]
                'dsaxISM': [psf_effl, otf_effl, psfex_satl, psfdetl, psfdet_array]
                'sim':  to be done
                'light-sheet': to be done
                'ptychography': to  be done

    :EXAMPLE:
    =========
    psfpara = nip.PSF_PARAMS()
    psfpara.wavelength = 488
    psfpara.NA = 1.4
    psfpara.n = 1.518
    psfpara.pinhole = 0.6 # in Airy-Units of PSFem (widefield)
    psfparams=[psfpara,psfpara]
    psf = calculatePSF(im,psf_params=psfparams,method='confocal',amplitude=False)

    '''
    psfp = False
    if psf_params == None:
        psf_params = nip.PSF_PARAMS()
        psf_params.wavelength = 488
        psf_params.NA = 1.4
        psf_params.n = 1.518
        psf_params = [psf_params, ]
    else:
        psfp = True

    if method == 'brightfield':
        psf_res = nip.psf(obj, psf_params[0])
        if psf_res.pixelsize == None:
            psf_res.pixelsize = obj.pixelsize

    elif method == 'confocal':
        # fish entries from input-list
        psfex = None if not 'psfex' in kwargs else kwargs['psfex']
        psfdet = None if not 'psfdet' in kwargs else kwargs['psfdet']
        pinhole = None if not 'pinhole' in kwargs else kwargs['pinhole']
        psfp = False if len(psf_params) < 2 else psfp

        # calculate PSF and return
        psf, psfex, psfdet, pinhole = calculatePSF_confocal(
            obj=obj, psf_params=psf_params, psfp=psfp, **kwargs)
        return psf, psfex, psfdet, pinhole

    elif method == '2photon':
        pass

    elif method == 'ism':
        # fish entries from input-list
        psfex = None if not 'psfex' in kwargs else kwargs['psfex']
        psfdet = None if not 'psfdet' in kwargs else kwargs['psfdet']
        psfdet_array = None if not 'psfdet_array' in kwargs else kwargs['psfdet_array']
        shift_offset = None if not 'shift_offset' in kwargs else kwargs['shift_offset']
        nbr_det = None if not 'nbr_det' in kwargs else kwargs['nbr_det']
        fmodel = None if not 'fmodel' in kwargs else kwargs['fmodel']
        pinhole = None if not 'pinhole' in kwargs else kwargs['pinhole']

        # calculate confocal PSF
        if psfex is None and psfdet is None:
            psf, psfex, psfdet, pinhole = calculatePSF(
                obj=obj, psf_params=psf_params, method='confocal', amplitude=amplitude, psfex=psfex, psfdet=psfdet)

        # calculate resulting ISM-PSF
        psf_eff, otf_eff, psfdet_array, shifts = calculatePSF_ism(
            psfex=psfex, psfdet=psfdet, psfdet_array=psfdet_array, shift_offset=shift_offset, nbr_det=nbr_det, fmodel=fmodel, pinhole=pinhole)
        return psf_eff, otf_eff, psfex, psfdet_array

    elif method == 'sax':
        # fish entries from input-list
        k_fluo = 1/6 if not 'k_fluo' in kwargs else kwargs['k_fluo']
        psfex = None if not 'psfex' in kwargs else kwargs['psfex']
        psfdet = None if not 'psfdet' in kwargs else kwargs['psfdet']

        # sanity check-parameters
        for m in range(len(psf_params)):
            psf_params[m] = sanityCheck_structure(
                psf_params[m], params={'k_fluo': k_fluo})

        # caculate saturated PSF
        if psfex is None:
            psfex = calculatePSF(obj=obj, psf_params=psf_params,
                                 method='brightfield', amplitude=amplitude, **kwargs)[0]
        psfex_sat = calculatePSF_sax(psfex, k_fluo=k_fluo)

        # calculate total (confocal) PSF
        psf, psfex_sat, psfdet, pinhole = calculatePSF(
            obj=obj, psf_params=psf_params, method='confocal', amplitude=amplitude, psfex=psfex_sat, psfdet=psfdet)
        return psf, psfex, psfex_sat, psfdet, pinhole

    elif method == 'dsax':
        # fish entries from input-list
        k_fluo = 1/6 if not 'k_fluo' in kwargs else kwargs['k_fluo']
        excitation_ratio = [
            1, 0.1, 0.01] if not 'excitation_ratio' in kwargs else kwargs['excitation_ratio']
        order = 2 if not 'order' in kwargs else kwargs['order']

        # check sanity of entries of params_dict
        for m in range(len(psf_params)):
            psf_params[m] = sanityCheck_structure(psf_params[m], params={
                                                  'k_fluo': k_fluo, 'excitation_ratio': excitation_ratio, 'order': order})

        # calculate dsax orders

        psf, psfex, psfex_sat, psfdet, pinhole = calculatePSF(
            obj, psf_params=psf_params, method='sax', amplitude=amplitude, k_fluo=k_fluo)
        psf_satl = [psf, ]
        psfex_satl = [psfex, ]
        psfdetl = [psfdet, ]
        for m in range(1, len(excitation_ratio)):
            psft, _, psfex_satt, psfdett, pinhole = calculatePSF(
                obj, psf_params=psf_params, method='sax', amplitude=amplitude, k_fluo=k_fluo, psfex=psfex*excitation_ratio[m], psfdet=psfdet)
            psf_satl.append(psft)
            psfex_satl.append(psfex_satt)
            psfdetl.append(psfdett)
        #psf_satl = nip.image(np.array(psf_satl))
        #psfex_satl = nip.image(np.array(psfex_satl))
        #psfdetl = nip.image(np.array(psfdetl))

        # done?
        return psf_satl, psfex_satl, psfdetl, pinhole

    elif method == 'dsaxISM':
        # calculate DSAX-ISM PSF for amount of orders (with different excitation factors)
        psf_satl, psfex_satl, psfdetl, pinhole = calculatePSF(
            obj, psf_params=psf_params, method='dsax', amplitude=amplitude, **kwargs)

        # test and add entries
        sanityCheck_structure(kwargs, {'psfex': None, 'psfdet': None})

        # calculate dsaxISM-PSF
        psf_effl = []
        otf_effl = []
        psfdet_array = None
        #pinhole_array = None
        for m in range(len(psfex_satl)):
            kwargs['psfex'] = psfex_satl[m]
            kwargs['psfdet'] = psfdetl[m]
            kwargs['psfdet_array'] = psfdet_array
            #kwargs['pinhole_array'] = pinhole_array
            psf_eff, otf_eff, _, psfdet_array = calculatePSF(
                obj, psf_params=psf_params, method='ism', amplitude=amplitude, **kwargs)
            psf_effl.append(psf_eff)
            otf_effl.append(otf_eff)

        # done?
        return psf_effl, otf_effl, psfex_satl, psfdetl, psfdet_array

    elif method == 'SIM':
        pass
    elif method == 'ptychography':
        pass
    else:
        raise ValueError('Method not implemented yet.')


def generate_pinhole(psf, psf_params, pshape='circular', pedge='hard'):
    '''
    Calculates and generates a pinhole from the simulation properties.

    :PARAMS:
    ========
    :psf:           (IMAGE) PSF to provide shape and pixel-size
    :psf_params:    (LIST) of STRUCTS from nip.PSF_PARAMS-type -> contains all necessary simulation parameters 
    :pshape:        (STRING) possible shapes for pinhole
                    'circular': 
                    'rect': rectangular pinhole -> to be implemented
                    'hexagon': hexagonal pinhole-> to be implemented
    :pedge:         (STRING) properties of pinhole edges 
                    'hard': just a hard edge
                    'gauss': gaussian damped edge -> to be implemented
                    'sinc': sinc damped edge -> to be implemented
                    'invgauss': gaussian increased edges -> to be implemented

    :OUTPUT:
    ========
    :pinhole:       (IMAGE) calculated pinhole

    :EXAMPLE:
    =========

    '''
    # pinhole-size from theoretical AU by: 1.22 * lambda / NA
    airyUNIT = 1.22 * psf_params.wavelength / psf_params.NA
    pinhole_radius = psf_params.pinhole * \
        airyUNIT / (2 * np.array(psf.pixelsize[-2:]))

    if pshape == 'circular':
        pinhole = nip.rr(psf.shape) <= pinhole_radius[-1]
    else:
        raise ValueError(f"Used pshape={pshape} not implemented yet ")

    if pedge == 'hard':
        pass
    else:
        raise ValueError(f"Used pedge={pedge} not implemented yet ")

    # done?
    return pinhole


def ismR_defaultPSF(obj, lex=488, lem=520, shift_offset=[2, 2], nbr_det=[3, 3]):
    # generate PSFs
    para = nip.PSF_PARAMS()
    PSF_SIM_PARA(para)  # add formerly defined PSF-paramaters

    # non-aberrated excitation
    para.lambdaEx = 488
    para.wavelength = 488
    para.aplanar = para.apl.excitation
    psfex = nip.psf(obj, para)

    # non-aberrated emission
    para.lambdaEx = 520
    para.wavelength = 520
    para.aplanar = para.apl.emission
    psfem = nip.psf(obj, para)

    # generate ISM total PSF and normalize to 1 (because if there is a photon reaching the detector it will be detected)
    psfem_array = shiftby_list(
        psfem, shift_offset=shift_offset, nbr_det=nbr_det)
    psf_eff = psfex[np.newaxis] * np.real(psfem_array)
    psf_eff /= np.sum(psf_eff, keepdims=True)
    otf_eff = rft3dz(psf_eff)

    # centers
    psfc = np.array(np.floor(np.array(psf_eff.shape)/2.0), dtype=np.uint)

    return psf_eff, otf_eff, psfc


# %%
# ------------------------------------------------------------------
#                       FWD-Model
# ------------------------------------------------------------------

def forward_model(obj, psf, fmodel='fft', retreal=True, is_list=False, **kwargs):
    '''
    A simple forward model calculation using either fft or rft.

    :PARAMS:
    =======
    :obj:       (IMAGE) real image input
    :psf:       (IMAGE) real psf input
    :fmodel:    (STRING) 'fft' or 'rft'
    :retreal:   (BOOL) wether output should be real

    :OUTPUT:
    ========
    :res:       (IMAGE) simulated image
    '''
    if fmodel == 'fft':
        # normalization for 3dim ortho-convolution normalization on last 3 axes
        res = nip.ift3d(nip.ft3d(
            obj[np.newaxis])*nip.ft3d(psf)) * np.sqrt(np.prod(list(psf.shape[-3:]))) * np.sqrt(np.prod(psf.shape[-3:]))
    else:
        # normalization for 2dim ortho-convolution on last 2 (=fft) axes; RFT is automatically ok due to only normalizing in back-path
        res = irft3dz(rft3dz(obj[np.newaxis])*rft3dz(psf),
                      s=psf.shape) * np.sqrt(np.prod(psf.shape[-2:]))

    if retreal:
        res = res.real

    return res

# %%
# ------------------------------------------------------------------
#                       Stack-Generators
# ------------------------------------------------------------------


def gen_defocusStack(im, blurRange=[-12, 12], noise=[nip.poisson, 100], Nz=41):
    '''
    Generates a defocus-stack.
    '''
    # 3D stack
    ims = nip.repmat(im, [Nz, ] + [1, ]*im.ndim)

    # defocus by simple Gauss-blurring
    blR = np.arange(blurRange[0], blurRange[1],
                    step=(blurRange[1]-blurRange[0])/Nz)
    for m in range(ims.shape[0]):
        ims[m] = nip.gaussian_filter(ims[m], np.abs(blR[m]))

    # add  noise
    imsn = noise[0](ims, NPhot=noise[1])

    # done?
    return ims, imsn
# %%
# ------------------------------------------------------------------
#                       Noise-Generators
# ------------------------------------------------------------------


def add_symmetric_point(im, pos, val):
    '''
    Adds a symmetric point around the Fourier-origin. Takes evenness of image-sizes into account. For now: only works on same dimensionality of im and pos.

    Note: No boundary-checks etc done
    '''
    offsets = np.mod(np.array(im.shape), 2)
    pos2 = tuple([(im.shape[m] - offsets[m]) - va for m, va in enumerate(pos)])
    im[pos] = val
    im[pos2] = val
    return im, pos2


def generate_pickupNoise(imshape, pickup_pos, pickup_level):
    '''
    Generates a pickupNoise-image.

    PARAMS:
    =======
    :imshape:       (TUPLE) shape of output-image
    :pickup_pos:    (LIST) of 
    :pickup_level:  (LIST) of values for pickup

    OUTPUTS:
    ========
    :im_pnFT:       (IMAGE) generated fourier-image
    :im_pn:         (IMAGE) noise-term-image
    :poslist:       (LIST) of Tuples with positions used

    EXAMPLE:
    ========
    im_pnFT, im_pn = generate_pickupNoise(imshape=(5,5,8),pickup_pos=[[2,3,4],[0,1,5]],pickup_level=[10,20])
    '''
    poslist = []
    im_pnFT = nip.image(np.zeros(imshape))
    for m, pos in enumerate(pickup_pos):
        pos = tuple(pos)
        im_pnFT, pos2 = add_symmetric_point(im_pnFT, pos, pickup_level[m])
        poslist.append([pos, pos2])
        im_pn = np.real(nip.ft(im_pnFT))
        im_pn -= np.min(im_pn, keepdims=True)

    # done?
    return im_pnFT, im_pn, poslist

# %%
# ------------------------------------------------------------------
#                       SAMPLING-EFFECTS
# ------------------------------------------------------------------


def sim_pixelSampling(im, method='kernel', lensing_kernel=None, strides=None, keepsize=False):
    '''
    Effect by pixel-sampling. If a lensing_kernel is provided the convolution method is used, else the strided version. Implemented for 2D, so if it should be used for nD, just np.transpose the target 2-dimensions to dimension [0,1]. 

    :PARAMS:
    ========
    :im:                (IMAGE) input image
    :method:            (string) method to use 
                                'kernel' (convolution based),
                                'strideSUM' (sums within strides),
                                'stride' (just every n-th pixel)
    :lensing_kernel:    (ARRAY) kernel to be used 
    :strides:           (ARRAY) Strides per dimension

    OUTPUTS:
    ========
    :imo:               (IMAGE) sampled image

    Example:
    ========
    # parameters
    strides = [4,4]
    lensing_kernel = np.ones(strides); lensing_kernel[0,:] = 0; lensing_kernel[:,0] = 0; 

    # read image
    im = nip.readim()

    # eval
    imL = sim_pixelSampling(im,lensing_kernel=lensing_kernel,strides=None)
    imS = sim_pixelSampling(im,lensing_kernel=None,strides=strides)

    # compare results
    print(np.allclose(imL,imS))


    TODO:
    =====
    1)  in "stridesSUM" -> 1-pixels shift for uneven pixel-pitch between extract-routine and mere selection
    '''
    imh = nip.image(np.zeros(im.shape))
    if method == 'kernel':
        [kernel_center, kernel_stride] = [
            np.array(lensing_kernel.shape)//2, lensing_kernel.shape]
        imo = nip.convolve(im, nip.extract(lensing_kernel, im.shape))
        imo = nip.image(imo[kernel_center[0]::kernel_stride[0],
                            kernel_center[1]::kernel_stride[1]])
        imh[kernel_center[0]::kernel_stride[0],
            kernel_center[1]::kernel_stride[1]] = imo
    elif method == 'stridesSUM':
        imo = nip.image(
            np.zeros((im.shape[0]//strides[0], im.shape[1]//strides[1]), dtype=np.float32))
        for m in range(1, strides[0]):
            for n in range(1, strides[1]):
                try:
                    imo += im[m::strides[1], n::strides[0]]
                except:
                    imo += nip.extract(im[m::strides[1],
                                          n::strides[0]], ROIsize=imo.shape)
        imh[strides[0]//2::strides[0], strides[1]//2::strides[1]] = imo
    elif method == 'strides':
        imh[strides[0]//2::strides[0], strides[1]//2::strides[1]
            ] = im[strides[0]//2::strides[0], strides[1]//2::strides[1]]
    else:
        print("Selection not implemented yet.")

    if keepsize:
        imo = imh

    return imo
