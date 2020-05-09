'''
All data generating functions will be found in here. 
'''
import numpy as np
import NanoImagingPack as nip

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


# %%
# ------------------------------------------------------------------
#                       TEST-OBJECT generators
# ------------------------------------------------------------------

def generate_testobj(test_object=3):
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
        #im[imc-5] = nip.shift2Dby(im[imc-5],[np.floor(im.shape[1]/4.0),0])
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
        #im[imc+5] = nip.shift2Dby(im[imc+5],[np.floor(im.shape[1]/6.0),0])
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


# %%
# ------------------------------------------------------------------
#                       PSF-generators
# ------------------------------------------------------------------

# %%
# ------------------------------------------------------------------
#                       General-generators
# ------------------------------------------------------------------


def gen_shift_npfunc(soff, pix):
    '''
    Calculates coordinates for an equally spaced rect-2D-array. Use with e.g. shiftby_list to generate a shifted set of images.

    NOTE: Slower than gen_shift_loop

    :PARAMS:
    ========
    :soff:      (LIST) offset between pixel-array (2D)
    :pix:       (LIST) number of pixels per direction (2D)

    :OUT:
    =====
    :c:         (NP.NDARRAY) calculated array spacing

    Test with ipython:
    ==================
    %timeit c1 = gen_shift_npfunc([1,1],[5,5])
    %timeit c2 = gen_shift_loop([1,1],[5,5])
    #29.7 µs ± 883 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    #18.7 µs ± 1.09 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
    '''
    a = np.repeat(
        np.arange(-int(pix[0]/2.0), int(pix[0]/2.0)+1)[np.newaxis, :], pix[1], 0).flatten()
    b = np.repeat(
        np.arange(-int(pix[1]/2.0), int(pix[1]/2.0)+1)[:, np.newaxis], pix[0], 1).flatten()
    c = [[soff[0]*m, soff[1]*n] for m, n in zip(b, a)]
    return c


def gen_shift_loop(soff, pix):
    '''
    Calculates coordinates for an equally spaced rect-2D-array. Use with e.g. shiftby_list to generate a shifted set of images.

    TODO: 
        1) generalize for different dimensions (e.g. give unit-cell and generate pattern)
        2) generalize for nD-input

    :PARAMS:
    ========
    :soff:      (LIST) offset between pixel-array (2D)
    :pix:       (LIST) number of pixels per direction (2D)

    :OUT:
    =====
    :c:         (NP.NDARRAY) calculated array spacing
    '''
    c = []
    xo = -int(pix[1]/2.0)
    yo = -int(pix[0]/2.0)
    for jj in range(pix[0]):
        for kk in range(pix[1]):
            c.append([soff[0]*(yo+jj), soff[1]*(xo+kk)])
    return np.array(c)


# %%
# ------------------------------------------------------------------
#                       FWD-Model
# ------------------------------------------------------------------

def forward_model(im, psf, **kwargs):
    image = nip.ift3d(nip.ft3d(im[np.newaxis])*nip.ft3d(psf))
    return image
