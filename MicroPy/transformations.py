'''
Transformations that were helpful will be collected here. No matter whether Hough, Radon, Fourier, Affine, ...
'''
import NanoImagingPack as nip
from scipy.fftpack import dct

# %%
# ------------------------------------------------------------------
#               FOURIER (-like) TRAFOS
# ------------------------------------------------------------------


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
    return nip.irft(nip.ift(im, axes=(-2, -1)), axes=-3, s=s)


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
def lp_norm(im, p=2):
    '''
    Calculates the LP-norm.
    '''
    return (np.sum(np.abs(im)**p))**(1.0/p)


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
