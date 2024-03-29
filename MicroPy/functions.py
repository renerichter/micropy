'''
Relevant functions are implemented here. 
'''
from typing import Optional, Tuple, List, Union, Generator, Callable
from .general_imports import *
from .numbers import harmonic_mean


def gaussian1D_func(x,mu,sigma):
    gaussian1D=1.0 / np.sqrt(2*np.pi*sigma) * \
        np.exp(- (x - mu)**2 / (2 * sigma**2))
    return gaussian1D

def gaussian1D(size=10, mu=0, sigma=20, axis=-1, norm='sum'):
    '''
    Calculates a 1D-gaussian.

    For testing: size=100;mu=0;sigma=20;
    '''
    xcoords = nip.ramp1D(mysize=size, placement='center', ramp_dim=axis)
    gaussian1D = gaussian1D_func(xcoords,mu,sigma)
    if norm == 'sum':
        gaussian1D /= np.sum(gaussian1D)
    return gaussian1D


def gaussian2D(size=[], mu=[], sigma=[]):
    '''
    Calculates a 2D gaussian. 
    Note that mu and sigma can be different for the two directions and hence have to be input explicilty.

    Example: 
    size=[100,100];mu=[0,5];sigma=[2,30];
    '''
    # just to make sure
    if not (type(size) in [tuple, list]):
        size = [size, size]
    if not (type(mu) in [tuple, list]):
        mu = [mu, mu]
    if not (type(sigma) in [tuple, list]):
        sigma = [sigma, sigma]
    gaussian2D = gaussian1D(size=size[0], mu=mu[0], sigma=sigma[0], axis=-2) * \
        gaussian1D(size=size[1], mu=mu[1], sigma=sigma[1], axis=-1)
    return gaussian2D


def gaussianND(size: np.ndarray,
               mu: Union[int, float, Tuple, List, np.ndarray] = 0,
               sigma: Union[int, float, Tuple, List, np.ndarray] = 1):
    '''
    Calculates an ND gaussian. 
    Note that mu and sigma can be different for the two directions and hence have to be input explicilty.

    Example: 
    size=[32,64,128];mu=0;sigma=[2,10,5];
    nip.v5(mipy.gaussianND(size,mu,sigma))
    '''
    # sanity
    mu = [mu, ]*len(size) if type(mu) in [int, float] else np.array(mu)
    sigma = [sigma, ]*len(size) if type(sigma) in [int, float] else np.array(sigma)

    # calculate gaussian
    gaussianND = np.ones(size)
    dimlist = list(np.arange(gaussianND.ndim))
    for m, msize in enumerate(size):
        gaussianND *= np.expand_dims(gaussian1D(size=msize,
                                                mu=mu[m], sigma=sigma[m], axis=m), dimlist[m+1:])

    # done?
    return gaussianND

def gaussProd_mu(λ1, λ2):
    return 1/(1+λ2*λ2/(λ1*λ1))

def gaussProd_sigma(σ1, σ2):
    return harmonic_mean(σ1, σ2)