'''
Relevant functions are implemented here. 
'''
import NanoImagingPack as nip
import numpy as np


def gaussian1D(size=10, mu=0, sigma=20, axis=-1, norm='sum'):
    '''
    Calculates a 1D-gaussian.

    For testing: size=100;mu=0;sigma=20;
    '''
    xcoords = nip.ramp1D(mysize=size, placement='center', ramp_dim=axis)
    gaussian1D = 1.0 / np.sqrt(2*np.pi*sigma) * \
        np.exp(- (xcoords - mu)**2 / (2 * sigma**2))
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
    gaussian2D = gaussian1D(size=size[0], mu=mu[0], sigma=sigma[0], axis=-1) * \
        gaussian1D(size=size[1], mu=mu[1], sigma=sigma[1], axis=-2)
    return gaussian2D
