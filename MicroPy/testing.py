'''
    Holds different test functions. 
'''
from .functions import gaussian1D, gaussian2D
from .general_imports import *
# %% -------------------------------------------------------
# speedtesting and comparison
# ----------------------------------------------------------
# 1D vs 2D convolution -> 1D seems slightly faster ->


def conftest1(im):
    a1 = nip.convolve(im, gaussian1D(
        size=im.shape[-1], mu=0, sigma=20, axis=-1), axes=-1)
    a2 = nip.convolve(a1, gaussian1D(
        size=im.shape[-2], mu=0, sigma=20, axis=-2), axes=-2)
    return a1, a2


def conftest2(im):
    b = nip.convolve(im, gaussian2D(size=im.shape[-2:], mu=0, sigma=20))
    return b


def compare_conf(im):
    from timeit import timeit
    c1, c2 = conftest1(im)
    c3 = conftest2(im)
    nip.catE((c1, c2, c3, c2-c3))
    # compare times/speed
    # 1.9 ms ± 199 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    timeit('c1, c2 = conftest1()', number=1000)
    # 2.14 ms ± 121 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    timeit('c2 = conftest2()')
