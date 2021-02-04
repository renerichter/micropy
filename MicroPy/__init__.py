from .basicTools import *
from .experimental import *
from .filters import *
from .fitting import *
from .functions import *
from .old.imageAnalysis import *
from .inout import *
from .microscopyCalculations import *
from .numbers import *
from .processingISM import *
from .simulation import *
from .stackProcess import *
from .testing import *
from .transformations import *
from .utility import *
from .config import *

if useDECON:
    from .deconvolution import *


# %%
# -------------------------------------------------------------------------
# Module-wide functions
# -------------------------------------------------------------------------
#


def lookfor(searchstring, show_doc=False):
    """To search for a function in the Micropy-Toolbox. 

    Note
    ----
    This function is a 1:1 copy of the NanoImagingPack-Toolbox function "lookfor"!

    Parameters
    ----------
    searchstring : string
        function to search
    show_doc : bool, optional
        whether to show documentation, by default False

    Returns
    -------
    what it finds...

    Example:
    --------
    >>> import Micropy as mipy
    >>> mipy.lookfor("Fourier")
    """
    from inspect import getmembers, isfunction, isclass
    import sys
    from difflib import SequenceMatcher

    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    flist = getmembers(sys.modules[__name__], isfunction)  # function_list
    clist = getmembers(sys.modules[__name__], isclass)  # class_list
    fl = []
    for f in flist:
        ratio = similar(searchstring, f[0])
        ds = f[1].__doc__
        if type(ds) == str:
            s = ds.split()
            for el in s:
                r = similar(searchstring, el)
                if r > ratio:
                    ratio = r
        if ratio >= 0.9:
            fl.append((f[0], f[1], ratio))
    fl.sort(key=lambda x: x[2], reverse=True)

    cl = []
    for c in clist:
        ratio = similar(searchstring, c[0])
        ds = c[1].__doc__
        if type(ds) == str:
            s = ds.split()
            for el in s:
                r = similar(searchstring, el)
                if r > ratio:
                    ratio = r
        if ratio >= 0.5:
            cl.append((c[0], c[1], ratio))
    cl.sort(key=lambda x: x[2], reverse=True)

    print('')
    print('Did you mean:')
    print('')
    print('Functions:')
    print('==========')
    for f in fl:
        print(f[0]+'\t\t'+f[1].__module__)
        if show_doc == True:
            print(f[1].__doc__)
    print('')
    print('Classes:')
    print('========')
    for c in cl:
        print(c[0]+'\t\t'+str(c[1]))
        if show_doc == True:
            print(c[1].__doc__)
    return()
