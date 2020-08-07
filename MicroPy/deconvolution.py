'''
This module shall only be installed/used if really needed as it requires Tensorflow to be working.
'''

# %% imports
import InverseModelling as im


def ismR_deconvolution(imfl,psfl,method='multi',regl=None,lambdal=None,NIter=100,tflog_name=None):
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
    if regl==None:
        regl = ['TV','GR','GS']
        lambdal = [2e-4,1e-3,2e-4]

    if method=='combi':
        #BorderSize = np.array([30, 30])
        #res = im.Deconvolve(nimg, psfParam, NIter=NIter, regularization=regl, mylambda = rev[m], BorderRegion=0.1, useSeparablePSF=False)
        pass
    elif method == 'multi':
        res = im.Deconvolve(imfl, psfl, NIter=NIter, regularization=regl, mylambda = lambdal, BorderRegion=0.1, useSeparablePSF=False)
        #print('test')
    else: 
        raise ValueError('Method unknown. Please specify your needs.')
    return res