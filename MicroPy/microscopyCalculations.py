'''
Some easy calculations regarding resolution and magnification for different imaging systems are included here. 
'''
# %% ------------------------------------------------------
# ---         RESOLUTION ESTIMATIONS             ---
# ---------------------------------------------------------
#
import numpy as np
from .numbers import harmonic_sum

def calculate_na_fincorr(obj_M_new=5, obj_M=10, obj_NA=0.25, obj_N=1, obj_di=160, roundout=0, printout=False):
    '''
    Calculates the NA of a finite-corrected objective given the standard and the new (calibration) measurements. 

    :param:
    =======
    :obj_M:FLOAT: magnification of objective
    :obj_NA:FLOAT: 0.25 
    :obj_N:FLOAT: refractive index -> air
    :obj_di:FLOAT: distance to image 

    :output:
    ========
    NA:FLOAT:   new, calculated NA

    Innline Testing: 
    obj_M_new = 3.3;obj_M=10;obj_NA=0.25;obj_N=1;obj_di=160;printout=True;roundout=3
    '''

    # here: finite corrected, hence:
    obj_ds = obj_di / obj_M
    # now use thin-lense approximation for objective to get a guess -> assume objective as "thin-lens"
    obj_f = harmonic_sum(obj_ds, obj_di)  # focal-plane distance
    obj_alpha = np.arcsin(obj_NA/obj_N)  # half opening-angle
    obj_D2 = obj_ds * np.tan(obj_alpha)  # half Diameter of "lens"
    # calc new NA and distances
    # f *M1 / (M1+1) = g1 -> given f is constant
    obj_ds_new = obj_f * (obj_M_new + 1) / obj_M_new
    obj_di_new = obj_M_new * obj_ds_new
    obj_alpha_new = np.arctan(obj_D2 / obj_ds_new)
    obj_na_new = obj_N * np.sin(obj_alpha_new)
    res_list = [obj_na_new, obj_alpha_new, obj_ds_new, obj_di_new]
    if roundout > 0:
        res_list = [round(res_list[m], roundout) for m in range(len(res_list))]
    if printout == True:
        print("NA_new=\t\t{}\nOBJ_alpha_new=\t{}\nOBJ_DS_new=\t{}\nOBJ_DI_new=\t{}".format(
            res_list[0], res_list[1], res_list[2], res_list[3]))
    return res_list


def calculate_magnification(pixel_range=1000, counted_periods=10, pixel_size=1.4, period_length_real=100, printout=False):
    '''
    Calculates the magnification using a grid-sample. 

    :param:
    =======
    :pixel_range:       Length of the used set
    :counted_periods:   Number of periods in measured length
    :period_length_real: Real period length
    :pixel_size:        size of pixel -> ideally in same dimension as real period length

    :out:
    ====
    :magnification:   

    '''
    period_length_pixel = pixel_range/counted_periods
    pixel_size_in_sample = period_length_real / period_length_pixel
    magnification = period_length_pixel * pixel_size / period_length_real
    if printout == True:
        print("New Magnification is: M={}.\nPixelsize in Sample-Coordinates is {}um.".format(
            np.round(magnification, 5), pixel_size_in_sample))
    return magnification


def calculate_maxbinning(res_lateral=100, obj_M=10, pixel_size=6.5, printout=True):
    '''
    Calculates the maximum bin-size possible to ensure (at least) correct Sampling, ignoring sub-pixel sampling.
    '''
    dmax_sampling_detector = res_lateral/2 * \
        obj_M  # to be correctly nyquist sampled on the detector
    max_binning = np.floor(dmax_sampling_detector / pixel_size)
    if printout == True:
        print("Maximum sampling steps on detector are dmax={}.\nHence Maximum binning is b={}.".format(
            np.round(dmax_sampling_detector, 2), max_binning))
    return max_binning, dmax_sampling_detector


def calculate_resolution(obj_na=0.25, obj_n=1, wave_em=525, technique='Brightfield', criterium='Abbe', cond_na=0, fluorescence=False, wave_ex=488, printout=False):
    '''
    Calculates the resolution for the selected technique with the given criteria in lateral xy and axial z.  

    For inline testing: obj_na=0.25;obj_n=1;wave_em=525;technique='brightfield';criterium='Abbe'; cond_na=0; fluorescence=False; wave_ex=488; printout=False

    technique: Brightfield,Confocal
    criterium: Abbe,AU,FWHM,Rayleigh,Sparrow
    '''
    # params and preparations
    fact_dict={'Abbe': 1, 'Rayleigh': 1.22, 'AU': 2*1.22, 'Sparrow': 0.95, 'FWHM': 1.02}

    res = np.zeros(3)
    # get right na
    if fluorescence:
        na = 2*obj_na
    else:
        if cond_na:
            na = cond_na + obj_na
        else:
            na = obj_na
    alpha = np.arcsin(obj_na/obj_n)

    # calculate Abbe-support-limit
    if technique == 'Brightfield':
        res[0] = wave_em/(obj_n*(1-np.cos(alpha)))
        res[1] = wave_em/na
        res[2] = res[1]
    elif technique == 'Confocal':
        # assume to be in incoherent case right now
        if fluorescence:
            leff = harmonic_sum(wave_ex, wave_em)
        else:
            leff = wave_em
        res[0] = leff/(obj_n*(1-np.cos(alpha)))
        res[1] = leff / na
        res[2] = res[1]
    else:
        raise ValueError("Selected technique not implemented yet.")
    
    # multiply factors for criteria
    res *= fact_dict[criterium]
    
    # print out
    if printout == True:
        print(f"~~~~~~Resolution results:~~~~~~\n~~~\tTechnique={technique}\n~~~\tCriterium={criterium}\n~~~\tResolution [z,y,x]={np.round(res,4)} nm.")
    
    # finally, return result
    return res

def convert_x_to_k(imsize, dx):
    '''
    imsize could eg be 2*nbr_bins used for calculating radial-summation or represent k-space from x-space
    '''
    k_max = 1.0/np.array(dx)
    dk = 2*k_max/(np.array(imsize))
    return k_max, dk

def convert_dict():
    conv_dict = {'h': [6.6261*10**(-34), 'J*s'], 'c': [2.99*10**8, 'm/s']}
    return conv_dict


def convert_lambda2energy(λ: float):
    '''in Joule; λ in m'''
    convd = convert_dict()
    return convd['h'][0]*convd['c'][0]/(λ*10**(-9))


def convert_phot2int(λ: float, NPhot: float = 1e9, area: float = 1e-4, time: float = 1):
    '''in Phot*J/m^2/s'''
    E = convert_lambda2energy(λ)
    return NPhot*E/area/time


def convert_int2phot(λ: float, I: float = 1e3, area: float = 1e-4, time: float = 1):
    ''''''
    E = convert_lambda2energy(λ)
    return I*time*area/E