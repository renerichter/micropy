'''
Some easy calculations regarding resolution and magnification for different imaging systems are included here. 
'''

# %% ------------------------------------------------------
# ---         RESOLUTION ESTIMATIONS             ---
# ---------------------------------------------------------
#


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


def calculate_resolution(obj_na=0.25, obj_n=1, wave_em=525, technique='brightfield', criterium='Abbe', cond_na=0, fluorescence=False, wave_ex=488, printout=False):
    '''
    Calculates the resolution for the selected technique with the given criteria in lateral xy and axial z.  

    For inline testing: obj_na=0.25;obj_n=1;wave_em=525;technique='brightfield';criterium='Abbe'; cond_na=0; fluorescence=False; wave_ex=488; printout=False
    '''
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
    if technique == 'brightfield':
        res[0] = wave_em/na
        res[1] = res[0]
        res[2] = wave_em/(obj_n*(1-np.cos(alpha)))
    elif technique == 'confocal':
        # assume to be in incoherent case right now
        if fluorescene:
            leff = harmonic_sum(wave_ex, wave_em)
        else:
            leff = wave_em
        res[0] = leff / na
        res[1] = res[0]
        res[2] = leff/(obj_n*(1-np.cos(alpha)))
    else:
        raise ValueError("Selected technique not implemented yet.")
    # multiply factors for criteria
    if criterium == 'Abbe':
        res = res
    else:
        raise ValueError("Selected criterium not implemented yet.")
    # print out
    if printout == True:
        print("The calculated resolution is: x={}, y={}, z={}".format(
            res[0], res[1], res[2]))
    # finally, return result
    return res
