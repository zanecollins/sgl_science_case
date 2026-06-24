#! /usr/bin/env python3
"""
WARNING: This is copied directly from POSEIDON and is a wrapper of SpectRes that allows you to bin to a given R
"""
import numpy as np
from spectres import spectres

def bin_spectrum(wl_native, spectrum_native, R_bin, err_data = []):
    '''
    Bin a model spectrum down to a specific spectral resolution. 
    
    This is a wrapper around the Python package SpectRes (for details on the 
    resampling algorithm, see https://arxiv.org/abs/1705.05165).

    Args:
        wl_native (np.array of float): 
            Input wavelength grid (μm).
        spectrum_native (np.array of float): 
            Input spectrum.
        R_bin (float or int):
            Spectral resolution (R = wl/dwl) to re-bin the spectrum onto.
        err_data (np.array of float):
            1σ errors on the spectral data.

    Returns:
        wl_binned (np.array of float): 
            New wavelength grid spaced at R = R_bin (μm).
        spectrum_binned (np.array of float):
            Re-binned spectrum at resolution R = R_bin.
        err_binned (np.array of float):
            Re-binned errors at resolution R = R_bin.

    '''
        
    # Create binned wavelength grid at resolution R_bin
    delta_log_wl_bins = 1.0/R_bin
    N_wl_bins = (np.log(wl_native[-1]) - np.log(wl_native[0])) / delta_log_wl_bins
    N_wl_bins = np.around(N_wl_bins).astype(np.int64)
    log_wl_binned = np.linspace(np.log(wl_native[0]), np.log(wl_native[-1]), N_wl_bins)    
    wl_binned = np.exp(log_wl_binned)
    
    # Call Spectres routine
    if (len(err_data) != 0):
        spectrum_binned, err_binned = spectres(wl_binned, wl_native, spectrum_native,
                                               spec_errs = err_data, verbose = False)

        # Cut out first and last values to avoid SpectRes boundary NaNs
        wl_binned = wl_binned[1:-1]
        spectrum_binned = spectrum_binned[1:-1]
        err_binned = err_binned[1:-1]

        # Replace Spectres boundary NaNs with second and penultimate values
     #   err_binned[0] = err_binned[1]
     #   err_binned[-1] = err_binned[-2]

    # Call Spectres routine
    else:
        spectrum_binned = spectres(wl_binned, wl_native, spectrum_native,
                                   verbose = False)

        # Cut out first and last values to avoid SpectRes boundary NaNs
        wl_binned = wl_binned[1:-1]
        spectrum_binned = spectrum_binned[1:-1]
        err_binned = None

    return wl_binned, spectrum_binned, err_binned