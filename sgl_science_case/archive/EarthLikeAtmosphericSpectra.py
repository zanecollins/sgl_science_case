import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import photutils
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle
from scipy.spatial import cKDTree
from pyhdf.SD import SD, SDC
from matplotlib.widgets import Button
from photutils.datasets import (apply_poisson_noise,
                                make_4gaussians_image)
import glob
import os

filename = '/Users/ZaniacCollins/Desktop/VenusProject/sgl/TransmissionSpectra/summed_spectrum.txt'

data = np.genfromtxt(filename, skip_header=2)
wavenumber = np.genfromtxt(filename, skip_header=1, usecols=0, delimiter = '')
wavelengths = 10**4/wavenumber
effective_height = np.genfromtxt(filename, skip_header = 1, usecols = 1, delimiter = '')

# --- Noise settings ---
TARGET_SNR = 1.0           # target SNR at a reference level
NOISE_MODEL = "package"   # "gaussian_additive" | "gaussian_fractional" | "poisson"
REF_KIND = "median"          # how to set the reference level: "median" | "mean"
RNG_SEED = 42                # for reproducibility

rng = np.random.default_rng(RNG_SEED)

y_true = effective_height.copy()

def ref_level(y, kind="median"):
    if kind == "median":
        return np.median(y)
    elif kind == "mean":
        return np.mean(y)
    else:
        raise ValueError("REF_KIND must be 'median' or 'mean'")

if NOISE_MODEL == "gaussian_additive":
    # Choose sigma so that SNR_ref = mu / sigma = TARGET_SNR
    mu = ref_level(y_true, REF_KIND)
    sigma = np.abs(mu) / (TARGET_SNR + 1e-30)
    noise = rng.normal(0.0, sigma, size=y_true.shape)
    y_noisy = y_true + noise

elif NOISE_MODEL == "gaussian_fractional":
    # Fractional (multiplicative) noise: y * (1 + N(0, 1/TARGET_SNR))
    frac_sigma = 1.0 / (TARGET_SNR + 1e-30)
    noise_frac = rng.normal(0.0, frac_sigma, size=y_true.shape)
    y_noisy = y_true * (1.0 + noise_frac)

elif NOISE_MODEL == "poisson":
    # Poisson with background b(λ)
    b = np.zeros_like(y_true)          # <-- replace with your background spectrum
    y_nonneg = np.clip(y_true, 0, None)
    b_nonneg = np.clip(b, 0, None)

    mu = max(ref_level(y_nonneg, REF_KIND), 1e-30)
    kappa = (TARGET_SNR**2) * (mu + ref_level(b_nonneg, REF_KIND)) / (mu**2)
    S = kappa * y_nonneg
    B = kappa * b_nonneg
    N = rng.poisson(S + B)
    y_noisy = (N - B) / kappa          # background-subtracted estimate

elif NOISE_MODEL == "package":
    scale_factor = 10e25  # Adjust based on your data
    y_scaled = y_true * scale_factor
    y_nonneg = np.clip(y_scaled, 0, None)
    y_noisy = np.random.poisson(lam=y_nonneg) / scale_factor
    #y_noisy = np.random.poisson(lam = y_true)

# --- Plot ---
fig, ax = plt.subplots(figsize=(12,8))
ax.set_title('High Resolution Atmospheric Spectra for Earth (R >= 10^6)')
ax.set_xlim(0.4, 15)
ax.set_xlabel('Wavelength (µm)')
ax.set_ylim(np.min(y_noisy), np.max(y_noisy) + np.max(y_noisy)/10)
ax.set_ylabel('Effective height (km)')


ax.plot(wavelengths, y_noisy, color='black', linewidth=0.4, alpha=0.8, label=f'Noisy ({NOISE_MODEL})', zorder=1)

plt.tight_layout()
plt.show()