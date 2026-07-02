import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ============== USER PARAMETERS - TWEAK THESE ==============
# This simple model demonstrates how finite spectral resolution
# causes close spectral features (e.g. two absorption lines) to blend
# together, making them indistinguishable.
#
# Relevant to exoplanet transmission spectroscopy, molecular line
# identification, and concepts like SGL spectral imaging.

lambda_c = 1.0          # reference wavelength in microns (e.g. near-IR)
line_separation = 0.0002 # separation between the two line centers (microns)
line_depth = 0.30       # depth of each absorption feature (0-1)
intrinsic_fwhm = 0.00015 # intrinsic FWHM of each narrow line (microns)
R_low = 1000
R_high = 50000
wl_min = 0.985
wl_max = 1.015
dlambda_fine = 0.00002  # fine grid step (much smaller than intrinsic width)

# ========================================================

print("Building simple spectral resolution demonstration model...")

# High-resolution "truth" wavelength grid
wl_fine = np.arange(wl_min, wl_max + dlambda_fine/2, dlambda_fine)

# Positions of the two features
c1 = lambda_c - line_separation / 2.0
c2 = lambda_c + line_separation / 2.0

# Convert FWHM to sigma for Gaussian
intrinsic_sigma = intrinsic_fwhm / (2 * np.sqrt(2 * np.log(2)))

def make_gaussian_absorption(wl, center, sigma, depth):
    """Create a Gaussian absorption profile (depth is max absorption)."""
    return depth * np.exp( -0.5 * ((wl - center) / sigma)**2 )

abs1 = make_gaussian_absorption(wl_fine, c1, intrinsic_sigma, line_depth)
abs2 = make_gaussian_absorption(wl_fine, c2, intrinsic_sigma, line_depth)

# Simple transmission-like spectrum (continuum = 1.0)
# Note: for non-overlapping lines this is fine; real radiative transfer more complex
spectrum_true = 1.0 - abs1 - abs2

def simulate_observed_spectrum(wl, spec_true, R, lambda_ref=None):
    """
    Simulate finite spectral resolution by convolving with Gaussian LSF.
    LSF FWHM = lambda_ref / R
    """
    if lambda_ref is None:
        lambda_ref = lambda_c
    fwhm = lambda_ref / float(R)
    sigma_wl = fwhm / (2 * np.sqrt(2 * np.log(2)))  # sigma of Gaussian LSF
    dlam = wl[1] - wl[0]
    sigma_pix = sigma_wl / dlam
    # Smooth the high-res spectrum
    spec_smoothed = gaussian_filter1d(spec_true, sigma_pix, mode='nearest') #Gaussian kernel for convolving high res to low res
    return spec_smoothed, fwhm

# Generate the two cases
spec_low, fwhm_low = simulate_observed_spectrum(wl_fine, spectrum_true, R_low)
spec_high, fwhm_high = simulate_observed_spectrum(wl_fine, spectrum_true, R_high)

# ============== DIAGNOSTIC OUTPUT ==============
print(f"\n--- Key Parameters ---")
print(f"Line separation: {line_separation:.4f} μm ({line_separation*1000:.1f} nm)")
print(f"Intrinsic line FWHM: {intrinsic_fwhm:.5f} μm")
print(f"\nLOW RESOLUTION (R={R_low}):")
print(f"  Instrument FWHM (resolution element) = {fwhm_low:.4f} μm")
print(f"  Separation / FWHM = {line_separation / fwhm_low:.2f}")
if line_separation < fwhm_low:
    print("  >>> FEATURES ARE BLENDED: You cannot reliably distinguish two separate lines.")
else:
    print("  >>> Marginally resolved at this R.")

print(f"\nHIGH RESOLUTION (R={R_high}):")
print(f"  Instrument FWHM (resolution element) = {fwhm_high:.4f} μm")
print(f"  Separation / FWHM = {line_separation / fwhm_high:.2f}")
if line_separation > fwhm_high * 1.5:
    print("  >>> FEATURES ARE CLEARLY RESOLVED: Two distinct lines visible.")
else:
    print("  >>> Only marginally resolved.")

# ============== PLOTTING ==============
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, 
                                gridspec_kw={'hspace': 0.28}, constrained_layout=True)

plot_configs = [
    (ax1, spec_low, R_low, fwhm_low, 'Low Spectral Resolution\n(Close features blend together)'),
    (ax2, spec_high, R_high, fwhm_high, 'High Spectral Resolution\n(Features can be distinguished)')
]

for ax, spec_obs, R, fwhm, title in plot_configs:
    # High-resolution truth (faint)
    ax.plot(wl_fine, spectrum_true, color='#555555', lw=1, alpha=0.55, 
            label='True high-res spectrum')
    
    # Smoothed / observed at this R
    ax.plot(wl_fine, spec_obs, color='#1f77b4', lw=1, 
            label=f'Observed at R = {R:.1e}')
    
    # Mark true line centers
    ax.axvline(c1, color='#d62728', linestyle='--', alpha=0.65, lw=1.0)
    ax.axvline(c2, color='#d62728', linestyle='--', alpha=0.65, lw=1.0)
    
    # Visual indicator of resolution element size (FWHM)
    y_bar = 0.965
    left = lambda_c - fwhm/2
    right = lambda_c + fwhm/2
    # Horizontal bar
    ax.plot([left, right], [y_bar, y_bar], 
            color='black', lw=1, solid_capstyle='round', 
            label=f'Bin size')
    
    # Vertical caps at ends
    cap_height = 0.025
    ax.plot([left, left], [y_bar - cap_height/5, y_bar + cap_height/5], 
            color='black', lw=1)
    ax.plot([right, right], [y_bar - cap_height/5, y_bar + cap_height/5], 
            color='black', lw=1)
    
    # R value in bottom left
    ax.text(wl_min + 0.005, 0.62, f'R = {R}', 
            ha='left', va='bottom', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'))
    
    ax.set_ylabel('Normalized Flux')
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle=':')
    ax.set_ylim(0.58, 1.10)

ax2.set_xlabel('Wavelength (μm)')

fig.suptitle('Demonstration: How Spectral Resolution Limits Our Ability to\n'
             'Distinguish Between Close Spectral Features',
             fontsize=13, fontweight='bold')

save_path = '/Users/zaniaccollins/Research/SGL/sgl_science_case/Figures/spec_res_demo'
plt.savefig(save_path, dpi=170, bbox_inches='tight', facecolor='white')
plt.show()
print(f"\n✓ Plot saved to: {save_path}")
print("Open the PNG to visually see the blending effect at low vs high resolution.")
print("\nModel complete. Edit the parameters at the top of this script and re-run to explore other cases.")