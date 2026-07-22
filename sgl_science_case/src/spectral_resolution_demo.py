import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ============== USER PARAMETERS - TWEAK THESE ==============
# This simple model demonstrates how finite spectral resolution
# causes close spectral features (e.g. two absorption lines) to blend
# together, making them indistinguishable. Now includes Gaussian noise
# and error bars on observed data to show SNR effects.

lambda_c = 1.0          # reference wavelength in microns (e.g. near-IR)
line_separation = 0.0002 # separation between the two line centers (microns)
line_depth = 0.30       # depth of each absorption feature (0-1)
intrinsic_fwhm = 0.00015 # intrinsic FWHM of each narrow line (microns)
R_low = 1000
R_high = 50000
SNR = 10                # Signal-to-noise ratio (continuum or per resolution element; add Gaussian noise)
wl_min = 0.998
wl_max = 1.002
dlambda_fine = 0.00002  # fine grid step (much smaller than intrinsic width)

# ========================================================

print("Building simple spectral resolution demonstration model with SNR and error bars...")

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
spectrum_true = 1.0 - abs1 - abs2

def simulate_observed_spectrum(wl, spec_true, R, snr=None, lambda_ref=None):
    """
    Simulate finite spectral resolution by convolving with Gaussian LSF.
    Optionally add Gaussian noise based on SNR (per-bin / depth-dependent).
    """
    if lambda_ref is None:
        lambda_ref = lambda_c
    fwhm = lambda_ref / float(R)
    sigma_wl = fwhm / (2 * np.sqrt(2 * np.log(2)))
    dlam = wl[1] - wl[0]
    sigma_pix = sigma_wl / dlam
    spec_smoothed = gaussian_filter1d(spec_true, sigma_pix, mode='nearest')
    
    spec_noisy = None
    sigma = None
    if snr is not None:
        depth = 1 - spec_smoothed  # absorption depth
        sigma = depth / snr        # std dev per bin (your "per-bin" SNR)
        noise = np.random.normal(0, sigma, len(spec_smoothed))
        spec_noisy = spec_smoothed + noise
    
    return spec_smoothed, spec_noisy, fwhm, sigma

# Generate the two cases
spec_low, spec_low_noisy, fwhm_low, error_low = simulate_observed_spectrum(wl_fine, spectrum_true, R_low, SNR)
spec_high, spec_high_noisy, fwhm_high, error_high = simulate_observed_spectrum(wl_fine, spectrum_true, R_high, SNR)

# ============== DIAGNOSTIC OUTPUT ==============
print(f"\n--- Key Parameters ---")
print(f"Line separation: {line_separation:.4f} μm ({line_separation*1000:.1f} nm)")
print(f"Intrinsic line FWHM: {intrinsic_fwhm:.5f} μm")
print(f"SNR = {SNR}")
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
    (ax1, spec_low, spec_low_noisy, error_low, R_low, fwhm_low, 'Low Spectral Resolution\n(Close features blend together)'),
    (ax2, spec_high, spec_high_noisy, error_high, R_high, fwhm_high, 'High Spectral Resolution\n(Features can be distinguished)')
]

for ax, spec_obs, spec_noisy, error, R, fwhm, title in plot_configs:
    # High-resolution truth (faint)
    ax.plot(wl_fine, spectrum_true, color='#555555', lw=1, alpha=0.55, 
            label='True high-res spectrum')
    
    # Smoothed / observed at this R
    ax.plot(wl_fine, spec_obs, color='#1f77b4', lw=1.5, alpha = 0.05,
            label=f'Smoothed observed at R = {R:.1e}')
    
    # Noisy observed with error bars (decimated to avoid overcrowding)
    if spec_noisy is not None:
        step = max(1, len(wl_fine) // 80)  # sample points for errorbars
        ax.errorbar(wl_fine[::step], spec_noisy[::step], yerr=error[::step], 
                    fmt='o', color='#1f77b4', alpha=0.9, markersize=3, 
                    capsize=1.5, elinewidth=1, label=f'Noisy data (SNR={SNR})')
    
    # Mark true line centers
    ax.axvline(c1, color='#d62728', linestyle='--', alpha=0.65, lw=1.0)
    ax.axvline(c2, color='#d62728', linestyle='--', alpha=0.65, lw=1.0)
    
    # Visual indicator of resolution element size (FWHM)
    y_bar = 0.965
    left = lambda_c - fwhm/2
    right = lambda_c + fwhm/2
    # Horizontal bar
    ax.plot([left, right], [y_bar, y_bar], 
            color='black', lw=0.7, solid_capstyle='round', 
            label=f'Resolution element (FWHM)')
    # Vertical caps at ends
    cap_height = (left - right)*50
    ax.plot([left, left], [y_bar - cap_height/2, y_bar + cap_height/2], 
            color='black', lw=0.7)
    ax.plot([right, right], [y_bar - cap_height/2, y_bar + cap_height/2], 
            color='black', lw=0.7)
    
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
             'Distinguish Between Close Spectral Features (with SNR noise & error bars)',
             fontsize=13, fontweight='bold')

fig.show()
# save_path = '/home/workdir/artifacts/spec_res_demo_with_snr.png'
# plt.savefig(save_path, dpi=170, bbox_inches='tight', facecolor='white')
# print(f"\n✓ Plot saved to: {save_path}")
# print("Open the PNG to visually see the blending effect + noise/error bars at low vs high resolution.")
# print("\nModel complete. Edit the parameters at the top of this script and re-run to explore other cases.")