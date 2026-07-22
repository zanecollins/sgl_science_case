import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from adjustText import adjust_text

# ============== USER PARAMETERS ==============
np.random.seed(42)   # Reproducible random lines for demo

wl_min = 0.95
wl_max = 1.05
dlambda_fine = 0.0000001
wl = np.arange(wl_min, wl_max, dlambda_fine)

N_lines_A = 80
N_lines_B = 88
R_low = 100000
R_high = 5000000
intrinsic_fwhm = 0.000001   # narrow intrinsic lines
line_depth_range = (0.18, 0.48)

# ============================================

print("Generating two mock spectra with many random lines to demonstrate spectral degeneracy...")

# Generate line positions for Spectrum A (e.g. "Molecule X")
pos_A = np.sort(np.random.uniform(wl_min + 0.008, wl_max - 0.008, N_lines_A))

# Generate for Spectrum B (e.g. "Molecule Y")
pos_B = np.sort(np.random.uniform(wl_min + 0.008, wl_max - 0.008, N_lines_B))

# Deliberately place some B lines close to A lines (within ~0.7 nm)
# This creates potential for blending degeneracy at low resolution
num_close = min(28, N_lines_A)
for i in range(num_close):
    pos_B[i] = pos_A[i] + np.random.uniform(-0.00001, 0.000001)
pos_B = np.sort(pos_B)

depths_A = np.random.uniform(*line_depth_range, N_lines_A)
depths_B = np.random.uniform(*line_depth_range, N_lines_B)

intrinsic_sigma = intrinsic_fwhm / (2 * np.sqrt(2 * np.log(2)))

def build_mock_spectrum(wl, positions, depths, sigma):
    """Build absorption spectrum from many Gaussian lines."""
    spec = np.ones_like(wl, dtype=float)
    for p, d in zip(positions, depths):
        spec -= d * np.exp(-0.5 * ((wl - p) / sigma)**2)
    return np.clip(spec, 0.0, 1.0)

spec_A_true = build_mock_spectrum(wl, pos_A, depths_A, intrinsic_sigma)
spec_B_true = build_mock_spectrum(wl, pos_B, depths_B, intrinsic_sigma)

# Combined spectrum: includes lines from BOTH A and B (superposition / multiple molecules)
all_pos = np.concatenate([pos_A, pos_B])
all_depths = np.concatenate([depths_A, depths_B])
spec_combined_true = build_mock_spectrum(wl, all_pos, all_depths, intrinsic_sigma)

def simulate_resolution(wl, spec_true, R, lambda_ref=1.0):
    """Convolve with Gaussian LSF of width lambda/R."""
    fwhm = lambda_ref / float(R)
    sigma_wl = fwhm / (2 * np.sqrt(2 * np.log(2)))
    dlam = wl[1] - wl[0]
    sigma_pix = sigma_wl / dlam
    spec_smoothed = gaussian_filter1d(spec_true, sigma_pix, mode='nearest')
    return spec_smoothed, fwhm

spec_A_low, fwhm_low = simulate_resolution(wl, spec_A_true, R_low)
spec_B_low, _ = simulate_resolution(wl, spec_B_true, R_low)
spec_combined_low, _ = simulate_resolution(wl, spec_combined_true, R_low)
spec_A_high, fwhm_high = simulate_resolution(wl, spec_A_true, R_high)
spec_B_high, _ = simulate_resolution(wl, spec_B_true, R_high)
spec_combined_high, _ = simulate_resolution(wl, spec_combined_true, R_high)

# Simple quantitative measure of similarity (degeneracy)
corr_A_B_low = np.corrcoef(spec_A_low, spec_B_low)[0, 1]
corr_A_B_high = np.corrcoef(spec_A_high, spec_B_high)[0, 1]
corr_combined_A_low = np.corrcoef(spec_combined_low, spec_A_low)[0, 1]
corr_combined_B_low = np.corrcoef(spec_combined_low, spec_B_low)[0, 1]

print(f"\n--- Mock Spectra Summary ---")
print(f"Lines in Spectrum A: {N_lines_A}")
print(f"Lines in Spectrum B: {N_lines_B}")
print(f"Combined spectrum includes lines from BOTH (total ~{len(all_pos)} lines, some close).")
print(f"Some B lines deliberately placed close to A lines (within ~0.7 nm) for demo.")
print(f"\nAt R={R_low} (low resolution):")
print(f"  Resolution element FWHM ≈ {fwhm_low:.4f} μm")
print(f"  Corr A_low vs B_low: {corr_A_B_low:.3f}")
print(f"  Corr combined_low vs A_low: {corr_combined_A_low:.3f}")
print(f"  Corr combined_low vs B_low: {corr_combined_B_low:.3f}  (high similarity → combined looks like it could be single-molecule)")
print(f"\nAt R={R_high} (high resolution):")
print(f"  Resolution element FWHM ≈ {fwhm_high:.4f} μm")
print(f"  Corr A_high vs B_high: {corr_A_B_high:.3f} (much lower → distinguishable patterns)")

# ============== PLOTTING ==============
fig, (ax_high, ax_low) = plt.subplots(2, 1, figsize=(11, 9), sharex=True, gridspec_kw={'hspace': 0.25}, constrained_layout=True)

# High-resolution panel: focus on COMBINED (both molecules) vs pures for contrast
ax_high.plot(wl, spec_combined_high, color='#2ca02c', lw=1, alpha=0.5, label='Combined (A + B lines)')
ax_high.plot(wl, spec_A_high, color='#1f77b4',ls = '--', lw=1, alpha=0.7, label='Pure A (faded)')
ax_high.plot(wl, spec_B_high, color='#ff7f0e', ls = '--', lw=1, alpha=0.7, label='Pure B (faded)')
ax_high.set_ylabel('Normalized Flux')
ax_high.set_title(f'High Spectral Resolution\n(Dense overlapping lines from both "molecules" visible → clear multi-component nature) R = {R_high}', fontsize=11, pad=6)
ax_high.legend(loc='upper right', fontsize=8, framealpha=0.95)
ax_high.grid(True, alpha=0.3, linestyle=':')
ax_high.set_ylim(0.40, 1.08)
# ax_high.text(0.02, 0.97, 
#              'High-res combined shows superposition of both line sets.\n'
#              'You can clearly see contributions from multiple species.',
#              transform=ax_high.transAxes, fontsize=9, verticalalignment='top',
#              bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.85))

# Low-resolution panel: COMBINED at low R vs pure A and pure B at low R
# This shows the indistinguishability / degeneracy
ax_low.plot(wl, spec_combined_low, color='#2ca02c', lw=1, label='Combined (A+B) at low R')
ax_low.plot(wl, spec_A_low, color='#1f77b4', ls = '--', lw=1, alpha=0.7, label='Pure A at low R')
ax_low.plot(wl, spec_B_low, color='#ff7f0e', ls = '--', lw=1, alpha=0.7, label='Pure B at low R')
ax_low.set_ylabel('Normalized Flux')
ax_low.set_xlabel('Wavelength (μm)')
ax_low.set_title(f'Low Spectral Resolution\n(Combined spectrum blends into shapes very similar to single-molecule cases → degeneracy) R = {R_low}', fontsize=11, pad=6)
ax_low.legend(loc='upper right', fontsize=8, framealpha=0.95)
ax_low.grid(True, alpha=0.3, linestyle=':')
ax_low.set_ylim(0.40, 1.08)

# Explanatory text on low-res panel
# ax_low.text(0.02, 0.97, 
#             f'At low R the combined (realistic multi-molecule) spectrum\n'
#             f'produces broad blended features that closely resemble what\n'
#             f'a pure single-molecule spectrum would look like.\n'
#             f'Corr(combined_low, A_low) = {corr_combined_A_low:.2f} | '
#             f'Corr(combined_low, B_low) = {corr_combined_B_low:.2f}\n'
#             f'→ Easy to misattribute the observed spectrum to just one molecule.',
#             transform=ax_low.transAxes, fontsize=9, verticalalignment='top',
#             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))

fig.suptitle('Mock Spectra with Combined (A + B): Demonstrating Indistinguishability at Low Resolution\n'
             '(When close lines from multiple molecules blend, the observed spectrum can be attributed to the wrong single species)',
             fontsize=12, fontweight='bold', y=1.2)

plt.show()
# save_path = '/home/workdir/artifacts/spectral_degeneracy_demo.png'
# plt.savefig(save_path, dpi=160, bbox_inches='tight', facecolor='white')
# print(f"\n✓ Plot saved to: {save_path}")
print("Open the PNG to see how low resolution creates degeneracy / indistinguishability for combined spectra.")
print("\nYou can change N_lines, R values, or the closeness offset in the code to explore further.")