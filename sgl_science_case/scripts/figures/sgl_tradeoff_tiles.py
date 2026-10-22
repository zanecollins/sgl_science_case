"""Spectral resolution vs SNR vs spatial resolution (number of image tiles) for the SGL, 3-4 um thermal.
Same model and parameters as the SGL curve in final_heatmap_ch4.py:
  n_p = Phi(3-4 um) * G(lam) * A * t / N      planet photons per tile (point-source gain)
  n_c = corona_flux * A * t                   corona photons per tile pointing
  mu  = 0.891 / sqrt(N)                       deconvolution penalty on SNR
  R   = (lam / W) * n_p / (S / mu)^2 * n_p / (n_p + n_c)"""
import os, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad

OUT = os.path.expanduser('~/orcd/pool/sgl_science_case/eps_sys_red_noise')
WL0, WL1 = 3, 4
h, c, kB, pc, R_earth = 6.62607015e-34, 2.99792458e8, 1.380649e-23, 3.08568e16, 6.371e6
T_PLANET, EMIS, R_G = 255.0, 0.95, 2953.25
LAM, W_BAND = 3.5e-6, (WL1 - WL0) * 1e-6
A_TEL, T_TILE, CORONA_FLUX, D_PC = 1.0, 300.0, 1e5, 10.0
G_SGL = 4 * np.pi**2 * R_G / LAM
B_phot = lambda l: 2 * c / l**4 / np.expm1(h * c / (l * kB * T_PLANET))
PHI = EMIS * np.pi * quad(B_phot, WL0 * 1e-6, WL1 * 1e-6)[0] * (R_earth / (D_PC * pc))**2
S = np.logspace(0, 3, 400)

def sgl_R(n_side):
    N = n_side**2
    n_p = PHI * G_SGL * A_TEL * T_TILE / N
    n_c = CORONA_FLUX * A_TEL * T_TILE
    mu = 0.891 / np.sqrt(N)
    return (LAM / W_BAND) * n_p / (S / mu)**2 * n_p / (n_p + n_c), n_p, n_c

fig, ax = plt.subplots(figsize=(10, 7))

N_SIDES = [1, 3, 10, 20, 30]   # 100x100 gives R < 10 at SNR >= 1 (off the plot)
cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(N_SIDES)))
for n_side, col in zip(N_SIDES, cols):
    R, n_p, n_c = sgl_R(n_side)
    lab = f'{n_side}×{n_side} tiles' + (' (disk-integrated)' if n_side == 1 else '')
    ax.plot(R, S, c=col, lw=2.4, zorder=4, label=lab)
    print(f'{n_side}x{n_side}: n_p/tile = {n_p:.3g}, n_c = {n_c:.3g}, R(SNR=10) = {np.interp(10, S, R):.3g}')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(10, 1e9); ax.set_ylim(1, 1e3)
ax.set_xlabel('Spectral resolution $R$'); ax.set_ylabel('SNR per bin (after deconvolution)')
ax.grid(True, which='major', alpha=0.3)
ax.set_title(f'SGL trade-off: spectral resolution, SNR and spatial resolution ({WL0}–{WL1} µm)\n'
             rf'Earth-like 255 K at {D_PC:g} pc, $A$ = {A_TEL:g} m$^2$, {T_TILE:g} s/tile, '
             rf'$\mu$ = 0.891/$\sqrt{{N}}$, corona {CORONA_FLUX:.0e} ph m$^{{-2}}$ s$^{{-1}}$', fontsize=11)
ax.legend(loc='upper right', frameon=True, framealpha=0.95, edgecolor='0.6', fontsize=9)
plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'{OUT}/sgl_tradeoff_tiles_3-4um.{ext}', dpi=200)
