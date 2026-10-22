"""Final figure: CH4 detection heat map, peak-scaled hydrocarbons, eps_sys = 1e-3, single stored noise draw, 3-4 um,
R <= 46415, SNR <= 100, with SGL SNR-R curves (3-4 um thermal, per tile, deconvolution penalty + corona)."""
import os, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mpatches, seaborn as sns
from scipy.interpolate import interp1d
from matplotlib.colors import TwoSlopeNorm

OUT = os.path.expanduser('~/orcd/pool/sgl_science_case/eps_sys_red_noise')
R_MAX, SNR_MAX, SIGMA_THRESH, WL0, WL1 = 46415, 100, 10.0, 3, 4
df = pd.read_csv(f'{OUT}/single_draw_detection_eps1e-3_3-4um.csv')
df = df[(df.resolution <= R_MAX) & (df.snr <= SNR_MAX)]

# SGL SNR-R curves, per tile, with deconvolution penalty and corona (user's whole-image model, N tiles):
#   n_p = Phi(3-4 um) * G(lam) * A * t / N_tiles      planet photons per tile
#   n_c = corona_flux * A * t                          corona photons per tile pointing (not lensed)
#   SNR after deconvolution = mu * SNR_raw, mu = 0.891 / sqrt(N_tiles)
#   R = (lam / W) * n_p / (S / mu)^2 * n_p / (n_p + n_c)
# Assumes the point-source gain G for the planet light; ignores throughput/QE and detector noise.
from scipy.integrate import quad
h, c, kB, pc, R_earth = 6.62607015e-34, 2.99792458e8, 1.380649e-23, 3.08568e16, 6.371e6
T_PLANET, EMIS, R_G = 255.0, 0.95, 2953.25
LAM, W_BAND = 3.5e-6, (WL1 - WL0) * 1e-6
A_TEL = 1.0                 # m^2
T_TILE = 300.0              # s per tile
N_SIDE = 10
N_TILES = N_SIDE**2
MU = 0.891 / np.sqrt(N_TILES)
CORONA_FLUX = 1e5           # photons m^-2 s^-1 in the band
G_SGL = 4 * np.pi**2 * R_G / LAM
B_phot = lambda l: 2 * c / l**4 / np.expm1(h * c / (l * kB * T_PLANET))   # photons s^-1 m^-2 sr^-1 m^-1
PHI_1PC = EMIS * np.pi * quad(B_phot, WL0 * 1e-6, WL1 * 1e-6)[0] * (R_earth / pc)**2
SGL_CASES = [(10, '-')]
SGL_COLORS = ['#00e5ff', '#ffd60a', '#7cfc00']
S_nrbin = np.logspace(0, 2, 400)
def sgl_R(D_pc):
    n_p = PHI_1PC / D_pc**2 * G_SGL * A_TEL * T_TILE / N_TILES
    n_c = CORONA_FLUX * A_TEL * T_TILE
    return (LAM / W_BAND) * n_p / (S_nrbin / MU)**2 * n_p / (n_p + n_c), n_p, n_c

pivot = df.pivot_table(index='snr', columns='resolution', values='sigma', aggfunc='mean').sort_index(ascending=False)
annot = pivot.map(lambda v: f'{v:.1f}' if np.isfinite(v) else '')
fig, ax = plt.subplots(figsize=(11, 8))
# diverging scale centred on the 10-sigma threshold: red below, blue above, saturating at 100 sigma
norm = TwoSlopeNorm(vmin=0.0, vcenter=SIGMA_THRESH, vmax=100.0)
sns.heatmap(pivot, ax=ax, cmap='RdBu', norm=norm, annot=annot, fmt='', linewidths=0.6, linecolor='0.9',
            cbar_kws={'label': r'Detection significance $\sigma$', 'extend': 'max', 'ticks': [0, 2.5, 5, 7.5, 10, 25, 50, 75, 100], 'format': '%g'})
ax.set_xticklabels([f'{r:.0e}' for r in pivot.columns], rotation=0)
ax.set_yticklabels([f'{s:g}' for s in pivot.index], rotation=0)

# columns/rows are log-spaced, so map (R, SNR) to cell index by interpolating in log space
x_from_R = interp1d(np.log10(np.asarray(pivot.columns, float)), np.arange(pivot.shape[1], dtype=float), bounds_error=False, fill_value=np.nan)
y_from_snr = interp1d(np.log10(np.asarray(pivot.index, float)), np.arange(pivot.shape[0], dtype=float), bounds_error=False, fill_value=np.nan)
for (D_pc, ls), col in zip(SGL_CASES, SGL_COLORS):
    R_sgl, n_p, n_c = sgl_R(D_pc)
    x, y = x_from_R(np.log10(R_sgl)) + 0.5, y_from_snr(np.log10(S_nrbin)) + 0.5
    m = np.isfinite(x) & np.isfinite(y)
    print(f'D = {D_pc} pc: n_p = {n_p:.3g}, n_c = {n_c:.3g} per tile, R(SNR=10) = {np.interp(10, S_nrbin, R_sgl):.3g}, {m.sum()} points in grid')
    if m.any():
        ax.plot(x[m], y[m], color=col, ls=ls, lw=2.5, zorder=4, label=f'SGL-accessible properties \nfor exo-earth at {D_pc} pc')

# detection-significance contours on cell centres
cs = ax.contour(np.arange(pivot.shape[1]) + 0.5, np.arange(pivot.shape[0]) + 0.5, pivot.values,
                levels=[5, 10], colors='k', linewidths=[1.6, 2.0], linestyles=['dashed', 'solid'], zorder=3)
for lev, ls, lw in ((5, 'dashed', 1.6), (10, 'solid', 2.0)):
    ax.plot([], [], c='k', ls=ls, lw=lw, label=rf'{lev}$\sigma$ contour')

first = True
for j, Rv in enumerate(pivot.columns):
    ok = pivot[Rv][pivot[Rv] >= SIGMA_THRESH]
    if ok.empty:
        continue
    i = list(pivot.index).index(ok.index.min())
    ax.add_patch(mpatches.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', linewidth=2.5,
                                    label=rf'min SNR for $\sigma\geq{SIGMA_THRESH:.0f}$' if first else None))
    first = False
# pinned to the top-right corner of the heat map
ax.legend(loc='upper right', bbox_to_anchor=(pivot.shape[1] - 0.04, 0.04), bbox_transform=ax.transData, ncol=1,
          frameon=True, framealpha=0.92, edgecolor='0.6', fontsize=8.5, labelspacing=0.25, borderpad=0.35)
ax.set_xlabel('Spectral Resolution'); ax.set_ylabel('SNR per bin')
ax.set_title(f'Log-Likelihood Detection Score ({WL0}–{WL1} µm)\n'
             r'CH4 with peak-scaled hydrocarbons, $\epsilon_{\rm sys}=10^{-3}$, single noise draw')
plt.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(f'{OUT}/final_heatmap_ch4_scaled_eps1e-3_3-4um.{ext}', dpi=200)
print(pivot.round(1).to_string())
