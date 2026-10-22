"""Forward model + stored noisy draw at R = 774, SNR = 46, eps_sys = 1e-3, 3-4 um, with the detection score recomputed."""
import os, pickle, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
D = os.path.expanduser('~/orcd/pool/sgl_science_case/spectra_dict_cache_verified_model_eps_sys')
OUT = os.path.expanduser('~/orcd/pool/sgl_science_case/eps_sys_red_noise')
R, SNR, WL0, WL1 = 774, 46, 3.0, 4.0
dA = pickle.load(open(f'{D}/CH3+C2H4+C2H2+C2H6+C4H2_peakscaled_fixed_eps1e-3.pkl', 'rb'))[R]
dB = pickle.load(open(f'{D}/CH4+CH3+C2H4+C2H2+C2H6+C4H2_peakscaled_fixed_eps1e-3.pkl', 'rb'))[R]
print('keys:', [k for k in dB if not k.startswith(('flux_snr', 'error_snr'))])
wl = np.asarray(dB['wavelength_grid'], float); m = (wl >= WL0) & (wl <= WL1); w = wl[m]
fa = np.asarray(dA['flux_clean'], float)[m]; fb = np.asarray(dB['flux_clean'], float)[m]
obs = np.asarray(dB[f'flux_snr{SNR}'], float)[m]; err = np.asarray(dB[f'error_snr{SNR}'], float)[m]

def logL(t, g):  # notebook compute_log_likelihood, a = 1
    f = t - t.mean(); g = g - g.mean(); N = f.size
    return -0.5 * N * np.log(np.var(g - f)), g - f
lA, rA = logL(fa, obs); lB, rB = logL(fb, obs); d = lB - lA
sig = np.sqrt(2 * max(d, 0))
print(f'N pix = {m.sum()}  logL_A = {lA:.3f}  logL_B = {lB:.3f}  dlogL = {d:.3f}  sigma = {sig:.2f}')
print(f'rms resid vs A {rA.std():.3e}  vs B {rB.std():.3e}  median err {np.median(err):.3e}  '
      f'rms CH4 signal (mean-sub) {np.std((fb - fb.mean()) - (fa - fa.mean())):.3e}')

BLUE, ORANGE, INK = '#2a78d6', '#eb6834', '#3d3d3a'
fig, (a0, a1, a2) = plt.subplots(3, 1, figsize=(11, 10), sharex=True, gridspec_kw={'height_ratios': [2.2, 1, 1.2]})
a0.errorbar(w, obs, yerr=err, fmt='o', ms=3, c=INK, ecolor='0.65', elinewidth=0.8, label=f'noisy draw (with CH4), SNR = {SNR}', zorder=1)
a0.plot(w, fa, c=BLUE, lw=1.6, label='template A: hydrocarbons, no CH4', zorder=3)
a0.plot(w, fb, c=ORANGE, lw=1.6, label='template B: hydrocarbons + CH4', zorder=3)
a0.set_ylabel('Flux (erg s$^{-1}$ cm$^{-2}$)'); a0.legend(frameon=False, loc='best')
a0.set_title(f'R = {R}, SNR = {SNR}, $\\epsilon_{{\\rm sys}}=10^{{-3}}$, {WL0:g}–{WL1:g} µm, {m.sum()} pixels:  '
             f'$\\Delta\\log L$ = {d:.1f}  →  $\\sigma$ = {sig:.1f}', loc='left', fontsize=11)
a1.plot(w, (fb - fb.mean()) - (fa - fa.mean()), c=ORANGE, lw=1.4, label='CH4 signal: B − A (mean-subtracted)')
a1.fill_between(w, -err, err, color='0.85', label='±1σ per-pixel noise')
a1.axhline(0, c='0.5', lw=0.8); a1.set_ylabel('Flux difference'); a1.legend(frameon=False, loc='best')
a2.plot(w, rA, 'o-', ms=3, lw=0.8, c=BLUE, label=f'data − A (rms {rA.std():.2e})')
a2.plot(w, rB, 'o-', ms=3, lw=0.8, c=ORANGE, label=f'data − B (rms {rB.std():.2e})')
a2.axhline(0, c='0.5', lw=0.8); a2.set_ylabel('Residual (mean-sub)'); a2.set_xlabel('Wavelength (µm)')
a2.legend(frameon=False, loc='best'); a2.set_xlim(WL0, WL1)
for a in (a0, a1, a2): a.grid(alpha=.3)
plt.tight_layout()
for ext in ('png', 'pdf'): fig.savefig(f'{OUT}/verify_forward_model_R{R}_snr{SNR}_eps1e-3.{ext}', dpi=200)
