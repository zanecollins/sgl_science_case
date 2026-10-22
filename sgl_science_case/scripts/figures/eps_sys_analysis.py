"""CH4 vs. peak-scaled hydrocarbons: detection / false-positive results for several systematic (red-noise) levels eps_sys.

A = CH3+C2H4+C2H2+C2H6+C4H2 (each scaled to the CH4 3-4 um peak coefficient), B = A + CH4.
Statistic = logL_B - logL_A with logL = -N/2 ln var(data - template) after mean subtraction
(notebooks/hitran_xsc.ipynb compute_log_likelihood, a = 1); score = sqrt(2 dlogL) for dlogL > 0.

1) single draw: the flux_snr* arrays stored in the pickles (what the notebook heat map shows)
2) many draws: NDRAW fresh noise realisations per (R, SNR), drawn the way 02_generate_spectra.py does it
   (white + eps_sys * median|F| * unit-std 1/f series generated over the full 1-17 um spectrum, then cut to the window;
    the red amplitude does not depend on SNR).

usage: python eps_sys_analysis.py [wl_min wl_max]
"""
import os, sys, pickle, time
import numpy as np, pandas as pd
import colorednoise as cn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.patches as mpatches, seaborn as sns

D = os.path.expanduser('~/orcd/pool/sgl_science_case/spectra_dict_cache_verified_model_eps_sys')
OUT = os.path.expanduser('~/orcd/pool/sgl_science_case/eps_sys_red_noise')
A_STEM = 'CH3+C2H4+C2H2+C2H6+C4H2_peakscaled_fixed_eps'
B_STEM = 'CH4+CH3+C2H4+C2H2+C2H6+C4H2_peakscaled_fixed_eps'
EPS_SYS = ['0', '1e-3', '1e-2']   # photon-only reference first: its error arrays are sigma_w
SW_REF = {}
NDRAW = 1000
SIGMA_THRESH = 10.0
WL0, WL1 = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (3.0, 4.0)
TAG = f'{WL0:g}-{WL1:g}um'
rng = np.random.default_rng(20261012)


def compute_log_likelihood(template, test):
    f = np.asarray(template, float); g = np.asarray(test, float)
    m = np.isfinite(f) & np.isfinite(g); f, g = f[m], g[m]
    f = f - np.mean(f); g = g - np.mean(g); N = f.size
    if np.dot(f, f) <= 0 or N < 2:
        return -np.inf
    sf2 = np.dot(f, f) / N; sg2 = np.dot(g, g) / N; R = np.dot(f, g) / N
    return -0.5 * N * np.log(sf2 - 2.0 * R + sg2)


def score(d):
    return np.sqrt(2 * np.clip(d, 0, None))


def single_draw(dA, dB, dtest):
    rec = []
    for r in sorted(set(dA) & set(dB)):
        wl = np.asarray(dtest[r]['wavelength_grid'], float)
        m = (wl >= WL0) & (wl <= WL1)
        if m.sum() < 10:
            continue
        fa = np.asarray(dA[r]['flux_clean'], float)[m]; fb = np.asarray(dB[r]['flux_clean'], float)[m]
        for key in dtest[r]:
            if not key.startswith('flux_snr'):
                continue
            ns = np.asarray(dtest[r][key], float)[m]
            d = compute_log_likelihood(fb, ns) - compute_log_likelihood(fa, ns)
            rec.append(dict(resolution=r, snr=int(key.split('snr')[-1]), delta_logL=d, sigma=float(score(d)), n_pix=int(m.sum())))
    return pd.DataFrame(rec)


def many_draws(dA, dB, eps):
    rec = []; t0 = time.time()
    for r in sorted(set(dA) & set(dB)):
        wl = np.asarray(dA[r]['wavelength_grid'], float)
        m = (wl >= WL0) & (wl <= WL1); N = int(m.sum()); Nfull = wl.size
        if N < 10:
            continue
        fa = np.asarray(dA[r]['flux_clean'], float)[m]; fb = np.asarray(dB[r]['flux_clean'], float)[m]
        snrs = sorted(int(k.split('snr')[-1]) for k in dA[r] if k.startswith('error_snr'))
        # stored error = sqrt(sigma_w^2 + sigma_r^2); sigma_w is the error array of the eps_sys = 0 run
        if eps == 0:
            SW_REF[r] = {(t, s): np.asarray(d[r][f'error_snr{s}'], float)[m]
                         for t, d in (('A', dA), ('B', dB)) for s in snrs}
        sw = SW_REF[r]
        # sigma_r = eps_sys * median|F| over the full spectrum, as in inject_white_red_noise
        sr = {t: eps * np.nanmedian(np.abs(np.asarray(d[r]['flux_clean'], float))) for t, d in (('A', dA), ('B', dB))}
        null = {s: np.empty(NDRAW) for s in snrs}; det = {s: np.empty(NDRAW) for s in snrs}
        step = max(1, int(2e7 // Nfull)); i = 0
        while i < NDRAW:
            k = min(step, NDRAW - i)
            z = rng.standard_normal((k, N))
            red = 0.0
            if eps > 0:
                red = cn.powerlaw_psd_gaussian(1.0, (k, Nfull), random_state=rng)
                red /= red.std(axis=1, keepdims=True) + 1e-30
                red = red[:, m].copy()
            for s in snrs:
                for truth, key, store in ((fa, 'A', null), (fb, 'B', det)):
                    g = truth + z * sw[(key, s)] + sr[key] * red
                    store[s][i:i + k] = -0.5 * N * (np.log((g - fb).var(axis=1)) - np.log((g - fa).var(axis=1)))
            i += k
        for s in snrs:
            n, d = null[s], det[s]
            rec.append(dict(resolution=r, snr=s, n_pix=N,
                            null_mean=n.mean(), null_std=n.std(ddof=1), null_max_score=score(n).max(),
                            fp_prefers_ch4=(n > 0).mean(), fp_score_ge3=(score(n) >= 3).mean(),
                            fp_score_ge5=(score(n) >= 5).mean(), fp_score_ge10=(score(n) >= 10).mean(),
                            det_mean=d.mean(), det_std=d.std(ddof=1), det_median_score=np.median(score(d)),
                            det_frac_score_ge10=(score(d) >= 10).mean(),
                            det_frac_above_null_q999=(d > np.quantile(n, 0.999)).mean(),
                            z_calibrated=(d.mean() - n.mean()) / np.sqrt(0.5 * (n.var(ddof=1) + d.var(ddof=1)))))
        print(f'  eps_sys={eps} R={r} N={N} done, {time.time() - t0:.0f}s', flush=True)
    return pd.DataFrame(rec)


def heat(df, col, title, fn, label, vmax=None, thresh=None, fmt='{:.1f}'):
    pivot = df.pivot_table(index='snr', columns='resolution', values=col, aggfunc='mean').sort_index(ascending=False)
    annot = pivot.map(lambda v: fmt.format(v) if np.isfinite(v) else '')
    if vmax is None:
        vmax = max(np.nanpercentile(pivot.values, 90), 1.0)
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(pivot, ax=ax, cmap='RdBu_r', center=0, vmin=0.0, vmax=vmax, annot=annot, fmt='',
                linewidths=0.6, linecolor='0.9', cbar_kws={'label': label})
    ax.set_xticklabels([f'{r:.0f}' for r in pivot.columns], rotation=0)
    ax.set_yticklabels([f'{s:g}' for s in pivot.index], rotation=0)
    if thresh is not None:
        first = True
        for j, R in enumerate(pivot.columns):
            ok = pivot[R][pivot[R] >= thresh]
            if ok.empty:
                continue
            i = list(pivot.index).index(ok.index.min())
            ax.add_patch(mpatches.Rectangle((j, i), 1, 1, fill=False, edgecolor='red', linewidth=2.5,
                                            label=f'min SNR with value ≥ {thresh:g}' if first else None))
            first = False
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(h, l, loc='upper left', frameon=True)
    ax.set_xlabel('Resolution (R)'); ax.set_ylabel('SNR per bin'); ax.set_title(title)
    plt.tight_layout(); fig.savefig(f'{OUT}/{fn}', dpi=130); plt.close(fig)
    print('\n' + fn); print(pivot.round(3).to_string())


def planck(wl, T):
    wn = 1e4 / wl
    return 1.191042972e-5 * wn**3 / np.expm1(1.4387769 * wn / T) * np.pi / 2.99792458e10


def forward_plot(dA, dB):
    BLUE, ORANGE, MUTED = '#2a78d6', '#eb6834', '#6b6a63'
    Rs = [r for r in (774, 5994, 46415) if r in dA and r in dB]
    fig, axs = plt.subplots(len(Rs), 2, figsize=(15, 3.6 * len(Rs)), sharex=True, squeeze=False)
    for (ax, ax2), r in zip(axs, Rs):
        w = np.asarray(dB[r]['wavelength_grid'], float); m = (w >= WL0) & (w <= WL1)
        fa = np.asarray(dA[r]['flux_clean'], float)[m]; fb = np.asarray(dB[r]['flux_clean'], float)[m]; bb = planck(w[m], 300.)
        ax.plot(w[m], bb, ls=(0, (4, 3)), c=MUTED, lw=1.1, label='300 K surface blackbody')
        ax.plot(w[m], fa, c=BLUE, lw=1.4, label='scaled hydrocarbons')
        ax.plot(w[m], fb, c=ORANGE, lw=1.0, label='scaled hydrocarbons + CH4')
        ax.set_ylabel('Clean flux (erg s$^{-1}$ cm$^{-2}$)'); ax.set_title(f'R = {r}', loc='left', fontsize=10); ax.grid(alpha=.3)
        ax2.plot(w[m], (fb - fa) / fa, c=ORANGE, lw=1.0)
        ax2.set_ylabel('(with − without) / without'); ax2.set_title(f'R = {r}: CH4 signal', loc='left', fontsize=10); ax2.grid(alpha=.3)
        ax.set_xlim(WL0, WL1)
        rel = (fb - fa) / fa; rel0 = rel - rel.mean()
        print(f'R={r}: CH4 relative signal min {rel.min():.3e} max {rel.max():.3e} rms(mean-sub) {rel0.std():.3e}; '
              f'max F/BB with {np.max(fb / bb):.3f} without {np.max(fa / bb):.3f}')
    axs[0, 0].legend(loc='upper left', frameon=False)
    for a in axs[-1]:
        a.set_xlabel('Wavelength (µm)')
    fig.suptitle('Forward model: hydrocarbons peak-scaled to CH4 (3–4 µm), with and without CH4', x=0.01, ha='left', fontsize=11)
    plt.tight_layout(); fig.savefig(f'{OUT}/forward_model_ch4_{TAG}.png', dpi=130); plt.close(fig)


pd.set_option('display.width', 250)
summary = []
for k, rs in enumerate(EPS_SYS):
    pa, pb = f'{D}/{A_STEM}{rs}.pkl', f'{D}/{B_STEM}{rs}.pkl'
    if not (os.path.exists(pa) and os.path.exists(pb)):
        print(f'missing pickles for eps_sys={rs}, skipping'); continue
    print(f'\n######## eps_sys = {rs} ########', flush=True)
    dA = pickle.load(open(pa, 'rb')); dB = pickle.load(open(pb, 'rb'))
    if k == 0:
        forward_plot(dA, dB)
    eps = float(rs); base = rf'A: scaled hydrocarbons  |  B: + CH4  |  {WL0:g}–{WL1:g} µm, $\epsilon_{{\rm sys}}$ = {rs}'
    det1 = single_draw(dA, dB, dB); fp1 = single_draw(dA, dB, dA)
    det1.to_csv(f'{OUT}/single_draw_detection_eps{rs}_{TAG}.csv', index=False)
    fp1.to_csv(f'{OUT}/single_draw_falsepositive_eps{rs}_{TAG}.csv', index=False)
    heat(det1, 'sigma', f'Log-likelihood detection score, single stored draw\n{base}  |  test: with CH4',
         f'heatmap_single_detection_eps{rs}_{TAG}.png', r'Detection score $\sigma$', thresh=SIGMA_THRESH)
    heat(fp1, 'sigma', f'False-positive check, single stored draw\n{base}  |  test: without CH4',
         f'heatmap_single_falsepositive_eps{rs}_{TAG}.png', r'Detection score $\sigma$', thresh=SIGMA_THRESH)
    md = many_draws(dA, dB, eps); md.insert(0, 'eps_sys', eps)
    md.to_csv(f'{OUT}/many_draws_eps{rs}_{TAG}.csv', index=False); summary.append(md)
    t = f'{NDRAW} noise draws\n{base}'
    heat(md, 'det_median_score', f'Median detection score (truth: with CH4), {t}', f'heatmap_median_score_eps{rs}_{TAG}.png',
         r'Median detection score $\sigma$', thresh=SIGMA_THRESH)
    heat(md, 'z_calibrated', f'Calibrated separation of with/without-CH4 dlogL distributions, {t}',
         f'heatmap_calibrated_z_eps{rs}_{TAG}.png', 'Calibrated z', thresh=5.0)
    heat(md, 'fp_score_ge3', f'False-positive rate: score ≥ 3 when CH4 is absent, {t}', f'heatmap_fp_rate_eps{rs}_{TAG}.png',
         'Fraction of null draws with score ≥ 3', vmax=1.0, fmt='{:.3f}')
    heat(md, 'det_frac_above_null_q999', f'Completeness at 0.1% false-positive threshold, {t}',
         f'heatmap_completeness_eps{rs}_{TAG}.png', 'Fraction of detections above null 99.9th pct', vmax=1.0, thresh=0.95, fmt='{:.2f}')
    del dA, dB
if summary:
    pd.concat(summary).to_csv(f'{OUT}/many_draws_all_{TAG}.csv', index=False)
print('\ndone')
