#!/usr/bin/env python3
"""
exo_spec_sim.py  (v4 - box model, wavelength-native, bin_spectrum)

Simple high-resolution exoplanet spectral simulation framework.

Setup:
  - "Atmosphere in a box": a single uniform slab of gas at fixed T, P, with a
    given column density. Beer-Lambert through the box.
  - Arbitrary light source.
  - HAPI/HITRAN cross-sections (CO2 first), renormalized to per-molecule.
  - Everything in WAVELENGTH (microns) immediately after HAPI.
  - We use HITRAN's native grid (pinned via OmegaRange so isotopologues match),
    convert to wavelength, and reduce resolution with bin_spectrum.
  - Gaussian noise only.
  - Cross-correlation to test when isotopologues are distinguishable.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

from hapi import (
    db_begin, fetch, absorptionCoefficient_Voigt, ISO, ISO_INDEX, tableList,
)
from bin_spec import bin_spectrum

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATA_DIR = "hitran_data"

MOL_CO2 = 2
CO2_ISOTOPOLOGUES = {
    "12C16O2": 1,    
    "13C16O2": 2,    
    # "16O12C18O": 3,  
}

MOL_CH4 = 6
CH4_ISOTOPOLOGUES = {
    "12CH4": 1,    
    "13CH4": 2,    
}


# Wavelength window (microns). 3.3 um CH4 band.
# WAV_MIN = 3.0   # um  
# WAV_MAX = 4.0   # um  
# NATIVE_WSTEP = 0.001     # cm^-1, native HITRAN-resolution sampling (HAPI side)


# Wavelength window (microns). 4.3 um CO2 band.
WAV_MIN = 3.0 #4.17   # um  (<-> ~2398 cm^-1)
WAV_MAX = 5.0 #4.45   # um  (<-> ~2198 cm^-1)
NATIVE_WSTEP = 0.001     # cm^-1, native HITRAN-resolution sampling (HAPI side)

# HAPI works in wavenumber; derive the matching window once.
WAVENUMBER_MIN = 1e4 / WAV_MAX   # cm^-1
WAVENUMBER_MAX = 1e4 / WAV_MIN


# -----------------------------------------------------------------------------
# 1. HITRAN data retrieval (fetch only if not already cached)
# -----------------------------------------------------------------------------
def init_hapi(data_dir=DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)
    db_begin(data_dir)


def _table_is_cached(table_name, data_dir=DATA_DIR):
    """Cached if both .header and .data exist AND table is registered in HAPI."""
    header = os.path.join(data_dir, f"{table_name}.header")
    data = os.path.join(data_dir, f"{table_name}.data")
    files_exist = os.path.isfile(header) and os.path.isfile(data)
    return files_exist and (table_name in tableList())


def fetch_co2_isotopologue(iso_name, nu_min=WAVENUMBER_MIN, nu_max=WAVENUMBER_MAX,
                           data_dir=DATA_DIR, force=False):
    """Fetch line data for a CO2 isotopologue, only if not already cached."""
    iso_local_id = CO2_ISOTOPOLOGUES[iso_name]
    table_name = f"CO2_{iso_name}"

    if not force and _table_is_cached(table_name, data_dir):
        print(f"[fetch] {iso_name}: using cached table '{table_name}' "
              f"(skipping download)")
        return table_name

    global_iso_id = ISO[(MOL_CO2, iso_local_id)][0]
    print(f"[fetch] {iso_name}: downloading table={table_name}, "
          f"global_iso_id={global_iso_id}, range=[{nu_min:.2f}, {nu_max:.2f}] cm^-1")
    fetch(table_name, MOL_CO2, iso_local_id, nu_min, nu_max)
    return table_name


def fetch_ch4_isotopologue(iso_name, nu_min=WAVENUMBER_MIN, nu_max=WAVENUMBER_MAX,
                           data_dir=DATA_DIR, force=False):
    """Fetch line data for a CH4 isotopologue, only if not already cached."""
    iso_local_id = CH4_ISOTOPOLOGUES[iso_name]
    table_name = f"CH4_{iso_name}"

    if not force and _table_is_cached(table_name, data_dir):
        print(f"[fetch] {iso_name}: using cached table '{table_name}' "
              f"(skipping download)")
        return table_name

    global_iso_id = ISO[(MOL_CH4, iso_local_id)][0]
    print(f"[fetch] {iso_name}: downloading table={table_name}, "
          f"global_iso_id={global_iso_id}, range=[{nu_min:.2f}, {nu_max:.2f}] cm^-1")
    fetch(table_name, MOL_CH4, iso_local_id, nu_min, nu_max)
    return table_name


# -----------------------------------------------------------------------------
# Abundance + cross-section
# -----------------------------------------------------------------------------
def terrestrial_abundance(iso_name):
    """Terrestrial isotopic abundance fraction used to weight HITRAN intensities."""
    if iso_name not in CO2_ISOTOPOLOGUES and iso_name not in CH4_ISOTOPOLOGUES:
        raise ValueError(f"Unknown isotopologue: {iso_name}")
    
    if iso_name in CO2_ISOTOPOLOGUES:
        iso_local_id = CO2_ISOTOPOLOGUES[iso_name]
        return ISO[(MOL_CO2, iso_local_id)][ISO_INDEX['abundance']]
    elif iso_name in CH4_ISOTOPOLOGUES:
        iso_local_id = CH4_ISOTOPOLOGUES[iso_name]
        return ISO[(MOL_CH4, iso_local_id)][ISO_INDEX['abundance']]


def compute_cross_section(table_name, T, P_atm, iso_name,
                          wstep=NATIVE_WSTEP, renormalize=True):
    """
    Voigt cross-section at temperature T (K), pressure P (atm), on HITRAN's
    native grid, converted to ASCENDING WAVELENGTH (microns).

    OmegaRange is pinned to the configured window so every isotopologue is
    evaluated on the SAME native grid (same start, same step, same length) ->
    no interpolation needed to combine them.

    HITRAN intensities are weighted by terrestrial abundance; with
    renormalize=True we divide it out to recover the per-molecule cross-section.

    Returns
    -------
    wav  : wavelength grid (um), ascending
    xsec : cross-section (cm^2/molecule)
    """
    nu, xsec = absorptionCoefficient_Voigt(
        SourceTables=table_name,
        Environment={"T": T, "p": P_atm},
        WavenumberStep=wstep,
        OmegaRange=[WAVENUMBER_MIN, WAVENUMBER_MAX],
        HITRAN_units=True,
    )

    if renormalize:
        xsec = xsec / terrestrial_abundance(iso_name)

    # wavenumber (cm^-1) -> wavelength (um), ascending
    wav = 1e4 / nu
    order = np.argsort(wav)
    return wav[order], xsec[order]


# -----------------------------------------------------------------------------
# 2. Light source (arbitrary)
# -----------------------------------------------------------------------------
def light_source(wav, kind="flat", level=1.0, slope=0.0):
    """Arbitrary incident light source I0(wav). 'flat' or linear 'slope'."""
    if kind == "flat":
        return np.full_like(wav, level)
    elif kind == "slope":
        x = (wav - wav.min()) / (wav.max() - wav.min())
        return level * (1.0 + slope * (x - 0.5))
    else:
        raise ValueError(f"unknown source kind: {kind}")


# -----------------------------------------------------------------------------
# 3. Box radiative transfer (Beer-Lambert through a uniform slab)
# -----------------------------------------------------------------------------
def box_transmission(xsec, column_density):
    """transmission = exp(-N * xsec). column_density in molec/cm^2."""
    return np.exp(-column_density * xsec)


def observed_spectrum(wav, xsec, column_density, source_kind="flat",
                      source_level=1.0, source_slope=0.0):
    """Light source * box transmission = observed spectrum (pre-noise)."""
    I0 = light_source(wav, kind=source_kind, level=source_level, slope=source_slope)
    return I0 * box_transmission(xsec, column_density), I0


def check_saturation(xsec, column_density, label="", warn_tau=3.0):
    """Report peak optical depth; warn if saturating (aim peak tau ~ 0.1-1)."""
    peak_tau = column_density * xsec.max()
    print(f"[saturation] {label}: peak tau = {peak_tau:.3f}, "
          f"trans floor = {np.exp(-peak_tau):.4f}")
    if peak_tau > warn_tau:
        print(f"  WARNING: peak tau > {warn_tau:.0f} -> saturating. Reduce N.")
    return peak_tau


def suggest_column_density(xsec, target_peak_tau=0.1):
    """Column density (molec/cm^2) putting the strongest line at target peak tau."""
    N = target_peak_tau / xsec.max()
    print(f"[suggest] N for peak tau={target_peak_tau}: {N:.3e} molec/cm^2")
    return N


# -----------------------------------------------------------------------------
# 4. Resolution reduction with bin_spectrum
# -----------------------------------------------------------------------------
def resample_to_R(wav, spectrum, R):
    """Reduce a native (HITRAN) spectrum on ascending wavelength grid to R."""
    new_wav, new_spec, _ = bin_spectrum(wav, spectrum, R, err_data=[])
    return new_wav, new_spec


# -----------------------------------------------------------------------------
# 5. Noise (Gaussian only)
# -----------------------------------------------------------------------------
def claude_add_gaussian_noise(spectrum, snr, rng=None):

    if rng is None:
        rng = np.random.default_rng()
    
    level = np.nanmedian(spectrum)
    
    sigma = level / snr

    return spectrum + rng.normal(0.0, sigma, size=spectrum.shape), sigma

def add_gaussian_noise(data, SNR, seed = 8):
    depth = 1.0 - data          # or np.abs(1.0 - data) if baseline isn't exactly 1
    signal_level = np.max(depth)   # Peak depth
    # signal_level = np.sqrt(np.mean(depth**2))   # RMS depth
    
    sigma = signal_level / SNR
    # print(f"Signal level: {signal_level:.4e}, Noise std dev: {sigma:.4e} for SNR={SNR}")

    nchan = len(data)
    rng = np.random.default_rng(seed)

    random_pertubations = sigma*rng.standard_normal(nchan)
    # print(f"Actual noise std: {random_pertubations.std():.4e} | Max |noise|: {np.abs(random_pertubations).max():.4e}")

    synthetic_atmosphere = data + random_pertubations 
    synthetic_atmosphere = np.clip(synthetic_atmosphere, 0, None) 
    
    errorbars = np.full_like(data, sigma) 
    
    return synthetic_atmosphere, sigma, errorbars


# -----------------------------------------------------------------------------
# 6. Cross-correlation comparison
# -----------------------------------------------------------------------------
# def claude_cross_correlate(template, data):
#     """Normalized CCF of continuum-removed template and data."""
#     good = np.isfinite(template) & np.isfinite(data)
#     t = template[good] - np.nanmean(template[good])
#     d = data[good] - np.nanmean(data[good])
#     norm = np.sqrt(np.sum(t**2) * np.sum(d**2))
#     if norm == 0:
#         return np.arange(-len(t) + 1, len(t)), np.zeros(2 * len(t) - 1)
#     ccf = np.correlate(d, t, mode="full") / norm
#     lags = np.arange(-len(t) + 1, len(t))
#     return lags, ccf

def cross_correlate(template, data):
    """Compute the cross-correlation function (CCF) of a template and data."""
    good = np.isfinite(template) & np.isfinite(data)
    t = template[good] - np.nanmean(template[good])
    d = data[good] - np.nanmean(data[good])

    template_norm = (t - np.mean(t)) / np.std(t) # Zero Mean Cross Correlation
    test_norm   = (d - np.mean(d)) / np.std(d)
    
    # Compute the cross-correlation using numpy's correlate function
    ccf = np.correlate(test_norm, template_norm, mode='full')

    # Normalize the CCF
    norm = np.sqrt(np.sum(template_norm**2) * np.sum(test_norm**2))
    if norm == 0:
        return np.arange(-len(t) + 1, len(t)), np.zeros(2 * len(t) - 1)
    ccf /= norm  # Normalize to max correlation of 1

    # Create an array of lag values corresponding to the CCF
    lags = np.arange(-len(t) + 1, len(t))

    return lags, ccf


def claude_detection_significance(ccf):
    peak_idx = np.argmax(np.abs(ccf))
    peak = ccf[peak_idx]
    mask = np.ones_like(ccf, dtype=bool)
    win = max(1, len(ccf) // 100)
    mask[max(0, peak_idx - win): min(len(ccf), peak_idx + win + 1)] = False
    noise = np.std(ccf[mask])
    return peak, (peak / noise if noise > 0 else np.inf)


def detection_significance(ccf):
    """
    Estimate the detection significance of a cross-correlation function (CCF).

    Parameters
    ----------
    ccf : array
        The cross-correlation function.

    Returns
    -------
    peak : float
        The peak value of the CCF.
    snr : float
        The signal-to-noise ratio (SNR) of the peak, calculated as the peak
        divided by the standard deviation of the CCF excluding a window around
        the peak.
    """

    max_corr = (ccf[len(ccf)//2])

    # SNR and resolution results
    cc_snr = max_corr / np.std(ccf) if np.std(ccf) > 0 else 0
    cc_uncertainty = np.std(ccf)

    # peak_idx = np.argmax(np.abs(ccf))
    # peak = ccf[peak_idx]

    # # Exclude a window around the peak to estimate noise
    # mask = np.ones_like(ccf, dtype=bool)
    # win = max(1, len(ccf) // 50)  # Window size is 1/50th of the CCF length
    # mask[max(0, peak_idx - win): min(len(ccf), peak_idx + win + 1)] = False

    # noise = np.std(ccf[mask])
    # snr = peak / noise if noise > 0 else np.inf

    # return peak, snr

    return max_corr, cc_snr #, cc_uncertainty


# -----------------------------------------------------------------------------
# TEST / SANITY-CHECK FUNCTIONS
# -----------------------------------------------------------------------------
def test_fetch_and_xsec(iso_name="12C16O2", T=300.0, P_atm=0.1):
    """Fetch one isotopologue and plot its cross-section vs wavelength."""
    init_hapi()
    if iso_name == "12C16O2":
        table = fetch_co2_isotopologue(iso_name)
    elif iso_name == "12CH4":
        table = fetch_ch4_isotopologue(iso_name)
    else:
        raise ValueError(f"Unknown isotopologue: {iso_name}")

    wav, xsec = compute_cross_section(table, T, P_atm, iso_name)

    print(f"[test_fetch_and_xsec] {iso_name}  (T={T} K, P={P_atm} atm)")
    print(f"  n points : {len(wav)}")
    print(f"  wav range: [{wav.min():.4f}, {wav.max():.4f}] um")
    print(f"  max xsec : {xsec.max():.3e} cm^2/molec at "
          f"{wav[np.argmax(xsec)]:.4f} um")
    plt.figure()
    plt.plot(wav, xsec, lw=0.5)
    plt.xlabel(r"Wavelength ($\mu$m)")
    plt.ylabel(r"Cross-section (cm$^2$/molec)")
    plt.title(f"{iso_name} cross-section")
    plt.tight_layout(); plt.show()
    return wav, xsec


def test_box_spectrum(iso_name="12C16O2", T=300.0, P_atm=0.1,
                      target_peak_tau=0.1, source_kind="slope", source_slope=0.0):
    """Build the observed box spectrum; show native vs binned (R=100)."""
    init_hapi()
    if iso_name == "12C16O2":
        table = fetch_co2_isotopologue(iso_name)
    elif iso_name == "12CH4":
        table = fetch_ch4_isotopologue(iso_name)

    wav, xsec = compute_cross_section(table, T, P_atm, iso_name)

    N = suggest_column_density(xsec, target_peak_tau)
    check_saturation(xsec, N, label=iso_name)

    obs, I0 = observed_spectrum(wav, xsec, N,
                                source_kind=source_kind, source_slope=source_slope)
    print(f"[test_box_spectrum] {iso_name}: min/max obs = "
          f"{obs.min():.4f} / {obs.max():.4f}")

    plt.figure(figsize=(9, 5))
    
    # plt.plot(wav, obs, lw=0.4, color="k", label="native HITRAN res")
    
    for R in (2000, 1000, 500, 250, 125):
        wav_R, obs_R = resample_to_R(wav, obs, R)
        plt.plot(wav_R, obs_R, lw=0.9, label=f"bin R={R}")
        print(f"  R={R}: {len(wav_R)} points")
    
    plt.xlabel(r"Wavelength ($\mu$m)"); plt.ylabel("Observed flux (arb.)")
    plt.title(f"{iso_name} box spectrum (source={source_kind})")
    plt.legend(); plt.tight_layout(); plt.show()
    
    return wav, obs


def test_isotopologue_5050(iso_name="12C16O2", iso_minor="13C16O2", T=300.0, P_atm=0.1, target_peak_tau=0.1):
    """
    Sanity-check plot: 12CO2 alone, 13CO2 alone, and a 50/50 combined slab.

    Uses renormalized (per-molecule) cross-sections so the 50/50 abundance
    weighting is physically meaningful, and picks a column density that keeps
    the strongest line unsaturated (peak tau ~ target_peak_tau).
    """
    init_hapi()

    # --- per-molecule (renormalized) cross-sections, same native grid ---
    if iso_name == "12C16O2":
        t_main = fetch_co2_isotopologue("12C16O2")
        wav, xs_12 = compute_cross_section(t_main, T, P_atm, "12C16O2")
        t_minor = fetch_co2_isotopologue("13C16O2")
        wav2, xs_13 = compute_cross_section(t_minor, T, P_atm, "13C16O2")
    
    elif iso_name == "12CH4":
        t_main = fetch_ch4_isotopologue("12CH4")
        wav, xs_12 = compute_cross_section(t_main, T, P_atm, "12CH4")

        t_minor = fetch_ch4_isotopologue("13CH4")
        wav2, xs_13 = compute_cross_section(t_minor, T, P_atm, "13CH4")

    # OmegaRange pins both to the same grid; assert it to be safe
    assert np.allclose(wav, wav2), "isotopologue grids differ unexpectedly"

    print(f"[test_isotopologue_5050]  T={T} K, P={P_atm} atm")
    print(f"  terrestrial abundance {iso_name} : {terrestrial_abundance(iso_name):.4e}")
    print(f"  terrestrial abundance {iso_minor} : {terrestrial_abundance(iso_minor):.4e}")
    print(f"  peak xsec {iso_name} (renorm)    : {xs_12.max():.3e} cm^2/molec")
    print(f"  peak xsec {iso_minor} (renorm)    : {xs_13.max():.3e} cm^2/molec")

    # --- choose column density from the STRONGER isotopologue to avoid saturation ---
    xs_max = max(xs_12.max(), xs_13.max())
    N = target_peak_tau / xs_max
    print(f"  column density (peak tau={target_peak_tau}): {N:.3e} molec/cm^2")

    f12, f13 = 1.0, 1.0
    tau_12   = N * f12 * xs_12
    tau_13   = N * f13 * xs_13
    trans_12   = np.exp(-tau_12)
    trans_13   = np.exp(-tau_13)

    f12, f13 = 0.5, 0.5
    tau_12   = N * f12 * xs_12
    tau_13   = N * f13 * xs_13
    tau_comb = tau_12 + tau_13          # combined slab = sum of opacities
    
    trans_comb = np.exp(-tau_comb)

    # saturation sanity check
    for lbl, tau in [("12CO2 (x1)", tau_12),
                     ("13CO2 (x1)", tau_13),
                     ("combined (50/50)",     tau_comb)]:
        print(f"  peak tau {lbl:14s}: {tau.max():.3f} "
              f"-> trans floor {np.exp(-tau.max()):.4f}")

    # --- plot: native resolution (wav already ascending microns) ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    if iso_name == "12C16O2":
        nice_label = r"$^{12}$CO$_{2}$"
        nice_label_minor = r"$^{13}$CO$_{2}$"
    elif iso_name == "12CH4":
        nice_label = r"$^{12}$CH$_{4}$"
        nice_label_minor = r"$^{13}$CH$_{4}$"

    ax.plot(wav, trans_12,   lw=0.5, label=f"{nice_label} alone (x1)")
    ax.plot(wav, trans_13,   lw=0.5, label=f"{nice_label_minor} alone (x1)")
    ax.plot(wav, trans_comb, lw=0.7, color="k", label="combined 50/50")

    # --- plot: resampled to R=100 ---
    R = 500
    wav_bin, trans_12_bin,   __ = bin_spectrum(wav, trans_12,   R, err_data=[])
    wav_bin, trans_13_bin,   __ = bin_spectrum(wav, trans_13,   R, err_data=[])
    wav_bin, trans_comb_bin, __ = bin_spectrum(wav, trans_comb, R, err_data=[])

    ax.plot(wav_bin, trans_12_bin,   lw=1.0, ls="--", label=f"{nice_label} binned R={R}")
    ax.plot(wav_bin, trans_13_bin,   lw=1.0, ls="--", label=f"{nice_label_minor} binned R={R}")
    ax.plot(wav_bin, trans_comb_bin, lw=1.0, ls="--", color="k",
            label=f"combined 50/50 binned R={R}")

    ax.set_xlabel(r"Wavelength ($\mu$m)")
    ax.set_ylabel("Transmission")
    # ax.set_title(f"CO2 isotopologues, 50/50 abundance "
    #              f"(N={N:.2e} cm$^{{-2}}$)")
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout(); plt.show()
    return wav, trans_12, trans_13, trans_comb


def test_isotopologue_crosscorr(iso_main="12C16O2", iso_minor="13C16O2",
                                T=300.0, P_atm=0.1, target_peak_tau=0.5,
                                R=500, snr=10.0, abundance_ratio=0.5, seed=42):
    """
    Can we distinguish the minor isotopologue via cross-correlation?

    data     = box with main + scaled-minor isotopologue + Gaussian noise
    template = box with minor isotopologue only
    Both binned to R before correlating (shared grid since same native input).
    """
    init_hapi()
    rng = np.random.default_rng(seed)

    if iso_main == "12C16O2":
        t_main = fetch_co2_isotopologue(iso_main)
        t_minor = fetch_co2_isotopologue(iso_minor)
    elif iso_main == "12CH4":
        t_main = fetch_ch4_isotopologue(iso_main)
        t_minor = fetch_ch4_isotopologue(iso_minor)
    wav, xs_main = compute_cross_section(t_main, T, P_atm, iso_main)
    wav2, xs_minor = compute_cross_section(t_minor, T, P_atm, iso_minor)
    assert np.allclose(wav, wav2), "isotopologue grids differ unexpectedly"

    # column density from the main isotopologue to avoid saturation
    N = target_peak_tau / xs_main.max()
    print("N:", N)

    # combined slab: main + scaled-minor opacity
    I0 = light_source(wav, kind="flat", level=1.0)
    data_clean = I0 * np.exp(-N * (xs_main * (1 - abundance_ratio) + xs_minor * abundance_ratio))

    # template: minor isotopologue only # FIXME: I am not sure if this is what we want to comapre
    template_clean = I0 * np.exp(-N * abundance_ratio * xs_minor)

    # bin to R (shared grid), then add noise to data
    wav_R, data_R = resample_to_R(wav, data_clean, R)
    _,     tmpl_R = resample_to_R(wav, template_clean, R)

    # data_noisy, sigma = claude_add_gaussian_noise(data_R, snr, rng=rng)
    # print(data_noisy, sigma)

    data_noisy, sigma, errorbars = add_gaussian_noise(data_R, snr) #, rng=rng)
    # print(data_noisy, sigma)

    

    lags, ccf = cross_correlate(tmpl_R, data_noisy) # FIXME: verify this function!
    peak, ccf_snr = detection_significance(ccf) # FIXME: verify this function!

    print(f"[test_isotopologue_crosscorr] {iso_main} vs {iso_minor}")
    print(f"  R={R:.0e}, SNR={snr}, abundance_ratio={abundance_ratio}")
    print(f"  N={N:.3e} molec/cm^2, noise sigma={sigma:.3e}")
    print(f"  CCF peak={peak:.4f}, detection S/N={ccf_snr:.2f}  (>~5 distinguishable)")

    if iso_main == "12C16O2":
        nice_label = r"$^{12}$CO$_{2}$"
        nice_label_minor = r"$^{13}$CO$_{2}$"
    elif iso_main == "12CH4":
        nice_label = r"$^{12}$CH$_{4}$"
        nice_label_minor = r"$^{13}$CH$_{4}$"

    fig, ax = plt.subplots(2, 1, figsize=(9, 7))
    ax[0].errorbar(wav_R, data_noisy, yerr=errorbars, fmt=".", label="noisy data (main+minor)")
    ax[0].plot(wav_R, tmpl_R, lw=0.9, label=f"{nice_label_minor} template")
    ax[0].set_xlabel(r"Wavelength ($\mu$m)"); ax[0].set_ylabel("Flux (arb.)")
    ax[0].legend(); ax[0].set_title(f"Spectra (binned to R={R}, SNR={snr})")
    ax[1].plot(lags, ccf, lw=0.8)
    ax[1].set_xlabel("Lag (pixels)"); ax[1].set_ylabel("CCF")
    ax[1].set_title(f"CCF (peak={peak:.3f}, S/N={ccf_snr:.1f})")
    plt.tight_layout(); plt.show()
    return ccf_snr


# def grid_study(R_list=(3e4, 1e5, 3e5), snr_list=(10, 50, 100),
#                ratio_list=(0.001, 0.011, 0.05)):
#     """Sweep resolution x SNR x abundance ratio; tabulate CCF detection S/N."""
#     print("\n=== grid_study: CCF detection S/N for 13C16O2 ===")
#     print(f"{'R':>8} {'SNR':>6} {'ratio':>8} {'ccf_snr':>10}")
#     results = []
#     for R in R_list:
#         for snr in snr_list:
#             for ratio in ratio_list:
#                 ccf_snr = test_isotopologue_crosscorr(
#                     R=R, snr=snr, abundance_ratio=ratio)
#                 plt.close("all")
#                 print(f"{R:8.0e} {snr:6.0f} {ratio:8.3f} {ccf_snr:10.2f}")
#                 results.append((R, snr, ratio, ccf_snr))
#     return results


def plot_slice(iso_main="12C16O2", iso_minor="13C16O2", R_list=[100, 500, 1000, 5000], abundance_ratio=0.5, snr=10.0):
    """Plot CCF detection S/N vs R for a fixed abundance ratio and SNR."""
    ccf_snrs = []
    for R in R_list:
        ccf_snr = test_isotopologue_crosscorr(
            iso_main=iso_main, iso_minor=iso_minor, R=R, snr=snr, abundance_ratio=abundance_ratio)
        plt.close("all")
        ccf_snrs.append(ccf_snr)
    plt.figure()
    plt.plot(R_list, ccf_snrs, marker="o")
    plt.xscale("log"); plt.xlabel("Resolution R")
    plt.ylabel("CCF detection S/N")
    plt.title(f"Abundance ratio={abundance_ratio}, SNR={snr}")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # test_fetch_and_xsec("12CH4") #12C16O2")
    # test_box_spectrum("12CH4") #"12C16O2")
    # test_isotopologue_5050("12CH4", "13CH4")
    # test_isotopologue_crosscorr("12C16O2", "13C16O2") #"12CH4", "12CH4")
    # test_isotopologue_crosscorr("12C16O2", "12C16O2")
    # grid_study()
    plot_slice(R_list=[100, 500, 1000, 5000], abundance_ratio=0.5, snr=10.0)