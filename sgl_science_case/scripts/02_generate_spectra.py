#!/usr/bin/env python3
"""
Generate reflectivity spectra from precomputed abs coefs, bin to requested R,
inject noise, and save per-scenario pickles.

Example:
  python scripts/02_generate_spectra.py \\
    --atmosphere ~/sgl_science_case/.../atmosphere_profile.csv \\
    --abs-coef-dir ~/orcd/scratch/sgl_science_case/abs_coef_cache \\
    --out-dir ~/orcd/pool/sgl_science_case/spectra_cache \\
    --ref_therm Thermal \\
    --cloud-top 8.0 --albedo 0.3 \\
    --resolutions 1e5 1e6 1e7 5e7 \\
    --snrs 3 5 10 15 20 25 50 \\
    --scenarios H2O+CH4 H2O+CH4+N2O \\
"""

from __future__ import annotations
import argparse, gc, os, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import re
import time
from bin_spec import bin_spectrum
from astropy import constants as const
from numpy.lib.stride_tricks import sliding_window_view


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atmosphere", required=True)
    p.add_argument("--abs-coef-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ref_therm", required=True)
    p.add_argument("--cloud-top", type=float, default=8.0)
    p.add_argument("--albedo", type=float, default=0.3)
    p.add_argument("--resolutions", nargs="+", type=float, default=np.logspace(2,6,10))#[1e5, 1e6, 1e7])
    p.add_argument("--snrs", nargs="+", type=float, default=np.logspace(0,3,10))#[5, 10, 25, 50])
    p.add_argument("--scenarios", nargs="+", default=["H2O+CH4", "H2O+CH4+N2O"],
                   help="e.g. H2O+CH4  H2O+CH4+N2O")
    p.add_argument("--xsc-dir", default="",
                   help="Directory of HITRAN .xsc files (for isoprene, etc.)")
    p.add_argument("--xsc-species", nargs="*", default=["Isoprene"],
                   help="Species that use XSC instead of LBL npz (names as in scenario string)")
    p.add_argument("--molecule_isotope", default = None)
    p.add_argument("--isotope_ratio", default = None)
    p.add_argument("--sigma_r_frac",type=float, default = 0)

    return p.parse_args()

def scenario_out_name(scen_str: str) -> str:
    s = scen_str.strip().upper()
   
    # if s in {"", "BLACKBODY", "BB", "NONE", "CONTINUUM"}:
    #     return "BLACKBODY"
    # parts = {p.strip() for p in scen_str.split("+") if p.strip()}
    # if "DMS:1" in parts or "DMS" in parts:
    #     return "ALL_WITH_DMS"
    # if "DMS:1" not in parts:
    #     return "ALL_WITHOUT_DMS"
    # if "CH4" in parts:
    #     return "ALL_WITH_CH4"
    # if "CH4" not in partS:
    #     return "ALL_WITHOUT_CH4"
    return f"hydrocarbons_with_CH4_R_0"

#### xsc helpers:


def parse_hitran_xsc(path: Path):
    lines = path.read_text(errors="replace").splitlines()
    floats = []
    for tok in lines[0].split():
        try:
            floats.append(float(tok))
        except ValueError:
            continue
        if len(floats) >= 3:
            break
    if len(floats) < 3:
        raise ValueError(f"Bad XSC header: {path}")
    numin, numax, npoints = floats[0], floats[1], int(floats[2])
    vals = []
    for line in lines[1:]:
        for tok in line.split():
            try:
                vals.append(float(tok))
            except ValueError:
                pass
    xsec = np.asarray(vals[:npoints], dtype=np.float64)
    wn = np.linspace(numin, numax, npoints)
    return wn, xsec


# filename stems for isoprene
XSC_ALIASES = {
    "Isoprene": ["C5-H8", "C5H8", "isoprene"],
    
    # Close confusers / biogenics
    "Butadiene": ["C4H6"],          # 1,3-butadiene
    "Propene": ["C3H6"],
    "Butene": ["C4H8"],              # C4H8 isomer files (1-butene / 2-butene / isobutene — same stem)
    "1-Butyne": ["HC=CCH2CH3", "HCCCH2CH3"],

    # Monoterpenes (C10H16) — same formula, different files
    "Limonene": ["C10H16", "C10-H16"],
    "Pinene": ["C10H16", "C10-H16"],   # if you only have formula stems, both share these

    # Aromatics
    "Benzene": ["C6H6"],
    "Toluene": ["C6H5CH3"],
    "Trimethylbenzene": ["C6H3(CH3)3"],
    "Tetramethylbenzene": ["(C6H2)(CH3)4"],

    # Longer alkene
    "1-Decene": ["CH2CH(CH2)7CH3"],
    
    "DMS": ["C2H6S"],
    "DMDS": ["C2H6S2"],
    "CS2": ["CS2"],
    "SF6": ["SF6"],
    "SO2F2": ["SO2F2"],
    "1-Propanethiol": ["CH3(CH2)2SH"],
    "2-Methyl-1-propanethiol": ["(CH3)2CHCH2SH"],
    "2-Propanethiol": ["(CH3)2CH(HS)"],
    "Benzenethiol": ["C6H5SH"],
    "Cyclohexanethiol": ["C6H11SH"],
    "DiethylSulfide": ["(CH3CH2)2S"],
    "DMSO": ["C2H6OS"],
    "EthylMercaptan": ["CH3CH2SH"],
    "Methanethiol": ["CH4S"],
    "MethylIsothiocyanate": ["C2H3NS"],
    "Tetrahydrothiophene": ["(CH2)4S"],
    "Thiophene": ["C4H4S"],
    "tert-Butylmercaptan": ["(CH3)3CSH"],

    # --- XSC on disk but no (or weak) profile column unless you add one ---
    "DiethylSulfate": ["(C2H5O)2SO2"],
    "DimethylSulfate": ["C2H6O4S"],
    "PropyleneSulfide": ["C3H6S"],
    "Thiophosgene": ["CCl2S"],
    "PerchloromethylMercaptan": ["CCl3SCl"],
    "EthyleneSulfide": ["CH2CH2S"],
    "MethanesulfonylChloride": ["CH3SO2Cl"],
    "Thioglycol": ["HS(CH2)2OH"],
    "SO2Cl2": ["SO2Cl2"],
    "SOF2": ["SOF2"],
    "SPCl3": ["SPCl3"],
    
}


def find_xsc_file(xsc_dir: Path, mol: str) -> Path:
    stems = {s.lower() for s in XSC_ALIASES.get(mol, [mol])}
    matches = []
    for f in sorted(Path(xsc_dir).iterdir()):
        if f.name.startswith("._") or not f.is_file():
            continue
        if not (f.suffix == ".txt" or f.name.endswith(".xsc") or f.name.endswith(".xsc.txt")):
            continue
        stem0 = f.name.split("_")[0].lower()
        if stem0 in stems or any(s in f.name.lower() for s in stems):
            matches.append(f)
    if not matches:
        raise FileNotFoundError(f"No XSC for {mol!r} in {xsc_dir}")
    return matches[0]


def add_xsc_molecule_tau(
    delta_tau_layers,
    wn_ref,
    above_df,
    dz_cm,
    density,
    mol,
    xsc_dir,
    *,
    use_density=True,
):
    """
    Add isothermal XSC opacity onto existing delta_tau_layers (LBL grid).
    XSC σ is cm^2/molecule → Δτ = σ * n * VMR * dz if use_density
    """
    path = find_xsc_file(Path(xsc_dir), mol)
    wn_x, sig = parse_hitran_xsc(path)
    print(f"→ {mol} (XSC) {path.name}  wn=[{wn_x.min():.1f},{wn_x.max():.1f}]")

    sig_on_grid = interp1d(wn_x, sig, kind="linear", bounds_error=False, fill_value=0.0)(wn_ref)

    # column name: prefer Isoprene_ppmv, then Isoprene_iso1_ppmv
    col = None
    for c in (f"{mol}_ppmv", f"{mol}_iso1_ppmv"):
        if c in above_df.columns:
            col = c
            break
    if col is None:
        print(f"ERROR: missing VMR column for XSC species {mol}")
        return delta_tau_layers

    ppmv_col = above_df[col].astype(float).values

    n_layers = len(above_df)
    for i in range(n_layers):
        if i < n_layers - 1:
            ppmv_avg = 0.5 * (ppmv_col[i] + ppmv_col[i + 1])
            n_avg = 0.5 * (density[i] + density[i + 1])
        else:
            ppmv_avg = ppmv_col[i]
            n_avg = density[i]
        vmr = ppmv_avg * 1e-6
        if use_density:
            delta_tau_layers[i] += sig_on_grid * n_avg * vmr * dz_cm[i]
        else:
            delta_tau_layers[i] += sig_on_grid * vmr * dz_cm[i]
    return delta_tau_layers


def load_abs_coef(mol, iso, alt, cache_dir, isotope_molecule=None, isotope_ratio=None):
    tag = ""
    if (
        isotope_molecule
        and mol.upper() == isotope_molecule.upper()
        and isotope_ratio is not None
    ):
        tag = f"_r{float(isotope_ratio):g}"
    path = Path(cache_dir) / f"{mol}_iso{iso}_alt{alt:03d}{tag}.npz"
    data = np.load(path)
    return data["wn"], data["coef"]

def bin_data(wave_data, flux_data, R_bin,  err_data=[]):
    """Bin the data to resolution R_bin (and err_data if provided)."""
    wav_binned, flux_binned, __ = bin_spectrum(wave_data, flux_data, R_bin, err_data=err_data)
    return wav_binned, flux_binned


def inject_white_red_noise(
    signal,
    snr_white,
    sigma_r_frac=0.5,
    corr_bins=8,
    floor_frac=0.05,
    rng=None,
):
    """
    White + red noise in the spectral domain (Pont-style).

    White noise is local (photon-like):
        sigma_w[i] = max(|signal[i]|, floor_frac * median|signal|) / snr_white

    Red noise is a correlated component with RMS
        sigma_r[i] = sigma_r_frac * sigma_w[i]
    built by smoothing a unit white series, then scaling per bin.

    Parameters
    ----------
    snr_white : float
        Target SNR per bin relative to the local reference level.
    sigma_r_frac : float
        Red amplitude as a fraction of local white sigma (0 = white only).
    corr_bins : int
        Correlation length in spectral bins.
    floor_frac : float
        Floor for the reference level as a fraction of median |signal|
        (avoids sigma → 0 in deep cores / zeros).
    """
    rng = np.random.default_rng() if rng is None else rng
    y = np.asarray(signal, dtype=np.float64)

    med = np.nanmedian(np.abs(y))
    if not np.isfinite(med) or med <= 0:
        med = 1.0

    # per-bin reference (Poisson-like) with continuum floor
    ref = np.maximum(np.abs(y), floor_frac * med)
    sigma_w = ref / float(snr_white)  # shape (N,)

    # white component (independent draws, local amplitude)
    white = rng.normal(0.0, 1.0, size=y.shape) * sigma_w

    if sigma_r_frac <= 0:
        noise = white
        sigma_eff = sigma_w.copy()
    else:
        k = max(1, int(corr_bins))
        pad = k // 2

        # smooth a unit-variance white series → correlated structure
        unit = rng.normal(0.0, 1.0, size=y.shape)
        wpad = np.pad(unit, pad, mode="edge") # padding to avoid edge-effects
        kernel = np.ones(k) / k # correlating
        red_unit = np.convolve(wpad, kernel, mode="valid") # correlating neighboring bins
        
        #more error safeties
        if len(red_unit) > len(y):
            red_unit = red_unit[: len(y)]
        elif len(red_unit) < len(y):
            red_unit = np.pad(red_unit, (0, len(y) - len(red_unit)), mode="edge")

        # normalize structure, then apply local red amplitude
        red_unit = red_unit / (np.std(red_unit) + 1e-30)
        sigma_r = sigma_r_frac * sigma_w          # per-bin
        red = red_unit * sigma_r

        noise = white + red
        # effective per-bin sigma (white and red independent)
        sigma_eff = np.sqrt(sigma_w**2 + sigma_r**2)

    noisy = y + noise
    return noisy.astype(np.float32), sigma_eff.astype(np.float32)


def pont_V(n, sigma_w, sigma_r):
    """Paper eq. (9): variance of the mean of n correlated samples."""
    n = np.asarray(n, dtype=np.float64)
    return sigma_w**2 / n + sigma_r**2


def planck_wn(wn_cm, T):
    """Planck function in wavenumber units [wn in cm^-1, T in K].
    Returns B_wn in erg/s/cm^2/sr/cm^-1 (cgs). Scale is arbitrary for template matching.
    """
    # B_ν = 2 h c^2 ν^3 / (exp(hcν/kT) - 1)
    # ν in cm^-1; use hc/k = 1.4388 cm K
    wn = np.asarray(wn_cm, dtype=np.float64)
    c1 = 1.191042972e-5   # 2hc^2 in appropriate cgs scaling for wn
    c2 = 1.4387769          # hc/k [cm K]
    x = c2 * wn / T
    # stable for large x
    return c1 * wn**3 / np.expm1(x)

#HELPERS FOR NORMALIZING HYDROCARBON AMPLITUDES

WL_MIN, WL_MAX = 3.0, 4.0  # µm  — degeneracy window
TARGET_MOL = ("CH4", 1)     # match this amplitude

def window_mask(wn_cm, wl_min=WL_MIN, wl_max=WL_MAX):
    wl = 1e4 / np.asarray(wn_cm, dtype=np.float64)
    return (wl >= wl_min) & (wl <= wl_max)

def compute_thermal_emission(
    molecules_isotopes,
    df_atm,
    abs_coef_dir,
    cloud_top=8.0,
    T_surface=None,
    xsc_dir="",
    xsc_species=None,
    isotope_molecule=None,
    isotope_ratio=None,
):
    if xsc_species is None:
        xsc_species = []
    xsc_set = {s.upper() for s in xsc_species}

    above_df = df_atm[df_atm["ALT_km"] >= cloud_top].copy()
    print(f"Computing thermal emission above {cloud_top} km ({len(above_df)} layers)", flush=True)

    alt_km = above_df["ALT_km"].values
    dz_km = np.diff(alt_km, append=alt_km[-1] + np.median(np.diff(alt_km)))
    dz_cm = dz_km * 1e5
    temps = above_df["TEMP_K"].values
    density = above_df["DENSITY_cm3"].astype(float).values
    if T_surface is None:
        T_surface = float(temps[0])

    delta_tau_layers = None
    wn_grid_ref = None
    n_layers = len(above_df)

    lbl_species = [(m, i) for m, i in molecules_isotopes if m.upper() not in xsc_set]
    xsc_only = [m for m, i in molecules_isotopes if m.upper() in xsc_set]

#________Precompute scales from one mid-atmosphere layer________________________________________________________________
    ref_idx = len(above_df) // 2
    scales = {}
    wn_ref, coef_ch4 = load_abs_coef("CH4", 1, above_df.index[ref_idx], abs_coef_dir)
    m = window_mask(wn_ref)
    peak_ch4 = np.max(coef_ch4[m]) * (above_df[f"CH4_iso1_ppmv"].iloc[ref_idx] * 1e-6)
#________________________________________________________________________________________________________________________

    # --- LBL ---
    for mol, iso in lbl_species:
        col_name = f"{mol}_iso{iso}_ppmv"
        if col_name not in df_atm.columns:
            print(f"ERROR: missing {col_name}")
            continue
        print(f"→ {mol} iso{iso} (LBL)")
        
# Scaling tests _________________________________________________________________________________________________________________________
        
        wn_grid, coef = load_abs_coef(mol,iso,above_df.index[ref_idx],abs_coef_dir)
        m = window_mask(wn_grid)
        peak = np.max(coef[m]) * (above_df[f"{mol}_iso{iso}_ppmv"].iloc[ref_idx] * 1e-6)
        scales[(mol, iso)] = peak_ch4 / peak if peak > 0 else 0.0
        print(f"Scale for {mol} = {scales[(mol,iso)]}")

# _________________________________________________________________________________________________________________________________________________________
                
        ppmv_col = above_df[col_name].astype(float).values
        for layer_pos, (orig_idx, row) in enumerate(above_df.iterrows()):
            wn_grid, coef = load_abs_coef(mol, iso, orig_idx, abs_coef_dir, isotope_molecule=isotope_molecule, isotope_ratio = isotope_ratio)
            wn_grid = np.asarray(wn_grid, dtype=np.float64).ravel()
            coef = np.asarray(coef, dtype=np.float64).ravel()
            if wn_grid_ref is None:
                wn_grid_ref = wn_grid
                delta_tau_layers = np.zeros((n_layers, wn_grid_ref.size), dtype=np.float64)                
                
            #NORMALIZING FOR HYDROCARBONS            
            delta_tau_layers[layer_pos] += coef * dz_cm[layer_pos] * scales[(mol,iso)] 

    # --- if no LBL: seed grid from first XSC ---
    if wn_grid_ref is None:
        if not xsc_only:
            if wn_grid_ref is None and not lbl_species and not xsc_only:
                # pure continuum: build a default wavenumber grid
                wl_lo, wl_hi, dwn = 1.0, 17.0, 1e-4
                wn_hi, wn_lo = 1e4 / wl_lo, 1e4 / wl_hi
                wn_grid_ref = np.arange(wn_lo, wn_hi + 0.5 * dwn, dwn, dtype=np.float64)
                wn_grid_ref = np.sort(np.unique(wn_grid_ref))

                wavelengths_um = (1e4 / wn_grid_ref).astype(np.float32)
                flux = planck_wn(wn_grid_ref, T_surface).astype(np.float32)
                order = np.argsort(wavelengths_um)
                return wavelengths_um[order], flux[order]
            
        if not xsc_dir:
            raise ValueError(f"XSC species {xsc_only} requested but --xsc-dir not set")
        path0 = find_xsc_file(Path(xsc_dir), xsc_only[0])
        wn0, _ = parse_hitran_xsc(path0)
        wn_grid_ref = np.asarray(wn0, dtype=np.float64).ravel()
        order = np.argsort(wn_grid_ref)
        wn_grid_ref = wn_grid_ref[order]
        wn_grid_ref = wn_grid_ref[np.concatenate([[True], np.diff(wn_grid_ref) > 0])]
        delta_tau_layers = np.zeros((n_layers, wn_grid_ref.size), dtype=np.float64)
        print(f"No LBL grid — using XSC grid from {path0.name} (N={wn_grid_ref.size})", flush=True)

    wn_grid_ref = np.asarray(wn_grid_ref, dtype=np.float64).ravel()

    # --- XSC ---
    if xsc_only and not xsc_dir:
        raise ValueError(f"XSC species {xsc_only} requested but --xsc-dir not set")
    for mol in xsc_only:
        delta_tau_layers = add_xsc_molecule_tau(
            delta_tau_layers, wn_grid_ref, above_df, dz_cm, density,
            mol, xsc_dir, use_density=True,
        )

    if delta_tau_layers is None:
        raise ValueError("No tau accumulated")

    # --- thermal sum ---
    tau_above = np.zeros_like(delta_tau_layers)
    for i in range(n_layers - 2, -1, -1):
        tau_above[i] = tau_above[i + 1] + delta_tau_layers[i + 1]
    tau_tot = delta_tau_layers.sum(axis=0)

    I = planck_wn(wn_grid_ref, T_surface) * np.exp(-np.minimum(tau_tot, 50.0))
    for i in range(n_layers):
        dtau = np.minimum(delta_tau_layers[i], 50.0)
        B_i = planck_wn(wn_grid_ref, temps[i])
        I += B_i * np.exp(-np.minimum(tau_above[i], 50.0)) * (1.0 - np.exp(-dtau))

    wavelengths_um = (1.e4 / wn_grid_ref).astype(np.float32)
    flux = I.astype(np.float32) * np.pi / const.c.cgs.value
    
    order = np.argsort(wavelengths_um)
    wavelengths_um = wavelengths_um[order]
    flux = flux[order]
    print(f"flux range: {flux.min():.3e} – {flux.max():.3e}")
    print("tau_tot: min/med/max", tau_tot.min(), np.median(tau_tot), tau_tot.max())
    print("frac tau>1", np.mean(tau_tot > 1))
    return wavelengths_um, flux

def parse_scenario(s: str):
    """
    Parse scenario strings into list of (molecule, isotope).

    Accepted forms:
      'H2O+CH4+N2O'      -> [('H2O', 1), ('CH4', 1), ('N2O', 1)]
      'CO2:1'            -> [('CO2', 1)]
      'H2O:1+CH4:1'      -> [('H2O', 1), ('CH4', 1)]
      'H2O+CH4:1'        -> [('H2O', 1), ('CH4', 1)]
    """
    s = s.strip()
    if not s or s.upper() in {"BLACKBODY", "BB", "NONE", "CONTINUUM"}:
        return []  # no molecules
    out = []
    for part in s.split("+"):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            mol, iso = part.split(":", 1)
            mol, iso = mol.strip(), iso.strip()
            if not mol or not iso:
                raise ValueError(f"Bad segment in scenario {s!r}: {part!r}")
            out.append((mol, int(iso)))
        else:
            out.append((part, 1))

    if not out:
        raise ValueError(f"No molecules parsed from {s!r}")
    return out

def main():
    args = parse_args()
    abs_dir = os.path.expanduser(args.abs_coef_dir)
    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    xsc_dir = os.path.expanduser(args.xsc_dir) if getattr(args, "xsc_dir", "") else ""
    xsc_species = list(getattr(args, "xsc_species", []) or [])

    df_atm = pd.read_csv(args.atmosphere)
    
    t_sc = time.time()
    
    isotope_molecule = args.molecule_isotope
    isotope_ratio = args.isotope_ratio
    
    sigma_r_frac = args.sigma_r_frac
    
    print("isotope / isotope_ratio: ", isotope_molecule, isotope_ratio)


    for scen_str in args.scenarios:
        scenario = parse_scenario(scen_str)
        print("Scenario:", scenario)
        # avoid ":" in filenames on some filesystems
        safe_name = scenario_out_name(scen_str)
        if args.molecule_isotope and args.isotope_ratio is not None:
            if args.molecule_isotope.upper() in safe_name.upper():
                safe_name = f"{safe_name}_r{float(args.isotope_ratio):g}"
        out_path = out_dir / f"{safe_name}.pkl"

        print(f"\n=== Scenario {scen_str} ===")

        if args.ref_therm.lower() == "thermal":

            wl_high, flux_high = compute_thermal_emission(
                scenario,
                df_atm,
                abs_dir,
                cloud_top=args.cloud_top,
                xsc_dir=xsc_dir,
                xsc_species=xsc_species,
                isotope_molecule=isotope_molecule,
                isotope_ratio=isotope_ratio,
            )
            
#         else:
#             wl_high, flux_high = compute_reflectivity(
#                 scenario,
#                 df_atm,
#                 abs_dir,
#                 args.cloud_top,
#                 args.albedo,
#             )
            
            print(f"High-res spectrum done in {time.time()-t_sc:.1f}s  N={len(wl_high)}", flush=True)

        scenario_dict = {}
        for R in args.resolutions:
            print(f"  Binning to R={R:.2e}")
            wl_b, flux_b = bin_data(wl_high, flux_high, R)
            entry = {
                "wavelength_grid": wl_b.astype(np.float32),
                "flux_clean": flux_b.astype(np.float32),
                "resolution": int(R),
            }
            for snr in args.snrs:
                # noisy, err = inject_noise(flux_b, snr)
                
                # Red-noise injection
                # σ_r is the red noise, where we have taken its contribution as half of the σ of the white noise
                noisy, err = inject_white_red_noise(
                    flux_b, snr_white=snr, sigma_r_frac=sigma_r_frac, corr_bins=10
                )
                entry[f"flux_snr{int(snr)}"] = noisy
                entry[f"error_snr{int(snr)}"] = err

            scenario_dict[int(R)] = entry
            del wl_b, flux_b, entry
            gc.collect()

        with open(out_path, "wb") as f:
            pickle.dump(scenario_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved → {out_path}")

        del wl_high, flux_high, scenario_dict
        gc.collect()

    print("\nAll scenarios done.")


if __name__ == "__main__":
    main()