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

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atmosphere", required=True)
    p.add_argument("--abs-coef-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ref_therm", required=True)
    p.add_argument("--cloud-top", type=float, default=8.0)
    p.add_argument("--albedo", type=float, default=0.3)
    p.add_argument("--resolutions", nargs="+", type=float, default=[1e5, 1e6, 1e7])
    p.add_argument("--snrs", nargs="+", type=float, default=[5, 10, 25, 50])
    p.add_argument("--scenarios", nargs="+", default=["H2O+CH4", "H2O+CH4+N2O"],
                   help="e.g. H2O+CH4  H2O+CH4+N2O")
    p.add_argument("--xsc-dir", default="",
                   help="Directory of HITRAN .xsc files (for isoprene, etc.)")
    p.add_argument("--xsc-species", nargs="*", default=["Isoprene"],
                   help="Species that use XSC instead of LBL npz (names as in scenario string)")
    return p.parse_args()

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


def load_abs_coef(mol, iso, alt, cache_dir):
    path = Path(cache_dir) / f"{mol}_iso{iso}_alt{alt:03d}.npz"
    data = np.load(path)
    return data["wn"], data["coef"]

def bin_spectrum_robust(wl_native, spectrum_native, R_bin, err_data=None):
    """
    Robust flux-conserving-ish rebinning for extremely large native grids.
    Avoids SpectRes entirely.
    """
    # if err_data is None:
    #     err_data = []

    wl_native = np.asarray(wl_native, dtype=np.float64)
    spectrum_native = np.asarray(spectrum_native, dtype=np.float64)

    # Ensure increasing wavelength
    if wl_native[0] > wl_native[-1]:
        wl_native = wl_native[::-1]
        spectrum_native = spectrum_native[::-1]
        # if len(err_data) > 0:
        #     err_data = np.asarray(err_data)[::-1]

    # Approx native resolution
    native_R = np.median(wl_native[:-1] / np.diff(wl_native))
    
    # If requested R is close to or higher than native, just return native
    if R_bin >= 0.95 * native_R:
        print(f"Requested R={R_bin:.2e} ≥ native R≈{native_R:.2e} → returning native grid")
        # if len(err_data) > 0:
        #     return wl_native, spectrum_native, np.asarray(err_data)
        return wl_native, spectrum_native #, None

    # Build new wavelength grid at constant R
    log_wl_min = np.log(wl_native[0])
    log_wl_max = np.log(wl_native[-1])
    delta_log_wl = 1.0 / R_bin
    n_bins = int(np.floor((log_wl_max - log_wl_min) / delta_log_wl)) + 1
    
    # Safety: never create more bins than native points
    n_bins = min(n_bins, len(wl_native) - 2)
    
    log_wl_binned = np.linspace(log_wl_min, log_wl_max, n_bins)
    wl_binned = np.exp(log_wl_binned)

    # Simple but stable interpolation (linear in wavelength)
    # For most detection/metric work this is sufficient
    interp = interp1d(wl_native, spectrum_native, kind='linear', 
                      bounds_error=False, fill_value='extrapolate')
    spectrum_binned = interp(wl_binned)

    # if len(err_data) > 0:
    #     err_interp = interp1d(wl_native, err_data, kind='linear',
    #                           bounds_error=False, fill_value='extrapolate')
    #     err_binned = err_interp(wl_binned)
    #     return wl_binned, spectrum_binned, err_binned

    return wl_binned, spectrum_binned

def inject_poisson_noise(signal, snr, seed=42, mode="gaussian_approx"):
    """
    SNR defined on the continuum/mean signal level.
    Works for transmission or thermal radiance.
    """
    rng = np.random.default_rng(seed)
    signal = np.asarray(signal, dtype=np.float64)

    # positive reference level (avoid zeros)
    ref = np.nanmedian(np.abs(signal))
    if ref <= 0:
        ref = np.nanmax(np.abs(signal))
    if ref <= 0:
        ref = 1.0

    noise_std = ref / float(snr)   # constant σ per bin
    # or wavelength-dependent: noise_std = np.maximum(np.abs(signal), ref) / snr

    if mode == "gaussian_approx":
        noise = rng.normal(0.0, noise_std, size=signal.shape)
        noisy = signal + noise
        errorbars = np.full_like(signal, noise_std)
    else:
        raise ValueError("mode must be 'gaussian_approx'")

    return noisy.astype(np.float32), errorbars.astype(np.float32)

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

def compute_thermal_emission(
    molecules_isotopes,
    df_atm,
    abs_coef_dir,
    cloud_top=8.0,
    T_surface=None,
    xsc_dir="",
    xsc_species=None,
):
    if xsc_species is None:
        xsc_species = []
    xsc_set = {s.upper() for s in xsc_species}

    t0 = time.time()
    above_df = df_atm[df_atm["ALT_km"] >= cloud_top].copy()
    print(f"Computing thermal emission above {cloud_top} km ({len(above_df)} layers)", flush = True)
    
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

    # --- LBL species only ---
    lbl_species = [(m, i) for m, i in molecules_isotopes if m.upper() not in xsc_set]
    xsc_only = [m for m, i in molecules_isotopes if m.upper() in xsc_set]

    for mol, iso in lbl_species:
        col_name = f"{mol}_iso{iso}_ppmv"
        t_mol = time.time()
        if col_name not in df_atm.columns:
            print(f"ERROR: missing {col_name}")
            continue
        print(f"→ {mol} iso{iso} (LBL)")
        ppmv_col = above_df[col_name].astype(float).values
        for layer_pos, (orig_idx, row) in enumerate(above_df.iterrows()):
            wn_grid, coef = load_abs_coef(mol, iso, orig_idx, abs_coef_dir)
            if wn_grid_ref is None:
                wn_grid_ref = wn_grid
                delta_tau_layers = np.zeros((n_layers, len(coef)), dtype=np.float64)
            if layer_pos < n_layers - 1:
                ppmv_avg = 0.5 * (ppmv_col[layer_pos] + ppmv_col[layer_pos + 1])
            else:
                ppmv_avg = ppmv_col[layer_pos]
            vmr_avg = ppmv_avg * 1e-6
            
            if layer_pos % 10 == 0 or layer_pos == n_layers - 1:
                print(
                    f"  layer {layer_pos+1}/{n_layers}  alt={above_df['ALT_km'].iloc[layer_pos]:.1f} km  "
                    f"elapsed={time.time()-t_mol:.1f}s",
                    flush=True,
                )
                
            # HAPI coef in cm^-1 style (your working CO2 convention)
            delta_tau_layers[layer_pos] += coef * dz_cm[layer_pos] #* vmr_avg
            

    if delta_tau_layers is None:
        raise ValueError("No LBL tau accumulated — need at least one LBL species to set the grid")

        
    # --- isothermal XSC species (e.g. isoprene) ---
    if xsc_only and not xsc_dir:
        raise ValueError(f"XSC species {xsc_only} requested but --xsc-dir not set")
    for mol in xsc_only:
        # XSC is cm^2/molecule → use density
        delta_tau_layers = add_xsc_molecule_tau(
            delta_tau_layers, wn_grid_ref, above_df, dz_cm, density,
            mol, xsc_dir, use_density=True,
        )

    # --- thermal sum (unchanged) ---
    tau_above = np.zeros_like(delta_tau_layers)
    for i in range(n_layers - 2, -1, -1):
        tau_above[i] = tau_above[i + 1] + delta_tau_layers[i + 1]

    tau_tot = delta_tau_layers.sum(axis=0)
    I = planck_wn(wn_grid_ref, T_surface) * np.exp(-np.minimum(tau_tot, 50.0))
    for i in range(n_layers):
        dtau = np.minimum(delta_tau_layers[i], 50.0)
        B_i = planck_wn(wn_grid_ref, temps[i])
        I += B_i * np.exp(-np.minimum(tau_above[i], 50.0)) * (1.0 - np.exp(-dtau))

    wavelengths_um = (1e4 / wn_grid_ref).astype(np.float32)
    radiance = I.astype(np.float32)
    print(f"Radiance range: {radiance.min():.3e} – {radiance.max():.3e}")
    print("tau_tot: min/med/max", tau_tot.min(), np.median(tau_tot), tau_tot.max())
    print("frac tau>1", np.mean(tau_tot > 1))
    return wavelengths_um, radiance

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
    if not s:
        raise ValueError("Empty scenario string")

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

    for scen_str in args.scenarios:
        scenario = parse_scenario(scen_str)
        # avoid ":" in filenames on some filesystems
        safe_name = scen_str.replace(":", "")
        out_path = out_dir / f"{safe_name}.pkl"

        print(f"\n=== Scenario {scen_str} ===")

        if args.ref_therm.lower() == "thermal":
            wl_high, radiance_high = compute_thermal_emission(
                scenario,
                df_atm,
                abs_dir,
                cloud_top=args.cloud_top,
                xsc_dir=xsc_dir,
                xsc_species=xsc_species,
            )
        else:
            wl_high, radiance_high = compute_reflectivity(
                scenario,
                df_atm,
                abs_dir,
                args.cloud_top,
                args.albedo,
            )
            
            print(f"High-res spectrum done in {time.time()-t_sc:.1f}s  N={len(wl_high)}", flush=True)

        scenario_dict = {}
        for R in args.resolutions:
            print(f"  Binning to R={R:.2e}")
            wl_b, rad_b = bin_spectrum_robust(wl_high, radiance_high, R)
            entry = {
                "wavelength_grid": wl_b.astype(np.float32),
                "radiance_clean": rad_b.astype(np.float32),
                "resolution": int(R),
            }
            # Optional noise:
            for snr in args.snrs:
                noisy, err = inject_poisson_noise(rad_b, snr)
                entry[f"radiance_snr{int(snr)}"] = noisy
                entry[f"error_snr{int(snr)}"] = err

            scenario_dict[int(R)] = entry
            del wl_b, rad_b, entry
            gc.collect()

        with open(out_path, "wb") as f:
            pickle.dump(scenario_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved → {out_path}")

        del wl_high, radiance_high, scenario_dict
        gc.collect()

    print("\nAll scenarios done.")


if __name__ == "__main__":
    main()