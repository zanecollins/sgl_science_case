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
    return p.parse_args()

def load_abs_coef(mol, iso, alt, cache_dir):
    path = Path(cache_dir) / f"{mol}_iso{iso}_alt{alt:03d}.npz"
    data = np.load(path)
    return data["wn"], data["coef"]

def bin_spectrum_robust(wl_native, spectrum_native, R_bin, err_data=None):
    """
    Robust flux-conserving-ish rebinning for extremely large native grids.
    Avoids SpectRes entirely.
    """
    if err_data is None:
        err_data = []

    wl_native = np.asarray(wl_native, dtype=np.float64)
    spectrum_native = np.asarray(spectrum_native, dtype=np.float64)

    # Ensure increasing wavelength
    if wl_native[0] > wl_native[-1]:
        wl_native = wl_native[::-1]
        spectrum_native = spectrum_native[::-1]
        if len(err_data) > 0:
            err_data = np.asarray(err_data)[::-1]

    # Native resolving power (approximate)
    native_R = np.median(wl_native[:-1] / np.diff(wl_native))
    
    # If requested R is close to or higher than native, just return native
    if R_bin >= 0.95 * native_R:
        print(f"Requested R={R_bin:.2e} ≥ native R≈{native_R:.2e} → returning native grid")
        if len(err_data) > 0:
            return wl_native, spectrum_native, np.asarray(err_data)
        return wl_native, spectrum_native, None

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

    if len(err_data) > 0:
        err_interp = interp1d(wl_native, err_data, kind='linear',
                              bounds_error=False, fill_value='extrapolate')
        err_binned = err_interp(wl_binned)
        return wl_binned, spectrum_binned, err_binned

    return wl_binned, spectrum_binned

def inject_poisson_noise(trans, snr, seed=42, mode='gaussian_approx'):

    """
    Inject noise into a transmission or reflectivity spectrum.
    
    Parameters:
        signal: array of transmission or reflectivity (0 to 1)
        snr: desired signal-to-noise ratio (scalar or per-bin array)
        mode: 'gaussian_approx' (recommended) or 'true_poisson'
    """
    np.random.seed(seed)
    
    signal = 1 - trans

    if mode == 'gaussian_approx':
        # Most common approach for spectra
        noise_std = signal / snr          # relative noise
        noise = np.random.normal(0, noise_std)
        noisy = trans + noise
        errorbars = noise_std
        
    elif mode == 'true_poisson':
        # True Poisson if you have absolute photon counts
        # Assume max_signal corresponds to e.g. 1e6 photons
        max_counts = 1e9                   # SGL high photon count!!
        counts = signal * max_counts
        noisy_counts = np.random.poisson(counts)
        noisy = noisy_counts / max_counts 
        errorbars = np.sqrt(counts) / max_counts   # sqrt(N) / N0
    
    else:
        raise ValueError("mode must be 'gaussian_approx' or 'true_poisson'")
    
    # Clip to physical range
    noisy = np.clip(noisy, 0.0, 1.0)
    
    return noisy, errorbars

def compute_reflectivity(scenario, df_atm, abs_coef_dir, cloud_top, albedo):
    """scenario: list of (mol, iso)"""
    above = df_atm[df_atm["ALT_km"] >= cloud_top].reset_index(drop=True)

    alt_km = above["ALT_km"].values
    dz_km = np.diff(alt_km, append=alt_km[-1] + np.median(np.diff(alt_km)))
    dz_cm = dz_km * 1e5

    total_tau = None
    wn_ref = None

    for mol, iso in scenario:
        col = f"{mol}_ppmv"
        if col not in df_atm.columns:
            raise KeyError(f"Missing column {col}")
        print(f"  → {mol} iso{iso}")
        for layer_pos, (orig_idx, row) in enumerate(above.iterrows()):
            wn, coef = load_abs_coef(mol, iso, orig_idx+8, abs_coef_dir)
            if wn_ref is None:
                wn_ref = wn
                
            # coef doesnt? include vmr already?
            # ppmv = float(row[col])
            # vmr = ppmv * 1e-6
            
            delta_tau = coef * dz_cm[layer_pos] #* vmr

            ppmv = float(row[col])
            vmr = ppmv * 1e-6
            delta_tau = coef * vmr 
            if total_tau is None:
                total_tau = np.zeros_like(coef, dtype=np.float64)
            total_tau += delta_tau

    tau_rt = 2.0 * total_tau
    transmission = np.exp(-tau_rt)
    reflectivity = (transmission).astype(np.float32)
    wavelengths = (1e4 / wn_ref).astype(np.float32)
    print(f"  max tau_rt={np.max(tau_rt):.3f}  refl range {reflectivity.min():.4f}–{reflectivity.max():.4f}")
    return wavelengths, reflectivity

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

def compute_thermal_emission(molecules_isotopes, df_atm, abs_coef_dir, cloud_top=8.0, T_surface=None):
    """
    Simple nadir thermal emission:
      I = B(T_surf) e^{-τ_tot} + Σ B(T_i) e^{-τ_above,i}
    Layers ordered from surface/cloud upward.
    """
    above_df = df_atm[df_atm["ALT_km"] >= cloud_top].copy()  # keep original index

    print(f"Computing thermal emission above {cloud_top} km ({len(above_df)} layers)")

    alt_km = above_df["ALT_km"].values
    dz_km = np.diff(alt_km, append=alt_km[-1] + np.median(np.diff(alt_km)))
    dz_cm = dz_km * 1e5
    temps = above_df["TEMP_K"].values

    # Surface/cloud temperature
    if T_surface is None:
        T_surface = float(temps[0])  # cloud-top temperature

    # --- accumulate per-layer delta_tau (sum over molecules) ---
    delta_tau_layers = None  # shape (n_layers, n_wn)
    wn_grid_ref = None

    n_layers = len(above_df)
    for mol, iso in molecules_isotopes:
        col_name = f"{mol}_iso{iso}_ppmv"
        if col_name not in df_atm.columns:
            print(f"ERROR: missing {col_name}")
            continue
        print(f"→ {mol} iso{iso}")

        ppmv_col = above_df[col_name].astype(float).values  # all layers once

        for layer_pos, (orig_idx, row) in enumerate(above_df.iterrows()):
            wn_grid, coef = load_abs_coef(mol, iso, orig_idx, abs_coef_dir)  # fix index if needed
            if wn_grid_ref is None:
                wn_grid_ref = wn_grid
                delta_tau_layers = np.zeros((n_layers, len(coef)), dtype=np.float64)

            # Match your cache convention (VMR in or out of coef)

            #Average the VMRs of the layers
            if layer_pos < n_layers - 1:
                ppmv_avg = 0.5 * (ppmv_col[layer_pos] + ppmv_col[layer_pos + 1])
            else:
                # topmost slab: only one interface
                ppmv_avg = ppmv_col[layer_pos]

            vmr_avg = ppmv_avg * 1e-6
            delta_tau_layers[layer_pos] += coef * vmr_avg * dz_cm[layer_pos]
            # if VMR already in coef:
            # delta_tau_layers[layer_pos] += coef * dz_cm[layer_pos]

    if delta_tau_layers is None:
        raise ValueError("No tau accumulated")

    # --- thermal RT: surface + layers, bottom → top ---
    # tau_above[i] = optical depth from top of layer i to space
    # For bottom-up: start from surface, attenuate and add layer emission

    I = planck_wn(wn_grid_ref, T_surface)  # surface / cloud emission no need for averaging here

    # layers from bottom (cloud) to top
    for i in range(n_layers):
        dtau = delta_tau_layers[i]
        # avoid overflow
        transm = np.exp(-dtau)
        B = planck_wn(wn_grid_ref, temps[i])
        # I from below, attenuated through layer, plus layer emission
        I = I * transm + B * (1.0 - transm)

    wavelengths_um = (1e4 / wn_grid_ref).astype(np.float32)
    radiance = I.astype(np.float32)

    print(f"Radiance range: {radiance.min():.3e} – {radiance.max():.3e}")
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

    df_atm = pd.read_csv(args.atmosphere)

    for scen_str in args.scenarios:
        scenario = parse_scenario(scen_str)
        out_path = out_dir / f"{scen_str}.pkl"
        # if out_path.exists():
        #     print(f"Skipping existing {out_path.name}")
        #     continue

        print(f"\n=== Scenario {scen_str} ===")

        if args.ref_therm.lower() == "thermal":
            wl_high, radiance_high = compute_thermal_emission(
                scenario, df_atm, abs_dir, args.cloud_top
            )
        else:
            wl_high, radiance_high = compute_reflectivity(
                scenario, df_atm, abs_dir, args.cloud_top, args.albedo
            )

        scenario_dict = {}
        for R in args.resolutions:
            print(f"  Binning to R={R:.2e}")
            wl_b, rad_b = bin_spectrum_robust(wl_high, radiance_high, R)
            entry = {
                "wavelength_grid": wl_b.astype(np.float32),
                "radiance_clean": rad_b.astype(np.float32),
                "resolution": int(R),
            }
            
            # for snr in args.snrs:
            #     noisy, err = inject_poisson_noise(rad_b, snr)
            #     entry[f"radiance_snr{int(snr)}"] = noisy
            #     entry[f"error_snr{int(snr)}"] = err
                
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