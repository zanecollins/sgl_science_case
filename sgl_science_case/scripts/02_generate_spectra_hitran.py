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
import re
from scipy.interpolate import interp1d

SPECIES_ALIASES = {
    # Hydrocarbons


    # Sulfur Containing
    "DMS": ["C2H6S"],
    "DMDS": ["C2H6S2"],
    "DiethylSulfate": ["(C2H5O)2SO2"],
    "Tetrahydrothiophene": ["(CH2)4S"],
    "2-Propanethiol": ["(CH3)2CH(HS)"],
    "2-Methyl-1-propanethiol": ["(CH3)2CHCH2SH"],
    "tert-Butylmercaptan": ["(CH3)3CSH"],
    "DiethylSulfide": ["(CH3CH2)2S"],
    "MethylIsothiocyanate": ["C2H3NS"],
    "DimethylSulfate": ["C2H6O4S"],
    "DMSO": ["C2H6OS"],
    "PropyleneSulfide": ["C3H6S"],
    "Thiophene": ["C4H4S"],
    "Cyclohexanethiol": ["C6H11SH"],
    "Benzenethiol": ["C6H5SH"],
    "Thiophosgene": ["CCl2S"],
    "PerchloromethylMercaptan": ["CCl3SCl"],
    "EthyleneSulfide": ["CH2CH2S"],
    "1-Propanethiol": ["CH3(CH2)2SH"],
    "EthylMercaptan": ["CH3CH2SH"],
    "MethanesulfonylChloride": ["CH3SO2Cl"],
    "Methanethiol": ["CH4S"],
    "CS2": ["CS2"],
    "Thioglycol": ["HS(CH2)2OH"],
    "SF6": ["SF6"],
    "SO2Cl2": ["SO2Cl2"],
    "SO2F2": ["SO2F2"],
    "SOF2": ["SOF2"],
    "SPCl3": ["SPCl3"],

    # Target
    "Isoprene": ["C5-H8", "C5H8"],

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


def find_xsc_file(xsc_dir: Path, mol: str) -> Path:
    alias_map = {k.upper(): v for k, v in SPECIES_ALIASES.items()}
    stems = {a.lower() for a in alias_map.get(mol.upper(), [mol])}

    matches = []
    for f in sorted(Path(xsc_dir).iterdir()):
        if f.name.startswith("._") or not f.is_file():
            continue
        if not (f.name.endswith(".txt") or f.name.endswith(".xsc")):
            continue
        if f.name.split("_")[0].lower() in stems:
            matches.append(f)
    if not matches:
        raise FileNotFoundError(f"No XSC for {mol!r} in {xsc_dir}")

    def score(p):
        m = re.search(r"(\d+\.?\d*)K", p.name)
        return abs(float(m.group(1)) - 298.0) if m else 999.0
    return sorted(matches, key=score)[0]


# module-level cache so each molecule is read once
_XSEC_CACHE = {}

def load_abs_coef(mol, iso, alt, cache_dir):
    """
    Drop-in replacement: ignore iso and alt.
    Returns (wn, sigma) from HITRAN XSC [cm^2 / molecule], single T.
    """
    key = mol.upper()
    if key not in _XSEC_CACHE:
        path = find_xsc_file(Path(cache_dir), mol)
        wn, xsec = parse_hitran_xsc(path)
        _XSEC_CACHE[key] = (wn, xsec)
        print(f"Loaded XSC {mol}: {path.name}")
    return _XSEC_CACHE[key]

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
            return wl_native, spectrum_native#, np.asarray(err_data)
        return wl_native, spectrum_native#, None

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

def compute_thermal_emission(molecules_isotopes, df_atm, abs_coef_dir, cloud_top=0.0, T_surface=None):
    above_df = df_atm[df_atm["ALT_km"] >= cloud_top].copy().reset_index(drop=True) # From the dataframe of Temperature, Pressure, and VMRs of molecules at Altitude, take the values above input cloud top altitude
    print(f"Computing thermal emission above {cloud_top} km ({len(above_df)} layers)")

    alt_km = above_df["ALT_km"].values
    dz_km = np.diff(alt_km, append=alt_km[-1] + np.median(np.diff(alt_km)))
    dz_cm = dz_km * 1e5 # Convert from km layer width to cm
    temps = above_df["TEMP_K"].values
    density = above_df["DENSITY_cm3"].astype(float).values

    if T_surface is None:
        T_surface = float(temps[0])

    # --- 1) load all XSCs ---
    raw = {}
    for mol, iso in molecules_isotopes:
        wn, coef = load_abs_coef(mol, iso, 0, abs_coef_dir) # Returns (wn, sigma) from HITRAN XSC [cm^2 / molecule]
        raw[mol] = (wn, coef)

    # --- 2) common overlapping grid (take only the range of wavenumbers that have overlap with all molecule cross sections)---
    wn_min = max(wn.min() for wn, _ in raw.values())
    wn_max = min(wn.max() for wn, _ in raw.values())
    n_grid = int(min(200_000, max(20_000, (wn_max - wn_min) / 0.05)))
    wn_grid_ref = np.linspace(wn_min, wn_max, n_grid)

    xsec = {}
    for mol, (wn, coef) in raw.items():
        xsec[mol] = interp1d(wn, coef, kind="linear",
                             bounds_error=False, fill_value=0.0)(wn_grid_ref)

    # --- 3) Δτ on common grid ---
    n_layers = len(above_df)
    delta_tau_layers = np.zeros((n_layers, len(wn_grid_ref)))

    for mol, iso in molecules_isotopes:
        col = f"{mol}_ppmv"   # take column of VMR for molecule {mol}
        if col not in above_df.columns:
            print(f"ERROR: missing {col}")
            continue
        ppmv_col = above_df[col].astype(float).values
        sigma = xsec[mol]

        # This block takes the average between the two adjacent layers
        for i in range(n_layers):
            if i < n_layers - 1:
                ppmv_avg = 0.5 * (ppmv_col[i] + ppmv_col[i + 1])
                n_avg = 0.5 * (density[i] + density[i + 1])
            else:
                ppmv_avg = ppmv_col[i]
                n_avg = density[i]
            vmr = ppmv_avg * 1e-6
            # σ [cm²] * n [cm⁻³] * VMR * dz [cm]
            delta_tau_layers[i] += sigma * n_avg * vmr * dz_cm[i] # For this layer, add the optical depth from each molecule

    # --- 4) thermal sum (same as before) ---
    tau_above = np.zeros_like(delta_tau_layers)
    for i in range(n_layers - 2, -1, -1): # Computes the optical depth above each layer
        tau_above[i] = tau_above[i + 1] + delta_tau_layers[i + 1]

    tau_tot = delta_tau_layers.sum(axis=0)#Compute total optical depth
    I = planck_wn(wn_grid_ref, T_surface) * np.exp(-np.minimum(tau_tot, 50.0)) #Surface intensity contribution
    for i in range(n_layers): #Second term in derived intensity (derived with Sara): I_\lambda = B(T_\text{Surface})\cdot e^{-\tau_\text{tot}} + \sum_i^{N} B(T_i) e^{-\tau_\text{i}} \cdot\Delta \tau
        I += planck_wn(wn_grid_ref, temps[i]) * np.exp(-tau_above[i],) * (1-delta_tau_layers[i])

    wl = (1e4 / wn_grid_ref).astype(np.float32)
    order = np.argsort(wl)
    return wl[order], I[order].astype(np.float32)

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