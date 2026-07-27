#!/usr/bin/env python3
"""
Generate atmospheric spectra using HITRAN absorption cross-section (.xsc) files
(single-T approximation) + layered Earth-like profile.

Example:
  python scripts/02_generate_spectra_xsc.py \
    --atmosphere data/atmosphere_profile.csv \
    --xsc-dir data/external/hydrocarbon_abs_cross_sec_hitran \
    --out-dir data/spectra_xsc_atm \
    --ref_therm Thermal \
    --cloud-top 0.0 \
    --resolutions 1000 10000 100000 \
    --snrs 5 10 25 50 \
    --scenarios Isoprene+Propene+Butadiene  Propene+Butadiene
"""

from __future__ import annotations
import argparse, gc, os, re, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Friendly name -> filename stem(s) before first "_"
SPECIES_ALIASES = {
    "Isoprene": ["C5-H8", "C5H8"],
    "Butadiene": ["C4H6"],
    "Propene": ["C3H6"],
    "Butene": ["C4H8"],
    "1-Butyne": ["HC=CCH2CH3"],
    "Limonene": ["C10H16", "C10-H16"],
    "Pinene": ["C10H16", "C10-H16"],
    "Benzene": ["C6H6"],
    "Toluene": ["C6H5CH3"],
    "Trimethylbenzene": ["C6H3(CH3)3"],
    "Tetramethylbenzene": ["(C6H2)(CH3)4"],
    "1-Decene": ["CH2CH(CH2)7CH3"],
    # add sulfur / others as needed
    "DMS": ["C2H6S"],
    "DMDS": ["C2H6S2"],
    "CS2": ["CS2"],
    "SF6": ["SF6"],
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atmosphere", required=True)
    p.add_argument("--xsc-dir", required=True, help="Folder of HITRAN .xsc.txt files")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ref_therm", required=True, choices=["Thermal", "Reflectivity", "thermal", "reflectivity"])
    p.add_argument("--cloud-top", type=float, default=0.0)
    p.add_argument("--albedo", type=float, default=0.3)
    p.add_argument("--resolutions", nargs="+", type=float, default=[1e3, 1e4, 1e5])
    p.add_argument("--snrs", nargs="+", type=float, default=[5, 10, 25, 50])
    p.add_argument("--scenarios", nargs="+", required=True,
                   help="e.g. Isoprene+Propene+Butadiene  Propene+Butadiene")
    p.add_argument("--default-ppmv", type=float, default=1.0,
                   help="If SPECIES_ppmv column missing, fill with this constant")
    p.add_argument("--scale", type=float, default=1.0, help="Global multiplier on all σ")
    return p.parse_args()


# ---------------------------------------------------------------------------
# XSC I/O
# ---------------------------------------------------------------------------

def parse_hitran_xsc(path: Path):
    """HITRAN XSC: header has νmin, νmax, npoints; body is σ values only."""
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
    if xsec.size < npoints:
        raise ValueError(f"{path.name}: expected {npoints} pts, got {xsec.size}")
    wn = np.linspace(numin, numax, npoints)
    return wn, xsec


def find_xsc_file(xsc_dir: Path, species: str) -> Path:
    alias_map = {k.upper(): v for k, v in SPECIES_ALIASES.items()}
    aliases = alias_map.get(species.upper(), [species])
    stems = {a.lower() for a in aliases}

    matches = []
    for f in sorted(xsc_dir.iterdir()):
        if not f.is_file() or f.name.startswith("._") or f.name.startswith("."):
            continue
        if not (f.name.endswith(".xsc") or f.name.endswith(".xsc.txt") or f.name.endswith(".txt")):
            continue
        stem = f.name.split("_")[0]
        if stem.lower() in stems:
            matches.append(f)

    if not matches:
        raise FileNotFoundError(f"No XSC for {species!r} in {xsc_dir} (stems tried: {stems})")

    # prefer ~298 K if multiple
    def score(p: Path):
        m = re.search(r"(\d+\.?\d*)K", p.name)
        return abs(float(m.group(1)) - 298.0) if m else 999.0

    return sorted(matches, key=score)[0]


def load_species_xsec(xsc_dir: Path, species_list, scale=1.0):
    """Load each species once; interpolate onto common overlapping wn grid."""
    raw = {}
    for sp in species_list:
        path = find_xsc_file(xsc_dir, sp)
        wn, xsec = parse_hitran_xsc(path)
        raw[sp] = (wn, xsec * scale)
        print(f"  {sp}: {path.name}  ({len(xsec)} pts, "
              f"{wn.min():.1f}–{wn.max():.1f} cm^-1)")

    wn_min = max(wn.min() for wn, _ in raw.values())
    wn_max = min(wn.max() for wn, _ in raw.values())
    if wn_min >= wn_max:
        raise ValueError("No overlapping wavenumber range among species")

    # ~0.01 cm^-1 sampling over overlap (adjust if memory-heavy)
    n = int(min(500_000, max(10_000, (wn_max - wn_min) / 0.05)))
    wn_grid = np.linspace(wn_min, wn_max, n)

    xsec_grid = {}
    for sp, (wn, xsec) in raw.items():
        xsec_grid[sp] = interp1d(wn, xsec, kind="linear",
                                 bounds_error=False, fill_value=0.0)(wn_grid)

    return wn_grid, xsec_grid


# ---------------------------------------------------------------------------
# Atmosphere helpers
# ---------------------------------------------------------------------------

def ensure_vmr_columns(df: pd.DataFrame, species_list, default_ppmv: float):
    """Add SPECIES_ppmv columns if missing (constant default)."""
    df = df.copy()
    for sp in species_list:
        col = f"{sp}_ppmv"
        if col not in df.columns:
            print(f"  WARNING: no {col} — filling with {default_ppmv} ppmv")
            df[col] = default_ppmv
    return df


def parse_scenario(s: str):
    """'Isoprene+Propene+Butadiene' -> ['Isoprene','Propene','Butadiene']"""
    parts = [p.strip() for p in s.split("+") if p.strip()]
    # strip optional :iso
    out = []
    for p in parts:
        out.append(p.split(":")[0].strip())
    return out


# ---------------------------------------------------------------------------
# Radiative transfer
# ---------------------------------------------------------------------------

def planck_wn(wn_cm, T):
    wn = np.asarray(wn_cm, dtype=np.float64)
    c1 = 1.191042972e-5
    c2 = 1.4387769
    return c1 * wn**3 / np.expm1(c2 * wn / T)

def layer_delta_tau(wn_grid, xsec_grid, species_list, above_df, dz_cm):
    """
    Δτ[layer, wn] = Σ_sp σ_sp(wn) * n_air * VMR_sp * Δz
    Single-T σ for all layers (approximation).
    """
    n_layers = len(above_df)
    n_wn = len(wn_grid)
    delta_tau = np.zeros((n_layers, n_wn), dtype=np.float64)

    density = above_df["DENSITY_cm3"].astype(float).values  # cm^-3

    for sp in species_list:
        col = f"{sp}_ppmv"
        ppmv = above_df[col].astype(float).values
        sigma = xsec_grid[sp]  # cm^2 / molecule

        for i in range(n_layers):
            if i < n_layers - 1:
                ppmv_avg = 0.5 * (ppmv[i] + ppmv[i + 1])
                n_avg = 0.5 * (density[i] + density[i + 1])
            else:
                ppmv_avg = ppmv[i]
                n_avg = density[i]
            vmr = ppmv_avg * 1e-6
            delta_tau[i] += sigma * n_avg * vmr * dz_cm[i]

    return delta_tau


def compute_thermal_emission(species_list, df_atm, wn_grid, xsec_grid, cloud_top=0.0):
    above = df_atm[df_atm["ALT_km"] >= cloud_top].reset_index(drop=True)
    print(f"Thermal RT above {cloud_top} km ({len(above)} layers)")

    alt_km = above["ALT_km"].values
    dz_km = np.diff(alt_km, append=alt_km[-1] + np.median(np.diff(alt_km)))
    dz_cm = dz_km * 1e5
    temps = above["TEMP_K"].values
    T_surface = float(temps[0])

    dtau = layer_delta_tau(wn_grid, xsec_grid, species_list, above, dz_cm)
    n_layers, n_wn = dtau.shape

    # τ from top of layer i to TOA
    tau_above = np.zeros_like(dtau)
    for i in range(n_layers - 2, -1, -1):
        tau_above[i] = tau_above[i + 1] + dtau[i + 1]

    tau_tot = dtau.sum(axis=0)
    # thin-layer form you preferred: I = B_s e^{-τ_tot} + Σ B_i e^{-τ_above,i} Δτ_i
    I = planck_wn(wn_grid, T_surface) * np.exp(-np.minimum(tau_tot, 50.0))
    for i in range(n_layers):
        B_i = planck_wn(wn_grid, temps[i])
        I += B_i * np.exp(-np.minimum(tau_above[i], 50.0)) * dtau[i]

    wl = (1e4 / wn_grid).astype(np.float32)
    # sort by increasing wavelength
    order = np.argsort(wl)
    print(f"  max τ={tau_tot.max():.3e}  radiance {I.min():.3e}–{I.max():.3e}")
    return wl[order], I[order].astype(np.float32)


def compute_reflectivity(species_list, df_atm, wn_grid, xsec_grid, cloud_top, albedo):
    above = df_atm[df_atm["ALT_km"] >= cloud_top].reset_index(drop=True)
    print(f"Reflectivity above {cloud_top} km ({len(above)} layers)")

    alt_km = above["ALT_km"].values
    dz_km = np.diff(alt_km, append=alt_km[-1] + np.median(np.diff(alt_km)))
    dz_cm = dz_km * 1e5

    dtau = layer_delta_tau(wn_grid, xsec_grid, species_list, above, dz_cm)
    tau_tot = dtau.sum(axis=0)
    tau_rt = 2.0 * tau_tot
    refl = (albedo * np.exp(-np.minimum(tau_rt, 50.0))).astype(np.float32)

    wl = (1e4 / wn_grid).astype(np.float32)
    order = np.argsort(wl)
    print(f"  max τ_rt={tau_rt.max():.3e}  refl {refl.min():.4f}–{refl.max():.4f}")
    return wl[order], refl[order]


def bin_spectrum_robust(wl_native, spectrum_native, R_bin):
    wl = np.asarray(wl_native, dtype=np.float64)
    sp = np.asarray(spectrum_native, dtype=np.float64)
    if wl[0] > wl[-1]:
        wl, sp = wl[::-1], sp[::-1]

    native_R = np.median(wl[:-1] / np.diff(wl))
    if R_bin >= 0.95 * native_R:
        return wl, sp

    log_a, log_b = np.log(wl[0]), np.log(wl[-1])
    n_bins = int(np.floor((log_b - log_a) * R_bin)) + 1
    n_bins = max(2, min(n_bins, len(wl) - 2))
    wl_b = np.exp(np.linspace(log_a, log_b, n_bins))
    sp_b = interp1d(wl, sp, kind="linear", bounds_error=False, fill_value=0.0)(wl_b)
    return wl_b, sp_b


def inject_noise(spec, snr, seed=42):
    np.random.seed(seed)
    spec = np.asarray(spec, dtype=float)
    continuum = np.nanmedian(np.abs(spec)) + 1e-30
    noise_std = continuum / snr
    noisy = spec + np.random.normal(0.0, noise_std, size=spec.shape)
    return noisy.astype(np.float32), np.full_like(spec, noise_std, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    xsc_dir = Path(os.path.expanduser(args.xsc_dir))
    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    df_atm = pd.read_csv(args.atmosphere)
    # drop empty trailing columns if present
    df_atm = df_atm.loc[:, ~df_atm.columns.str.match(r"^Unnamed")]

    # union of all species across scenarios (load XSC once)
    all_species = []
    parsed = {}
    for s in args.scenarios:
        parsed[s] = parse_scenario(s)
        for sp in parsed[s]:
            if sp not in all_species:
                all_species.append(sp)

    print("Loading XSC library...")
    wn_grid, xsec_grid = load_species_xsec(xsc_dir, all_species, scale=args.scale)

    df_atm = ensure_vmr_columns(df_atm, all_species, args.default_ppmv)

    for scen_str, species_list in parsed.items():
        print(f"\n=== Scenario {scen_str} ===")
        mode = args.ref_therm.lower()
        if mode == "thermal":
            wl_high, rad_high = compute_thermal_emission(
                species_list, df_atm, wn_grid, xsec_grid, args.cloud_top
            )
        else:
            wl_high, rad_high = compute_reflectivity(
                species_list, df_atm, wn_grid, xsec_grid, args.cloud_top, args.albedo
            )

        scenario_dict = {}
        for R in args.resolutions:
            print(f"  Binning R={R:.2e}")
            wl_b, rad_b = bin_spectrum_robust(wl_high, rad_high, R)
            entry = {
                "wavelength_grid": wl_b.astype(np.float32),
                "radiance_clean": rad_b.astype(np.float32),
                "resolution": int(R),
            }
            for snr in args.snrs:
                noisy, err = inject_noise(rad_b, snr)
                entry[f"radiance_snr{int(snr)}"] = noisy
                entry[f"error_snr{int(snr)}"] = err
            scenario_dict[int(R)] = entry

        # short filename for long combines
        if scen_str.lower().startswith("isoprene+"):
            safe = "ALL_WITH_ISOPRENE"
        elif "Isoprene" not in species_list and scen_str.count("+") >= 2:
            safe = "ALL_WITHOUT_ISOPRENE"
        else:
            safe = scen_str.replace("/", "_").replace(" ", "")
            if len(safe) > 80:
                safe = "combo_" + str(abs(hash(scen_str)) % 10_000_000)

        out_path = out_dir / f"{safe}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(scenario_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved → {out_path}")
        del wl_high, rad_high, scenario_dict
        gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()