#!/usr/bin/env python3
"""
Pre-compute absorption coefficients for each (molecule, isotope, altitude).
Saves compressed .npz files. Resumable (skips existing files).

Example:
  python scripts/01_precompute_abs_coefs.py \\
    --atmosphere ~/sgl_science_case/sgl_science_case/data/atmosphere_profile.csv \\
    --hapi-db ~/sgl_science_case/sgl_science_case/notebooks/HAPI_DB \\
    --out-dir ~/orcd/pool/sgl_science_case/abs_coef_cache \\
    --dwn 1e-5 \\
    --cloud-top 8.0 \\
    --molecules H2O:1 CH4:1 N2O:1
"""

from __future__ import annotations
import argparse, gc, os
from pathlib import Path
import numpy as np
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--atmosphere", required=True)
    p.add_argument("--hapi-db", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dwn", type=float, default=1e-5)
    p.add_argument("--wl-min", type=float, default=7.0)
    p.add_argument("--wl-max", type=float, default=8.5)
    p.add_argument("--cloud-top", type=float, default=0.0)
    p.add_argument("--molecules", nargs="+", default=["H2O:1", "CH4:1", "N2O:1"])
    p.add_argument("--isotope-molecule", default=None,
                   help="Molecule for iso1/iso2 ratio (e.g. CO2). None = no ratio scaling.")
    p.add_argument("--isotope-ratio", type=float, default=None,
                   help="VMR_iso1 / VMR_iso2 for --isotope-molecule (e.g. 99).")
    return p.parse_args()

# Isotope ratio handling ================================================================================================================================================

def apply_isotope_ratio_vmrs(vmr1, vmr2, ratio):
    """
    Keep total VMR fixed; set vmr1/vmr2 = ratio.
    ratio = VMR_iso1 / VMR_iso2.
    """
    total = float(vmr1) + float(vmr2)
    if total <= 0 or ratio is None or ratio <= 0:
        return float(vmr1), float(vmr2)
    v1 = total * ratio / (ratio + 1.0)
    v2 = total / (ratio + 1.0)
    return v1, v2

def ratio_tag(isotope_molecule, isotope_ratio, mol):
    if isotope_molecule and mol.upper() == isotope_molecule.upper() and isotope_ratio is not None:
        # filesystem-safe tag, e.g. _r99 or _r0.5
        return f"_r{isotope_ratio:g}"
    return ""

# =======================================================================================================================================================================

def get_molecule_id(name):
    from hapi import ISO
    for (M, I), data in ISO.items():
        if data[4].upper() == name.upper():
            return M
    raise ValueError(f"Molecule {name} not found")

def compute_absorption_coefficient(
    molecule, isotope, dwn, altitude_idx, df_atm, wn_min, wn_max,
    isotope_molecule=None, isotope_ratio=None,
):
    from hapi import absorptionCoefficient_Voigt
    M = get_molecule_id(molecule)
    table_name = f"{molecule}_iso{isotope}"

    # safe layer average (handles alt 0)
    i = altitude_idx
    j = max(i - 1, 0)

    p_atm = df_atm["PRES_mb"].iloc[i] / 1013.25
    p_below = df_atm["PRES_mb"].iloc[j] / 1013.25
    T = float(df_atm["TEMP_K"].iloc[i])
    T_below = float(df_atm["TEMP_K"].iloc[j])
    P_average = 0.5 * (p_atm + p_below)
    T_average = 0.5 * (T + T_below)

    col1 = f"{molecule}_iso1_ppmv"
    col2 = f"{molecule}_iso2_ppmv"
    col_this = f"{molecule}_iso{isotope}_ppmv"

    # default: this isotope's CSV VMR
    vmr = float(df_atm[col_this].iloc[i]) / 1e6
    vmr_below = float(df_atm[col_this].iloc[j]) / 1e6
    vmr_average = 0.5 * (vmr + vmr_below)

    # optional iso1/iso2 rebalancing for one molecule
    if (
        isotope_molecule
        and molecule.upper() == isotope_molecule.upper()
        and isotope_ratio is not None
        and col1 in df_atm.columns
        and col2 in df_atm.columns
    ):
        v1 = float(df_atm[col1].iloc[i]) / 1e6
        v2 = float(df_atm[col2].iloc[i]) / 1e6
        v1b = float(df_atm[col1].iloc[j]) / 1e6
        v2b = float(df_atm[col2].iloc[j]) / 1e6
        v1, v2 = apply_isotope_ratio_vmrs(v1, v2, isotope_ratio)
        v1b, v2b = apply_isotope_ratio_vmrs(v1b, v2b, isotope_ratio)
        if isotope == 1:
            vmr_average = 0.5 * (v1 + v1b)
        elif isotope == 2:
            vmr_average = 0.5 * (v2 + v2b)

    print(
        f" {molecule} iso{isotope} alt={altitude_idx} "
        f"T={T_average:.1f}K p={P_average:.4f} vmr={vmr_average:.3e}"
        + (f" (ratio={isotope_ratio:g})" if isotope_molecule and molecule.upper()==isotope_molecule.upper() else "")
    )

    wn, coef = absorptionCoefficient_Voigt(
        Components=[(M, isotope, vmr_average)],
        Diluent={"self": vmr_average, "air": 1.0 - vmr_average},
        SourceTables=table_name,
        WavenumberRange=(wn_min, wn_max),
        WavenumberStep=dwn,
        Environment={"T": T_average, "p": P_average},
        HITRAN_units=False,
    )
    return wn, coef

def ensure_tables(molecules_isotopes, wn_min, wn_max, hapi_db):
    from hapi import db_begin, tableList, fetch, ISO
    db_begin(hapi_db)
    existing = set(tableList())
    def mol_id(name):
        for (M, I), data in ISO.items():
            if data[4].upper() == name.upper():
                return M
        raise ValueError(name)
    for mol, iso in molecules_isotopes:
        tname = f"{mol}_iso{iso}"
        if tname in existing:
            print(f"✓ {tname} present")
            continue
        print(f"→ Fetching {tname} ...")
        fetch(tname, mol_id(mol), iso, wn_min, wn_max)

def main():

    #parse the arguments and initialize the output directories
    args = parse_args()
    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    wn_min, wn_max = 1e4 / args.wl_max, 1e4 / args.wl_min
    molecules_isotopes = [(m.split(":")[0], int(m.split(":")[1])) for m in args.molecules]

    #read in atmospheric profile table 
    df_atm = pd.read_csv(args.atmosphere)
    altitudes = df_atm.index[df_atm["ALT_km"] >= args.cloud_top].tolist()
    print(f"Layers >= {args.cloud_top} km: {len(altitudes)}")

    ensure_tables(molecules_isotopes, wn_min, wn_max, os.path.expanduser(args.hapi_db))
    
    iso_mol = args.isotope_molecule
    iso_ratio = args.isotope_ratio

    tasks = []
    for mol, iso in molecules_isotopes:
        tag = ratio_tag(iso_mol,iso_ratio,mol)
        for alt in altitudes:
            f = out_dir / f"{mol}_iso{iso}_alt{alt:03d}{tag}.npz"
            if f.exists():
                print(f"Skipping {f.name}")
            else:
                tasks.append((mol, iso, alt))
                
    print(f"Tasks remaining: {len(tasks)}")

    for i, (mol, iso, alt) in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {mol} iso{iso} alt {alt}")
        wn, coef = compute_absorption_coefficient(mol, iso, args.dwn, alt, df_atm, wn_min, wn_max, isotope_molecule = iso_mol, isotope_ratio = iso_ratio)
        print(f"Molecule {mol} iso{iso} alt{alt} min/max/mean: {np.min(coef)}/{np.max(coef)}/{np.mean(coef)}")   
        out = out_dir / f"{mol}_iso{iso}_alt{alt:03d}{tag}.npz"        
        np.savez_compressed(out,
                            wn=wn.astype(np.float32), coef=coef.astype(np.float32))
        print(f" -> saved {out.name} ({len(wn)} points)")
        del wn, coef
        gc.collect()
    print("Done.")

if __name__ == "__main__":
    main()