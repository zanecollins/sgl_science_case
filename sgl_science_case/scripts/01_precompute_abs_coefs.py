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
    return p.parse_args()

def get_molecule_id(name):
    from hapi import ISO
    for (M, I), data in ISO.items():
        if data[4].upper() == name.upper():
            return M
    raise ValueError(f"Molecule {name} not found")

def compute_absorption_coefficient(molecule, isotope, dwn, altitude_idx, df_atm, wn_min, wn_max):
    from hapi import absorptionCoefficient_Voigt #Computes absorption cross sections through Hapi

    # Retrieve molecule id and table of linelists
    M = get_molecule_id(molecule)
    table_name = f"{molecule}_iso{isotope}"

    #From atmosphere profile table, retrive pressure, vmr, and temperature at input layer
    p_atm = df_atm["PRES_mb"].iloc[altitude_idx] / 1013.25
    vmr = df_atm[f"{molecule}_iso{isotope}_ppmv"].iloc[altitude_idx] / 1e6
    T = df_atm["TEMP_K"].iloc[altitude_idx]
    print(f"  {molecule} iso{isotope} alt={altitude_idx} T={T:.1f}K p={p_atm:.4f} vmr={vmr:.3e}")

    #Compute absorption coefficient - this is the most computationally expensive.
    wn, coef = absorptionCoefficient_Voigt(
        Components=[(M, isotope, vmr)],
        Diluent={"self": vmr, "air": 1.0 - vmr},
        SourceTables=table_name,
        WavenumberRange=(wn_min, wn_max),
        WavenumberStep=dwn,
        Environment={"T": T, "p": p_atm},
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

    tasks = []
    for mol, iso in molecules_isotopes:
        for alt in altitudes:
            f = out_dir / f"{mol}_iso{iso}_alt{alt:03d}.npz"
            if f.exists():
                print(f"Skipping {f.name}")
            else:
                tasks.append((mol, iso, alt))
    print(f"Tasks remaining: {len(tasks)}")

    for i, (mol, iso, alt) in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {mol} iso{iso} alt {alt}")
        wn, coef = compute_absorption_coefficient(mol, iso, args.dwn, alt, df_atm, wn_min, wn_max)
        np.savez_compressed(out_dir / f"{mol}_iso{iso}_alt{alt:03d}.npz",
                            wn=wn.astype(np.float32), coef=coef.astype(np.float32))
        print(f"  → saved ({len(wn)} points)")
        del wn, coef
        gc.collect()
    print("Done.")

if __name__ == "__main__":
    main()