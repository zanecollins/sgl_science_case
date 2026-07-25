#!/usr/bin/env python3
"""
Build simple "spectra" from HITRAN absorption cross-section (.xsc) files
for cross-correlation tests (e.g. DMS, DMDS).

No atmosphere / thermal RT — just σ(ν) on a common grid, optional sums,
binning, and noise.

Example:
  python scripts/02_generate_xsc_spectra.py \
    --xsc-dir ~/Volumes/externalssd/hitran_xsc \
    --out-dir ~/data/xsc_spectra \
    --species DMS DMDS \
    --combine DMS+DMDS \
    --resolutions 1000 10000 100000 \
    --snrs 5 10 25 50 \
    --wl-min 7.0 --wl-max 8.5
"""

from __future__ import annotations
import argparse, gc, os, pickle, re
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d


# Map friendly names -> tokens that appear in HITRAN filenames
SPECIES_ALIASES = {
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
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--xsc-dir", required=True, help="Folder of *.xsc / *.xsc.txt files")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--species", nargs="+", required=True,
                   help="Friendly names or filename tokens, e.g. DMS DMDS")
    p.add_argument("--combine", nargs="*", default=[],
                   help="Optional sums, e.g. DMS+DMDS")
    p.add_argument("--resolutions", nargs="+", type=float, default=[1e3, 1e4, 1e5])
    p.add_argument("--snrs", nargs="+", type=float, default=[5, 10, 25, 50])
    p.add_argument("--wl-min", type=float, default=None, help="µm; default = data range")
    p.add_argument("--wl-max", type=float, default=None, help="µm")
    p.add_argument("--scale", type=float, default=1.0,
                   help="Multiply all σ by this (proxy column density)")
    return p.parse_args()


def parse_hitran_xsc(path: Path):
    """
    HITRAN IR cross-section format:
      line 1: name  νmin  νmax  npoints  T  P  ...
      rest:   xsec values (cm^2/molecule), row-wrapped
    Returns wn [cm^-1], xsec
    """
    lines = path.read_text(errors="replace").splitlines()
    if not lines:
        raise ValueError(f"Empty file: {path}")

    header = lines[0].split()
    # Find first three floats: numin, numax, npoints
    floats = []
    for tok in header:
        try:
            floats.append(float(tok))
        except ValueError:
            continue
        if len(floats) >= 3:
            break
    if len(floats) < 3:
        raise ValueError(f"Could not parse header in {path}: {lines[0][:120]}")

    numin, numax, npoints = floats[0], floats[1], int(floats[2])

    vals = []
    for line in lines[1:]:
        for tok in line.split():
            try:
                vals.append(float(tok))
            except ValueError:
                pass

    xsec = np.asarray(vals, dtype=np.float64)
    if xsec.size < npoints:
        raise ValueError(f"{path.name}: expected {npoints} points, got {xsec.size}")
    xsec = xsec[:npoints]

    wn = np.linspace(numin, numax, npoints)
    return wn, xsec

def find_xsc_files(xsc_dir: Path, species_token: str):
    # case-insensitive alias lookup
    alias_map = {k.upper(): v for k, v in SPECIES_ALIASES.items()}
    aliases = alias_map.get(species_token.upper(), [species_token])
    alias_stems = {a.lower() for a in aliases}

    files = []
    for f in sorted(xsc_dir.iterdir()):
        if not f.is_file() or f.name.startswith("._") or f.name.startswith("."):
            continue
        name = f.name
        if not (name.endswith(".xsc") or name.endswith(".xsc.txt") or name.endswith(".txt")):
            continue

        stem = name.split("_")[0]  # e.g. (C2H5O)2SO2  or  C2H6S
        if stem.lower() in alias_stems:
            files.append(f)

    return files


def pick_one_xsc(files, prefer_T=298.0):
    """If multiple T/P files exist, pick closest to prefer_T if T is in the name."""
    if not files:
        return None
    if len(files) == 1:
        return files[0]

    def score(p: Path):
        m = re.search(r"(\d+\.?\d*)K", p.name)
        if not m:
            return abs(0 - prefer_T)
        return abs(float(m.group(1)) - prefer_T)

    return sorted(files, key=score)[0]


def to_common_grid(spectra: dict, wl_min=None, wl_max=None, n_grid=200000):
    """
    spectra: name -> (wn_cm, xsec)
    Returns wl_um grid and dict name -> xsec on that grid.
    """
    # work in wavenumber, then convert
    wn_lo, wn_hi = [], []
    for wn, _ in spectra.values():
        wn_lo.append(wn.min())
        wn_hi.append(wn.max())
    wn_min = max(wn_lo)
    wn_max = min(wn_hi)
    if wn_min >= wn_max:
        raise ValueError("Cross sections have no overlapping wavenumber range")

    # optional wavelength window
    if wl_min is not None:
        wn_max = min(wn_max, 1e4 / wl_min)
    if wl_max is not None:
        wn_min = max(wn_min, 1e4 / wl_max)
    if wn_min >= wn_max:
        raise ValueError("Requested wavelength window has no overlap with data")

    wn_grid = np.linspace(wn_min, wn_max, n_grid)
    # guard zeros
    wn_grid = wn_grid[wn_grid > 0]
    wl_grid = 1e4 / wn_grid
    order = np.argsort(wl_grid)
    wl_grid, wn_grid = wl_grid[order], wn_grid[order]

    out = {}
    for name, (wn, xsec) in spectra.items():
        f = interp1d(wn, xsec, kind="linear", bounds_error=False, fill_value=0.0)
        out[name] = f(wn_grid).astype(np.float64)
    return wl_grid.astype(np.float64), out


def bin_spectrum_robust(wl_native, spectrum_native, R_bin):
    wl_native = np.asarray(wl_native, dtype=np.float64)
    spectrum_native = np.asarray(spectrum_native, dtype=np.float64)
    if wl_native[0] > wl_native[-1]:
        wl_native = wl_native[::-1]
        spectrum_native = spectrum_native[::-1]

    native_R = np.median(wl_native[:-1] / np.diff(wl_native))
    # print(native_R)
    if R_bin >= 0.95 * native_R:
        return wl_native, spectrum_native

    log_wl_min, log_wl_max = np.log(wl_native[0]), np.log(wl_native[-1])
    n_bins = int(np.floor((log_wl_max - log_wl_min) * R_bin)) + 1
    n_bins = max(2, min(n_bins, len(wl_native) - 2))
    wl_binned = np.exp(np.linspace(log_wl_min, log_wl_max, n_bins))
    interp = interp1d(wl_native, spectrum_native, kind="linear",
                      bounds_error=False, fill_value=0.0)
    return wl_binned, interp(wl_binned)


def inject_noise_continuum(spec, snr, seed=42):
    np.random.seed(seed)
    spec = np.asarray(spec, dtype=float)
    continuum = np.nanmedian(np.abs(spec)) + 1e-30
    noise_std = continuum / snr
    noisy = spec + np.random.normal(0.0, noise_std, size=spec.shape)
    return noisy.astype(np.float32), np.full_like(spec, noise_std, dtype=np.float32)


def main():
    args = parse_args()
    xsc_dir = Path(os.path.expanduser(args.xsc_dir))
    out_dir = Path(os.path.expanduser(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load requested species ---
    loaded = {}
    for sp in args.species:
        files = find_xsc_files(xsc_dir, sp)
        if not files:
            print(f"WARNING: no XSC files matched {sp!r} in {xsc_dir}")
            print("  Available examples:", [f.name for f in list(xsc_dir.iterdir())[:10]])
            continue
        chosen = pick_one_xsc(files)
        print(f"{sp}: using {chosen.name}")
        wn, xsec = parse_hitran_xsc(chosen)
        loaded[sp] = (wn, xsec * args.scale)

    if not loaded:
        raise SystemExit("No species loaded.")

    wl_grid, xsec_on_grid = to_common_grid(
        loaded, wl_min=args.wl_min, wl_max=args.wl_max
    )

    # optional combinations
    for combo in args.combine:
        parts = [p.strip() for p in combo.split("+") if p.strip()]
        if not all(p in xsec_on_grid for p in parts):
            print(f"Skipping combine {combo}: missing species")
            continue
        xsec_on_grid[combo] = sum(xsec_on_grid[p] for p in parts)

    # --- bin + noise + save ---
    for name, xsec in xsec_on_grid.items():
        scenario_dict = {}
        for R in args.resolutions:
            wl_b, y_b = bin_spectrum_robust(wl_grid, xsec, R)
            entry = {
                "wavelength_grid": wl_b.astype(np.float32),
                "xsec_clean": y_b.astype(np.float32),
                "resolution": int(R),
            }
            for snr in args.snrs:
                noisy, err = inject_noise_continuum(y_b, snr)
                entry[f"xsec_snr{int(snr)}"] = noisy
                entry[f"error_snr{int(snr)}"] = err
            scenario_dict[int(R)] = entry
            del wl_b, y_b
            gc.collect()

                # short names for big combinations
        if name.count("+") >= 3:
            if name.startswith("DMS+"):
                safe = "ALL_SULFUR_WITH_DMS"
            else:
                safe = "ALL_SULFUR_NO_DMS"
        else:
            safe = name.replace("/", "_").replace(" ", "")

        out_path = out_dir / f"{safe}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(scenario_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()