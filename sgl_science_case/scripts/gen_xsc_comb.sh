#!/usr/bin/env bash
# generate_sulfur_xsc_spectra.sh
# Local run: build XSC "spectra" for HITRAN sulfur species (except SO2, SF5CF3)

set -euo pipefail

# ---------- edit these paths ----------
XSC_DIR="${XSC_DIR:-data/external/sulfur_abs_cross_sec_hitran}"
OUT_DIR="${OUT_DIR:-data/xsc_spectra}"
SCRIPT="${SCRIPT:-scripts/sulfur_abscoef_test.py}"

# Optional wavelength window (µm); leave empty for full overlap of files
WL_MIN="${WL_MIN:-}"
WL_MAX="${WL_MAX:-}"

RESOLUTIONS=(1000 10000 10000 20000)
SNRS=(5 10 25 50)

# All sulfur XSC species from HITRAN list except SO2 and SF5CF3
SPECIES=(
  DMS DMDS DiethylSulfate Tetrahydrothiophene
  2-Propanethiol 2-Methyl-1-propanethiol tert-Butylmercaptan
  DiethylSulfide MethylIsothiocyanate DimethylSulfate DMSO
  PropyleneSulfide Thiophene Cyclohexanethiol Benzenethiol
  Thiophosgene PerchloromethylMercaptan EthyleneSulfide
  1-Propanethiol EthylMercaptan MethanesulfonylChloride Methanethiol
  CS2 Thioglycol SF6 SO2Cl2 SO2F2 SOF2 SPCl3
)

COMBINE=(
  "DMS+CS2+SF6+Methanethiol+DMSO"
  "CS2+SF6+Methanethiol+DMSO"
  "DMS+DMDS+CS2+SF6"
  "DMDS+CS2+SF6"
)

mkdir -p "$OUT_DIR"

if [[ ! -d "$XSC_DIR" ]]; then
  echo "ERROR: XSC dir not found: $XSC_DIR"
  exit 1
fi

if [[ ! -f "$SCRIPT" ]]; then
  echo "ERROR: script not found: $SCRIPT"
  exit 1
fi

echo "XSC_DIR = $XSC_DIR"
echo "OUT_DIR = $OUT_DIR"
echo "Species count = ${#SPECIES[@]}"

python "$SCRIPT" \
  --xsc-dir "$XSC_DIR" \
  --out-dir "$OUT_DIR" \
  --species "${SPECIES[@]}" \
  --combine "${COMBINE[@]}" \
  --resolutions "${RESOLUTIONS[@]}" \
  --snrs "${SNRS[@]}"

echo "Done. Outputs in: $OUT_DIR"
ls -lh "$OUT_DIR" | head