#!/bin/bash
#SBATCH --job-name=sgl_spectra_co2_debugging
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=logs/sgl_spectra_co2_debugging-%j.out
#SBATCH --error=logs/sgl_spectra_co2_debugging-%j.err

module load deprecated-modules
module load anaconda3/2022.05-x86_64
source activate sgl-science
cd ~/sgl_science_case/sgl_science_case

export PYTHONUNBUFFERED=1

python -u scripts/02_generate_spectra.py \
  --atmosphere ~/sgl_science_case/sgl_science_case/data/atmosphere_profile.csv \
  --abs-coef-dir ~/orcd/pool/sgl_science_case/abs_coef_cache_dwn1e-3_carbon \
  --out-dir ~/orcd/pool/sgl_science_case/spectra_dict_cache_dwn1e-3_hydrocarbs \
  --ref_therm Thermal \
  --cloud-top 0 \
  --albedo 0 \
  --resolutions  1e3 1e4 \
  --snrs  0 \
  --scenarios "CO2:1" \
  
