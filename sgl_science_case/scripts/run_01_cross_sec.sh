#!/bin/bash
#SBATCH --job-name=sgl_cross_sec_co2_iso2_1e-4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cross_sec_CO2_iso2_1e-4-%j.out
#SBATCH --error=logs/cross_sec_CO2_iso2_1e-4-%j.err

module load deprecated-modules
module load anaconda3/2022.05-x86_64
source activate sgl-science

cd ~/sgl_science_case/sgl_science_case

python scripts/01_precompute_abs_coefs.py \
  --atmosphere ~/sgl_science_case/sgl_science_case/data/atmosphere_profile.csv \
  --hapi-db ~/sgl_science_case/sgl_science_case/notebooks/HAPI_DB \
  --out-dir ~/orcd/pool/sgl_science_case/abs_coef_cache_dwn1e-4_carbon \
  --dwn 1e-4 \
  --wl-min 0.5 \
  --wl-max 20 \
  --cloud-top 0 \
  --molecules CO2:2 \