#!/bin/bash
#SBATCH --job-name=sgl_cross_sec_carbons_dwn1e-3_CO2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sgl_cross_sec_carbons_dwn1e-3_CO2-%j.out
#SBATCH --error=logs/sgl_cross_sec_carbons_dwn1e-3_CO2-%j.err

module load deprecated-modules
module load anaconda3/2022.05-x86_64
source activate sgl-science

cd ~/sgl_science_case/sgl_science_case

python scripts/01_precompute_abs_coefs.py \
  --atmosphere ~/sgl_science_case/sgl_science_case/data/atmosphere_profile.csv \
  --hapi-db ~/sgl_science_case/sgl_science_case/notebooks/HAPI_DB \
  --out-dir ~/orcd/pool/sgl_science_case/abs_coef_cache_dwn1e-3_carbon \
  --dwn 1e-3 \
  --wl-min 13.5 \
  --wl-max 17 \
  --cloud-top 0 \
  --molecules  CO2:1