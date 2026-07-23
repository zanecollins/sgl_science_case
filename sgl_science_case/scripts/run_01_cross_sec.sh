#!/bin/bash
#SBATCH --job-name=sgl_cross_sec
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cross_sec-%j.out
#SBATCH --error=logs/cross_sec-%j.err

module load deprecated-modules
module load anaconda3/2022.05-x86_64
source activate sgl-science

cd ~/sgl_science_case/sgl_science_case

python scripts/01_precompute_abs_coefs.py \
  --atmosphere ~/sgl_science_case/sgl_science_case/data/atmosphere_profile_CO2.csv \
  --hapi-db ~/sgl_science_case/sgl_science_case/notebooks/HAPI_DB \\
  --out-dir ~/orcd/pool/sgl_science_case/abs_coef_cache_dwn1e-6 \
  --dwn 1e-6
  --cloud-top 8.0 \
  --molecules CO2:1 CO2:2 CH4:1 CH4:2