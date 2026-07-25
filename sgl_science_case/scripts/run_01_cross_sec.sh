#!/bin/bash
#SBATCH --job-name=sgl_cross_sec_Earth_like_dwn1e-6_7to8.5
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sgl_cross_sec_Earth_like_dwn1e-6_7to8.5-%j.out
#SBATCH --error=logs/sgl_cross_sec_Earth_like_dwn1e-6_7to8.5-%j.err

module load deprecated-modules
module load anaconda3/2022.05-x86_64
source activate sgl-science

cd ~/sgl_science_case/sgl_science_case

python scripts/01_precompute_abs_coefs.py \
  --atmosphere ~/sgl_science_case/sgl_science_case/data/atmosphere_profile.csv \
  --hapi-db ~/sgl_science_case/sgl_science_case/notebooks/HAPI_DB \
  --out-dir ~/orcd/pool/sgl_science_case/abs_coef_cache_dwn1e-6_earth \
  --dwn 1e-6 \
  --wl-min 7 \
  --wl-max 8.5 \
  --cloud-top 0 \
  --molecules  H2O:1 CH4:1 N2O:1\