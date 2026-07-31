#!/bin/bash
#SBATCH --job-name=sgl_cross_sec_carbons_dwn1e-4_hydrocarbons_2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sgl_cross_sec_carbons_dwn1e-4_hcs_2-%j.out
#SBATCH --error=logs/sgl_cross_sec_carbons_dwn1e-4_hcs_2-%j.err

module load deprecated-modules
module load anaconda3/2022.05-x86_64
source activate sgl-science

cd ~/sgl_science_case/sgl_science_case

python scripts/01_precompute_abs_coefs.py \
  --atmosphere ~/sgl_science_case/sgl_science_case/data/atmosphere_profile.csv \
  --hapi-db ~/sgl_science_case/sgl_science_case/notebooks/HAPI_DB \
  --out-dir ~/orcd/pool/sgl_science_case/abs_coef_cache_dwn1e-4_hydrocarbons \
  --dwn 1e-4 \
  --wl-min 13 \
  --wl-max 18 \
  --cloud-top 0 \
  --molecules  C2H6:1 C4H2:1 CH3:1