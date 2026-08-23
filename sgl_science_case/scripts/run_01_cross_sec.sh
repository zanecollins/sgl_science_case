#!/bin/bash
#SBATCH --mail-type=BEGIN,END #Mail when job starts and ends
#SBATCH --mail-user=zaniacco@mit.edu #email recipient
#SBATCH -p pi_seager
#SBATCH --job-name=sgl_cross_sec_carbons_dwn1e-4_CO2_iso1_r89
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sgl_cross_sec_carbons_dwn1e-4_CO2_iso1_r89-%j.out
#SBATCH --error=logs/sgl_cross_sec_carbons_dwn1e-4_CO2_iso1_r89-%j.err

module load deprecated-modules
module load anaconda3/2022.05-x86_64
source activate sgl-science

cd ~/sgl_science_case/sgl_science_case

python scripts/01_precompute_abs_coefs.py \
  --atmosphere ~/sgl_science_case/sgl_science_case/data/atmosphere_profile.csv \
  --hapi-db ~/sgl_science_case/sgl_science_case/notebooks/HAPI_DB \
  --out-dir ~/orcd/pool/sgl_science_case/abs_coef_cache_dwn1e-4_hcs_xsecs_and_lbl_isoprene_1to17 \
  --dwn 1e-4 \
  --wl-min 1 \
  --wl-max 17 \
  --cloud-top 0 \
  --molecules CO2:1 \
  --isotope-molecule CO2 \
  --isotope-ratio 89