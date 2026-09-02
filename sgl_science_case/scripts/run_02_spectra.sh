#!/bin/bash
#SBATCH --mail-type=END #Mail when job starts and ends
#SBATCH --mail-user=zacollins@g.hmc.edu #email recipient
#SBATCH -p pi_seager
#SBATCH --job-name=hcs_scaled_debug_test
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=logs/hcs_scaled_debug_test%j.out
#SBATCH --error=logs/hsc_scaled_debug_test-%j.err

module load deprecated-modules
module load anaconda3/2022.05-x86_64
source activate sgl-science
cd ~/sgl_science_case/sgl_science_case

export PYTHONUNBUFFERED=1

python -u scripts/02_generate_spectra.py \
  --atmosphere ~/sgl_science_case/sgl_science_case/data/atmosphere_profile.csv \
  --abs-coef-dir ~/orcd/pool/sgl_science_case/abs_coef_cache_dwn1e-4_hcs_xsecs_and_lbl_isoprene_1to17 \
  --out-dir ~/orcd/pool/sgl_science_case/spectra_dict_cache_hydrocarbons \
  --ref_therm Thermal \
  --cloud-top 0 \
  --albedo 0 \
  --scenarios "CH4+C2H2+C2H6+C2H4+C4H2+CH3" \
  --output_name "hcs_scaled_debug_test"
