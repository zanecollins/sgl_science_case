#!/bin/bash
#SBATCH --job-name=sgl_spectra_HIGH_RES_RUN_07_25_afternoon_H2O_CH4_N2O
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/spectra_earth_like-%j.out
#SBATCH --error=logs/spectra_earth_like-%j.err

module load deprecated-modules
module load anaconda3/2022.05-x86_64
source activate sgl-science

cd ~/sgl_science_case/sgl_science_case

python scripts/02_generate_spectra.py \
  --atmosphere ~/sgl_science_case/sgl_science_case/data/atmosphere_profile.csv \
  --abs-coef-dir ~/orcd/pool/sgl_science_case/abs_coef_cache_dwn1e-6 \
  --out-dir ~/orcd/pool/sgl_science_case/spectra_dict_cache_dwn1e-6_earth \
  --ref_therm Thermal \
  --cloud-top 0 \
  --albedo 0.3 \
  --resolutions 1e2 1e3 1e4 1e5 1e6 1e7 1e8 1e9\
  --snrs 3 5 10 15 20 25 50 100\
  --scenarios H2O:1+CH4:1+N2O:1
