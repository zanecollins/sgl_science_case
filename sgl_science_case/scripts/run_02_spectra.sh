#!/bin/bash
#SBATCH --job-name=sgl_spectra_hydrocarbons_lbl
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=logs/sgl_spectra_hydrocarbons_lbl-%j.out
#SBATCH --error=logs/sgl_spectra_hydrocarbons_lbl-%j.err

module load deprecated-modules
module load anaconda3/2022.05-x86_64
source activate sgl-science
cd ~/sgl_science_case/sgl_science_case

export PYTHONUNBUFFERED=1

python -u scripts/02_generate_spectra.py \
  --atmosphere ~/sgl_science_case/sgl_science_case/data/atmosphere_profile.csv \
  --abs-coef-dir ~/orcd/pool/sgl_science_case/abs_coef_cache_dwn1e-4_hydrocarbons \
  --out-dir ~/orcd/pool/sgl_science_case/spectra_dict_cache_dwn1e-4_hydrocarbons \
  --ref_therm Thermal \
  --cloud-top 0 \
  --albedo 0 \
  --resolutions  1e2 1e3 1e4 1e5 1e6 1e7 \
  --snrs  3 5 10 25 50 \
  --xsc-dir ~/sgl_science_case/sgl_science_case/data \
  --xsc-species Isoprene Butadine Propene Butene 1-Butyne Limonene Pinene Benzene Toluene Trimethylbenzene Tetramethylbenzene\
  --scenarios "CH3:1+C2H6:1+C2H4:1+C4H2:1+C2H2:1+Propene:1+Butene:1+1-Butyne:1+Limonene:1+Pinene:1+Benzene:1+Toluene:1+Trimethylbenzene:1+Tetramethylbenzene:1+Isoprene:1" \
              "CH3:1+C2H6:1+C2H4:1+C4H2:1+C2H2:1+Propene:1+Butene:1+1-Butyne:1+Limonene:1+Pinene:1+Benzene:1+Toluene:1+Trimethylbenzene:1+Tetramethylbenzene:1"
