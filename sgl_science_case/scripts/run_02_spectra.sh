#!/bin/bash
#SBATCH --mail-type=END #Mail when job starts and ends
#SBATCH --mail-user=zacollins@g.hmc.edu #email recipient
#SBATCH -p pi_seager
#SBATCH --job-name=eps_sys
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# One array task per red-noise level: photon-only reference, baseline, pessimistic.
# Override with sbatch --export=ALL,SCENARIO=...,MODEL_TAG=...,EXTRA_FLAGS=...,REPO_DIR=...,OUT_DIR=...
EPS_SYS_LIST=(0 1e-3 1e-2)
EPS_SYS=${EPS_SYS_LIST[${SLURM_ARRAY_TASK_ID:-1}]}
SCENARIO=${SCENARIO:-"Isoprene+CH3+C2H4+C2H2+C2H6+C4H2"}
MODEL_TAG=${MODEL_TAG:-earthlike}
EXTRA_FLAGS=${EXTRA_FLAGS:-}
REPO_DIR=${REPO_DIR:-~/sgl_science_case/sgl_science_case}
OUT_DIR=${OUT_DIR:-~/orcd/pool/sgl_science_case/spectra_dict_cache_verified_model_eps_sys}

module load deprecated-modules
module load anaconda3/2022.05-x86_64
source activate sgl-science
cd "$REPO_DIR"

export PYTHONUNBUFFERED=1

python -u scripts/02_generate_spectra.py \
  --atmosphere ~/sgl_science_case/sgl_science_case/data/atmosphere_profile.csv \
  --abs-coef-dir ~/orcd/pool/sgl_science_case/abs_coef_cache_dwn1e-4_hcs_xsecs_and_lbl_isoprene_1to17 \
  --out-dir "$OUT_DIR" \
  --ref_therm Thermal \
  --cloud-top 0 \
  --albedo 0 \
  --scenarios "$SCENARIO" \
  --xsc-dir ~/sgl_science_case/sgl_science_case/data \
  --output_name "${SCENARIO}_${MODEL_TAG}_eps${EPS_SYS}" \
  --eps_sys "$EPS_SYS" $EXTRA_FLAGS
