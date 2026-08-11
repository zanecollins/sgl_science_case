#!/bin/bash
#SBATCH --mail-type=BEGIN,END #Mail when job starts and ends
#SBATCH --mail-user=zaniacco@mit.edu #email recipient
#SBATCH -p pi_seager
#SBATCH --job-name=blackbody
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=logs/blackbody%j.out
#SBATCH --error=logs/blackbody-%j.err

module load deprecated-modules
module load anaconda3/2022.05-x86_64
source activate sgl-science
cd ~/sgl_science_case/sgl_science_case

export PYTHONUNBUFFERED=1

python -u scripts/02_generate_spectra.py \
  --atmosphere ~/sgl_science_case/sgl_science_case/data/atmosphere_profile.csv \
  --abs-coef-dir ~/orcd/pool/sgl_science_case/abs_coef_cache_dwn1e-4_hcs_xsecs_and_lbl_isoprene_1to17 \
  --out-dir ~/orcd/pool/sgl_science_case/spectra_dict_cache_dwn1e-4_hydrocarbons \
  --ref_therm Thermal \
  --cloud-top 0 \
  --albedo 0 \
  --xsc-dir ~/sgl_science_case/sgl_science_case/data \
  --xsc-species \
    Isoprene Butadiene Propene Butene 1-Butyne Limonene Pinene \
    Benzene Toluene Trimethylbenzene Tetramethylbenzene \
    DMS DMDS CS2 SF6 SO2F2 \
    1-Propanethiol 2-Methyl-1-propanethiol 2-Propanethiol \
    Benzenethiol Cyclohexanethiol DiethylSulfide DMSO \
    EthylMercaptan Methanethiol MethylIsothiocyanate \
    Tetrahydrothiophene Thiophene tert-Butylmercaptan \
  --scenarios \
    "Blackbody"

