python scripts/02_generate_spectra_hitran.py \
  --atmosphere data/external/atmosphere_profile.csv \
  --abs-coef-dir data \
  --out-dir /orcd/pool/sgl_science_case/spectra_dict_cache_dwn1e-4_hydrocarbons \
  --ref_therm Thermal \
  --cloud-top 0 \
  --resolutions 1000 10000 100000 120000\
  --snrs 5 10 25 50 100\
  --scenarios "Isoprene+Butadiene+Propene+Butene+Benzene+Toluene" \
              "Butadiene+Propene+Butene+Benzene+Toluene" \
#   --default-ppmv 1.0