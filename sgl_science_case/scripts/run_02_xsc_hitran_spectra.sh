python scripts/02_generate_spectra_hitran.py \
  --atmosphere data/external/atmosphere_profile.csv \
  --abs-coef-dir data/external/hydrocarbon_abs_cross_sec_hitran \
  --out-dir data/spectra_xsc_atm \
  --ref_therm Thermal \
  --cloud-top 0 \
  --resolutions 1000 10000 100000 \
  --snrs 5 10 25 50 \
  --scenarios "Isoprene+Butadiene+Propene+Butene+Benzene+Toluene" \
              "Butadiene+Propene+Butene+Benzene+Toluene" \
#   --default-ppmv 1.0