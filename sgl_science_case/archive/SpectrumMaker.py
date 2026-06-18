import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# === USER INPUTS ===
data_folder = "/Users/ZaniacCollins/Desktop/VenusProject/sgl/TransmissionSpectra/CarbonHydrogenContainingSpecies/HCOOH_HCN_HC3N_CH3CN_CH3CL"  # Folder containing line-by-line data files
molecule_weights = {
    "CH3BR : 0.5",
    "CH3CL : 0.5"
}

# === PARAMETERS ===
wavenumber_range = (1.6, 16.67)   # µm
bin_width = 0.001                 # resolution of your output spectrum in cm⁻¹

# === OUTPUT ARRAYS ===
bin_edges = np.arange(*wavenumber_range, bin_width)
binned_intensity = np.zeros_like(bin_edges[:-1])

# === MAIN LOOP ===
for filename in os.listdir(data_folder):
    if not filename.endswith(".txt"):
        continue
    
    molecule = filename.split('_')[0]
    if molecule not in molecule_weights:
        print(f"Skipping {molecule} (no weight provided)")
        continue
    
    weight = molecule_weights[molecule]
    filepath = os.path.join(data_folder, filename)

    # === PARSE FILE ===
    wn_list = np.genfromtxt(filepath, skip_header = 1, usecols = 0)
    intensity_list = np.genfromtxt(filepath, skip_header=1, usecols = 1)

    wn_array = np.array(wn_list)
    intensity_array = np.array(intensity_list) * weight
    print(f"{filename}: wavenumber range = {wn_array.min():.6f} – {wn_array.max():.6f} µm")


    # === BIN BY WAVELENGTH ===
    hist, _ = np.histogram(wn_array, bins=bin_edges, weights=intensity_array)
    binned_intensity += hist


# === PLOT RESULT ===
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

plt.plot(bin_centers, binned_intensity)
plt.xlabel("Wavelength (µm)")
plt.ylabel("Weighted Intensity")
plt.title("Summed Sulfur Compound Spectra H2S(0.3), OCS(0.3), SO2(0.3), SO3(0.1)")
plt.show()

# === SAVE TO TEXT FILE ===
output_file = "summed_spectrum.txt"
output_data = np.column_stack((bin_centers, binned_intensity))

np.savetxt(output_file, output_data, fmt=("%.6f,%.12e"), delimiter=' ', header="Wavenumber(cm^-1)  Intensity", comments='')

print(f"Saved spectrum to {output_file}")
