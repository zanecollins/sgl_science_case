import numpy as np
import matplotlib.pyplot as plt

file_path = '/users/ZaniacCollins/Desktop/VenusProject/sgl/TransmissionSpectra/Oxygen_HR_LBL.txt'


wavenumbers = np.genfromtxt(file_path, skip_header=1, usecols=0)
intensities = np.genfromtxt(file_path, skip_header = 1, usecols = 1)

wavelengths = [1e7 / wn for wn in wavenumbers if wn != 0]
intensities = [intensities[i] for i in range(len(wavenumbers)) if wavenumbers[i] != 0]

intensities = np.log10(np.clip(intensities,1*10**-40, None))

plt.figure(figsize=(10, 6))
plt.plot(wavelengths, intensities, 'b-', label='HITRAN Spectra')
plt.xlabel('Wavenumber (cm^-1)')
plt.ylabel('Absorption Cross Section')
plt.title('HITRAN Line-by-Line Spectra')

plt.tight_layout
plt.show()

# Example usage
# plot_hitran_spectra('path/to/hitran/data.txt')