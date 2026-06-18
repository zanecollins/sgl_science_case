import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Path to your GOME-2 LER NetCDF file
filename = '/users/ZaniacCollins/Desktop/VenusProject/sgl/GOME-2_MetOp-ABC_MSC_025x025_surface_LER_v4.1.nc'

# Load the dataset
ds = xr.open_dataset(filename)

# Choose your region
target_lat = 37.0
target_lon = 47.0
target_month = 'JULY'  # Must match the string in the file, e.g. 'JULY', 'JANUARY', etc.

# Find the closest actual grid point in file
closest_lat = ds['latitude'].values[np.abs(ds['latitude'].values - target_lat).argmin()]
closest_lon = ds['longitude'].values[np.abs(ds['longitude'].values - target_lon).argmin()]

# Select the data using .sel (coordinate-based selection)
LER = ds['minimum_LER'].sel(
    month=target_month,
    latitude=closest_lat,
    longitude=closest_lon
)

wavelengths = ds['wavelength'].values

plt.figure(figsize=(7,4))
plt.plot(wavelengths, LER, marker='o')
plt.xlabel('Wavelength (nm)')
plt.ylabel('LER (unitless)')
plt.title(f"GOME-2 Surface LER Spectrum at {closest_lat:.2f}N, {closest_lon:.2f}E, {target_month}")
plt.grid(True)
plt.show()