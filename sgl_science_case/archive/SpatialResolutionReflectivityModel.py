import numpy as np
import matplotlib.pyplot as plt
import random
import xarray as xr
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle
from scipy.spatial import cKDTree
from pyhdf.SD import SD, SDC
from matplotlib.widgets import Button
import glob
import os

filename = 'GOME-2_MetOp-ABC_MSC_025x025_surface_LER_v4.1.nc'
ler_file = "/Users/ZaniacCollins/Desktop/VenusProject/sgl/GOME-2_MetOp-ABC_MSC_025x025_surface_LER_v4.1.nc"
ds = xr.open_dataset(ler_file)

# Choose month (must be in the file, e.g., 'JULY')
month = "JULY"
LER = ds['minimum_LER'].sel(month=month)
wavelengths = ds['wavelength'].values
latitudes = ds['latitude'].values
longitudes = ds['longitude'].values
# Flatten LER for easy matching (shape: [nlon, nlat, nwavelength])
LER = LER.transpose('longitude', 'latitude', 'wavelength')  # (nlon, nlat, nwavelength)
ler_shape = LER.shape

# Create flat list of all (lon, lat) pairs in the LER grid
LER_coords = np.array([[lo, la] for i, lo in enumerate(longitudes) for j, la in enumerate(latitudes)])
LER_coords = LER_coords.reshape(len(longitudes)*len(latitudes), 2)

# Build KD-tree for matching disk cells to LER grid
tree = cKDTree(LER_coords)


# Circle and grid parameters
radius = 1
center = (0, 0)
grid_size = (20, 40)  # 20 rows, 40 columns
x_min, x_max = -1, 1
y_min, y_max = -1, 1
dx = (x_max - x_min) / grid_size[1]  # Width per cell
dy = (y_max - y_min) / grid_size[0]  # Height per cell

# Sun's angle parameters (in degrees)
theta_s_deg = 45
phi_s_deg = 0
theta_s = np.radians(theta_s_deg)
phi_s = np.radians(phi_s_deg)

# Sun vector
sun_direction = np.array([
    np.sin(theta_s) * np.cos(phi_s),
    np.sin(theta_s) * np.sin(phi_s),
    np.cos(theta_s)
])

# Observer's direction (along z-axis)
observer_direction = np.array([0, 0, 1])

# Placeholder wavelengths (AIRS channels, wavenumbers in cm^-1)
wavelengths = ds['wavelength'].values  # These are your x-axis values

# Convert disk coordinates to lat/lon
def disk_to_latlon(cx, cy):
    lat = cy * 90   # top (cy=1) is 90°, bottom is -90°
    lon = cx * 180  # right (cx=1) is 180°, left is -180°
    return lat, lon

# Initialize values matrix
values = np.zeros(grid_size)

# Assign AIRS spectra to disk cells and compute weights
assigned_spectra = []
poly_brdf_weights = {}
weight_values = []
for i in range(grid_size[1]):
    for j in range(grid_size[0]):
        cx = x_min + dx * (i + 0.5)
        cy = y_min + dy * (j + 0.5)
        r = np.sqrt(cx**2 + cy**2)
        if r <= radius:
            z = np.sqrt(1 - r**2)
            normal = np.array([cx, cy, z])
            cos_theta_i = max(np.dot(normal, sun_direction), 0)
            cos_theta_v = max(np.dot(normal, observer_direction), 0)
            weight = cos_theta_i * cos_theta_v
            weight_values.append(weight)
            lat, lon = disk_to_latlon(cx, cy)
            _, idx = tree.query([lat, lon])
            lon_idx = idx // len(latitudes)
            lat_idx = idx % len(latitudes)
            spectrum = LER[lon_idx, lat_idx, :].values
            assigned_spectra.append(spectrum)
            poly_idx = len(assigned_spectra) - 1
            values[j, i] = poly_idx
            if weight > 0:
                poly_brdf_weights[poly_idx] = poly_brdf_weights.get(poly_idx, 0.0) + weight

# Create figure
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 6, height_ratios=[1, 0.5])

# First row: SGL, Summed Spectrum
ax1 = fig.add_subplot(gs[0, :3])  # SGL spans first three columns
ax_sum = fig.add_subplot(gs[0, 3:6])  # Summed spectrum

# Second row: 5 random AIRS spectra
#poly_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]  # 4 random spectra
ax_click = fig.add_subplot(gs[1, 2])  # Clicked spectrum in last column
ax_click.set_title("Clicked Spectrum")
ax_click.set_xlabel('Wavelength (nm)')
ax_click.set_ylabel('Radiance (mW/m²/sr/cm⁻¹)')
ax_click.grid(True)

# --- Plot 1: SGL Image (Illumination + Projection) ---
theta = np.linspace(0, 2 * np.pi, 300)
circle_x = center[0] + radius * np.cos(theta)
circle_y = center[1] + radius * np.sin(theta)
ax1.plot(circle_x, circle_y, 'b')

circle_path = Path(np.vstack([circle_x, circle_y]).T, closed=True)

for i in range(grid_size[1]):
    for j in range(grid_size[0]):
        cx = x_min + dx * (i + 0.5)
        cy = y_min + dy * (j + 0.5)
        r = np.sqrt(cx**2 + cy**2)
        if r <= radius:
            color = (weight_values[int(values[j, i])] / max(weight_values + [1]),) * 3
            rect = Rectangle((x_min + i * dx, y_min + j * dy), dx, dy, edgecolor='black', facecolor=color)
            ax1.add_patch(rect)
            clip_patch = PathPatch(circle_path, facecolor='none')
            ax1.add_patch(clip_patch)
            rect.set_clip_path(clip_patch)

ax1.set_xlim(x_min, x_max)
ax1.set_ylim(y_min, y_max)
ax1.set_aspect('equal')
ax1.grid(False)
ax1.set_title(f'SGL Model (Sun θ={theta_s_deg}°)')
ax1.set_axis_off()

# --- Summed Spectrum Plot ---
total_weight = sum(poly_brdf_weights.values())
weights = [poly_brdf_weights.get(i, 0) / total_weight if total_weight > 0 else 0 for i in range(len(assigned_spectra))]
y_sum = np.ma.zeros(27)
for idx, weight in enumerate(weights):
    y_sum += weight * assigned_spectra[idx]
ax_sum.plot(wavelengths, y_sum, color='black', linewidth=2)
ax_sum.set_xlabel('Wavelength (nm)')
ax_sum.set_ylabel('LER (Unitless)')
ax_sum.set_title('Refl. summed over surface')
ax_sum.grid(True)
ax_sum.set_ylim(0, np.max(y_sum) * 1.2 if np.max(y_sum) > 0 else 1)

'''# --- Random AIRS Spectra Plots ---
random_indices = random.sample(range(len(assigned_spectra)), 4)  # Reduced to 4
for i, (idx, ax) in enumerate(zip(random_indices, poly_axes)):
    ax.plot(wavelengths, assigned_spectra[idx], color='blue', linewidth=1)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('LER (Unitless)')
    ax.grid(True)
    ax.set_ylim(0, np.max(assigned_spectra[idx]) * 1.2)'''

# Handle mouse click events
def on_click(event):
    if event.inaxes != ax1:
        return
    cx, cy = event.xdata, event.ydata
    if cx is None or cy is None:
        return

    # Find grid cell (i, j)
    i = int((cx - x_min) / dx)
    j = int((cy - y_min) / dy)
    if 0 <= i < grid_size[1] and 0 <= j < grid_size[0]:
        if values[j, i] >= 0:
            idx = int(values[j, i])
            spectrum = assigned_spectra[idx]
            ax_click.clear()
            ax_click.plot(wavelengths, spectrum, color='blue', linewidth=1)
            ax_click.set_title(f'(Cell {i}, {j})')
            ax_click.set_xlabel('Wavelength (nm)')
            ax_click.set_ylabel('LER (Unitless)')
            ax_click.grid(True)
            ax_click.set_ylim(0, np.max(spectrum) * 1.2 if np.max(spectrum) > 0 else 1)
            fig.canvas.draw_idle()

        #print coords of region clicked
    lat, lon = disk_to_latlon(cx, cy)  # <-- Get lat/lon from click location
    print(f"Clicked cell: ({i}, {j}) | Latitude: {lat:.2f}°, Longitude: {lon:.2f}°")  # <-- PRINT

# Connect the click event
fig.canvas.mpl_connect('button_press_event', on_click)
plt.tight_layout()
plt.show()

# Print results
print(f"Sun's angle: θ_s = {theta_s_deg}°, φ_s = {phi_s_deg}°")
print(f"Weight range (cos θ_i * cos θ_v): min = {min(weight_values):.3f}, max = {max(weight_values):.3f}")
print("Latitude range:", np.min(np.min(lat)), "to", np.max(lat))
print("Longitude range:", np.min(lon), "to", np.max(lon))