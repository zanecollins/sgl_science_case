import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
import os
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.colors import to_rgba
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

#load files
spectra_folder = '/users/ZaniacCollins/Desktop/sgl/Python/HAPI_DB/'

text_files= [
        os.path.join(spectra_folder, fname)
    for fname in os.listdir(spectra_folder)
    if fname.endswith('.data') or fname.endswith('.xsc.txt')
]

# Load the spreadsheet
asm_df = pd.read_excel('/users/ZaniacCollins/Desktop/sgl/ASM_5.0_release.xlsx')
asm_df['Name'] = asm_df['Unnamed: 4'].str.strip()
asm_df['Unnamed: 13'] = asm_df['Unnamed: 13'].str.upper().str.strip()
life_dict = dict(zip(asm_df['Unnamed: 4'], asm_df['Unnamed: 4']))
all_files = sorted(glob.glob(os.path.join(spectra_folder, "*.data")))

# Extract base molecule name (strip extension and path, lowercase)
def extract_name(file):
    return os.path.splitext(os.path.basename(file))[0].lower().split('.')[0]
molecule_file_info = [(file, extract_name(file)) for file in all_files]

pairs = [(extract_name(f), f) for f in text_files]
# Sort alphabetically by molecule name
pairs.sort(key=lambda x: x[0])

#Sorting based on life use
used_by_life = []
not_used_by_life = []
not_listed = []
for file, mol in molecule_file_info:
    if mol in life_dict:
        if life_dict[mol] == 'Y':
            used_by_life.append((file, mol))
        else:
            not_used_by_life.append((file, mol))
    else:
        not_listed.append((file, mol))

# Combine with order: life, not life, unlisted
life_ordered_files = used_by_life + not_used_by_life + not_listed

# Unpack sorted names and files if you want separate lists again
text_files_sorted = [file for file, mol in life_ordered_files]
molecule_names_sorted = [mol for file, mol in life_ordered_files]

wavelength_max = 16.6667
wavelength_min = 1.5

# Parameters
num_molecules = len(text_files)  # Keep manageable...update to as needed
wavelength = np.linspace(wavelength_min, wavelength_max, 20000)  # Wavelength in microns
molecule_names = molecule_names_sorted
spectra = []

# List of target molecules and their custom colormaps
target_molecules = {
    "methanethiol": plt.colormaps["Greens"],
    "dimethylsulfide": plt.colormaps["Greens"],
    "dimethyldisulfide": plt.colormaps["Greens"],
    "carbondisulfide": plt.colormaps["Greens"],
    "carbonylsulfide": plt.colormaps["Greens"],
}

# Helper function to map a molecule name to the right colormap
def get_colormap(mol):
    for key in target_molecules:
        if key in mol.replace("-", "").replace("_", ""):
            return target_molecules[key]
    # Default for everything else
    return red_orange_cmap

custom_log_y_limits = []
for i in range(len(text_files)):
    custom_log_y_limits.append((-23,-10))

def get_wn_range_and_intensities(filepath):
    intensities = []
    with open(filepath, 'r') as f:
        first_line = f.readline()
        tokens = first_line.split()
        floats = []
        for t in tokens:
            try:
                floats.append(float(t))
            except:
                pass
        wn_start, wn_end = floats[0], floats[1]
        for line in f:
            for val in line.split():
                try:
                    intensity = float(val.replace('D','E'))
                    intensity = np.clip(intensity, 1e-25, None)  # avoid log(0)
                    intensities.append(intensity)
                except ValueError:
                    pass
    return wn_start, wn_end, np.array(intensities)

def load_cross_section_file(filepath):
    """
    For lbl files
    """
    data = np.genfromtxt(
    filepath,
    comments="#",
    usecols=(1, 2),   # <-- IMPORTANT: use wavenumber and intensity columns
    invalid_raise=False
    )
    data = data[np.isfinite(data).all(axis=1)]

    wn_grid = data[:, 0]        # because usecols returns 2 cols now
    intensities = data[:, 1]
    wavelength_grid = 1e4 / wn_grid
    return wavelength_grid, intensities


#Line-by-line or not??
line_by_line = True

for i,(file, name) in enumerate(zip(text_files_sorted, molecule_names_sorted)):
    if line_by_line == True:
        lam, intensity = load_cross_section_file(file)
        print(name, "lam range:", lam.min(), lam.max())
        print(name, "intensity range:", intensity.min(), intensity.max())
    else:
        wn_start, wn_end, intensity = get_wn_range_and_intensities(file)
        wn_grid = np.linspace(wn_start, wn_end, len(intensity))
        wn_grid = wn_grid[wn_grid!= 0]
        lam = 1e4 / wn_grid

    # Ensure monotonic increasing for interpolation

    min_log_intensity, max_log_intensity = custom_log_y_limits[i]
    sort_idx = np.argsort(lam)
    lam = lam[sort_idx]
    intensity = intensity[sort_idx]
    intensity = np.clip(intensity, 1*10**min_log_intensity, None)
    log_intensity = np.log10(intensity)

        # --- Per-spectrum log-intensity limits ---
    
    if min_log_intensity is None:
        min_log_intensity = np.min(log_intensity)
    if max_log_intensity is None:
        max_log_intensity = np.max(log_intensity)
    log_intensity = np.clip(log_intensity, min_log_intensity, max_log_intensity)
    interp_log_intensity = np.interp(wavelength, lam, log_intensity)
    offset = -min_log_intensity
    interp_log_intensity += offset

    # Mask outside real spectrum range
    lam_min, lam_max = min(lam), max(lam)
    interp_log_intensity = np.where(
        (wavelength >= lam_min) & (wavelength <= lam_max),
        interp_log_intensity,
        min(interp_log_intensity)
    )
    print(f"{name}: min={np.min(log_intensity)}, max={np.max(log_intensity)}, lam_min={np.min(lam)}, lam_max={np.max(lam)}")
    spectra.append(interp_log_intensity)


# Plotting
fig, ax = plt.subplots(figsize=(14, 10)) # can make this larger if needed
offset = 1.5  # Vertical offset between spectra, adjust as needed
colors = cm.Oranges(np.linspace(1.5, 10.0, num_molecules))
colormaps = [
    plt.colormaps["Oranges"],
    plt.colormaps["Blues"],
    plt.colormaps["Greens"],
    plt.colormaps["Reds"],
]



# # Define custom red-to-orange colormap
# red_orange_cmap = LinearSegmentedColormap.from_list(
#     "RedOrangeSmooth",
#     [
#         # "#fffaf5",  # very light
#         "#ffe3c2",  # light orange
#         "#ffb573",  # orange
#         "#ff963a",  # deep orange
#         "#ff5d00",  # vivid orange
#         "#ff3000",  # red-orange
#         "#c41c00"   # deep red
#     ],
#     N=100  # More steps for a smoother gradient
# )

# This colormap is the most similar to the one used in the paper
red_orange_cmap = LinearSegmentedColormap.from_list(
    "RedOrangeSmooth",
    [
        "#fffbf5",  
        "#ffe7ce",  
        "#ffc587", 
        "#ffb058",  
        "#ffa233", 
        "#ff9701", 
        "#ff7f00",
        "#ff7200",
        "#ff5800",
        "#ff4900",
        "#ff4400",
        "#ff3b00",
        "#ff2a00",
        "#ff3600"
    ],
    N=500  # More steps for a smoother gradient
)

use_fill = False  # <-- Set this to False for line-only mode

for i, (spectrum, mol_name) in enumerate(zip(spectra, molecule_names_sorted)):
    x = wavelength
    y = spectrum - i * offset
    
    # Pick colormap for this molecule
    cmap = get_colormap(mol_name.lower().replace(" ", ""))

    # Draw baseline
    ax.plot(x, np.full_like(x, -i * offset), color='white', linewidth=0.7, zorder=0)
    avg_y = (y[:-1] + y[1:]) / 2
    n_shades = 500
    min_y = np.min(y)
    max_y = np.max(y)
    frac = (avg_y - min_y) / (max_y - min_y)
    darker_frac = 0.5 + 0.5 * frac  # Maps [0,1] → [0.5,1]

    # Gradient fill from bottom (min_y) to curve
    n_shades = 500
    min_y = np.min(y)
    max_y = np.max(y)
    norm = Normalize(min_y, max_y)
    y_shades = np.linspace(min_y, max_y, n_shades)

    # Create line segments from the curve to the baseline
    segments = []
    colors = []

    for y0, y1 in zip(y_shades[:-1], y_shades[1:]):
        # Mask to get part of curve between y0 and y1
        y_fill = np.clip(y, y0, y1)

        # Fractional height between min_y and max_y
        frac = (y1 - min_y) / (max_y - min_y) # stronger at top

        # Interpolate color from colormap (higher intensity near top)
        # color = cmap(0 + 0.85 * frac)  # Adjust this if you want to clip off part of the color map...shouldn't be necessary 
        color = cmap(frac)

        # Shade in the spectral peaks
        ax.fill_between(x, y0, y_fill, color=color, edgecolor=None, alpha = 1, zorder = i)

        # Add labels for each molecule
        #ax.text(0.1, -i * offset, molecule_names[i], va='center', ha='right', fontsize=7)

    # Plot the curve as a gradient-colored outline
    # Could play around with this to make it a little darker than the shading
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    avg_y = (y[:-1] + y[1:]) / 2
    frac = (avg_y - min_y) / (max_y - min_y)

    colors = cmap(frac)

    lc = LineCollection(segments, colors=colors, linewidths=1.2, zorder = i)
    ax.add_collection(lc)

# Set y-ticks at the baseline of each spectrum
ytick_positions = [-i * offset for i in range(num_molecules)]
ax.set_yticks(ytick_positions)
ax.set_yticklabels(molecule_names_sorted,  fontsize = 8)
ax.set_ylabel("Molecule")


wavelength_ticks=[]
wavelength_min = float(np.min(wavelength))
wavelength_max = float(np.max(wavelength))
num_ticks = 5
wavelength_ticks = np.linspace(wavelength_min, wavelength_max, num_ticks)


# Axes setup
ax.set_xlim(1.5, 13)
ax.set_xlabel("Wavelength (μm)")
ax.set_ylabel("Log(Cross-Section Absorption) + offset")
ax.set_xticks(wavelength_ticks)

# Add top axis for wavenumber
def micron_to_wavenumber(x): return 1e4 / x
def wavenumber_to_micron(x): return 1e4 / x

secax = ax.secondary_xaxis('top', functions=(micron_to_wavenumber, wavenumber_to_micron))
secax.set_xlabel("Wavenumber (cm⁻¹)") # upper x-axis label
# Set the ticks for the secondary x-axis

secax.set_xticks(1e4/wavelength_ticks)

plt.tight_layout()
plt.show()