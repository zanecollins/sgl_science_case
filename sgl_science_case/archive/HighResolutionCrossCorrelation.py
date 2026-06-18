import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.path import Path
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.colors import to_rgba
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import PathPatch, Rectangle
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from scipy.signal import correlate
from pyhdf.SD import SD, SDC
from matplotlib.widgets import Button
import glob
import os

#Here, tell the script if the data is line by line or cs_abs
Line_by_Line = False

molecule_weights = {
    "CH3Br": 1}

# AtmosphericSpectraFileName = '/Users/ZaniacCollins/Desktop/VenusProject/sgl/TransmissionSpectra/CarbonHydrogenContainingSpecies/CH3Br_CH3Cl/CH3Br_HR_LBL.txt'

# #_________________________ Parsing Observed Spectrum _________________________
# with open(AtmosphericSpectraFileName, "r") as f:
#     text = f.read()

# text = text.replace(",", " ")  # Replace commas with spaces

# with open(AtmosphericSpectraFileName, "w") as f:
#     f.write(text)

# AtmosphericSpectraData = np.genfromtxt(AtmosphericSpectraFileName, skip_header=1)
# AtmosphericSpectraWavenumber = np.genfromtxt(AtmosphericSpectraFileName, skip_header = 1, usecols=0)
# AtmosphericSpectraWavelengths = 10**4/AtmosphericSpectraWavenumber
# AtmosphericSpectraEffective_height = np.genfromtxt(AtmosphericSpectraFileName, skip_header = 1, usecols = 1)

#_________________________ Parsing Template Spectrum ____________________________________________________________________________________________________

TemplateSpectraFileName = '/Users/ZaniacCollins/Desktop/VenusProject/sgl/Fall_25_Transmission_Spectra/Isoprene_cs_abs.txt'
if Line_by_Line == True:
    TemplateSpectraWavenumber = np.genfromtxt(TemplateSpectraFileName, skip_header = 1, usecols = 0)
    TemplateSpectraWavenumber = np.array(TemplateSpectraWavenumber)
    TemplateSpectraWavelength = 10**4/TemplateSpectraWavenumber
    TemplateSpectraCrossSection = np.genfromtxt(TemplateSpectraFileName, skip_header = 1, usecols = 1)
    LogTemplateSpectraCrossSection= np.log10(np.clip(TemplateSpectraCrossSection,1*10**-50, None))
else:
    TemplateSpectraCrossSection = np.genfromtxt(TemplateSpectraFileName, skip_header = 1,usecols=1)
    TemplateSpectraCrossSection = np.array(TemplateSpectraCrossSection)
    with open(TemplateSpectraFileName, 'r') as f:
        header = f.readline().strip()
        header_items = header.split()
        Wavenumber_min = header_items[1]
        Wavenumber_max = header_items[2]
    TemplateSpectraWavenumber = np.linspace(float(Wavenumber_min), float(Wavenumber_max), len(TemplateSpectraCrossSection))
    TemplateSpectraWavelength = 10**4/TemplateSpectraWavenumber
    LogTemplateSpectraCrossSection= np.log10(np.clip(TemplateSpectraCrossSection,1*10**-50, None))
template_molecule_name = os.path.splitext(os.path.basename(TemplateSpectraFileName))[0].lower().split('_')[0]

#700 cm^-1 to 1078 cm^-1

#_________________________ Parsing Test Spectrum ____________________________________________________________________________________________________

TestSpectraFileName = '/Users/ZaniacCollins/Desktop/VenusProject/sgl/Fall_25_Transmission_Spectra/Isoprene_cs_abs.txt'
if Line_by_Line == True:
    TestSpectraWavenumber = np.genfromtxt(TestSpectraFileName, skip_header = 1, usecols = 0)
    TestSpectraWavenumber = np.array(TestSpectraWavenumber)
    TestSpectraWavelength = 10**4/TemplateSpectraWavenumber
    TestSpectraCrossSection = np.genfromtxt(TestSpectraFileName, skip_header = 1, usecols = 1)
    LogTestSpectraCrossSection= np.log10(np.clip(TestSpectraCrossSection,1*10**-50, None))
else:
    TestSpectraCrossSection = np.genfromtxt(TestSpectraFileName, skip_header = 1,usecols=1)
    TestSpectraCrossSection = np.array(TestSpectraCrossSection)
    with open(TestSpectraFileName, 'r') as f:
        header = f.readline().strip()
        header_items = header.split()
        Wavenumber_min = header_items[1]
        Wavenumber_max = header_items[2]
    TestSpectraWavenumber = np.linspace(float(Wavenumber_min), float(Wavenumber_max), len(TestSpectraCrossSection))
    TestSpectraWavelength = 10**4/TestSpectraWavenumber
    # if TemplateSpectraWavelength[0] > TemplateSpectraWavelength[-1]:
    #     TemplateSpectraWavelength = TemplateSpectraWavelength[::-1]
    #     TemplateSpectraCrossSection = TemplateSpectraCrossSection[::-1]
    LogTestSpectraCrossSection= np.log10(np.clip(TestSpectraCrossSection,1*10**-50, None))
test_molecule_name = os.path.splitext(os.path.basename(TestSpectraFileName))[0].lower().split('_')[0]

# Ensure both arrays are 1D and same length
#TemplateSpectraCrossSection = np.ravel(TemplateSpectraCrossSection)
#TestSpectraCrossSection = np.ravel(TestSpectraCrossSection)

#_________________________ Parsing files for Waterfall Plot _________________________

# ComponentsFolder = '/users/ZaniacCollins/Desktop/VenusProject/sgl/TransmissionSpectra/CarbonHydrogenContainingSpecies/CH3Br_CH3Cl'
# ComponentsSpectra = []

# text_files= [
#         os.path.join(ComponentsFolder, fname)
#     for fname in os.listdir(ComponentsFolder)
#     if fname.endswith('.txt') or fname.endswith('.xsc.txt')
# ]

# def extract_name(file):
#     return os.path.splitext(os.path.basename(file))[0].lower().split('_')[0]

# ComponentFormulas = [extract_name(file) for file in text_files]

#_________________________Helper Function for loading cross section data ____________________________
def load_cross_section_file(filepath):
    """
    For lbl files
    """
    data = np.genfromtxt(filepath, comments='#')
    wn_grid = data[:,0]
    intensities = data[:,1]
    wavelength_grid = 1e4 / wn_grid
    return wavelength_grid, intensities

#_________________________ Cross Section minimum and maximum cut off for waterfall plot _____________________
# custom_log_y_limits = []
# for i in range(len(2)):
#     custom_log_y_limits.append((-23,-10))

# #_________________________ Boolean set to format of data files __________________
# line_by_line = False

# for i,(file, name) in enumerate(zip(text_files, ComponentFormulas)):

#     lam, intensity = load_cross_section_file(file)

#     # Ensure monotonic increasing for interpolation
#     min_log_intensity, max_log_intensity = custom_log_y_limits[i]
#     sort_idx = np.argsort(lam)
#     lam = lam[sort_idx]
#     intensity = intensity[sort_idx]
#     intensity = np.clip(intensity, 1*10^min_log_intensity, None)
#     log_intensity = np.log10(intensity)

#         # --- Per-spectrum log-intensity limits ---
    
#     if min_log_intensity is None:
#         min_log_intensity = np.min(log_intensity)
#     if max_log_intensity is None:
#         max_log_intensity = np.max(log_intensity)
#     log_intensity = np.clip(log_intensity, min_log_intensity, max_log_intensity)
#     interp_log_intensity = np.interp(TemplateSpectraWavelengths, lam, log_intensity)
#     offset = -min_log_intensity
#     interp_log_intensity += offset

#     # Mask outside real spectrum range
#     lam_min, lam_max = min(lam), max(lam)
#     interp_log_intensity = np.where(
#         (TemplateSpectraWavelengths >= lam_min) & (TemplateSpectraWavelengths <= lam_max),
#         interp_log_intensity,
#         min(interp_log_intensity)
#     )
#     print(f"{name}: min={np.min(log_intensity)}, max={np.max(log_intensity)}, lam_min={np.min(lam)}, lam_max={np.max(lam)}")
#     ComponentsSpectra.append(interp_log_intensity)

# OverlapRange = True
# wl_min = 5.3   # µm
# wl_max = 8.5   # µm

#_________________________High Pass Filter__________________________

from scipy.signal import savgol_filter

def highpass(spectrum, window=51, polyorder=4): #Fits a polynomial to each point, considering the surrounding points in a set window 
    trend = savgol_filter(spectrum, window_length=window, polyorder=polyorder)
    return spectrum - trend

#_________________________Normalize________________________________

def normalize_spectrum(spectrum):

    normalized_spectrum = (spectrum - np.mean(spectrum))/ np.std(spectrum)

    return normalized_spectrum

#np.argmin(TemplateSpectraWavelength_Interpolated-wl_min)

#_________________________Clipping data to specific wavelength range _________________________

# if OverlapRange:
#     wl = TemplateSpectraWavelengths
#     band = (np.minimum(wl, wl[::-1][::-1]) >= wl_min) & (np.maximum(wl, wl[::-1][::-1]) <= wl_max)  # or just assume ascending:
#     # band = (wl >= wl_min) & (wl <= wl_max)
#     wl_cut  = wl[band]

#     TemplateSpectraEffective_height  = TemplateSpectraEffective_height[band]
#     TemplateSpectraWavelength = TemplateSpectraWavelength[band]
#     TemplateSpectraCrossSection = TemplateSpectraCrossSection[band]
#     TemplateSpectraWavelengths = TemplateSpectraWavelengths[band]
    
#     #Clip waterfall plot 
#     for i in range(len(ComponentsSpectra)):
#         ComponentsSpectra[i] = ComponentsSpectra[i][band]

#_________________________ Interpolate _________________________

# interp_func = interp1d(TemplateSpectraWavelength, TemplateSpectraCrossSection,
#                        kind='linear', bounds_error=False, fill_value=np.nan)
# TemplateSpectraWavelength_Interpolated = interp_func(TemplateSpectraWavelength)

#_________________________ Keep only points where interpolation worked _________________________
# valid = ~np.isnan(TemplateSpectraWavelength_Interpolated)

#Filter noise
#TemplateSpectraEffective_height = (highpass(TemplateSpectraEffective_height))
#LogTemplateSpectraCrossSection = (highpass(LogTemplateSpectraCrossSection))

#_________________________ Normalizing spectra ___________________________________________________________________________
# TemplateSpectraEffective_height = normalize_spectrum(TemplateSpectraCrossSection)
# LogTemplateSpectraCrossSection = normalize_spectrum(TemplateSpectraWavelength)
# TemplateSpectraCrossSection=normalize_spectrum(TemplateSpectraWavelength)

TemplateSpectraCrossSection = normalize_spectrum(TemplateSpectraCrossSection)
TestSpectraCrossSection = normalize_spectrum(TestSpectraCrossSection)

#Cross Correlating
ccf = correlate(TemplateSpectraCrossSection, TestSpectraCrossSection, mode='full') 

c = 2.99792548 * 10**5 #km/s
wavenumber_bin = np.mean(np.diff(TemplateSpectraWavelength))
central_wavenumber = np.mean(TemplateSpectraWavelength)

#lag_0 is the index where the cross correlation function doesn't shift wavelength (i.e. rest velocity)
lag_0 = len(TemplateSpectraCrossSection) - 1

#computing velocity shift
lags = np.arange(len(ccf)) - lag_0
wavenumber_shifts = lags * wavenumber_bin
velocity_shifts = -c * wavenumber_shifts / central_wavenumber  # km/s

max_index = np.argmax(ccf)
velocity_at_peak = velocity_shifts[max_index]
SNR = ccf[lag_0]/np.std(ccf)

#_________________________Plotting_________________________
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

# Template spectrum
# title_str = "molecule = weight: " + ", ".join(f"{k}={v:g}" for k, v in molecule_weights.items())
axs[0].plot(TemplateSpectraWavelength, TemplateSpectraCrossSection, color='red', linewidth=0.5)
axs[0].set_title('Template Cross Section of '+ template_molecule_name+ ' Template')
axs[0].set_ylabel('Normalized Cross Section')
axs[0].set_xlabel('Wavelength (µm)')

# Test Spectrum
axs[1].plot(TestSpectraWavelength, TestSpectraCrossSection, color='blue', linewidth=0.5)
axs[1].set_title('Cross Section of '+ test_molecule_name+ ' Test Spectrum')
axs[1].set_ylabel('Normalized Cross Section')
axs[1].set_xlabel('Wavelength (µm)')
axs[1].text(0.05, 0.95, f"SNR = {SNR:.4f}", transform=axs[1].transAxes,
            fontsize=12, verticalalignment='top')
axs[1].text(0.05, 0.85, f"CC value = {ccf[lag_0]:.4f}", transform=axs[1].transAxes,
            fontsize=12, verticalalignment='top')
axs[1].text(0.05, 0.75, f"CC value at 0 shift / CC max = {ccf[lag_0]/len(TemplateSpectraWavelength):.4f}", transform=axs[1].transAxes,
         fontsize=12, verticalalignment='top')

#_________________________ WaterfallPlotting _________________________

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

# Helper function to map a molecule name to the right colormap
def get_colormap(mol):

    return red_orange_cmap

offset = 1

# for i, (spectrum, mol_name) in enumerate(zip(ComponentsSpectra, ComponentFormulas)):
#     x = TemplateSpectraWavelengths
#     y = spectrum - i * offset
    
#     # Pick colormap for this molecule
#     cmap = get_colormap(mol_name[1].lower().replace(" ", ""))
#     # Draw baseline
#     axs[2].plot(x, np.full_like(x, -i * offset), color='white', linewidth=0.7, zorder=0)
#     avg_y = (y[:-1] + y[1:]) / 2
#     n_shades = 500
#     min_y = np.min(y)
#     max_y = np.max(y)
#     frac = (avg_y - min_y) / (max_y - min_y)
#     darker_frac = 0.5 + 0.5 * frac  # Maps [0,1] → [0.5,1]

#     # Gradient fill from bottom (min_y) to curve
#     n_shades = 500
#     min_y = np.min(y)
#     max_y = np.max(y)
#     norm = Normalize(min_y, max_y)
#     y_shades = np.linspace(min_y, max_y, n_shades)

#     # Create line segments from the curve to the baseline
#     segments = []
#     colors = []

#     for y0, y1 in zip(y_shades[:-1], y_shades[1:]):
#         # Mask to get part of curve between y0 and y1
#         y_fill = np.clip(y, y0, y1)

#         # Fractional height between min_y and max_y
#         frac = (y1 - min_y) / (max_y - min_y) # stronger at top

#         # Interpolate color from colormap (higher intensity near top)
#         # color = cmap(0 + 0.85 * frac)  # Adjust this if you want to clip off part of the color map...shouldn't be necessary 
#         color = cmap(frac)

#         # Shade in the spectral peaks
#         axs[2].fill_between(x, y0, y_fill, color=color, edgecolor=None, alpha = 1, zorder = i)

#         # Add labels for each molecule
#         #ax.text(0.1, -i * offset, molecule_names[i], va='center', ha='right', fontsize=7)

#     # Plot the curve as a gradient-colored outline
#     # Could play around with this to make it a little darker than the shading
#     points = np.array([x, y]).T.reshape(-1, 1, 2)
#     segments = np.concatenate([points[:-1], points[1:]], axis=1)

#     avg_y = (y[:-1] + y[1:]) / 2
#     frac = (avg_y - min_y) / (max_y - min_y)

#     colors = cmap(frac)

#     lc = LineCollection(segments, colors=colors, linewidths=1.2, zorder = i)
#     axs[2].add_collection(lc)

# # Set y-ticks at the baseline of each spectrum
# ytick_positions = [-i * offset for i in range(len(ComponentFormulas))]

# #Molecule Formula Labels
# axs[2].set_yticks(ytick_positions)
# axs[2].set_yticklabels(ComponentFormulas,  fontsize = 8)
# axs[2].set_ylabel("Molecule")

# #Set x-ticks
# wavelength_ticks=[]
# wavelength_min = float(np.min(TemplateSpectraWavelengths))
# wavelength_max = float(np.max(TemplateSpectraWavelengths))
# num_ticks = 5
# wavelength_ticks = np.linspace(wavelength_min, wavelength_max, num_ticks)

# # Axes setup
# axs[2].set_xlim(1.5, 13)
# axs[2].set_xlabel("Wavelength (μm)")
# axs[2].set_ylabel("Log(Cross-Section Absorption) + offset")
# axs[2].set_xticks(wavelength_ticks)

# # Add top axis for wavenumber
# def micron_to_wavenumber(x): return 1e4 / x
# def wavenumber_to_micron(x): return 1e4 / x
# secax = axs[2].secondary_xaxis('top', functions=(micron_to_wavenumber, wavenumber_to_micron))
# secax.set_xlabel("Wavenumber (cm⁻¹)") # upper x-axis label

# # Set the ticks for the secondary x-axis
# secax.set_xticks(1e4/wavelength_ticks)

plt.tight_layout()
plt.show()
