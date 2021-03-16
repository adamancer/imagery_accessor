"""Calculate and plot NDVI using the imagery accessor

Adapted from https://github.com/earthlab/earthpy/blob/main/examples/plot_calculate_classify_ndvi.py.
The original file contains more detail about the functions and
calculations shown here.
"""

import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import earthpy as et
import earthpy.plot as ep
import imagery_accessor as ixr


# Get data and set your home working directory
data = et.data.get_data("vignette-landsat")

# Change working directory
os.chdir(os.path.join(et.io.HOME, "earth-analytics"))

# Stack the Landsat 8 bands
stacked = ixr.open_imagery("data/vignette-landsat", source="landsat")

# Calculate NDVI using the accessor
ndvi = stacked.im.ndvi()

# Plot NDVI using the accessor
titles = ["Landsat 8 - Normalized Difference Vegetation Index (NDVI)"]
ndvi.im.plot_bands(cmap="RdYlGn", cols=1, title=titles, vmin=-1, vmax=1)

# Classify NDVI results
ndvi_class_bins = [-np.inf, 0, 0.1, 0.25, 0.4, np.inf]
ndvi_landsat_class = ndvi.im.npfunc("digitize", ndvi_class_bins)

# Set labels for the classified data
ndvi_colors = ["gray", "y", "yellowgreen", "g", "darkgreen"]
ndvi_cmap = ListedColormap(ndvi_colors)

# Define class names
ndvi_cat_names = [
    "No Vegetation",
    "Bare Area",
    "Low Vegetation",
    "Moderate Vegetation",
    "High Vegetation",
]

# Get list of classes
classes = np.unique(ndvi_landsat_class).tolist()[:5]

# Plot data
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_title(
    "Landsat 8 - Normalized Difference Vegetation Index (NDVI) Classes",
    fontsize=14,
)

# Plot classified NDVI
ndvi_landsat_class.im.plot_bands(ax=ax, cmap=ndvi_cmap, cbar=False)

# Draw a legend
ep.draw_legend(im_ax=ax.get_images()[0],
               classes=classes,
               titles=ndvi_cat_names)

# Show figure
plt.tight_layout()
plt.show()
