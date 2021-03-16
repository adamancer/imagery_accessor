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
import earthpy.spatial as es

import imagery_accessor as ixr


# Use the as_array wrapper to keep metadata when using numpy or similar
@ixr.as_xarray
def digitize(*args, **kwargs):
    return np.digitize(*args, **kwargs)


# Get data and set your home working directory
data = et.data.get_data("vignette-landsat")

# Set working directory
os.chdir(os.path.join(et.io.HOME, "earth-analytics"))

# Stack the Landsat 8 bands
arr_st = ixr.open_imagery("data/vignette-landsat", source="landsat")

# Calculate and plot NDVI
ndvi = (arr_st[4] - arr_st[3]) / (arr_st[4] + arr_st[3])


titles = ["Landsat 8 - Normalized Difference Vegetation Index (NDVI)"]
ixr.plot_bands(ndvi, cmap="RdYlGn", cols=1, title=titles, vmin=-1, vmax=1)

# Classify NDVI and plot classified data
ndvi_class_bins = [-np.inf, 0, 0.1, 0.25, 0.4, np.inf]
ndvi_landsat_class = digitize(ndvi, ndvi_class_bins)

# Define color map
nbr_colors = ["gray", "y", "yellowgreen", "g", "darkgreen"]
nbr_cmap = ListedColormap(nbr_colors)

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

# Plot your data
fig, ax = plt.subplots(figsize=(12, 12))
ixr.plot_bands(ndvi_landsat_class, ax=ax, cmap=nbr_cmap, cbar=None)
im = ax.get_images()[0]

# Draw legend
ep.draw_legend(im_ax=im, classes=classes, titles=ndvi_cat_names)
ax.set_title(
    "Landsat 8 - Normalized Difference Vegetation Index (NDVI) Classes",
    fontsize=14,
)
ax.set_axis_off()

# Auto adjust subplot to fit figure size
plt.tight_layout()
plt.show()
