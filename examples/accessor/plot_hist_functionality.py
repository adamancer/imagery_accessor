"""Calculate and plot histograms

Adapted from
https://github.com/earthlab/earthpy/blob/main/examples/plot_hist_functionality.py.
The original file contains more detail about the functions and
calculations shown here.
"""

import os
import matplotlib.pyplot as plt
import earthpy as et
import imagery_accessor as ixr


# Get data for example
data = et.data.get_data("vignette-landsat")

# Set working directory
os.chdir(os.path.join(et.io.HOME, "earth-analytics"))

# Stack the Landsat 8 bands
stacked = ixr.open_imagery("data/vignette-landsat", source="landsat")

# Remove the QA bands
stacked = stacked.im.no_qa()

# Create the list of color names for each band
colors = [
    "midnightblue",
    "Blue",
    "Green",
    "Red",
    "Maroon",
    "Purple",
    "Violet"
]

# Plot the histograms with the color and title lists you just created
# sphinx_gallery_thumbnail_number = 1
stacked.im.hist(colors=colors)
plt.show()

# Plot each histogram with 50 bins, arranged across three columns
stacked.im.hist(bins=50, cols=3)
plt.show()
