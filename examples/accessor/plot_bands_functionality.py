"""Plot bands

Adapted from
https://github.com/earthlab/earthpy/blob/main/examples/plot_bands_functionality.py.
The original file contains more detail about the functions and
calculations shown here.
"""

import os
import matplotlib.pyplot as plt
import earthpy as et
import earthpy.plot as ep
import imagery_accessor as ixr


# Get data for example
data = et.data.get_data("vignette-landsat")

# Set working directory
os.chdir(os.path.join(et.io.HOME, "earth-analytics"))

# Stack the Landsat 8 bands
stacked = ixr.open_imagery("data/vignette-landsat", source="landsat")
stacked = stacked.im.no_qa()

# Plot all bands in the stack
stacked.im.plot_bands()

# Plot one band in the stack
stacked.im.plot_bands("nir", cbar=False)

# Plot one band and scale it
stacked.im.plot_bands("nir", cbar=False, scale=True)

# Plot all bands in two columns
stacked.im.plot_bands(cols=2)
