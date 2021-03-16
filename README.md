imagery_accessor
================

Extends xarray to simplify metadata-aware remote-sensing calculations and
plotting. Leans heavily on rioxarray and earthpy. Work in progress.

Installation
------------

This package is currently unpublished but can be installed from GitHub
using conda as follows:

```
conda create --name imagery_accessor
conda activate imagery_accessor

git clone https://github.com/adamancer/imagery_accessor
cd imagery_accessor
conda env update -n imagery_accessor -f environment.yml
pip install .
```

Basic usage
-----------

### Accessor

The accessor is available on any xarray object after the image_accessor is
imported. It allows users to plot and run calculations on satellite images
using methods on the `im` attribute.

```python
import os

import earthpy as et
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import imagery_accessor as ixr


# Download sample data
data = et.data.get_data("vignette-landsat")

# Change working directory
os.chdir(os.path.join(et.io.HOME, "earth-analytics"))

# Load and plot an RGB image from a set of Landsat images
stacked = ixr.open_imagery("data/vignette-landsat", source="landsat")

# Calculate NDVI using built-in methods
ndvi = stacked.im.ndvi()
ndvi_class_bins = [-np.inf, 0, 0.1, 0.25, 0.4, np.inf]
ndvi_landsat_class = ndvi.im.npfunc("digitize", ndvi_class_bins)

# Set up color map for classified NDVI plot
ndvi_colors = ["gray", "y", "yellowgreen", "g", "darkgreen"]
ndvi_cmap = ListedColormap(ndvi_colors)

# Plot results using the decorated functions defined in the utils submodule
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

# Plot RGB from Landsat data
stacked.im.plot_rgb("CIR", ax=ax1, stretch=True, str_clip=0.5)

# Plot NDVI using plot_bands
ndvi.im.plot_bands(ax=ax2, cmap=ixr.BuGn, vmin=-1, vmax=1)

# Plot classified NDVI using plot_bands
ndvi_landsat_class.im.plot_bands(ax=ax3, title="NDVI Classified", cmap=ndvi_cmap)

plt.show()
```

The rio accessor from rioxarray is available for any object loaded using
open_imagery().


### Decorators and decorated functions

The accessor is complex and often relies on having satellite-specific metadata
available. A simpler approach is to use decorators to adapt existing
workflows to work more cleanly with xarray objects. For example, the decorated
equivalents of the earthpy plotting functions (ep.hist, ep.plot_bands, and
ep.plot_rgb) can be dropped in wherever the original functions are used:

```python
import os

import earthpy as et
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import imagery_accessor as ixr


# Use the as_array wrapper on numpy functions to preserve metadata
@ixr.as_xarray
def digitize(*args, **kwargs):
    return np.digitize(*args, **kwargs)


# Download sample data
data = et.data.get_data("vignette-landsat")

# Change working directory
os.chdir(os.path.join(et.io.HOME, "earth-analytics"))

# Load and plot an RGB image from a set of Landsat images. This still uses
# open_imagery function, but arrays loaded using the rioxarray.open_rasterio
# function should also work.
stacked = ixr.open_imagery("data/vignette-landsat", source="landsat")

# Calculate and classify NDVI
ndvi = (stacked[4] - stacked[3]) / (stacked[4] + stacked[3])
ndvi_class_bins = [-np.inf, 0, 0.1, 0.25, 0.4, np.inf]
ndvi_landsat_class = digitize(ndvi, ndvi_class_bins)

# Set up color map for classified NDVI plot
ndvi_colors = ["gray", "y", "yellowgreen", "g", "darkgreen"]
ndvi_cmap = ListedColormap(ndvi_colors)

# Plot results using the decorated functions defined in the utils submodule
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

# Plot RGB from Landsat data. Note that both plot_rgb and plot_bands will
# automatically calculate extent if the data was loaded using rioxarray.
ixr.plot_rgb(stacked,
             ax=ax1,
             title="CIR",
             rgb=(4, 3, 2),
             stretch=True,
             str_clip=0.5)

# Plot NDVI using plot_bands
ixr.plot_bands(ndvi,
               ax=ax2,
               title="NDVI",
               cmap=ixr.BuGn,
               vmin=-1,
               vmax=1)

# Plot classified NDVI using plot_bands
ixr.plot_bands(ndvi_landsat_class,
               ax=ax3,
               title="NDVI Classified",
               cmap=ndvi_cmap)

plt.show()
```

See the examples for examples adapted from the earthpy docs.
