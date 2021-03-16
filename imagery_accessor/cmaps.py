"""Defines custom color maps to use for spectral indices"""
from matplotlib.colors import LinearSegmentedColormap



# Based on the blue-green color map from the USGS Landsat indices
# FIXME: Seems more gradual than the source
BuGn = LinearSegmentedColormap(
    'BuGn',
    segmentdata={
        'red':   [[0.0, 0.07, 0.07],
                  [0.5, 1.00, 1.00],
                  [1.0, 0.14, 0.14]],
        'green': [[0.0, 0.27, 0.27],
                  [0.5, 1.00, 1.00],
                  [1.0, 0.44, 0.44]],
        'blue':  [[0.0, 0.47, 0.47],
                  [0.5, 1.00, 1.00],
                  [1.0, 0.05, 0.05]]},
    N=256)
