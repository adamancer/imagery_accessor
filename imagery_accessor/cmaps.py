"""Defines custom color maps to use for spectral indices"""
import os

from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image




def linear_cmap_from_list(name, colors, **kwargs):
    """Builds a linear color map from a list of colors using 0-255 per channel

    This converts colors using 255 bits per channel to floats between 0 and 1
    and passes them to LinearSegmentedColormap.from_list().

    Parameters
    ----------
    name: str
        name of the color map
    rgb: list
        list of colors as (r, g, b) or (value, (r, g, b)), where colors use
        255 bits per channel. If given, value is used to divide the range
        unevenly and each value must be a float between 0 and 1.
    kwargs
        keywords arguments accepted by LinearSegmentedColormap

    Returns
    -------
    matplotlib.LinearSegmentedColormap
        color map of given colors
    """
    segments = []
    for color in colors:
        # Color can be given either by itself as (value, color)
        if len(color) == 2:
            segments.append((color[0], [c / 255 for c in color[1]]))
        else:
            segments.append([c / 255 for c in color])
    return LinearSegmentedColormap.from_list(name, segments, **kwargs)


def linear_cmap_from_png(fp, names, **kwargs):
    """Builds linear color maps based on an image

    Each row in the file will be read into a separate LinearSegmentedColormap.

    Parameters
    ----------
    fp: str
        path to an image where each row contains color data for a map
    nams: list
        list of names for the color maps
    kwargs
        keywords arguments accepted by LinearSegmentedColormap

    Returns
    -------
    list of matplotlib.LinearSegmentedColormap
        list of color maps based on file
    """
    arr = np.array(Image.open(fp))

    if len(arr) != len(names):
        raise ValueError(
            f"Must provide {len(arr)} names for cmaps based on '{fp}'"
            f" (provided {names})"
        )

    cmaps = []
    for name, row in zip(names, arr):
        cmaps.append(linear_cmap_from_list(name, row, **kwargs))
    return cmaps




# Define color maps based on USGS spectral indices
BuGn, BuOr, RdYlGn = linear_cmap_from_png(
    os.path.join(os.path.dirname(__file__), "config", "usgs_landsat_cmaps.png"),
    ["BuGn", "BuOr", "RdYlGn"],
    N=256
)
