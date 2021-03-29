"""Defines decorators and plotting functions for xarrays

TODO: Add docstrings and params to utility functions
"""

from functools import wraps

import numpy as np
import earthpy.plot as ep
from rasterio.plot import plotting_extent as rasterio_plotting_extent
import xarray as xr




def as_xarray(func):
    """Wraps a non-xarray function so that metadata is maintained"""

    @wraps(func)
    def wrapped(*args, **kwargs):
        result = func(*args, **kwargs)

        # Copy metadata from original object and add band if needed
        xobj = copy_xr_metadata(args[0], result)
        if "band" not in xobj.coords and "band" not in xobj.dims:
            xobj = add_dim(xobj, dim="band", coords={"name" : ["result"]})

        return xobj
    return wrapped


def plot_xarray(func):
    """Wraps an xarray object to allow plotting with earthpy"""

    @wraps(func)
    def wrapped(*args, **kwargs):

        # Convert first argument to a masked array to plot with earthpy
        args = list(args)
        arr = to_numpy_array(args[0])

        # HACK: Masked arrays cannot be stretched because they are not
        # handled intuitively by the np.percentile function used by the
        # earthpy internals. To get around that, the decorator forces NaN
        # values to 0 when stretch is True.
        if kwargs.get("stretch"):
            pct_clip = np.nanmedian(arr)
            arr = to_numpy_array(args[0].fillna(0))
        else:
            arr = np.ma.masked_array(arr, np.isnan(arr))

        return func(arr, *args[1:], **kwargs)
    return wrapped


@plot_xarray
def hist(*args, **kwargs):
    """Plots histogram based on an xarray object"""
    return ep.hist(*args, **kwargs)


@plot_xarray
def plot_bands(*args, **kwargs):
    """Plots bands based on an xarray object"""
    # Automatically assigned extent using the rio accessor
    try:
        kwargs.setdefault("extent", plotting_extent(args[0]))
    except AttributeError:
        # Fails if rio accessor has not been loaded
        pass
    return ep.plot_bands(*args, **kwargs)


@plot_xarray
def plot_rgb(*args, **kwargs):
    """Plots RGB based on an xarray object"""
    # Automatically assigned extent using the rio accessor
    try:
        kwargs.setdefault("extent", plotting_extent(args[0]))
    except AttributeError:
        # Fails if rio accessor has not been loaded
        pass
    return ep.plot_rgb(*args, **kwargs)


def add_dim(xobj, dim="band", coords=None):
    """Adds an index dimension to an array

    Parameters
    ---------
    xobj: xarray.DataArray or xarray.Dataset
        an array without an index dimension
    dim: str
        the name of the index dimension
    coords: dict of list-like
        list of names for the bands in the given xarray. The length of each
        list must match that of the array.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
       Array with band as a dimensional coordinate or dataset with band as keys
    """

    # Convert dataset to array
    is_dataset = False
    if isinstance(xobj, xr.Dataset):
        xobj = xobj.to_array(dim=dim)

    # Check shape to see if it contains only one band
    if len(xobj.shape) == 2:
        xobj = [xobj]

    # Assign band
    layers = []
    for arr in xobj:
        arr[dim] = len(layers)
        layers.append(arr)
    new_xobj = xr.concat(layers, dim=dim)

    # Map any provided names
    if coords:
        coords = {k: (dim, list(v)) for k, v in coords.items()}
        new_xobj = new_xobj.assign_coords(**coords)

    return new_xobj.to_dataset(dim=dim) if is_dataset else new_xobj


def copy_xr_metadata(xobj, other):
    """Copies metadata from one xarray object to another

    Parameters
    ---------
    xarr: xarray.DataArray or xarray.Dataset
        the array/dataset to copy metadata from
    other: numpy.array or similar
        the object to copy metadata to

    Returns
    -------
    xarray.DataArray or xarray.Datset
       Copy of other converted to type of xobj with metadata applied
    """
    if isinstance(xobj, xr.DataArray):
        return copy_array_metadata(xobj, other)
    return copy_dataset_metadata(xobj, other)


def copy_array_metadata(xarr, other):
    """Copies metadata from an array to another object

    Looks at the shape and length of xarr and other to decide which
    metadata to copy.

    Parameters
    ---------
    xarr: xarray.DataArray
        the array to copy metadata from
    other: numpy.array or similar
        the object to copy metadata to

    Returns
    -------
    xarray.DataArray
       Copy of other with metadata applied
    """

    # Convert a list, etc. to an array
    if isinstance(other, (list, tuple)):
        other = np.array(other)

    # If arrays have the same shape, copy all metadata
    if xarr.shape == other.shape:
        return xarr.__class__(other, dims=xarr.dims, coords=xarr.coords)

    # If arrays have the same number of layers, copy scalar coordinates
    # and any other coordinates with same the length as the array
    if len(xarr) == len(other):
        coords = {
            k: v for k, v in xarr.coords.items()
            if not v.shape or v.shape[0] == len(xarr)
        }
        dims = [d for d in xarr.dims if d in coords]
        return xarr.__class__(other, dims=dims, coords=coords)

    # If arrays have the same dimensions, copy spatial and scalar coordinates
    xarr_sq = xarr.squeeze()
    other_sq = other.squeeze()
    if xarr_sq.shape == other_sq.shape:
        coords = {k: v for k, v in xarr_sq.coords.items()
                  if k not in xarr_sq.dims}
        return xarr.__class__(
            other_sq, dims=xarr_sq.dims, coords=xarr_sq.coords)

    raise ValueError("Could not copy xr metadata")


def copy_dataset_metadata(xdat, other):
    """Copies metadata from a dataset to another object

    Parameters
    ---------
    xarr: xarray.Dataset
        the dataset to copy metadata from
    other: numpy.array or similar
        the object to copy metadata to

    Returns
    -------
    xarray.Dataset
       Copy of other with metadata applied
    """
    xarr = xdat.to_array(dim="band")
    return copy_array_metadata(xarr, other).to_dataset(dim="band")


def to_numpy_array(obj, dim="band"):
    """Converts an object to a numpy array"""
    if isinstance(obj, xr.Dataset):
        xobj = xobj.to_array(dim=dim)
    if isinstance(obj, xr.DataArray):
        return obj.values
    if isinstance(obj, (list, tuple)):
        return np.array(obj)
    return obj


def plotting_extent(xobj):
    """Calculates plotting extent for an xarray object for matplotlib"""
    if isinstance(xobj, xr.DataArray):
        xarrs = xobj if len(xobj.shape) > 2 else [xobj]
    else:
        xarrs = xobj.values()
    for xarr in xarrs:
        return rasterio_plotting_extent(xarr, xarr.rio.transform())
