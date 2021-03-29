"""Defines decorators and plotting functions for xarrays

TODO: Add docstrings and params to utility functions
"""

from functools import wraps

import numpy as np
import earthpy.plot as ep
import rasterio
from rasterio.enums import Resampling
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


def scale(xobj, *scaling):
    """Scales an xarray object created using rioxarray

    Parameters
    ----------
    xobj: xarray.DataArray or xarray.Dataset
        the xarray object to scale
    scaling: float or list of floats
        scaling factors. If two are provided, the order is width, height.
        Values less than one downsample (shrink) the object, values greater
        than one upsample (stretch) the object.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        the scaled xarray object
    """
    width = xobj.sizes["x"]
    height = xobj.sizes["y"]

    if len(scaling) == 1:
        scaling = (scaling[0], scaling[0])
    if len(scaling) != 2:
        raise ValueError("Must provide 1-2 values for scaling")

    width = int(width * scaling[0])
    height = int(height * scaling[1])

    return xobj.rio.reproject(xobj.rio.crs, shape=(width, height))


def rotation(xobj, angle, pivot=None):
    """Rotates an xarray object created using rioxarray around a pivot

    Parameters
    ----------
    angle: float
        angle in degrees to rotate the object
    pivot: tuple
        coordinates around which to pivot. Use array coordinates, not
        the CRS coordinates. If not given, pivots around the center.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        the rotated xarray object
    """

    width = xobj.sizes["x"]
    height = xobj.sizes["y"]

    # If pivot isn't given, rotate around center
    if pivot is None:
        pivot = [d // 2 for d in (width, height)]

    # Use the maximum dimension of the current object to calculate a shape
    # that will accomodate any rotation. If the orientation of the data is
    # not the same as the array (for example, if the array has already been
    # rotated), this will overestimate the size of the canvas required. Rows
    # and columns consisting of all NaNs are trimmed later to account for this.
    diag = int((width ** 2 + height ** 2) ** 0.5)

    # Rotate object based on the existing transform, then translate the
    # rotated object to the center of the new, larger canvas
    transform = xobj.rio.transform()
    transform *= transform.rotation(angle, pivot)
    transform *= transform.translation((width - diag) // 2,
                                       (height - diag) // 2)

    # Rotate the object according to the transform
    rotated = xobj.rio.reproject(
        xobj.rio.crs, shape=(diag, diag), transform=transform
    )

    # Trim all-Nan rows and columns before returning
    return rotated.dropna("x", "all").dropna("y", "all")


def scale_file(path, scaled, *scaling):
    """Resamples and saves a raster

    Convenience function based on
    https://rasterio.readthedocs.io/en/latest/topics/resampling.html?highlight=resampling#up-and-downsampling.
    Resizing large rasters after loading is memory intensive and resizing
    while reading is often preferable.

    Parameters
    ----------
    path: str
        path to the original file
    scaled: str
        path to save the scaled file
    scaling: float
        scaling factors. If two values are provided, the order is width,
        height. Values less than one downsample (shrink) the object, values
        greater than one upsample (stretch) the object.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        if 1-2 scaling factors are not provided
    """

    if len(scaling) == 1:
        scaling = (scaling[0], scaling[0])
    if len(scaling) != 2:
        raise ValueError("Must provide either 1 or 2 values for scaling")

    with rasterio.open(path) as src:

        # Resample the data as it is read in
        imgdata = src.read(
            out_shape=(src.count,
                       int(src.height * scaling[1]),
                       int(src.width * scaling[0])),
            resampling=Resampling.bilinear)

        # Record the transform
        transform = src.transform * src.transform.scale(
            (src.width / imgdata.shape[-1]),
            (src.height / imgdata.shape[-2])
        )

        # Save the scaled image
        with rasterio.Env():
            profile = src.profile

            profile.update(
                width=imgdata.shape[-2],
                height=imgdata.shape[-1],
                transform=transform
            )

            with rasterio.open(scaled, "w", **profile) as dst:
                dst.write(imgdata)
