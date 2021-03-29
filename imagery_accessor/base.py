"""Defines accessor with methods for preserving metadata

TODO: Test on arrays with more than three dimensions
TODO: Figure out when to use copying
TODO: Add inplace kwarg?
TODO: Verify docstrings, especially type being returned
"""
import numpy as np
import xarray as xr

from .metadata import MetadataRef
from .utils import add_dim, copy_xr_metadata




class BaseAccessor:
    """Extends xarray objects to parse and run calculations on satellite data

    This accessor makes limited use of type checking where functionality of
    arrays and datasets differs.

    Attributes
    ----------
    metadata_coord: str
        name of the coord that stores metadata for this accessor
    name: str
        name of the accessor as registered with xarray. Assigned on
        subclasses.
    metadata: dict
        metadata stored on the imagery_ref scalar variable
    """


    def __init__(self, xobj):
        self._obj = xobj
        self._metadata = None

        # Create metadata_ref coord if it does not exist
        try:
            self._obj.coords["metadata_ref"]
        except KeyError:
            self._obj.coords["metadata_ref"] = 0


    def __iter__(self):
        """Iterates across layers of this object"""
        if isinstance(self._obj, xr.DataArray):
            obj = self._obj if len(self._obj.shape) > 2 else [self._obj]
            return iter(obj)
        return iter(self._obj.values())


    @property
    def metadata(self):
        """Returns imagery metadata, deriving it from the filename if possible

        Metadata is attached to the metadata_ref scalar variable, which allows
        it to carry over on many array operations. If that scalar variable
        does not exist, it is added the first time the metadata property
        is accessed.
        """
        if not self._metadata:
            self._metadata = MetadataRef(self._obj)
        return self._metadata


    @metadata.setter
    def metadata(self, metadata):
        self._obj.coords["metadata_ref"].attrs = dict(metadata)
        self._metadata = MetadataRef(self._obj)
        self._metadata.propagate()


    @property
    def name(self):
        return self.metadata["ACC_NAME"]


    @name.setter
    def name(self, val):
        self._obj.coords["metadata_ref"].attrs["ACC_NAME"] = val


    def concat(self, xarrs, **kwargs):
        """Concatenates arrays into an object of the same type as original

        Parameters
        ----------
        xarrs: list-like
            list of arrays
        kwargs:
            any keywrod argument accepted by xr.concat. Only used when
            concatenating arrays.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Concatenated array or dataset
        """
        if isinstance(self._obj, xr.DataArray):
            return xr.concat(xarrs, **kwargs)

        return xr.Dataset(dict(zip(self._obj.keys(), xarrs)),
                          coords=self._obj.coords,
                          attrs=self._obj.attrs)


    def npfunc(self, npfunc, *args, **kwargs):
        """Runs a numpy function on the xarray

        This method allows the user to run certain numpy functions without
        losing the xarray attributes, coordinates, etc. managed by the
        accessor. The numpy function must return an array with the same
        shape as the original.

        Parameters
        ----------
        npfunc: str or callable
            either the name of a numpy function or the function itself. The
            function must take the xarray object as the first argument.
        args:
            any additional arguments accepted by the numpy function
        kwargs:
            any additional jeyword arguments accepted by the numpy function

        Returns
        -------
        xarray.DataArray
            New array with results of the numpy function and all original
            metadata
        """

        # Convert string to function
        if isinstance(npfunc, str):
            npfunc = getattr(np, npfunc)

        obj = self._obj
        if isinstance(self._obj, xr.Dataset):
            obj = obj.to_array(dim="band")

        arr = npfunc(obj, *args, **kwargs)
        return self.copy_xr_metadata(arr, name=f"np.{npfunc.__name__}")


    def copy_xr_metadata(self, other, name="array"):
        """Copies coords and dims from current object to another array

        This allows metadata stored in the original to be reintegrated after
        using a function that drops them (like any numpy function).

        Parameters
        ----------
        other: numpy.array
            an array with the same shape as the current data array

        Returns
        -------
        xarray.DataArray
            Array with coords and dims copied over
        """
        xobj = copy_xr_metadata(self._obj, other)
        if "band" not in xobj.coords and "band" not in xobj.dims:
            xobj = add_dim(xobj, dim="band", coords={"name" : [name]})
        return xobj
