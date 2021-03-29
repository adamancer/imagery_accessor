"""Defines accessor for plotting and preserving metadata on satellite data

TODO: Identify metadata available through the rio accessor (like resolution)
"""
from datetime import datetime
from glob import glob
import os

import earthpy.spatial as es
import numpy as np
import pandas as pd
from rasterio.plot import plotting_extent
import rioxarray as rxr
from shapely.geometry import box
import xarray as xr

from .base import BaseAccessor
from .instruments import InstrumentAccessor
from .utils import add_dim, copy_xr_metadata, hist, plot_bands,plot_rgb




@xr.register_dataarray_accessor("im")
@xr.register_dataset_accessor("im")
class ImageryAccessor(BaseAccessor):
    """Extends xarray objects to parse and run calculations on satellite data

    This accessor makes limited use of type checking where functionality of
    arrays and datasets differs.

    Attributes
    ----------
    red: xarray.DataArray
        red band
    green: xarray.DataArray
        green band
    blue: xarray.DataArray
        blue band
    nir: xarray.DataArray
        near-infrared band
    pixel_qa: xarray.DataArray
        pixel QA band for Landsat
    radsat_qa: xarray.DataArray
        radsat QA band for Landsat 8
    extent: tuple
        plotting bounds in matplotlib order
    metadata: dict
        metadata stored on the metadata_ref scalar variable

    Class Attributes
    ----------------
    instruments: pandas.DataFrame
        contains metadata for bands available on various instruments
    """
    instruments = pd.read_csv(
        os.path.join(__file__, "..", "config", "instruments.csv"))


    def __init__(self, xobj):
        super().__init__(xobj)
        self.name = "im"

        # Compile ad hoc indices for any coord that is the same size as band.
        # These are used (1) to map from non-dimensional coordinates to
        # dimensions for use with xobj.sel() and (2) to preserve coordinates
        # across internal methods that can drop them. The second part only
        # kind of works (see the mask_from_pixel_qa() method below for an
        # example of where it fails.
        #
        # TODO: Check if this is reimplementing built-in xarray functionality
        # TODO: Investigate reintegrating coords/dims from metadata
        self._coord_to_dim = {}
        self._dim_to_coord = {}
        try:
            bands = self.band_ids()
        except (KeyError, TypeError):
            # Pull values from metadata if possible
            try:
                indexes = self.metadata["ACC_INDEXES"]
                self._coord_to_dim = indexes["coord_to_dim"]
                self._dim_to_coord = indexes["dim_to_coord"]
            except KeyError:
                pass
        else:
            for key, val in xobj.coords.items():
                try:
                    if key != "band" and len(val) == len(bands):
                        vals = list(val.values)
                        self._coord_to_dim[key] = dict(zip(vals, bands))
                        self._dim_to_coord[key] = dict(zip(bands, vals))
                except TypeError:
                    pass
            # FIXME: Recursion error in metadata if ACC_INDEXES propagates
            self.metadata["ACC_INDEXES"] = {
                "coord_to_dim": self._coord_to_dim,
                "dim_to_coord": self._dim_to_coord
            }



    @property
    def blue(self):
        try:
            return self.get_band_by_name("blue")
        except ValueError as exc:
            try:
                return self.get_band_by_wavelength((450, 495))
            except:
                raise exc


    @property
    def green(self):
        try:
            return self.get_band_by_name("green")
        except ValueError as exc:
            try:
                return self.get_band_by_wavelength((495, 570))
            except:
                raise exc


    @property
    def red(self):
        try:
            return self.get_band_by_name("red")
        except ValueError as exc:
            try:
                return self.get_band_by_wavelength((620, 750))
            except:
                raise exc


    @property
    def nir(self):
        try:
            return self.get_band_by_name("nir")
        except ValueError as exc:
            try:
                return self.get_band_by_wavelength((750, 900))
            except:
                raise exc


    @property
    def pixel_qa(self):
        return self.get_band_by_name("pixel_qa")


    @property
    def radsat_qa(self):
        return self.get_band_by_name("radsat_qa")


    @property
    def extent(self):
        """Returns extent of first child array"""
        for obj in self:
            return plotting_extent(obj, obj.rio.transform())


    def parse_metadata(self):
        """Parses metadata in filename if possible

        Returns
        -------
        None
        """
        for method in [
            self._parse_landsat_product_id
        ]:
            try:
                self.metadata.update(self._parse_landsat_product_id())
                break
            except (KeyError, ValueError):
                raise


    def get_band_metadata(self, band_id):
        """Gets band corresponding to the given name

        Parameters
        ----------
        band_id: int
            the id of a band

        Returns
        -------
        xarray.DataArray
            the array corresponding to the band

        Raises
        ------
        ValueError
            if name does not match a defined band
        """

        return self.instruments.im.get_band_by_id(
            band_id,
            self.metadata["instrument"],
            self.metadata.get("sensor")
        )


    def get_band_by_name(self, name):
        """Gets band corresponding to the given name

        Parameters
        ----------
        name: str
            the name of a band (red, green, blue, etc.)

        Returns
        -------
        xarray.DataArray
            the array corresponding to the name

        Raises
        ------
        ValueError
            if name does not match a defined non-dimensional coord
        """

        # Query the instrument dataframe for this name. This check will fail
        # if no instrument is provided.
        try:
            bands = self.instruments.im.get_band_by_name(
                name,
                self.metadata["instrument"],
                self.metadata.get("sensor"),
            )
        except KeyError:
            bands = None

        # Only allow unique matches to be retrieved by name
        if bands is not None and len(bands) > 1:
            raise ValueError(f"'{self.__class__.__name__}' object has"
                             f" multiple bands that mach '{name}'")

        try:
            if isinstance(self._obj, xr.DataArray):
                try:
                    return self._obj.sel(band=self._coord_to_dim["name"][name])
                except (IndexError, KeyError):
                    return self._obj[bands.iloc[0].band_id]

            try:
                return self._obj[self._coord_to_dim["name"][name]]
            except KeyError:
                return self._obj[bands.iloc[0]._dataset_key]

        except (AttributeError, IndexError, KeyError):
            # IndexError if no bands match
            # KeyError if band/key not found
            raise ValueError(f"'{self.__class__.__name__}' object has"
                             f" no data in attribute '{name}'")


    def get_band_by_wavelength(self, wavelengths):
        """Gets band corresponding to the given wavelength(s)

        Parameters
        ----------
        wavelengths: int
            wavelengths in nm to match

        Returns
        -------
        xarray.DataArray
            Array corresponding to the wavelengths

        Raises
        ------
        KeyError
            if no instrument is provided in the metadata attribute
        ValueError
            if wavelength does not match a band
        """

        # Query the instrument dataframe for this wavelength
        bands = self.instruments.im.get_band_by_wavelength(
            wavelengths,
            self.metadata["instrument"],
            self.metadata.get("sensor"),
        )

        try:
            bands = [b for _, b in bands.iterrows()]

            if isinstance(self._obj, xr.DataArray):
                arrs = [self._obj.sel(band=b.band_id) for b in bands]
            else:
                arrs = [self._obj.sel(band=b._dataset_key) for b in bands]

            # Return the middle of the returned rows
            return arrs[int(len(arrs) / 2)]

        except (IndexError, KeyError):
            # IndexError if no bands match
            # KeyError if band/key not found
            raise ValueError(f"'{self.__class__.__name__}' object has"
                             f" no band matching wavelengths '{wavelengths}'")


    def bands(self, colors, delim=None):
        """Returns new array with bands and corresponding title

        Parameters
        ----------
        colors: list-like or str
            ordered list of colors as either a string ("rgb" or "cir"),
            a list of strings (["red", "green", "blue"]), a list of band IDs
            ([1, 2, 3]), or a list of wavelengths ([450, 550, 650]).
        delim: str
            delim to use to join the title. If None, the title is returned
            as a list.

        Returns
        -------
        tuple
            Tuple of (array of bands, title(s))
        """

        # Define one-letter abbreviations for bands
        abbrs = {c[0]: c for c in ["red", "green", "blue", "nir"]}

        # Split strings into lists using available abbreviations
        if isinstance(colors, str):
            colors = list(colors.lower())

            # If any letter does not correspond to a color band, use the
            # whole string as-is
            if set(colors) - abbrs.keys():
                colors = ["".join(colors)]

        elif isinstance(colors, (float, int)):
            colors = [colors]

        # Evaluate lists of numbers
        if not isinstance(colors[0], str):

            # Assume wavelengths if numeric values greater than array length
            if any([c > len(self._obj) for c in colors]):
                bands = [self.get_band_by_wavelength(c) for c in colors]

                # Build title from actual (as opposed to provided) wavelengths
                # FIXME: Dataset will probably fail here
                meta = [self.get_band_metadata(b["band"].item()) for b in bands]
                wvlns = [sorted({m.min_wavelength_nm,
                                 m.max_wavelength_nm}) for m in meta]

                title = [f"{':'.join([str(s) for s in w])} nm" for w in wvlns]

            # Otherwise assume they're band numbers
            elif all([c < len(self._obj) for c in colors]):
                bands = [self._obj.sel(band=c) for c in colors]
                title = [self._dim_to_coord["name"][c] for c in colors]

        # CIR is a special case
        elif colors == ["cir"]:
            bands = [self.nir, self.red, self.green]
            title = ["nir", "red", "green"]

        # Evaluate lists of strings
        else:
            bands = []
            title = []
            for color in colors:
                color = color.lower()
                if color in abbrs.values():
                    # Color is a valid name
                    bands.append(self.get_band_by_name(color))
                    title.append(color)
                else:
                    # Color is an abbreviation (r, g, b)
                    bands.append(self.get_band_by_name(abbrs[color]))
                    title.append(abbrs[color])

        # Format and capitalize title as string if delimiter is given
        if delim is not None:
            if colors == ["cir"]:
                title = "CIR"
            elif "nir" not in title and not title[0][0].isnumeric():
                title = "".join([t[0] for t in title]).upper()
            else:
                title = delim.join(title) \
                              .replace("nir", "NIR") \
                              .replace("swir", "SWIR")
                title = title[0].upper() + title[1:]

        # Assign zero-indexed band
        for i, band in enumerate(bands):
            # Can't reassign coordinates on a DataArray, so must clear band
            if isinstance(self._obj, xr.DataArray):
                del band["band"]
                band = band.squeeze()

            band["band"] = i
            bands[i] = band

        return xr.concat(bands, dim="band"), title


    def clean_all_bands(self, **kwargs):
        """Cleans all bands in the array

        Parameters
        ----------
        kwargs:
            any keyword argument accepted by clean_band()

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Array or dataset with out-of-range data set to nan
        """
        # Clean each band and build a new DataArray
        out_xr = []
        for arr in self:
            if not self.is_qa(arr):
                out_xr.append(self.clean_band(arr, **kwargs))

        # If a QA bands exist, include but do not clean them
        for band in ("pixel_qa", "radsat_qa"):
            try:
                out_xr.append(self.get_band_by_name(band).copy())
            except ValueError:
                pass

        return self.concat(out_xr, dim="band")


    def is_qa(self, xobj):
        """Tests if band is a QA band (Landsat only)

        FIXME: Will fail with dataset
        FIXME: Will fail with anything except Landsat

        Returns
        -------
        bool
            True if band is a QA band, false if not
        """
        try:
            return not isinstance(xobj.band.item(), int)
        except AttributeError:
            return False


    def no_qa(self):
        """Returns copy of array with QA bands removed (Landsat only)

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Array or dataset with no QA bands
        """
        return self.concat([a for a in self if not self.is_qa(a)], dim="band")


    def mask_from_pixel_qa(self, vals):
        """Masks array using the QA layer (Landsat only)

        Parameters
        ----------
        vals: list-like
            either a list of pixels to mask or a list of Landsat flags to use
            to derive the mask based on the QA band. If the latter, flags are
            converted to pixels used the earthpy.mask module.

        Returns
        -------
        xarray.DataArray or xarray.Dataset
            Array or dataset with specified features set to nan

        Raises
        ------
        ValueError
            if flags are invalid for the given Landsat satellite
        """

        # Map flags to values if strings given, otherwise use the values as-are
        if isinstance(vals[0], str):

            # Format key for earthy.mask lookup from satellite number
            satnum = self.metadata["satellite"].lstrip("0")
            key = "L{}".format(satnum)
            if key != "L8":
                key = "L47"  # Landsat 4-7?

            # Align casing for specified flags and read pixel values. Throw an
            # error if any key is not found.
            avail = {k.lower(): k for k in self._get_landsat_flags()}
            mask = []
            for flag in vals:
                flag = flag.lower()
                try:
                    mask.extend(em.pixel_flags["pixel_qa"][key][avail[flag]])
                except KeyError:
                    raise ValueError(f"QA flags for Landsat {satnum}"
                                     f" must be one of {avail.values()}"
                                     f" ({flags} given)")
        else:
            mask = vals[:]

        # Mask each band and build a new DataArray
        out_xr = []
        for arr in self:
            if not self.is_qa(arr):
                out_xr.append(arr.where(~self.pixel_qa.isin(mask)))

                # Ad hoc attributes are lost here, so set band manually
                out_xr[-1]["band"] = arr.band

        # If QA bands exist, include them in the output but do not clean them
        for band in ("pixel_qa", "radsat_qa"):
            try:
                out_xr.append(self.get_band_by_name(band).copy())
            except ValueError:
                pass

        # Reattach non-dimensional coordinates using the precompiled indexes.
        # This only works for DataArrays but will not cause an error with
        # Datasets since the indexes are created for both.
        coords = {}
        for key, val in self._dim_to_coord.items():
            coords[key] = ("band", list(val.values()))

        return self.concat(out_xr, dim="band").assign_coords(**coords)


    def mask_clouds(self, confidence="high"):
        """Masks clouds down to the given confidence level (Landsat only)

        This is a convenience method for masking clouds in Landsat data. Use
        mask_from_pixel_qa() if you need more control over the exact pixels
        being masked.

        Parameters
        ----------
        confidence:
           level of confidence assigned by Landsat (high, medium, low)

        Returns
        -------
        xarray.DataArray
            Array with cloud features set to nan

        Raises
        ------
        ValueError
            if confidence is not one of high, medium, or low
        """
        flags = ["Cloud", "Cloud Shadow"]

        if confidence:
            conf_levels = ["high", "medium", "low"]
            if confidence and confidence.lower() not in conf_levels:
                raise ValueError(f"Confidence must be one of {conf_levels}"
                                 f"('{confidence}' given)")

            # Add terms for the given confidence to search, making sure
            # to catch higher confidence levels as well (so medium will
            # also catch high confidence pixels, for example)
            terms = []
            for conf in conf_levels:
                terms.extend([f"{conf} cloud", f"{conf} cirrus"])
                if conf == confidence.lower():
                    break

            flags.extend(self._get_landsat_flags(terms))

        return self.mask_from_pixel_qa(flags)


    def evi(self):
        """Calculates Enhanced Vegetation Index (EVI)

        Returns
        -------
        xarray.DataArray
            Array containing EVI values
        """

        # Some instruments have multiple infrared bands, so try to select
        # by wavelength before falling back to the named NIR band
        try:
            nir = self.get_band_by_wavelength(865)
        except ValueError:
            nir = self.nir

        xobj = 2.5 * (nir - self.red) / (nir + 6 * self.red - 7.5 * self.blue + 1)
        xobj = add_dim(xobj, dim="band", coords={"name": ["EVI"]})
        return xobj


    def nbr(self):
        """Calculates Normalized Burn Ratio (NBR)

        Returns
        -------
        xarray.DataArray
            Array containing NBR values
        """

        # Some instruments have multiple infrared bands, so try to select
        # by wavelength before falling back to the named NIR band
        try:
            nir = self.get_band_by_wavelength(865)
        except ValueError:
            nir = self.nir

        # Likewise, some instruments have multiple SWIR bands. Use the one
        # around 2125 nm.
        swir = self.get_band_by_wavelength(2125)

        xobj = (nir - swir) / (nir + swir)
        xobj = add_dim(xobj, dim="band", coords={"name": ["NBR"]})
        return xobj


    def nbr2(self):
        """Calculates Normalized Burn Ratio 2 (NBR2)

        Returns
        -------
        xarray.DataArray
            Array containing NBR2 values
        """

        swir_1 = self.get_band_by_wavelength(1610)
        swir_2 = self.get_band_by_wavelength(2125)

        xobj = (swir_1 - swir_2) / (swir_1 + swir_2)
        xobj = add_dim(xobj, dim="band", coords={"name": ["NBR2"]})
        return xobj


    def ndvi(self):
        """Calculates Normalized Difference Vegetation Index (NDVI)

        Returns
        -------
        xarray.DataArray
            Array containing NDVI values
        """

        # Some instruments have multiple infrared bands, so try to select
        # by wavelength before falling back to the named NIR band
        try:
            nir = self.get_band_by_wavelength(865)
        except ValueError:
            nir = self.nir

        xobj = (nir - self.red) / (nir + self.red)
        xobj = add_dim(xobj, dim="band", coords={"name": ["NDVI"]})
        return xobj


    def hist(self, bands=None, **kwargs):
        """Plots all bands in the array as a histogram

        Parameters
        ----------
        kwargs:
            any keyword argument accepted by ep.hist(). The title
            extent, and figsize arguments are provided if not given.

        Returns
        -------
        None
        """

        if bands:
            obj, titles = self.bands(bands)
        else:
            obj = self._obj
            titles = self.titles()

        if isinstance(obj, xr.Dataset):
            obj = obj.to_array()

        # Set defaults based on array/dataset metadata
        kwargs.setdefault("title", titles)

        return hist(obj, **kwargs)


    def plot_bands(self, bands=None, **kwargs):
        """Plots all bands in the array

        Parameters
        ----------
        kwargs:
            any keyword argument accepted by ep.plot_rgb(). The title
            extent, and figsize arguments are provided if not given.

        Returns
        -------
        None
        """

        if bands:
            obj, titles = self.bands(bands)
        else:
            obj = self._obj
            titles = self.titles()

        if isinstance(obj, xr.Dataset):
            obj = obj.to_array()

        # Set defaults based on array/dataset metadata
        kwargs.setdefault("extent", self.extent)
        kwargs.setdefault("figsize", self.figsize(12))
        kwargs.setdefault("title", titles)

        return plot_bands(obj, **kwargs)


    def plot_rgb(self, bands="rgb", **kwargs):
        """Creates an RGB plot using the given bands

        Parameters
        ----------
        bands: str or list-like
            list of colors to plot as string ("rgb") or list
            (["red", "green", "blue"]). The keyword "cir" is also valid.
        kwargs:
            any keyword argument accepted by ep.plot_rgb(). The title and
            extent arguments are provided if not given.

        Returns
        -------
        None
        """

        # Bands are always returned as an array, so don't need dataset check
        xarr, title = self.bands(bands, delim="-")
        kwargs.setdefault("title", title)

        # Extent can vary in a dataset, so use exact bands to set the default
        kwargs.setdefault("extent", xarr.im.extent)

        # Mask null values and plot
        return plot_rgb(xarr, **kwargs)


    def band_ids(self):
        """Returns list of band indices"""
        if isinstance(self._obj, xr.DataArray):
            return list(self._obj.coords["band"].values)
            #return [a.band.item() for a in self]
        else:
            return list(self._obj.keys())


    def titles(self):
        """Calculates titles based on names assigned to bands

        Returns
        -------
        list
            List of names or band ids to use as plot titles
        """
        bands = self.band_ids()
        try:
            return [self._dim_to_coord["name"].get(b, b) for b in bands]
        except KeyError:
            # Fall back to band IDs if name coord is not populated
            return bands


    def figsize(self, width=12, nrows=1, ncols=1):
        """Calculates size using aspect ratio of data and number of rows/cols

        Width is preserved and height varied to maintain aspect ratio.

        Parameters
        ----------
        dims: float-like
            width in inches. Defaults to 12, which is what earthpy uses,
            but if set to None will use the user default.

        Returns
        -------
        tuple
            Figsize as (width, height)
        """

        if width is None:
            width = mpl.rcParams["figure.figsize"][0]

        for obj in self:
            asp_ratio = obj.shape[1] / obj.shape[0]
            break
        height = width * asp_ratio if asp_ratio < 1 else width / asp_ratio

        # Scale and pad each dimension based on the number of rows and columns
        if ncols > 1:
            width *= 1.1 * ncols
        if nrows > 1:
            height *= 1.1 * nrows
        return (width, height)


    def all_nan(self):
        """Tests if all values in the array (except the QA bands) are NaN

        QA bands are excluded because they will rarely be masked and are kept
        as-is when used to mask other bands.

        Returns
        -------
        bool
            True if all values in array are NaN, False otherwise
        """
        try:
            self.pixel_qa
            self.radsat_qa

        except ValueError:
            # If no QA bands, check the whole array
            try:
                np.nanargmin(self._obj)  # throws ValueError if all NaN
            except ValueError:
                return True

        else:
            # If the QA bands exist, exclude them from the NaN check
            for arr in self:
                if isinstance(arr.band, int):
                    try:
                        np.nanargmin(arr)
                    except ValueError:
                        return True
        return False


    @staticmethod
    def clean_band(band, valid_range=None):
        """Cleans a single band

        Parameters
        -----------
        band: str or array
            Either an array or the path to the array to be opened
        valid_range: tuple (optional)
            A tuple of min and max range of values for the data. Default = None

        Returns
        -----------
        xarray.DataArray
            Array with out-of-range values set to nan
        """
        # Read band from file if passed as a string
        if isinstance(band, str):
            band = open_clip_sqeeze(band)

        # Remove pixels outside valid range if provided
        if valid_range:
            mask = ((band < min(valid_range)) | (band > max(valid_range)))
            band = band.where(~xr.where(mask, True, False))

        return band


    def _get_landsat_flags(self, term=None):
        """Gets flags available for the current Landsat satellite

        Parameters
        ---------
        term: str or list-like
           limit results to flags contianing the given term(s) (optional)

        Returns
        -------
        list
            List of flags
        """
        if isinstance(term, (list, tuple)):
            flags = []
            for term in term:
                flags.extend(self._get_landsat_flags(term))
            return flags

        # Format key for earthy.mask lookup from satellite number
        satnum = self.metadata["satellite"].lstrip("0")
        key = "L{}".format(satnum)
        if key != "L8":
            key = "L47"  # Landsat 4-7?

        flags = em.pixel_flags["pixel_qa"][key]
        if term:
            return [f for f in flags if term.lower() in f.lower()]
        return flags


    def _parse_landsat_product_id(self):
        """Parses acquisition and processing info from Landsat filename

        Returns
        -------
        dict
            Dictionary containing the parsed metadata
        """

        # Sensor lookup. The Landsat 8-9 values are hased because they
        # complicate the lookup and are not needed to distinguish the bands,
        # but they do create a semantic issues with the sensor.
        sensors = {
            #"C": None,
            #"O": "OLI",
            #"T": "TIRS",
            "E": "ETM+",
            "T": "TM",
            "M": "MSS",
        }

        # Get path from first child
        for obj in self:
            prod_id = os.path.basename(obj.coords["path"].item())
            break

        # Split into major parts
        sat, corr, wrs, acq, proc, coll_num, coll_cat = prod_id.split("_")[:7]

        return {
            # Cleaned-up metadata used for instrument lookups
            "instrument": f"Landsat {sat[3:]}",
            "sensor": sensors.get(sat[1]),
            # Original metadata
            "sensor_abbr": sat[1],
            "satellite": sat[3:],
            "wrs_path": wrs[:3],
            "wrs_row": wrs[3:],
            "acq_date": datetime.strptime(acq, "%Y%m%d"),
            "proc_date": datetime.strptime(proc, "%Y%m%d"),
            "coll_num": coll_num,
            "coll_cat": coll_cat
        }




# HACK: Populate imagery_accessor_class on the InstrumentAccessor so that
# class can update the attribute on non-inplace changes to the DataFrame.
InstrumentAccessor.imagery_accessor_class = ImageryAccessor




def open_imagery(path, source=None, bands=None, clip_bound=None, **kwargs):
    """Reads imagery from a path

    # FIXME: kwargs are being passed to multiple functions, seems bad?

    Parameters
    ---------
    path: str
        path to file or directory containing imagery
    source: str or dict (optional)
        name of the instrument or a dict with the instrument and sensor names
    bands: list-like (optional)
        list of colors in the order they appear in the imagery
    clip_bound: geopands.GeoDataFrame (optional)
        boundary to which to clip the imagery
    kwargs:
        keyword arguments to pass to function used to open images

    Returns
    -------
    xarray.DataArray or xarray.Dataset
         Array or dataset containing data from path
    """

    # Convert source to a dict if given as a string
    if isinstance(source, str):
        source = {"instrument": source}

    # Map named source to an opener or band order
    if source:
        handlers = {
            "landsat": stack_imagery,
            "modis": stack_imagery,
            "naip": ("red", "green", "blue", "nir"),
        }

        try:
            handler = handlers[source["instrument"].lower()]
        except KeyError:
            raise KeyError(
                f"instrument must be one of {list(handlers)}"
                f" ('{source}' provided)"
            )

        # Skip any path pointing to a file that wants to use stack_imagery
        if handler == stack_imagery and os.path.isfile(path):
            pass

        # Call the handler if it's callable
        elif callable(handler):
            return handler(path, clip_bound=clip_bound, **kwargs)

        # Otherwise the handler must be a list of bands.
        else:
            bands = handler

    # Open-clip-squeeze the file at path
    xobj = open_clip_squeeze(path, clip_bound=clip_bound, **kwargs)

    # If bands are given, add them as a dimension
    if bands:
        xobj = xobj.assign_coords(name=("band", list(bands)))

    # Record source in metadata
    if source:
        if isinstance(xobj, xr.DataArray):
            xobj.im.metadata = source
        else:
            for xdat in xobj:
                xdat.im.metadata = source

    return xobj


def open_clip_squeeze(path, clip_bound=None, **kwargs):
    """Opens, clips, and squeezes a raster

    Parameters
    ---------
    path: str
        path to an imagery file that can be opened by open_rasterio
    clip_bound: geopandas.GeoDataFrame
        boundary to which to clip the imagery
    kwargs:
        keyword arguments to pass to rioxarray.open_rasterio()

    Returns
    -------
    xarray.DataArray
         Clipped and squeezed array

    Raises
    ------
    ValueError
        if any xarray object does not have data in xobj.rio.crs
    """

    kwargs.setdefault("masked", True)
    xobjs = rxr.open_rasterio(path, **kwargs)

    # Datasets can return a list, so convert results to list for consistency
    if not isinstance(xobjs, list):
        xobjs = [xobjs]

    # Clip each object if a bound is given
    if clip_bound is not None:
        for i, xobj in enumerate(xobjs):

            # Raise an error if CRS is empty
            if xobj.rio.crs is None:
                raise ValueError("rio.crs attribue must not be empty")

            clip_box = box(*clip_bound.to_crs(xobj.rio.crs).total_bounds)
            xobjs[i] = xobj.rio.clip([clip_box],
                                     crs=xobj.rio.crs,
                                     all_touched=True,
                                     from_disk=True)

    # For arrays, squeeze and return the array
    if isinstance(xobjs[0], xr.DataArray):
        return xobjs[0].squeeze()

    # For datasets, always return a list
    return xobjs


def stack_imagery(path, clip_bound=None, min_band=1, max_band=7, **kwargs):
    """Combines TIFFs in a directory into an xarray

    Parameters
    ---------
    path: str
        path to directory containing a set of TIFFs
    clip_bound: geopands.GeoDataFrame
        boundary to which to clip the imagery
    min_band: int
        minimum band to retrieve
    max_band: int
        maximum band to retrieve

    Returns
    -------
    xarray.DataArray
         Array merging all matching TIFFs into one object

    Raises
    ------
    IOError
        If wrong number of bands found
    """

    # Define filename pattern to match both Landsat and MODIS
    pattern = f"*b[a0]*[{min_band}-{max_band}]*.tif"

    # Read TIFFs along path matching selected bands
    paths = []
    for fp in glob(os.path.join(path, pattern)):
        paths.append(fp)

    # Raise an error if any of the expected bands are missing
    expected = (max_band - min_band + 1)
    if len(paths) != expected:
        raise IOError(f"Wrong number of TIFFs found in {os.path.abspath(path)}"
                      f" (found {len(paths)}, expected={expected})")

    # Combine sorted list of TIFFs into an xarray
    out_xr = []
    for fp in sorted(paths):
        out_xr.append(open_clip_squeeze(fp, clip_bound=clip_bound, **kwargs))

        # Add min_band so that the band id matches original
        out_xr[-1]["band"] = len(out_xr) + min_band - 1

    # Get names for each band
    # TODO: Use instruments to get names (but need satellite for that)
    if "band" in paths[0]:
        names = ["aerosols", "blue", "green", "red", "nir", "swir_1", "swir_2"]
    else:
        names = ["red", "blue_green", "nir_2",
                 "green", "nir_5", "mir_6", "mir_7"]
    names = names[min_band - 1:max_band]

    # Check for Landsat QA bands
    for qa_band in ("pixel_qa", "radsat_qa"):
        try:
            fp = glob(os.path.join(path, f"*{qa_band}*.tif"))[0]
        except IndexError:
            pass
        else:
            out_xr.append(open_clip_squeeze(fp, clip_bound=clip_bound))
            out_xr[-1]["band"] = qa_band

            # Add QA band to list of dimensions
            paths.append(fp)
            names.append(qa_band)

    coords = {"path": ("band", [os.path.realpath(p) for p in paths])}
    if names:
        coords["name"] = ("band", names)

    # Create array and add additional coords
    xobj = xr.concat(out_xr, dim="band").assign_coords(**coords)

    # FIXME: Find another way to assign the satellite
    if "blue_green" in names:
        xobj.im.metadata["instrument"] = "MODIS"
    else:
        xobj.im.parse_metadata()

    return xobj


def stack_images(paths, names=None, clip_bound=None, **kwargs):
    out_xr = []
    for fp in sorted(paths):
        out_xr.append(open_clip_squeeze(fp, clip_bound=clip_bound, **kwargs))
        # Add min_band so that the band id matches original
        out_xr[-1]["band"] = len(out_xr)

    coords = {"path": ("band", [os.path.realpath(p) for p in paths])}
    if names:
        coords["name"] = ("band", names)

    return xr.concat(out_xr, dim="band").assign_coords(**coords)
