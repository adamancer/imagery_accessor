"""Defines dataframe accessor to store and query instrument metadata"""
import numpy as np
import pandas as pd




@pd.api.extensions.register_dataframe_accessor("im")
class InstrumentAccessor:
    """Extends dataframe to store and retrieve instrument band metadata"""
    imagery_accessor_class = None


    def __init__(self, df):
        self._obj = df

        # Index and expand the dataframe when first loaded
        if "_instrument" not in self._obj.columns:

            # Use None for nulls instead of NaN
            for series in (self._obj.band_name, self._obj.sensor):
                series.where(series.notna(), None, inplace=True)

            # Add column to store keys for datasets
            self._obj["_dataset_key"] = None

            # Add lower case columns to simplify lookups
            self._obj["_instrument"] = self._obj["instrument"].str.lower()
            self._obj["_sensor"] = self._obj["sensor"].str.lower()
            self._obj["_band_name"] = self._obj["band_name"].str.lower()

            # Split ranges (e.g., Landsat 8-9)
            rows = self._obj[self._obj.instrument.str.startswith("Landsat")]
            for _, row in rows.iterrows():
                nums = row.instrument.rsplit(" ", 1)[-1].split("-")
                if len(nums) > 1:
                    nums = [int(n) for n in nums]
                    for num in range(min(nums), max(nums) + 1):

                        # Ignore keys with leading underscore
                        new = {k: v for k, v in row.to_dict().items()
                               if not k.startswith("_")}

                        new["instrument"] = f"Landsat {num}"
                        self.add_band(**new)

                    # Filter out the original row
                    self._obj = self._obj[
                        self._obj.instrument != row.instrument]

            # Set index as instrument-sensor-band_id
            self._obj["_"] = self._obj["instrument"].str.lower().str.cat(
                [self._obj["sensor"].str.lower(),
                 self._obj["band_id"].astype(str)],
                sep="-",
                na_rep="None")
            self._obj.set_index("_", inplace=True)

            # Sort and index the array
            self._obj.sort_values(["instrument", "sensor", "band_id"],
                                  inplace=True)

            # Update the attribute on the ImageryAccessor
            self.update_imagery_accessor()


    def get_band_by_id(self, band_id, instrument, sensor=None):
        """Gets band corresponding to the given ID

        Parameters
        ----------
        band_id: int
            the id of a band
        instrument: str
            name of an instrument
        sensor: str
            the shorthand name of a sensor

        Returns
        -------
        xarray.DataArray
            the array corresponding to the name
        """

        bands = self._obj.loc[
            (self._obj._instrument == instrument.lower())
            & (self._obj.band_id == band_id)
        ]

        # Filter results by sensor if given
        if sensor:
            bands = bands.loc[self._obj._sensor == sensor.lower()]

        return bands.iloc[0]


    def get_band_by_name(self, name, instrument, sensor=None):
        """Gets band corresponding to the given name

        Parameters
        ----------
        name: str
            the name of a band (red, green, blue, etc.)
        instrument: str
            name of an instrument
        sensor: str
            the shorthand name of a sensor

        Returns
        -------
        xarray.DataArray
            the array corresponding to the name
        """

        bands = self._obj.loc[
            (self._obj._instrument == instrument.lower())
            & (self._obj._band_name == name.lower())
        ]

        # Filter results by sensor if given
        if sensor:
            bands = bands.loc[self._obj._sensor == sensor.lower()]

        return bands


    def get_band_by_wavelength(self, wavelengths, instrument, sensor=None):
        """Gets bands corresponding to the given wavelength(s)

        FIXME: For single wavelengths, return best match

        Parameters
        ----------
        wavelengths: int or list-like
            wavelengths in nm to match
        instrument: str
            name of an instrument
        sensor: str
            the shorthand name of a sensor

        Returns
        -------
        xarray.DataArray
            Array corresponding to the wavelengths
        """

        # Calculate mean wavelength for comparisons
        mean_wl = np.mean(wavelengths)
        min_wl = np.min(wavelengths)
        max_wl = np.max(wavelengths)

        bands = self._obj.loc[
            (self._obj._instrument == instrument.lower())
            & (
                ((self._obj.min_wavelength_nm != self._obj.max_wavelength_nm)
                 & (mean_wl >= self._obj.min_wavelength_nm)
                 & (mean_wl <= self._obj.max_wavelength_nm))
                |
                ((self._obj.min_wavelength_nm == self._obj.max_wavelength_nm)
                 & (min_wl <= self._obj.min_wavelength_nm)
                 & (max_wl >= self._obj.max_wavelength_nm))
            )]

        # Filter results by sensor if given
        if sensor:
            bands = bands.loc[self._obj._sensor == sensor.lower()]

        return bands


    def add_band(self,
                 band_id,
                 band_name,
                 instrument,
                 sensor=None,
                 min_wavelength_nm=None,
                 max_wavelength_nm=None):
        """Adds a band to the instrument dataframe

        Parameters
        ----------
        band_id: int
            index of the band
        band_name: int
            name of the band
        instrument: str
            name of an instrument
        sensor: str
            the shorthand name of a sensor
        min_wavelength_nm: int
            mininum wavelength of band in nanometers
        max_wavelength_nm: int
            maximum wavelength of band in nanometers

        Returns
        -------
        None
        """
        self._obj = self._obj.append({
            "instrument": instrument,
            "sensor": sensor,
            "band_id": band_id,
            "band_name": band_name,
            "min_wavelength_nm": min_wavelength_nm,
            "max_wavelength_nm": max_wavelength_nm,
            "_instrument": instrument.lower(),
            "_sensor": sensor.lower() if sensor else None,
            "_band_name": band_name.lower() if band_name else None,
        }, ignore_index=True)

        self._obj["_"] = self._obj["instrument"].str.lower().str.cat(
                [self._obj["sensor"].str.lower(),
                 self._obj["band_id"].astype(str)],
                sep="-",
                na_rep="None")
        self._obj.set_index("_", inplace=True)

        self.update_imagery_accessor()


    def assign_dataset_keys(self, bands_to_keys, instrument, sensor=None):
        """Assigns dataset keys corresponding to a set of bands

        Parameters
        ----------
        bands_to_keys: dict or list-like
            dict or list mapping bands to keys. Lists are converted to
            dicts using a 1-based indexed (so index 0 maps to key 1)
        instrument: str
            name of an instrument
        sensor: str
            the shorthand name of a sensor

        Returns
        -------
        None
        """
        if not isinstance(bands_to_keys, dict):
            bands_to_keys = {i + 1: v for i, v in enumerate(bands_to_keys)}

        rows = []
        for band, dataset_key in bands_to_keys.items():
            idx = f"{instrument.lower()}-{sensor.lower() if sensor else None}-{band}"
            rows.append({
                "_": idx,
                "instrument": instrument,
                "sensor": sensor,
                "band_id": band,
                "_dataset_key": dataset_key
            })

        bands_to_keys = pd.DataFrame(rows)
        bands_to_keys.set_index("_", inplace=True)

        self._obj.update(bands_to_keys)


    def list(self):
        """Displays list of available instruments and bands"""
        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None
        ):
            print(self._obj.loc[:, :"max_wavelength_nm"])


    def update_imagery_accessor(self):
        """Updates the instruments attribute on the ImageryAccessor

        Some dataframe operations, like append, cannot be performed in place,
        which means the ImageryAccessor.instruments class attribute will not
        be updated when those operation are performed. Internal methods can
        use this method to update that attribute.
        """
        self.imagery_accessor_class.instruments = self._obj
