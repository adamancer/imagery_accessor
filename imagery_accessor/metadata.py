"""Defines dict to manage metadata on ImageryAccessor objects"""
import xarray as xr



class MetadataRef:

    def __init__(self, xobj):
        super().__init__()
        self._obj = xobj
        self._dict = self._obj.coords["metadata_ref"].attrs


    def __str__(self):
        return str(self._dict)


    def __repr__(self):
        return repr(self._dict)


    def __len__(self):
        return len(self._dict)


    def __contains__(self, key):
        return key in self._dict


    def __eq__(self, other):
        return self._dict == other._dict


    def __getattr__(self, attr):
        return getattr(self._dict, attr)


    def __getitem__(self, key):
        return self._dict[key]


    def __setitem__(self, key, val):
        self._dict[key] = val
        self.propagate()


    def __delitem__(self, key):
        del self._dict[key]
        self.propagate()


    def setdefault(self, key, val):
        try:
            self[key]
        except KeyError:
            self[key] = val


    def update(self, *args, **kwargs):
        for key, val in dict(*args, **kwargs).items():
            self[key] = val


    def propagate(self):
        """Propagates metadata to children of the wrapped object

        Automatically called when items are set or deleted.

        Returns
        -------
        None
        """
        # TODO: Verify that this is working as expected and is not overkill
        try:
            children = list(getattr(self._obj, self._dict["accessor_name"]))
        except TypeError:
            pass
        else:
            if (
                isinstance(self._obj, xr.Dataset)
                or len(self._obj.shape) > 2
                or len(children) == len(self._obj.coords["band"])
            ):
                for child in children:
                    child.coords["metadata_ref"].attrs = self._dict
