"""Tests for propagating metadata when using ImageryAccessor"""

from earthpy.plot import plot_rgb, _stretch_im
from earthpy.io import path_to_example
import numpy as np
import matplotlib.pyplot as plt
import pytest
import xarray as xr

import imagery_accessor as ixr


plt.show = lambda: None


@pytest.fixture
def rgb_array():
    """Fixture holding an RGB image for plotting."""
    channels = ["red", "green", "blue"]
    paths = [path_to_example(f"{ch}.tif") for ch in channels]
    stacked = ixr.stack_images(paths, channels)

    # Add custom metadata to verify that it carries over to children
    stacked.im.metadata["hello"] = "world"

    return stacked


@pytest.fixture
def rgb_dataset():
    """Fixture holding an RGB image for plotting."""
    channels = ["red", "green", "blue"]
    paths = [path_to_example(f"{ch}.tif") for ch in channels]
    stacked = ixr.stack_images(paths, channels).to_dataset(dim="band")

    # Add custom metadata to verify that it carries over to children
    stacked.im.metadata["hello"] = "world"

    return stacked


def test_array_to_dataset(rgb_array):
    arr = rgb_array.to_dataset(dim="band").to_array(dim="band")
    assert arr.attrs == rgb_array.attrs
    assert arr.dims == rgb_array.dims
    assert arr.im.metadata == rgb_array.im.metadata


@pytest.mark.parametrize("rgb_image", ["rgb_array", "rgb_dataset"])
def test_rgb_extent(rgb_image, request):
    """Tests plot_rgb whether respects axis limits when using extent

    This test was adapted from the earthpy library.
    """
    rgb_image = request.getfixturevalue(rgb_image)

    ax = rgb_image.im.plot_rgb(
        title="My Title",
        figsize=(5, 5),
    )
    # Get x and y lims to test extent
    plt_ext = ax.get_xlim() + ax.get_ylim()

    plt_array = ax.get_images()[0].get_array()

    assert ax.figure.bbox_inches.bounds[2:4] == (5, 5)
    assert ax.get_title() == "My Title"
    #assert np.array_equal(plt_array[0], rgb_image.transpose([1, 2, 0])[1])
    assert rgb_image.im.extent == plt_ext
    plt.close()


@pytest.mark.parametrize("rgb_image", ["rgb_array", "rgb_dataset"])
def test_obj_metadata(rgb_image, request):
    """Verifies that __init__ doesn't muck up inherited metadata"""
    rgb_image = request.getfixturevalue(rgb_image)
    assert rgb_image.coords["metadata_ref"].attrs == rgb_image.im._obj.coords["metadata_ref"].attrs


@pytest.mark.parametrize("rgb_image", ["rgb_array", "rgb_dataset"])
def test_xarr_copy(rgb_image, request):
    rgb_image = request.getfixturevalue(rgb_image)

    rgb_copy = rgb_image.copy()
    rgb_copy.im.metadata["hello"] = "universe"
    assert id(rgb_image.im.metadata) != id(rgb_copy.im.metadata)
    assert rgb_image.im.metadata["hello"] != rgb_copy.im.metadata["hello"]


@pytest.mark.parametrize("rgb_image", ["rgb_array", "rgb_dataset"])
def test_clean_all_bands(rgb_image, request):
    rgb_image = request.getfixturevalue(rgb_image)

    rgb_clean = rgb_image.im.clean_all_bands()
    rgb_clean.im.metadata["hello"] = "universe"
    assert id(rgb_image.im.metadata) != id(rgb_clean.im.metadata)
    assert rgb_image.im.metadata["hello"] != rgb_clean.im.metadata["hello"]


@pytest.mark.parametrize("rgb_image", ["rgb_array", "rgb_dataset"])
def test_metadata_copy_along_band(rgb_image, request):
    rgb_image = request.getfixturevalue(rgb_image)

    try:
        rgb_arr = [np.mean(a) for a in rgb_image.values]
    except TypeError:
        rgb_arr = [np.mean(a) for a in rgb_image.values()]

    rgb_arr = rgb_image.im.copy_xr_metadata(rgb_arr)

    # Make comparisons with arrays, not datasets, to simplify iteration
    if isinstance(rgb_image, xr.Dataset):
        rgb_image = rgb_image.to_array(dim="band")
        rgb_arr = rgb_arr.to_array(dim="band")

    for band, band_arr in zip(rgb_image, rgb_arr):

        # Metadata should be the same
        assert band.im.metadata == band_arr.im.metadata

        # Derived should have no dims
        assert not band_arr.dims

        # Non-dimensional coordinates should be the same
        for key, val in band_arr.coords.items():
            try:
                assert band.coords[key] == val
            except ValueError:
                # Simple equality fails with np.array
                assert np.array_equal(band.coords[key], val)


@pytest.mark.parametrize("rgb_image", ["rgb_array", "rgb_dataset"])
def test_metadata_to_npfunc(rgb_image, request):
    rgb_image = request.getfixturevalue(rgb_image)
    rgb_arr = rgb_image.im.npfunc("digitize", [-np.inf, 128, np.inf])

    # Make comparisons with arrays, not datasets, to simplify iteration
    if isinstance(rgb_image, xr.Dataset):
        rgb_image = rgb_image.to_array(dim="band")
        rgb_arr = rgb_arr.to_array(dim="band")

    for band, band_arr in zip(rgb_image, rgb_arr):
        # Metadata should be the same
        assert band.im.metadata == band_arr.im.metadata

        # Derived should have the same dims
        assert band.dims == band_arr.dims

        # Non-dimensional coordinates should be the same
        for key, val in band_arr.coords.items():
            try:
                assert band.coords[key] == val
            except ValueError:
                # Simple equality fails with np.array
                assert np.array_equal(band.coords[key], val)


@pytest.mark.parametrize("rgb_image", ["rgb_array", "rgb_dataset"])
def test_metadata_to_1d_arr(rgb_image, request):
    rgb_image = request.getfixturevalue(rgb_image)
    rgb_1d = rgb_image.sum(axis=0)

    # Non-index metadata should be the same
    assert rgb_image.im.metadata
    assert rgb_image.im.metadata["hello"] == rgb_1d.im.metadata["hello"]

    # Non-dimensional, non-scalar coords should transfer
    ignore_coords = list(rgb_image.dims)
    for key, val in rgb_1d.coords.items():
        if not rgb_image.coords[key].shape and key not in ignore_coords:
            assert rgb_image.coords[key] == val
