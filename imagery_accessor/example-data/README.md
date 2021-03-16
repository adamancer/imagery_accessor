Example datasets
================

**All files and attributions copied verbatim from earthpy**

This package contains some small datasets that are used in examples.
This README file describes these example datasets at a high level, including
their sources and any relevant licensing information.


# RGB imagery for Rocky Mountain National Park

Low resolution RGB satellite imagery over RMNP, as a three channel GeoTIFF, and
separate one channel GeoTIFF files.

### Filename

`rmnp-rgb.tif`, `red.tif`, `green.tif`, `blue.tif`

### Source

This Landsat imagery was originally showcased on NASA's Visible Earth website:
https://visibleearth.nasa.gov/view.php?id=88405.
The data provided in Earthpy have been spatially coarsened and reprojected to
reduce the file size, and we have also provided additional single band GeoTIFF
files (`red.tif`, `green.tif`, and `blue.tif`), to use in examples that stack
raster layers.

### License

There are no restrictions on the use of data received from the U.S. Geological
Survey's Earth Resources Observation and Science (EROS) Center or NASA's Land
Processes Distributed Active Archive Center (LP DAAC), unless expressly
identified prior to or at the time of receipt. More information on licensing
and Landsat data citation is available from USGS.
