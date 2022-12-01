from typing import List, Union, Tuple

from osgeo import gdal
import argparse
import numpy as np

from utils.ShapeFileHandler import ShapeFilesDirectoryHandler

def rasterize_shapefile(shape_file: gdal.Dataset, raster_file_name: str,
                        **args) -> gdal.Dataset:
    """ Rasterize vector shape file.
        Args:
            shape_file: The shape file to rasterize.
            raster_file_name: The output filename of the raster.
            args: Extra arguments to the rasterization method.
        Returns:
            gdal.Dataset: The raster as a GDAL Dataset.
    """
    pixel_size = 25

    # getting layer information of shapefile.
    shp_layer = shape_file.GetLayer()

    # get extent values to set size of output raster.
    x_min, x_max, y_min, y_max = shp_layer.GetExtent()

    # calculate size/resolution of the raster.
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)

    no_data_value = -9999

    source_layer: gdal.Dataset = shape_file.GetLayer()

    target_ds: gdal.Dataset = gdal.GetDriverByName('GTiff').Create(
        raster_file_name, x_res, y_res, 1, gdal.GDT_Float32)
    target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    target_ds.SetProjection(source_layer.GetSpatialRef().ExportToWkt())
    band: gdal.Band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(no_data_value)

    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], source_layer, **args)

    return target_ds


if __name__ == '__main__':
    shape_file_handler = ShapeFilesDirectoryHandler(r"C:\Users\hp\Desktop\UNV\fourth year\First sem\spatial data analysis\project\project 3\Spatial-interpolation-and-predication-\data")
    shape_file_data = shape_file_handler.read_shapefiles()

    extra_params = {
        "options": ['ATTRIBUTE=SNOWDEPTH']
    }

    rasterize_shapefile(shape_file_data['snowpoint'], "shit.tiff", **extra_params)

