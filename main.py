from typing import List, Union, Tuple

from osgeo import gdal
import argparse
import numpy as np
import pandas as pd

from utils.ShapeFileHandler import ShapeFilesDirectoryHandler
from idw import idw, get_point_positions

NO_DATA_VALUE = -9999


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

    x_min = 738107.6875
    y_max = 4334147.5
    x_res = 882
    y_res = 1111

    source_layer: gdal.Dataset = shape_file.GetLayer()

    target_ds: gdal.Dataset = gdal.GetDriverByName('GTiff').Create(
        raster_file_name, x_res, y_res, 1, gdal.GDT_Float32)
    target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    target_ds.SetProjection(source_layer.GetSpatialRef().ExportToWkt())
    band: gdal.Band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(NO_DATA_VALUE)

    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], source_layer, **args)

    return target_ds


def idw_wrapper(raster: gdal.Dataset) -> np.ndarray:
    """_summary_

    Args:
        raster (gdal.Dataset): _description_

    Returns:
        np.ndarray: _description_
    """
    raster_arr = np.array(raster.GetRasterBand(1).ReadAsArray())

    # contains the known points
    # [x, y, value] for each point
    points = get_point_positions(raster_arr, NO_DATA_VALUE)

    # cross validate the points.
    # dont forget to turn the point values to NO_DATA_VALUE
    # to "predict" them and compare

    # function changes values in place
    idw(raster_arr, points, NO_DATA_VALUE)
    print(raster_arr)


if __name__ == '__main__':
    shape_file_handler = ShapeFilesDirectoryHandler(
        r"C:\Users\salah\Downloads\Interpolaion and pre")
    shape_file_data = shape_file_handler.read_shapefiles()

    extra_params = {
        "options": ['ATTRIBUTE=SNOWDEPTH']
    }

    points_raster = rasterize_shapefile(
        shape_file_data['snowpoint'], "points.tiff", **extra_params)

    idw_wrapper(points_raster)
