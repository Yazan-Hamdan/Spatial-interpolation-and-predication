from typing import Tuple, Union, List
import logging

from osgeo import gdal
import argparse
import numpy as np

from utils.ShapeFileHandler import ShapeFilesDirectoryHandler
from algos.idw import idw, get_point_positions

NO_DATA_VALUE = -9999
RASTER_WIDTH = 752
RASTER_HEIGHT = 948


def get_classification_ranges(
    min_val: Union[float, int],
    max_val: Union[float, int],
    doubled=False
) -> List[Tuple[float, float]]:
    """ Splits values into ranges to classify into 9 categories.

    Args:
        min_val (number): Minimum values of the data.
        max_val (number): Maximum value of the data.
        doubled (bool): Make the center point have the highest value. \
            Defaults to False.

    Returns:
        List[Tuple[float, float]]: The ranges to classify with.
    """
    range_value = 17 if doubled else 9
    data_range = (max_val - min_val) / range_value
    return [(min_val + (data_range * i), min_val + (data_range * (i + 1)))
            for i in range(0, range_value)]


def classify_arr(arr: np.ndarray, data_min: int,
                 data_max: int, reverse=False,
                 doubled=False) -> np.ndarray:
    """Classify arr values.

    Args:
        arr (gdal.Band): The array band to classify.
        reverse (bool, optional): Whether to reverse classification values.
        Defaults to False.
        data_min (int): The lower bound of the resulting array.
        data_max (int): The upper bound of the resulting array.
        doubled (bool): Make the center point have the highest value. \
            Defaults to False.
    """
    band_data = arr
    final_data = band_data.copy()
    ranges = get_classification_ranges(data_min, data_max, doubled)
    current_class = 1 if not reverse else 9
    should_reverse = False

    for val_range in ranges:
        data_selection = \
            (final_data >= val_range[0]) & (final_data < val_range[1]) \
            if val_range[0] != ranges[-1][0] \
            else (final_data >= val_range[0]) & (final_data <= val_range[1])

        final_data[data_selection] = current_class

        if reverse:
            if not should_reverse:
                should_reverse = doubled and current_class == 1

            if not should_reverse:
                current_class -= 1
            else:
                current_class += 1
        else:
            if not should_reverse:
                should_reverse = doubled and current_class == 9

            if not should_reverse:
                current_class += 1
            else:
                current_class -= 1

    return final_data


def classify_band(band: gdal.Band, reverse=False, doubled=False) -> None:
    """Classify raster band values.

    Args:
        band (gdal.Band): The raster band to classify.
        reverse (bool, optional): Whether to reverse classification values. \
            Defaults to False.
        doubled (bool): Make the center point have the highest value. \
            Defaults to False.
    """
    [data_min, data_max, _, __] = band.GetStatistics(True, True)

    band_data = np.array(band.ReadAsArray())

    new_band_data = classify_arr(
        band_data, reverse, data_min, data_max, doubled)

    band.WriteArray(new_band_data)


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
    pixel_size = 30

    x_min = 738107.6875
    y_max = 4334147.5

    source_layer: gdal.Dataset = shape_file.GetLayer()

    target_ds: gdal.Dataset = gdal.GetDriverByName('GTiff').Create(
        raster_file_name, RASTER_WIDTH, RASTER_HEIGHT, 1, gdal.GDT_Float32)
    target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    target_ds.SetProjection(source_layer.GetSpatialRef().ExportToWkt())
    band: gdal.Band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(NO_DATA_VALUE)

    # Rasterize
    gdal.RasterizeLayer(target_ds, [1], source_layer, **args)

    return target_ds


def best_idw(raster: gdal.Dataset) -> np.ndarray:
    """Finds the best IDW interpolation.

    Args:
        raster (gdal.Dataset): The points to interpolate from.

    Returns:
        np.ndarray: The interpolated data as numpy array.
    """
    raster_arr = np.array(raster.GetRasterBand(1).ReadAsArray())
    points = get_point_positions(raster_arr, NO_DATA_VALUE)
    cutoff_point = int((points.shape[0] / 10) * 9)
    used_points = points[0: cutoff_point]
    test_points = points[cutoff_point: -1]
    closest_ks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 60, 80, 100]
    outputs: dict[int, dict] = {}

    logging.basicConfig(filename="idw_logs.csv",
                        filemode='w',
                        format='%(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)

    idw_logger = logging.getLogger("idw")
    idw_logger.debug("k, mse")

    for k in closest_ks:
        to_print = k if k > 0 else "ALL"
        prefix = "[IDW_INTERPOLATION]"
        print(f"{prefix} Calculating IDW with {to_print} closest points.")
        raster_to_use = raster_arr.copy()
        for test_point in test_points:
            raster_to_use[int(test_point[0]), int(
                test_point[1])] = NO_DATA_VALUE

        output_raster = idw(raster_to_use, used_points, NO_DATA_VALUE, k)

        mse = sum([pow(test_point[2] -
                       output_raster[int(test_point[0]), int(test_point[1])], 2
                       )
                   for test_point in test_points]) / test_points.shape[0]

        print(f"{prefix} {to_print} POINTS MSE = {mse}")

        outputs[k] = {
            "raster": output_raster,
            "mse": mse
        }

        idw_logger.debug(f"{k}, {mse}")

        print(f"{prefix} Saving {to_print} points raster.")
        save_arr_as_raster(f'idw_{k}.tiff',
                           raster.GetGeoTransform(),
                           raster.GetProjection(), output_raster)

    best_k = outputs[0]

    for k, result in outputs.items():
        if best_k is None or best_k["mse"] > result["mse"]:
            best_k = {
                **result,
                "id": k
            }

    best_k["raster"] = idw(raster_arr.copy(), used_points,
                           NO_DATA_VALUE, best_k["id"])

    return best_k


def save_arr_as_raster(
    name: str,
    geo_transform: Tuple[float, float, float, float, float, float],
    projection: str,
    arr: np.ndarray
) -> None:
    """Save numpy array as a GeoTIFF raster.

    Args:
        name (str): Name of the file to save.
        geo_transform (Tuple[float, float, float, float, float, float]):\
            The geotransform.
        projection (str): The projection.
        arr (np.ndarray): The raster data to save.
    """
    end_ds: gdal.Dataset = gdal.GetDriverByName('GTiff').Create(
        name, RASTER_WIDTH, RASTER_HEIGHT, 1, gdal.GDT_Float32)

    end_ds.SetGeoTransform(geo_transform)
    end_ds.SetProjection(projection)
    end_band: gdal.Band = end_ds.GetRasterBand(1)
    end_band.SetNoDataValue(NO_DATA_VALUE)
    end_band.WriteArray(arr, 0, 0)
    end_band.FlushCache()


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-is", "--interpolate_snow",
                            help="""Whether to interpolate snow data from points.
                            Must be run at least once.""",
                            required=False,
                            action=argparse.BooleanOptionalAction)

    arg_parser.add_argument("-sp", "--shape_path",
                            help="""The absolute path to the shape files.
                            Only used when finding best interpolated snow.""",
                            required=False)

    cmd_args = arg_parser.parse_args()

    if cmd_args.interpolate_snow:
        if cmd_args.shape_path is None:
            arg_parser.error(
                "--shape_path is required with --interpolate_snow")

        shape_file_handler = ShapeFilesDirectoryHandler(cmd_args.shape_path)
        shape_file_data = shape_file_handler.read_shapefiles()

        extra_params = {
            "options": ['ATTRIBUTE=SNOWDEPTH']
        }

        points_raster = rasterize_shapefile(
            shape_file_data['snowpoint'],
            "points.tiff",
            **extra_params)

        best_result = best_idw(points_raster)

        save_arr_as_raster('best_idw.tiff',
                           points_raster.GetGeoTransform(),
                           points_raster.GetProjection(),
                           best_result["raster"])

    gdal.DEMProcessing('slope.tiff', "data/arelev1.tif", 'slope')
    gdal.DEMProcessing('hillshade.tiff', "data/arelev1.tif", 'hillshade')

    rasters_to_process = {
        "best_idw.tiff": {
            "parameters": {"reverse": False, "doubled": True},
            "weight": 0.4
        },
        "slope.tiff":  {
            "parameters": {"reverse": False, "doubled": True},
            "weight": 0.1
        },
        "hillshade.tiff": {
            "parameters": {"reverse": True, "doubled": False},
            "weight": 0.5
        },
    }

    end_result: np.ndarray = None
    geo_transform_to_use: Tuple[float, float,
                                float, float, float, float] = None
    projection_to_use: str = None

    for raster_name, raster_info in rasters_to_process.items():
        raster_data: gdal.Dataset = gdal.Open(raster_name)
        raster_band: gdal.Band = raster_data.GetRasterBand(1)
        [raster_data_min, raster_data_max, _,
            __] = raster_band.GetStatistics(True, True)

        if geo_transform_to_use is None:
            geo_transform_to_use = raster_data.GetGeoTransform()

        if projection_to_use is None:
            projection_to_use = raster_data.GetProjection()

        no_data_value = raster_band.GetNoDataValue()
        raster_band_arr = np.array(raster_band.ReadAsArray())
        raster_band_arr[raster_band_arr == no_data_value] = 1

        classified_raster = classify_arr(raster_band_arr,
                                         raster_data_min, raster_data_max,
                                         **raster_info["parameters"])

        if end_result is None:
            end_result = raster_info["weight"] * classified_raster
        else:
            end_result += raster_info["weight"] * classified_raster

    elevation_raster: gdal.Dataset = gdal.Open("data/arelev1.tif")
    elevation_data_arr = np.array(
        elevation_raster.GetRasterBand(1).ReadAsArray())

    elevation_no_data_locations = elevation_data_arr == \
        elevation_raster.GetRasterBand(1).GetNoDataValue()

    end_result[elevation_no_data_locations] = NO_DATA_VALUE
    save_arr_as_raster("result.tiff", geo_transform_to_use,
                       projection_to_use, end_result)
