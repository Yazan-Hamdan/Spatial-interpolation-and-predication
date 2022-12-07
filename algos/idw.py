import numpy as np
from numpy.typing import NDArray
import algos._idw


def get_point_positions(raster: NDArray[np.float32], no_data_value: int
                        ) -> NDArray[np.float32]:
    """Return the [x, y, value] for each known point.

    Args:
        raster (NDArray[np.float32]): The raster with the known points.
        no_data_value (int): The no data value.

    Returns:
        NDArray[np.float32]: the list of [x, y, value] for each point.
    """
    return algos._idw.get_point_positions(raster, no_data_value)


def calculate_idw(x_coord: int, y_coord: int, known_points: NDArray[np.float32],
                  closest_k_points: int = 5) -> float:
    """Calculates the IDW value for one point given all known points.

    Args:
        x_coord (int): The X coordinate of the point.
        y_coord (int): The Y coordinate of the point.
        known_points (NDArray[np.float32]): The list of known points.
        closest_k_points (int): Only count closest K points. Default 5.

    Returns:
        float: The IDW value.
    """
    return algos._idw.calculate_idw(x_coord, y_coord, known_points,
                                    closest_k_points)


def idw(raster: NDArray[np.float32], points_pos: NDArray[np.float32],
        no_data_value: int, closest_k_points: int = 5) -> NDArray[np.float32]:
    """ Runs the IDW algorithim on a raster given known points.

    Changes numpy array inplace.

    Args:
        raster (NDArray[np.float32]): The raster to run the algorithm on.
        points_pos (NDArray[np.float32]): List of known points.
        no_data_value (int): The no data value.

    Returns:
        NDArray[np.float32]: Reference to the input numpy array
    """
    return algos._idw.idw(raster, points_pos, no_data_value,
                          closest_k_points)
