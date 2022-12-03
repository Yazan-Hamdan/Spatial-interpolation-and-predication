import numpy
cimport numpy
from libc.math cimport sqrt, pow, abs
  
cpdef get_point_positions(numpy.ndarray[numpy.float32_t, ndim=2] raster, int no_data_value):
  cdef int rows = raster.shape[0]
  cdef int columns = raster.shape[1]
  cdef int number_of_points = 0

  for row in range(rows):
    for column in range(columns):
      if raster[row, column] != no_data_value:
        number_of_points+=1

  cdef numpy.ndarray[numpy.float32_t, ndim=2] output = numpy.empty((number_of_points, 3), dtype=numpy.float32);

  cdef int current_row = 0
  for row in range(rows):
    for column in range(columns):
      if raster[row, column] != no_data_value:
        output[current_row, 0] = row
        output[current_row, 1] = column
        output[current_row, 2] = raster[row, column]
        current_row+=1

  return output

cpdef calculate_idw(
  int x,
  int y, 
  numpy.ndarray[numpy.float32_t, ndim=2] known_points
):
  cdef float numerator = 0;
  cdef float denominator = 0;
  cdef int num_of_elements = known_points.shape[0]

  for i in range(num_of_elements):
    distance = abs(sqrt(pow(x - known_points[i, 0], 2) + pow(y - known_points[i, 1], 2)))
    numerator += known_points[i, 2] / distance
    denominator += 1 / distance

  return numerator / denominator

cpdef idw(
  numpy.ndarray[numpy.float32_t, ndim=2] raster, 
  numpy.ndarray[numpy.float32_t, ndim=2] points_pos,
  int no_data_value
):
  for x in range(raster.shape[0]):
      for y in range(raster.shape[1]):
        if raster[x, y] == no_data_value:
          raster[x, y] = calculate_idw(x, y, points_pos)

  return raster