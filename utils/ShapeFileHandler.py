import os

from osgeo import ogr


class ShapeFilesDirectoryHandler:
    def __init__(self, directory_name: str):
        """
        Args:
            directory_name:[str] The name of directory where shapefiles are
                                 stored
        """
        self.__directory_name = directory_name

    def __get_shapefiles(self) -> list:
        """
        return all shapefiles of a directory
        Returns:
            shape_files: [list] all shapefiles into one list
        """
        shape_files = [f for f in os.listdir(self.__directory_name) if
                       f.endswith('.shp')]

        return shape_files

    def read_shapefiles(self) -> dict:
        """
        open a list of shapefiles into one list
        Returns: files_data: [dict] where key is the name of the file,
                                    value is the data stored in that file
        """
        shape_files = self.__get_shapefiles()
        files_data = {
            # -4 to exclude extension name from file name
            filename[:-4]: ogr.Open(os.path.join(self.__directory_name,
                                                 filename))
            for filename in shape_files}

        return files_data
