"""
It provides a set of utility functions for working with file paths
"""

import glob
import os


class PathUtils:
    """
    This class provides a set of utility functions for working with file paths
    """
    image_extensions = ['.jpg', '.jpeg', '.gif', '.png']
    detection_extensions = ['.json']

    @staticmethod
    def is_image_file(s):
        """
        It returns true if the file extension is in the list of image extensions
        :param s: the path to the file
        :return: A boolean value.
        """
        ext = os.path.splitext(s)[1]
        return ext.lower() in PathUtils.image_extensions

    @staticmethod
    def find_image_files(strings):
        """
        It returns a list of strings that are image files
        :param strings: a list of strings
        :return: A list of strings that are image files.
        """
        return [s for s in strings if PathUtils.is_image_file(s)]

    @staticmethod
    def find_images(dir_name, recursive=False):
        """
        Find all the image files in a directory
        :param dir_name: The directory to search for images
        :param recursive: If True, will search for images in subdirectories, defaults to False (optional)
        :return: A list of strings that are the paths to the images.
        """
        if recursive:
            strings = glob.glob(os.path.join(dir_name, '**', '*.*'), recursive=True)
        else:
            strings = glob.glob(os.path.join(dir_name, '*.*'))

        image_strings = PathUtils.find_image_files(strings)

        return image_strings

    @staticmethod
    def is_detection_file(s):
        """
        It returns true if the file extension is in the list of detection extensions
        :param s: the file name
        :return: A boolean value.
        """
        ext = os.path.splitext(s)[1]
        return ext.lower() in PathUtils.detection_extensions

    @staticmethod
    def find_detection_files(strings):
        """
        It returns a list of strings that are detection files
        :param strings: a list of strings
        :return: A list of strings that are detection files.
        """
        return [s for s in strings if PathUtils.is_detection_file(s)]

    @staticmethod
    def find_detections(dir_name, recursive=False):
        """
        Find all the detection files in a directory
        :param dir_name: The directory to search for detections in
        :param recursive: If True, will search all subdirectories of the given directory, defaults to False (optional)
        :return: A list of strings that are the paths to the detection files.
        """
        if recursive:
            strings = glob.glob(os.path.join(dir_name, '**', '*.*'), recursive=True)
        else:
            strings = glob.glob(os.path.join(dir_name, '*.*'))

        detection_strings = PathUtils.find_detection_files(strings)

        return detection_strings
