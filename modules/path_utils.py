import os
import glob



class PathUtils:
    """
    Una colección de funciones de utilidad que admiten este script independiente.
    """

    image_extensions = ['.jpg', '.jpeg', '.gif', '.png']
    detection_extensions = ['.json']

    @staticmethod
    def is_image_file(s):
        """
        Compara la extesión de un archivo con las extensiones admitadas en 
        image_extensions.
        """
        ext = os.path.splitext(s)[1]
        return ext.lower() in PathUtils.image_extensions

    @staticmethod
    def find_image_files(strings):
        """
        Devuelve una lista de nombres candidatos a ser ficheros de imágenes. Para los 
        nombres busca a partir de las extensiones incluidas en image_extension.
        """
        return [s for s in strings if PathUtils.is_image_file(s)]

    @staticmethod
    def find_images(dir_name, recursive=False):
        """
        Busca todos los ficheros que parecen imagénes dentro de un directorio.
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
        Compara la extesión de un archivo con las extensiones admitadas en 
        detection_extensions.
        """
        ext = os.path.splitext(s)[1]
        return ext.lower() in PathUtils.detection_extensions

    @staticmethod
    def find_detection_files(strings):
        """
        Devuelve una lista de nombres candidatos a ser ficheros de detecciones. Para los 
        nombres busca a partir de las extensiones incluidas en detection_extensions.
        """
        return [s for s in strings if PathUtils.is_detection_file(s)]

    @staticmethod
    def find_detections(dir_name, recursive=False):
        """
        Busca todos los ficheros que parecen detecciones dentro de un directorio.
        """
        if recursive:
            strings = glob.glob(os.path.join(dir_name, '**', '*.*'), recursive=True)
        else:
            strings = glob.glob(os.path.join(dir_name, '*.*'))

        detection_strings = PathUtils.find_detection_files(strings)

        return detection_strings