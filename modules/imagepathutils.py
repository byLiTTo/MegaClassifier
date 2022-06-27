import os
import glob



class ImagePathUtils:
    """
    Una colección de funciones de utilidad que admiten este script independiente.
    """

    # Pegue esto en los nombres de archivo antes de la extensión para el resultado renderizado
    DETECTION_FILENAME_INSERT = '_detections'

    image_extensions = ['.jpg', '.jpeg', '.gif', '.png']

    @staticmethod
    def is_image_file(s):
        """
        Compara la extesión de un archivo con las extensiones admitadas en 
        image_extensions.
        """
        ext = os.path.splitext(s)[1]
        return ext.lower() in ImagePathUtils.image_extensions

    @staticmethod
    def find_image_files(strings):
        """
        Devuelve una lista de nombres candidatos a ser ficheros de imágenes. Para los 
        nombres busca a partir de las extensiones incluidas en image_extension.
        """
        return [s for s in strings if ImagePathUtils.is_image_file(s)]

    @staticmethod
    def find_images(dir_name, recursive=False):
        """
        Busca todos los ficheros que parecen imagénes dentro de un directorio.
        """
        if recursive:
            strings = glob.glob(os.path.join(dir_name, '**', '*.*'), recursive=True)
        else:
            strings = glob.glob(os.path.join(dir_name, '*.*'))

        image_strings = ImagePathUtils.find_image_files(strings)

        return image_strings