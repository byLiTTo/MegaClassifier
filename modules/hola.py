"""
Módulo para ejecutar un modelo de detección de animales con TensorFlow.

La clase TFDetector contiene funciones para cargar el modelo de detección de Tensorflow
y ejecutar una instancia. La función main además renderiza los bounding boxes de las
predicciones y guardar los resultados.

Este modulo no es práctico para ejecutarlo con un gran número de imágenes.

Si no desea usar la GPU debe seleccionar la variable: CUDA_VISIBLE_DEVICES a -1

Este módulo solo considera detecciones por encima de 0.1 de umbral. El umbral que
se especifica es para las renderizaciones. Si desea trabajar con valores menores deberá
cambiar la variable: DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD
"""

###################################################################################################
#%% CONSTANTES IMPORTACIONES Y ENTORNO

import enum
import os
import glob
import numpy as np
import time
import humanfriendly
import statistics
import visualization.visualization_utils as viz_utils
import argparse
import sys

from ct_utils import truncate_float


# IMPORTACION TENSORFLOW --------------------------------------------------------------------------


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow.compat.v1 as tf

print('TensorFlow version:', tf.__version__)
print('Is GPU available? tf.test.is_gpu_available:', tf.test.is_gpu_available())


###################################################################################################
#%% CLASES

#--------------------------------------------------------------------------------------------------
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
            strings = glob.glob(os.path.join(dir_name, ' *.*'))
        
        image_strings = ImagePathUtils.find_image_files(strings)

        return image_strings

#--------------------------------------------------------------------------------------------------
class TFDetector:
    """
    Cargamos un modelo de detección en la inicialización.
    Está pensado para usar con MegaDetector (TF).
    El batch inference size se establece en 1. Para soportar tamaños superiores debe 
    ser modificado.
    """

    # Número de decimales a redondear para el umbral de confianza y las coordenadas del bounding box
    CONF_DIGITS = 3
    COORD_DIGITS = 4

    # MegaDetector fue entrenado con batch size de 1 y la función de redimensionado es una parte del gráfico de inferencia.
    BATCH_SIZE = 1

    # Lista de algunas causas de fallos
    FAILURE_TF_INFER = 'Failure TF inference'
    FAILURE_IMAGE_OPEN = 'Failure image access'

    DEFAULT_RENDERING_CONFIDENCE_THRESHOLD = 0.85 # Umbral de renderizado de BBoxes
    DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD = 0.1  # Umbral para incluir en fichero de salida

    # Tipos de detecciones disponibles en MegaDetector v4+
    DEFAULT_DETECTOR_LABEL_MAP = {
        '1': 'animal',
        '2': 'person',
        '3': 'vehicle'
    }

    # animal, persona, grupo, vehículo - Para la asignación de color
    NUM_DETECTOR_CATEGORIES = 4

    def __init__(self, model_path):
        """
        Carga un modelo a partir de una ruta y comienza una tf.Session con gráficos.
        Obtiene tensor handles de entrada y salida.
        """
        detection_graph = TFDetector.__load_model(model_path)
        self.tf_session = tf.Session(graph=detection_graph)

        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.box_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.score_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
        self.class_tensor = detection_graph.get_tensor_by_name('detection_classes:0')

    @staticmethod
    def round_and_make_float(d, precision=4):
        return truncate_float(float(d), precision=precision)

    @staticmethod
    def __convert_coords(tf_coords):
        """
        Convierte las coordenadas de la salida del modelo del formato [y1, x1, y2, x2]
        al formato usado por la API y MegaDB: [x1, y1, width, height]. Todas las 
        coordenadas están normalizadas entre [0, 1].

        Args:
            - tf_coords: np.array con las coordenadas de los BBoxes de las predicciones 
            del TF detector con el formato [y1, x1, y2, x2].

        Returns: Lista de float con las coordenadas de los BBoxes de las predicciones 
            en el formato [x1, y1, width, height].
        """
        # Cambia de [y1, x1, y2, x2] a [x1, y1, width, height]
        width = tf_coords[3] - tf_coords[1]
        height = tf_coords[2] - tf_coords[0]

        new = [tf_coords[1], tf_coords[0], width, height]  # debe ser una lista en vez de np.array

        # convertir numpy floats a Python floats
        for i, d in enumerate(new):
            new[i] = TFDetector.round_and_make_float(d, precision=TFDetector.COORD_DIGITS)
        return new

    @staticmethod
    def convert_to_tf_coords(array):
        """
        De [x1, y1, width, height] a [y1, x1, y2, x2], donde x1 is x_min, x2 is x_max

        Este es un paso superfluo como resultado del modelo [y1, x1, y2, x2]. pero lo 
        hemos convertido al formato de salida de la API - Solo para mantener la 
        interfaz de sincronización.
        """
        x1 = array[0]
        y1 = array[1]
        width = array[2]
        height = array[3]
        x2 = x1 + width
        y2 = y1 + height
        return [y1, x1, y2, x2]

    @staticmethod
    def __load_model(model_path):
        """
        Carga un modelo de detección (i.e., create a graph) de un fichero .pb.

        Args:
            - model_path: .pb del fichero del modelo.

        Returns: gráfico generado.
        """
        print('TFDetector: Cargando gráfico...')
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, ' rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        print('TFDetector: Gráfico de detecciones cargado.')

        return detection_graph

    def _generate_detections_one_image(self, image):
        np_im = np.asarray(image, np.uint8)
        im_w_batch_dim = np.expand_dims(np_im, axis=0)

        # need to change the above line to the following if supporting a batch size > 1 and resizing to the same size
        # np_images = [np.asarray(image, np.uint8) for image in images]
        # images_stacked = np.stack(np_images, axis=0) if len(images) > 1 else np.expand_dims(np_images[0], axis=0)

        # Realiza inferencia
        (box_tensor_out, score_tensor_out, class_tensor_out) = self.tf_session.run(
            [self.box_tensor, self.score_tensor, self.class_tensor],
            feed_dict={self.image_tensor: im_w_batch_dim})

        return box_tensor_out, score_tensor_out, class_tensor_out

    def generate_detections_one_image(self, image, image_id, 
        detection_threshold=DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD):
        """
        Aplicar el detector a una imagen.

        Args:
            - image: PIL del objeto de la imagen.
            - image_id: identificador de imagen; estará en el campo del fichero de salida.
            - detection_threshold: umbral de confianza por encima del cual se incluirá 
                la prosible de detección.

        Returns:
        A dict with the following fields, see the 'images' key in https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
            - 'file' (always present)
            - 'max_detection_conf'
            - 'detections', which is a list of detection objects containing keys 'category', 'conf' and 'bbox'
            - 'failure'
        """
        result = {
            'file': image_id
        }
        try:
            b_box, b_score, b_class = self._generate_detections_one_image(image)

            # Nuestro batch size es 1; se necesitaria hacer un bucle si batch size > 1
            boxes, scores, classes = b_box[0], b_score[0], b_class[0]

            detections_cur_image = [] # estará vacío si la imagen no tiene detecciones
            max_detection_conf = 0.0
            for b, s, c in zip(boxes, scores, classes):
                if s > detection_threshold:
                    detection_entry = {
                        'categoria': str(int(c)),  # usamos string para el valor de la categoría de detección, no usamos int
                        'conf': truncate_float(float(s),  # cast de float para json 
                                            precision=TFDetector.CONF_DIGITS),
                        'bbox': TFDetector.__convert_coords(b)
                    }
                detections_cur_image.append(detection_entry)
                if s > max_detection_conf:
                    max_detection_conf = s
            result['max_detection_conf'] = truncate_float(float(max_detection_conf), precision = TFDetector.CONF_DIGITS)
            result['detections'] = detections_cur_image
        except Exception as e:
            result[' failure'] = TFDetector.FAILURE_TF_INFER
            print('TFDetector: image {} failed during inference: {}'.format(image_id, str(e)))
        
        return result



###################################################################################################
#%% FUNCIÓN PRINCIPAL
            
def load_and_run_detector(model_file, image_file_names, output_dir,
                          render_confidence_threshold=TFDetector.DEFAULT_RENDERING_CONFIDENCE_THRESHOLD,
                          crop_images=False):
    """
    Carga y ejecuta el detector en imágenes y visualice los resultados.
    """
    if len(image_file_names) == 0:
        print('Warning: no files available')
        return

    start_time = time.time()
    tf_detector = TFDetector(model_file)
    elapsed = time.time() - start_time
    print('Cargado modelo en {}'.format(humanfriendly.format_timespan(elapsed)))

    detection_results = []
    time_load = []
    time_infer = []

    # Dictionary mapping output file names to a collision-avoidance count.
    #
    # Since we'll be writing a bunch of files to the same folder, we rename
    # as necessary to avoid collisions.
    output_filename_collision_counts = {}

    def input_file_to_detection_file(fn, crop_index=-1):
        """
        Crea nombres únicos para los ficheros de salida.

        Esta función hace 3 cosas:
        1) Si la opción --crop es usada, una imagen puede generar varios ficheros de 
            salida. Por ejemplo, si foo.jpg tiene 3 detecciones, esta función se ejecutará 
            3 veces, siendo la salida:
                foo_crop00_detecciones.jpg
                foo_crop01_detecciones.jpg
                foo_crop02_detecciones.jpg
        2) Si la opción --recursive es usada, entonces el mismo nombre fichero aparecerá 
            multiples veces y las salidas se añadiran a la misma carpeta. Para evitar esto 
            añadimos un prefijo numérico a los nombres duplicados, quedando:
                foo_crop00_detections.jpg
                0000_foo_crop00_detections.jpg
                0001_foo_crop00_detections.jpg
        3) Antepone el directorio de salida:
                out_dir/foo_crops00_detections.jpg

        Args:
            - fn: str, filename
            - crop_index: int, crop number

        Returns: Ruta del archivo de salida.
        """
        fn = os.path.basename(fn).lower()
        name, ext = os.path.splitext(fn)
        if crop_index >= 0:
            name += '_crop{:0>2d}'.format(crop_index)
        fn = '{}{}{}'.format(name, ImagePathUtils.DETECTION_FILENAME_INSERT, '.jpg')
        if fn in output_filename_collision_counts:
            n_collisions = output_filename_collision_counts[fn]
            fn = '{:0>4d}'.format(n_collisions) + '_' + fn
            output_filename_collision_counts[fn] += 1
        else:
            output_filename_collision_counts[fn] = 0
        fn = os.path.join(output_dir, fn)
        return fn

    for im_file in tqdm(image_file_names):

        try:
            start_time = time.time()

            image = viz_utils.load_image(im_file)

            elapsed = time.time() - start_time
            time_load.append(elapsed)

        except Exception as e:
            print('Image {} cannot be loaded. Exception: {}'.format(im_file, e))
            result = {
                'file': im_file,
                'failure': TFDetector.FAILURE_IMAGE_OPEN
            }
            detection_results.append(result)
            continue

        try:
            start_time = time.time()

            result = tf_detector.generate_detections_one_image(image, im_file)
            detection_results.append(result)

            elapsed = time.time() - start_time
            time_infer.append(elapsed)

        except Exception as e:
            print('An error occurred while running the detector on image {}. Exception: {}'.format(im_file, e))
            continue

        try:
            if crop_images:

                images_cropped = viz_utils.crop_image(result['detecciones'], image)

                for i_crop, cropped_image in enumerate(images_cropped):
                    output_full_path = input_file_to_detection_file(im_file, i_crop)
                    cropped_image.save(output_full_path)

            else:

                # image is modified in place
                viz_utils.render_detection_bounding_boxes(result['detecciones'], image,
                                                          label_map=TFDetector.DEFAULT_DETECTOR_LABEL_MAP,
                                                          confidence_threshold=render_confidence_threshold)
                output_full_path = input_file_to_detection_file(im_file)
                image.save(output_full_path)

        except Exception as e:
            print('Visualizing results on the image {} failed. Exception: {}'.format(im_file, e))
            continue

    # ...for each image

    ave_time_load = statistics.mean(time_load)
    ave_time_infer = statistics.mean(time_infer)
    if len(time_load) > 1 and len(time_infer) > 1:
        std_dev_time_load = humanfriendly.format_timespan(statistics.stdev(time_load))
        std_dev_time_infer = humanfriendly.format_timespan(statistics.stdev(time_infer))
    else:
        std_dev_time_load = 'not available'
        std_dev_time_infer = 'not available'
    print('De media, para esta imagen,')
    print('- loading took {}, std dev is {}'.format(humanfriendly.format_timespan(ave_time_load),
                                                    std_dev_time_load))
    print('- inference took {}, std dev is {}'.format(humanfriendly.format_timespan(ave_time_infer),
                                                      std_dev_time_infer))



###################################################################################################
#%% MAIN

def main():

    parser = argparse.ArgumentParser(
        description='Module to run a TF animal detection model on images')
    parser.add_argument(
        'detector_file',
        help='Path to .pb TensorFlow detector model file')
    group = parser.add_mutually_exclusive_group(required=True)  # must specify either an image file or a directory
    group.add_argument(
        '--image_file',
        help='Single file to process, mutually exclusive with --image_dir')
    group.add_argument(
        '--image_dir',
        help='Directory to search for images, with optional recursion by adding --recursive')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recurse into directories, only meaningful if using --image_dir')
    parser.add_argument(
        '--output_dir',
        help='Directory for output images (defaults to same as input)')
    parser.add_argument(
        '--threshold',
        type=float,
        default=TFDetector.DEFAULT_RENDERING_CONFIDENCE_THRESHOLD,
        help=('Confidence threshold between 0 and 1.0; only render boxes above this confidence'
              ' (but only boxes above 0.1 confidence will be considered at all)'))
    parser.add_argument(
        '--crop',
        default=False,
        action="store_true",
        help=('If set, produces separate output images for each crop, '
              'rather than adding bounding boxes to the original image'))
    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    assert os.path.exists(args.detector_file), 'detector_file specified does not exist'
    assert 0.0 < args.threshold <= 1.0, 'Confidence threshold needs to be between 0 and 1'  # Python chained comparison

    if args.image_file:
        image_file_names = [args.image_file]
    else:
        image_file_names = ImagePathUtils.find_images(args.image_dir, args.recursive)

    print('Ejecutando detector en {} imagenes...'.format(len(image_file_names)))

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        if args.image_dir:
            args.output_dir = args.image_dir
        else:
            # but for a single image, args.image_dir is also None
            args.output_dir = os.path.dirname(args.image_file)

    load_and_run_detector(model_file=args.detector_file,
                          image_file_names=image_file_names,
                          output_dir=args.output_dir,
                          render_confidence_threshold=args.threshold,
                          crop_images=args.crop)


if __name__ == '__main__':
    main()