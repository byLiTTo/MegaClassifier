"""
It takes a model file, a list of image file names, and an output directory, and it runs the model on the images, saving
the results in the output directory
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
import traceback
from datetime import datetime

import humanfriendly
import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm

import repos.CameraTraps.visualization.visualization_utils as visualization_utils
from dataset_utils import DatasetUtils
from path_utils import PathUtils
from repos.CameraTraps.ct_utils import truncate_float

print('')
print(
    '=======================================================================================================================================')
print('TensorFlow version:', tf.__version__)
# print('Is GPU available? tf.test.is_gpu_available:', tf.test.is_gpu_available())
print('Is GPU available? tf.test.is_gpu_available:', tf.config.list_physical_devices('GPU'))
print(
    '=======================================================================================================================================')
print('')


########################################################################################################################
# CLASES:

class TFDetector:
    """
    This class is a wrapper for the TensorFlow Object Detection API
    """
    # Número de decimales a redondear para el umbral de confianza y las coordenadas bbox
    CONF_DIGITS = 3
    COORD_DIGITS = 4

    # Lista de posibles fallos
    FAILURE_TF_INFER = 'Failure TF inference'
    FAILURE_IMAGE_OPEN = 'Failure image access'

    DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD = 0.1

    DEFAULT_DETECTOR_LABEL_MAP = {
        '1': 'animal',
        '2': 'person',
        '3': 'vehicle'  # Disponibles en mega detector v4+
    }

    def __init__(self, model_path):
        """
        We load the model, create a session, and get the tensors that we need to detect objects in images
        :param model_path: The path to the frozen graph file that contains the model
        """
        detection_graph = TFDetector.load_model(model_path)
        self.tf_session = tf.Session(graph=detection_graph)

        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.box_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.score_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
        self.class_tensor = detection_graph.get_tensor_by_name('detection_classes:0')

    @staticmethod
    def round_and_make_float(d, precision=4):
        """
        It takes a number, rounds it to the nearest integer, and then converts it to a float
        :param d: the number to be rounded
        :param precision: The number of decimal places to round to, defaults to 4 (optional)
        :return: A float with a precision of 4.
        """
        return truncate_float(float(d), precision=precision)

    @staticmethod
    def convert_coordinates(tf_coordinates):
        """
        It converts the coordinates from [y1, x1, y2, x2] to [x1, y1, width, height]
        :param tf_coordinates: [y1, x1, y2, x2]
        :return: The coordinates of the bounding box.
        """
        # Cambiamos de [y1, x1, y2, x2] a [x1, y1, width, height]
        width = tf_coordinates[3] - tf_coordinates[1]
        height = tf_coordinates[2] - tf_coordinates[0]

        new = [tf_coordinates[1], tf_coordinates[0], width, height]  # must be a list instead of np.array

        # Convierte numpy floats a Python floats
        for i, d in enumerate(new):
            new[i] = TFDetector.round_and_make_float(d, precision=TFDetector.COORD_DIGITS)
        return new

    @staticmethod
    def convert_to_tf_coordinates(array):
        """
        It takes in a list of coordinates in the format [x1, y1, width, height] and returns a list of coordinates in the
        format [y1, x1, y2, x2]
        :param array: the array of bounding boxes
        :return: the coordinates of the bounding box in the format that Tensorflow requires.
        """
        x1 = array[0]
        y1 = array[1]
        width = array[2]
        height = array[3]
        x2 = x1 + width
        y2 = y1 + height
        return [y1, x1, y2, x2]

    @staticmethod
    def load_model(model_path):
        """
        It takes the path to a frozen inference graph, loads the graph into the default TensorFlow graph, and returns
        the graph
        :param model_path: The path to the frozen inference graph
        :return: The graph is being returned.
        """
        print('')
        print('==========================================================================================')
        print('TFDetector: Cargando gráficos')
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        print('TFDetector: Gráficos de detección, cargados.')
        print('==========================================================================================')
        print('')

        return detection_graph

    def generate_detection(self, image):
        """
        It takes an image, adds a batch dimension to it, and then runs the image through the TensorFlow session
        :param image: the image to perform inference on
        :return: The bounding box coordinates, the confidence score, and the class of the object detected.
        """
        numpy_im = np.asarray(image, np.uint8)
        im_w_batch_dim = np.expand_dims(numpy_im, axis=0)

        # need to change the above line to the following if supporting a batch size > 1 and resizing to the same size
        # numpy_images = [np.asarray(image, np.uint8) for image in images]
        # images_stacked = np.stack(numpy_images, axis=0) if len(images) > 1 else np.expand_dims(numpy_images[0], axis=0)

        # performs inference
        (box_tensor_out, score_tensor_out, class_tensor_out) = self.tf_session.run(
            [self.box_tensor, self.score_tensor, self.class_tensor],
            feed_dict={self.image_tensor: im_w_batch_dim})

        return box_tensor_out, score_tensor_out, class_tensor_out

    def generate_detections_image(self, image_obj, image_path, detection_threshold=DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD):
        """
        It takes an image, runs it through the model, and returns a dictionary with the detections
        :param image_obj: the image object
        :param image_path: The path to the image file
        :param detection_threshold: The confidence threshold for detections
        :return: A dictionary with the following fields:
            - 'file' (always present)
            - 'max_detection_conf'
            - 'detections', which is a list of the detection objects with the keys 'category', 'conf' and 'bbox'
            - 'failure'
            For more details: https: // github.com / microsoft / CameraTraps / tree / master / api / batch_processing
        """
        result = {
            'file': image_path
        }
        try:
            b_box, b_score, b_class = self.generate_detection(image_obj)

            # Nuestro batch size es 1; need to loop the batch dim if supporting batch size > 1
            boxes, scores, classes = b_box[0], b_score[0], b_class[0]

            detections = []  # Estará vacío si no encontramos detecciones que satisfacen el umbral especificado
            max_detection_score = 0.0
            for b, s, c in zip(boxes, scores, classes):
                if s > detection_threshold:
                    detection_successful = {
                        'category': str(int(c)),  # usamos string para el número de clase, no int
                        'conf': truncate_float(float(s), precision=TFDetector.CONF_DIGITS),
                        # cast a float para añadirlo al json
                        'bbox': TFDetector.convert_coordinates(b),
                    }
                    detections.append(detection_successful)
                    if s > max_detection_score:
                        max_detection_score = s

            result['max_detection_conf'] = truncate_float(float(max_detection_score), precision=TFDetector.CONF_DIGITS)
            result['detections'] = detections

        except Exception as e:
            result['failure'] = TFDetector.FAILURE_TF_INFER
            print('TFDetector: image {} failed during inference: {}'.format(image_path, str(e)))
            print('------------------------------------------------------------------------------------------')
            print(traceback.format_exc())

        return result


########################################################################################################################
# FUNCIONES AUXILIARES

def generate_json(results, output_dir):
    """
    It takes the results of the detection and writes them to a JSON file
    :param results: The results of the detection
    :param output_dir: The directory where the output files will be saved
    """
    if platform.system() == 'Windows':
        windows = True
    else:
        windows = False

    final_output = {
        'images': results,
        'detection_categories': TFDetector.DEFAULT_DETECTOR_LABEL_MAP,
        'info': {
            'detection_completion_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'format_version': '1.0'
        }
    }

    name = datetime.utcnow().strftime('%Y-%m-%d_%H-%M')
    file_name = name + '.json'

    if windows:
        os.makedirs((output_dir + '\\' + 'registry'), exist_ok=True)
        output_file = (output_dir + '\\' + 'registry' + '\\' + file_name)
    else:
        os.makedirs((output_dir + '/' + 'registry'), exist_ok=True)
        output_file = (output_dir + '/' + 'registry' + '/' + file_name)

    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=1)

    for i_image in final_output['images']:
        fn = i_image['file']
        file_name = os.path.basename(fn).lower()
        name, ext = os.path.splitext(file_name)
        i_results = {
            'file': fn,
            'max_detection_conf': i_image['max_detection_conf'],
            'detections': i_image['detections'],
            'detection_categories': TFDetector.DEFAULT_DETECTOR_LABEL_MAP,
        }
        if windows:
            output_file = (output_dir + '\\' + name + '.json')
        else:
            output_file = (output_dir + '/' + name + '.json')

        with open(output_file, 'w') as f:
            json.dump(i_results, f, indent=1)


########################################################################################################################
# FUNCTION PRINCIPAL

def run(model_file, image_file_names, output_dir):
    """
    It takes a model file, a list of image file names, and an output directory, and it runs the model on the images,
    saving the results in the output directory
    :param model_file: The path to the model file
    :param image_file_names: A list of image file names
    :param output_dir: The directory where the output files will be saved
    """
    if len(image_file_names) == 0:
        print("WARNING: No hay ficheros disponibles")
        return

    start_time = time.time()
    tf_detector = TFDetector(model_file)
    elapsed_time = time.time() - start_time

    print('')
    print('Modelo cargado en: {}.'.format(humanfriendly.format_timespan(elapsed_time)))
    print('==========================================================================================')
    print('')

    all_results = []
    time_load = []
    time_infer = []

    for image_file in tqdm(image_file_names):
        try:
            start_time = time.time()
            image_obj = visualization_utils.load_image(image_file)
            elapsed_time = time.time() - start_time
            time_load.append(elapsed_time)
        except Exception as e:
            print('La imagen {} no ha podido ser cargada. Exception: {}'.format(image_file, e))
            print('------------------------------------------------------------------------------------------')
            print(traceback.format_exc())
            result = {
                'file': image_file,
                'failure': TFDetector.FAILURE_IMAGE_OPEN
            }
            all_results.append(result)
            continue

        try:
            print('')
            start_time = time.time()
            result = tf_detector.generate_detections_image(image_obj, image_file)
            all_results.append(result)
            elapsed_time = time.time() - start_time
            print('')
            print('Generadas detecciones de imagen {} en {}.'.format(image_file,
                                                                     humanfriendly.format_timespan(elapsed_time)))
            time_infer.append(elapsed_time)
        except Exception as e:
            print('')
            print('Ha ocurrido un error mientras se ejecutaba el detector en la imagen {}. EXCEPTION: {}'.format(
                image_file, e))
            print('------------------------------------------------------------------------------------------')
            print(traceback.format_exc())
            continue
    try:
        generate_json(all_results, output_dir)
        print('')
        print('==========================================================================================')
        print('Generado JSON con detecciones')
        print('==========================================================================================')
    except Exception as e:
        print(traceback.format_exc())
        print('!!!')
        print('Error al generar fichero JSON de salida en la ruta {}, EXCEPTION: {}'.format(output_dir, e))

    average_time_load = statistics.mean(time_load)
    average_time_infer = statistics.mean(time_infer)

    if len(time_load) > 1 and len(time_infer) > 1:
        std_dev_time_load = humanfriendly.format_timespan(statistics.stdev(time_load))
        std_dev_time_infer = humanfriendly.format_timespan(statistics.stdev(time_infer))
    else:
        std_dev_time_load = 'NO DISPONIBLE'
        std_dev_time_infer = 'NO DISPONIBLE'

    print('')
    print('==========================================================================================')
    print('De media, por cada imagen: ')
    print('Ha tomado {} en cargar, con desviación de {}'.format(humanfriendly.format_timespan(average_time_load),
                                                                std_dev_time_load))
    print('Ha tomado {} en procesar, con desviación de {}'.format(humanfriendly.format_timespan(average_time_infer),
                                                                  std_dev_time_infer))
    print('==========================================================================================')


########################################################################################################################
# Command-line driver
def main():
    parser = argparse.ArgumentParser(
        description='Modulo para ejecutar un modelo de detección en Tensorflow, sobre imágenes')
    parser.add_argument(
        'detector_file',
        help='Ruta al fichero .pb del modelo de detección de tensorflow'
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--image_file',
        help='Único fichero de imagen a procesar, para acceder a múltiples imágenes usar --image_dir'
    )
    group.add_argument(
        '--image_dir',
        help='Directorio donde se buscarán las imágenes a procesar, con la opción --recursive'
    )
    group.add_argument(
        '--csv_file',
        help='Fichero CSV con las rutas de las imágenes que forman el dataset'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Maneja directorios de forma recursiva, solo tiene sentido usarlo con --image_dir'
    )
    parser.add_argument(
        '--dataset_dir',
        help='Directorio donde se ubican las imágenes del dataset'
    )
    parser.add_argument(
        '--output_dir',
        help='Directorio de salida de los ficheros JSON, que contienen los datos de las detecciones'
    )

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    assert os.path.exists(args.detector_file), 'El fichero del modelo de detección no existe'

    if args.image_file:
        image_file_names = [args.image_file]
    else:
        if args.image_dir:
            image_file_names = PathUtils.find_images(args.image_dir, args.recursive)
        else:
            file_names, labels = DatasetUtils.load_csv(args.csv_file, 'utf-8')
            file_names, labels = DatasetUtils.convert_to_abspath(args.dataset_dir, file_names, labels)
            image_file_names = file_names

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        if args.image_dir:
            args.output_dir = args.image_dir
        else:
            args.output_dir = os.path.dirname(args.image_file)

    print('Ejecutando detector en {} imágenes...'
          .format(len(image_file_names)))

    run(model_file=args.detector_file, image_file_names=image_file_names, output_dir=args.output_dir)

    print('')
    print('Resultados guardados en: {}'.format(args.output_dir))
    print('')
    print('==========================================================================================')


if __name__ == '__main__':
    main()
