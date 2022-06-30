"""
Módulo para ejecutar un modelo de detección de animales con TensorFlow.

La clase TFDetector contiene funciones para cargar el modelo de detección de Tensorflow
y ejecutar una instancia. La función main además calculará los bounding boxes de las
predicciones y guardar los resultados.

Si no desea usar la GPU debe seleccionar la variable: CUDA_VISIBLE_DEVICES a -1

Este módulo solo considera detecciones por encima de 0.1 de umbral. El umbral que
se especifica es para las renderizaciones. Si desea trabajar con valores menores deberá
cambiar la variable: DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD
"""
###################################################################################################
# IMPORTs

import argparse
import humanfriendly
import imagepathutils as ip_utils
import numpy as np
import time
import visualization.visualization_utils as v_utils
import statistics
import platform
import json
import sys
import os
import traceback



###################################################################################################
# FROMs

from imagepathutils import ImagePathUtils as ip_utils
from ct_utils import truncate_float
from tqdm import tqdm
from datetime import datetime



###################################################################################################
# IMPORTACION TENSORFLOW

# Useful hack to force CPU inference
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow.compat.v1 as tf

print('==========================================================================================')
print('TensorFlow version:', tf.__version__)
#print('Is GPU available? tf.test.is_gpu_available:', tf.test.is_gpu_available())
print('Is GPU available? tf.test.is_gpu_available:',tf.config.list_physical_devices('GPU'))
print('==========================================================================================')


###################################################################################################
# CLASES:

class TFDetector:
    """
    Un modelo de detector cargado en el momento de la inicialización. Está destinado a ser 
    utilizado con el MegaDetector (TF). El tamaño del lote de inferencia se establece en 1 
    por defecto, si desea cambiarlo, tendrá que modificar el código.
    """

    # Number of decimal places to round to for confidence and bbox coordinates
    CONF_DIGITS = 3
    COORD_DIGITS = 4

    # MegaDetector was trained with batch size of 1, and the resizing function is a part
    # of the inference graph
    BATCH_SIZE = 1

    # Lista de posibles fallos
    FAILURE_TF_INFER = 'Failure TF inference'
    FAILURE_IMAGE_OPEN = 'Failure image access'

    DEFAULT_RENDERING_CONFIDENCE_THRESHOLD = 0.85  # to render bounding boxes
    DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD = 0.1  # to include in the output json file

    DEFAULT_DETECTOR_LABEL_MAP = {
        '1': 'animal',
        '2': 'person',
        '3': 'vehicle'  # Disponibles en megadetector v4+
    }

    NUM_DETECTOR_CATEGORIES = 4  # animal, person, group, vehicle - for color assignment

    def __init__(self, model_path):
        """
        Carga el modelo desde model_path e inicia tf.Session con este gráfico. Obtiene 
        tensor handles de entrada y salida.
        """
        detection_graph = TFDetector.load_model(model_path)
        self.tf_session = tf.Session(graph=detection_graph)

        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.box_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')
        self.score_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
        self.class_tensor = detection_graph.get_tensor_by_name('detection_classes:0')

    @staticmethod
    def round_and_make_float(d, precision=4):
        return truncate_float(float(d), precision=precision)

    @staticmethod
    def convert_coords(tf_coords):
        """
        Convierte las coordenadas del formato de salida del modelo: [y1, x1, y2, x2]
        al formato de coordenadas de la API de MegaDB: [x1, y1, width, height].

        Ambos formatos de coordenadas se cuentran normalizados entre [0, 1].

        Args:
        - tf_coords: np.array que contiene las coordenadas del bounding box que devuelve 
            TF detector, con el formato: [y1, x1, y2, x2].

        Returns: Listo de decimales (Python), con las coordenadas del bounding box, pero
            en formato: [x1, y1, width, height].
        """

        # Cambiamos de [y1, x1, y2, x2] a [x1, y1, width, height]
        width = tf_coords[3] - tf_coords[1]
        height = tf_coords[2] - tf_coords[0]

        new = [tf_coords[1], tf_coords[0], width, height]  # must be a list instead of np.array

        # convert numpy floats to Python floats
        for i, d in enumerate(new):
            new[i] = TFDetector.round_and_make_float(d, precision=TFDetector.COORD_DIGITS)
        return new   

    @staticmethod
    def convert_to_tf_coords(array):
        """
        A partir del formato de coordenadas: [x1, y1, width, height], las convierte al 
        formato: [y1, x1, y2, x2]. Donde x1 es x_min y x2 es x_max.

        Esta función la hemos creado para mantener la compatibilidad con la API.
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
        Carga el modelo de detección desde un fichero ".pb".

        Args:
            -model_path: ruta donde se encuentra el fichero ".pb" del modelo.
        Returns: Los gráficos cargados.
        """
        print('==========================================================================================')
        print('TFDetector: Loading graph...')
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        print('TFDetector: Detection graph loaded.')
        print('==========================================================================================')

        return detection_graph  

    def generate_detection(self, image):
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

    def generate_detections_image(self, image_obj, image_path, 
            detection_threshold=DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD):
        """
        Aplicar el detector a una imagen especificada.

        Args:
            image: the PIL Image object
            image_id: a path to identify the image; will be in the "file" field of the output object
            detection_threshold: confidence above which to include the detection proposal

        Returns:
        A dict with the following fields, see the 'images' key in https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
            - 'file' (always present)
            - 'max_detection_conf'
            - 'detections', which is a list of detection objects containing keys 'category', 'conf' and 'bbox'
            - 'failure'
        """
        result = {
            'file': image_path
        }
        try:
            b_box, b_score, b_class = self.generate_detection(image_obj)

            # our batch size is 1; need to loop the batch dim if supporting batch size > 1
            boxes, scores, classes = b_box[0], b_score[0], b_class[0]

            detections = []  # Estará vacío si no encontramos detecciones que satisfacen el umbral especificado
            max_detection_score = 0.0
            for b, s, c in zip(boxes, scores, classes):
                if s > detection_threshold:
                    detection_successful = {
                        'MD_class': str(int(c)),    # usamos string para el número de clase, no int
                        'score': truncate_float(float(s), precision=TFDetector.CONF_DIGITS),    # cast a float para añadirlo al json
                        'bbox': TFDetector.convert_coords(b),
                    }
                    detections.append(detection_successful)
                    if s > max_detection_score:
                        max_detection_score = s

            result['max_detection_conf'] = truncate_float(float(max_detection_score), 
                                                precision=TFDetector.CONF_DIGITS)
            result['detections'] = detections

        except Exception as e:
            print(traceback.format_exc())
            result['failure'] = TFDetector.FAILURE_TF_INFER
            print('TFDetector: image {} failed during inference: {}'
                .format(image_path, str(e)))

        return result



###################################################################################################
# FUNCIONES AUXIIARES

def generate_json(results, output_dir):
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
        output_file = (output_dir + '\\' + file_name)
    else:
        output_file = (output_dir + '/' + file_name)

    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=1)

    print('==========================================================================================')
    print('Fichero de salida guardado en {}'.format(output_file))
    print('==========================================================================================')




###################################################################################################
# FUNCION PRINCIPAL

def run(model_file, image_file_names, output_dir):
    """
    
    """
    
    if len(image_file_names) == 0:
        print("WARNING: No hay ficheros disponibles")
        return

    start_time = time.time()
    tf_detector = TFDetector(model_file)
    elapsed_time = time.time() - start_time

    print('==========================================================================================')
    print('Modelo cargado en: {}.'
        .format(humanfriendly.format_timespan(elapsed_time)))
    print('==========================================================================================')

    all_results = []
    time_load = []
    time_infer = []

    for image_file in tqdm(image_file_names):
        try:
            start_time = time.time()
            image_obj = v_utils.load_image(image_file)
            elapsed_time = time.time() - start_time
            time_load.append(elapsed_time)
        except Exception as e:
            print(traceback.format_exc())
            print('La imagen {} no ha podido ser cargada. Excepcion: {}'.format(image_file, e))
            result = {
                'file': image_file,
                'failure': TFDetector.FAILURE_IMAGE_OPEN
            }
            all_results.append(result)
            continue

        try:
            start_time = time.time()
            result = tf_detector.generate_detections_image(image_obj, image_file)
            all_results.append(result)
            elapsed_time = time.time() - start_time
            print('Generadas detecciones de imagen {} en {}.'
                .format(image_file, humanfriendly.format_timespan(elapsed_time)))
            time_infer.append(elapsed_time)
        except Exception as e:
            print(traceback.format_exc())
            print('Ha ocurrido un error mientras se ejecutaba el detector en la imagen {}. EXCEPTION: {}'
                .format(image_file, e))
            continue
    try:
        generate_json(all_results,output_dir)
    except Exception as e:
        print(traceback.format_exc())
        print('Error al generar fichero JSON de salida en la ruta {}, EXCEPTION: {}'
            .format(output_dir, e))

    average_time_load = statistics.mean(time_load)
    average_time_infer = statistics.mean(time_infer)

    if len(time_load) > 1 and len(time_infer) > 1:
        std_dev_time_load = humanfriendly.format_timespan(statistics.stdev(time_load))
        std_dev_time_infer = humanfriendly.format_timespan(statistics.stdev(time_infer))
    else:
        std_dev_time_load = 'NO DISPONIBLE'
        std_dev_time_infer = 'NO DISPONIBLE'

    print('==========================================================================================')
    print('De media, por cada imagen, ')
    print('Ha tomado {} en cargar, con desviación de {}'
        .format(humanfriendly.format_timespan(average_time_load), std_dev_time_load))
    print('Ha tomado {} en procesar, con desviación de {}'
        .format(humanfriendly.format_timespan(average_time_infer), std_dev_time_infer))
    print('==========================================================================================')



###################################################################################################
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
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Maneja directorios de forma recursiva, solo tiene sentido usarlo con --image_dir'
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
        image_file_names = ip_utils.find_images(args.image_dir, args.recursive)

    print('==========================================================================================')
    print('Ejecutando detector en {} imágenes...'
        .format(len(image_file_names)))

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        if args.image_dir:
            args.output_dir = args.image_dir
        else:
            args.output_dir = os.path.dirname(args.image_file)

    run(model_file=args.detector_file,
        image_file_names=image_file_names,
        output_dir=args.output_dir
    )
    print('==========================================================================================')



if __name__ == '__main__':
    main()



###################################################################################################