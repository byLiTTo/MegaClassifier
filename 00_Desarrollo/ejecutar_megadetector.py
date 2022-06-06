#%% Constantes, importaciones

import argparse
import os
import time
import warnings
import sys
import json

import humanfriendly
from tqdm import tqdm

#%% Importacion de TENSORFLOW

force_cpu = False
if force_cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from detection.run_tf_detector import ImagePathUtils, TFDetector
import visualization.visualization_utils as viz_utils

# Numpy FutureWarnings from tensorflow import
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow.compat.v1 as tf

print('TensorFlow version:', tf.__version__)
print('tf.test.is_gpu_available:', tf.test.is_gpu_available())

#%% FUNCIONES AUXILIARES

""""
Divide una lista en n partes pares.

Args
    - ls: list
    - n: int, # de partes
"""
def chunks_by_number_of_chunks(ls, n):
    for i in range(0, n):
        yield ls[i::n]



#%% Command-line driver

def main():
    parser = argparse.ArgumentParser(
        description='Módulo para ejecutar un modelo con TF para la detección de animales en varias imágenes'
    )
    parser.add_argument(
        'ruta_modelo',
        help='Ruta al fichero de modelo del detector TensorFlow.pb'
    )
    parser.add_argument(
        'ruta_entrada',
        help='Ruta a un único archivo de imagen, ruta a un archivo JSON que tiene una lista de rutas a imágenes o ruta a un directorio que contiene imágenes'
    )
    parser.add_argument(
        'ruta_salida',
        help='Ruta al archivo de resultados JSON de salida, debe tener una extensión .json'
    )
    parser.add_argument(
        '--recursive',
        action= 'sotre_true',
        help='Recorre los directorios, solo se ejecuta si ruta_entrada apunta a un directorio'
    )
    parser.add_argument(
        '--output_relative_filenames',
        action='store_true',
        help='Salida de nombres de archivos relativos, solo tiene importancia si ruta_entrada apunta a un directorio'
    )
    parser.add_argument(
        '--umbral',
        type=float,
        default=TFDetector.DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD,
        help="Umbral de confianza entre 0.0 y 1.0. (No incluye valores por debajo de este umbral). Por defecto es 0.1."
    )

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    assert os.path.exists(args.ruta_modelo), 'La ruta_modelo especificada no existe'

    assert 0.0 < args.umbral <= 1.0, 'El umbral de confianza debe estar entre 0 y 1'  # Python chained comparison

    assert args.ruta_salida.endswith('.json'), 'El archivo de salida especificado debe terminar con .json'

    if args.output_relative_filenames:
        assert os.path.isdir(args.image_file), 'image_file debe ser un directorio cuando se establece --output_relative_filenames'

    if os.path.exists(args.ruta_salida):
        print('Warning: ruta_salida {} ya existe y será sobrescrito.'.format(args.ruta_salida))

    resultados = []

    if os.path.isdir(args.ruta_entrada):
        image_file_names = ImagePathUtils.find_images(args.ruta_entrada, args.recursive)
        print('{} archivos de imagen encontrados en el directorio de entrada'.format(len(image_file_names)))
    elif os.path.isfile(args.ruta_entrada) and args.ruta_entrada.endswith('.json'):
        with open(args.ruta_entrada) as f:
            image_file_names = json.load(f)
        print('{} archivos de imagen encontrados en la lista json'.format(len(image_file_names)))
    elif os.path.isfile(args.ruta_entrada) and ImagePathUtils.is_image_file(args.ruta_entrada):
        image_file_names = [args.ruta_entrada]
        print('Una sola imagen en {} es el archivo de entrada'.format(args.ruta_entrada))
    else:
        raise ValueError('image_file especificado no es un directorio, una lista json o un archivo de imagen, '
                         '(o no tiene extensiones reconocibles).')

    assert len(image_file_names) > 0, 'El archivo de imagen especificado no apunta a archivos de imagen válidos'

    assert os.path.exists(image_file_names[0]), 'La primera imagen a puntuar no existe en {}'.format(image_file_names[0])

    output_dir = os.path.dirname(args.ruta_salida)

    if len(output_dir) > 0:
        os.makedirs(output_dir,exist_ok=True)
        
    assert not os.path.isdir(args.ruta_salida), 'El archivo de salida especificado es un directorio'

    checkpoint_path = None

    start_time = time.time()

    results = load_and_run_detector_batch(model_file=args.ruta_modelo,
                                          image_file_names=image_file_names,
                                          checkpoint_path=None,
                                          confidence_threshold=args.umbral,
                                          checkpoint_frequency=-1,
                                          results=results,
                                          n_cores=0,
                                          use_image_queue=False)

    elapsed = time.time() - start_time
    print('Inferencia terminada en {}'.format(humanfriendly.format_timespan(elapsed)))

    relative_path_base = None
    if args.output_relative_filenames:
        relative_path_base = args.ruta_entrada
    write_results_to_file(results, args.ruta_salida, relative_path_base=relative_path_base)


    print('Hecho!')


if __name__ == '__main__':
    main()


    


#%% FUNCIONES DE PROCESAMIENTO DE IMAGEN

""""
Ejecuta MegaDetector sobre una lista de archivos de imagen.

Args
- im_files: lista de str, ruta hacia los ficheros de imágenes
- tf_detector: TFDetector (modelo cargado) o str (ruta al fichero de modelo .pb)
- confidence_threshold: float, solo se devuelven las detecciones por encima de este umbral

Returns
    - results: lists de dict, cada dict representa detecciones en una imagen
        para ver el formato más completo: https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
"""
def procesa_imagenes(imagenes, tf_detector, umbral):
    if isinstance(tf_detector, str):
        inicio = time.time()
        tf_detector = TFDetector(tf_detector)
        transcurrido = time.time() - inicio
        print('Cargado modelo (batch level) en {}'.format(humanfriendly.format_timespan(transcurrido)))

        resultados = []
        for imagen in imagenes:
            resultados.append(procesa_imagen(imagen, tf_detector, umbral))
        return resultados

""""
Ejecuta MegaDetector sobre un solo archivo de imagen.

Args
- im_files: lista de str, ruta hacia los ficheros de imágenes
- tf_detector: TFDetector (modelo cargado) o str (ruta al fichero de modelo .pb)
- confidence_threshold: float, solo se devuelven las detecciones por encima de este umbral
- image: imagen precargada anteriormente, si está disponible

Returns
- results: lists de dict, cada dict representa detecciones en una imagen
    para ver el formato más completo: https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
"""
def proces_imagen(imagen, tf_detector, umbral, im=None):
    print('Procesando imagen {}'.format(imagen))

    if im is None:
        try:
            image = viz_utils.load_image(imagen)
        except Exception as e:
            print('Imagen {} no se puiede cargar. Excepcion: {}'.format(imagen, e))
            resultado = {
                'fichero': imagen,
                'error': TFDetector.FAILURE_IMAGE_OPEN
            }
            return resultado
    try:
        resultado = tf_detector.generate_detections_one_image(
            im, imagen, umbral_detecion=umbral)
    except Exception as e:
        print('Iamgen {} no puede ser procesada. Exception: {}'.format(imagen, e))
        resultado = {
                'fichero': imagen,
                'error': TFDetector.FAILURE_IMAGE_OPEN
            }
        return resultado

    return resultado



