#%% Constantes, importaciones

import argparse

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



