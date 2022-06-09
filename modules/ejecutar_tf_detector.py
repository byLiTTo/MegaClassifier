""""
Modulo para ejecutar el modelo de detección de animales con Tensorflow en paquetes de 
imágenes, escribir los resultados en un fichero con el formato que se indica en batch API:

https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing

Esto permite que los resultados se utilicen en nuestra cola de posprocesamiento, mirar:
api/batch_processing/postprocessing/postprocess_batch_results.py .

Este script tiene una funcionalidad *somewhat* probada para guardar los resultados en los 
puntos de control intermitentemente, en caso de que ocurra un desastre. Para activar esto, 
seleccionamos --checkpoint_frequency a n > 0, y los resultados serán guardados como 
checkpoint cada n imágenes. Los checkpoints serán guardados en un fichero en el mismo
directorio que output_file, después de que todas las imágenes sean procesadas, los
resultados finales serán escritos en el archivo output_file y el checkpoint temporal será
borrado. Si desea reanudar desde un punto de control, establezca la ruta del archivo de 
punto de control usando --resume_from_checkpoint.

El umbral (Threshold) se puede proporcionar como argumento. Es el umbral de confianza por 
encima del cual las detecciones se incluirá en el archivo de salida.

Tiene soporte preliminar de multiprocesamiento solo para CPUs, si la GPU está disponible,
usará GPU en lugar de CPUs y la opción --ncores será ignorada. La opción de realizar
checkpoints no estará disponible con el uso de multiprocessing.

Ejemplo de ejecución de comando:
python run_tf_detector_batch.py "d:\temp\models\md_v4.1.0.pb" "d:\temp\test_images" "d:\temp\out.json" --recursive

"""

#%% Constantes, importaciones y entorno

import argparse
import json
import os
import sys
import time
import copy
import shutil
import warnings
import itertools

from datetime import datetime
from functools import partial

import humanfriendly
from tqdm import tqdm

# from multiprocessing.pool import ThreadPool as workerpool
import multiprocessing
from threading import Thread
from multiprocessing import Process
from multiprocessing.pool import Pool as workerpool

# Número de imágenes a pre-fetch
max_queue_size = 10
use_threads_for_queue = False
verbose = True

# Truco útil para forzar la inferencia de la CPU
#
# Necesario hacer esto antes de cualquier importación de TF
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

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

#%% Funciones de soporte para multiprocesos

""""
Función de productor; solo se usa cuando se usa la cola de imágenes (opcional).

Lee hasta N imágenes del disco y las coloca en la cola de bloqueo para su procesamiento.
"""
def producer_func(q, image_files):
    if verbose:
        print('Comienza producción'); sys.stdout.flush()
    for im_file in image_files:
        try:
            if verbose:
                print('Cargando imagen {}'.format(im_file)); sys.stdout.flush()
            image = viz_utils.load_image(im_file)
        except Exception as e:
            print('Proceso de producción: imagen {} no puede ser cargada. Excepción: {}'.format(im_file, e))
            raise
        print('Imagen {} en cola'.format(im_file)); sys.stdout.flush()
        q.put([im_file,image])
    q.put(None)
    print('Finalizada carga de imagenes'); sys.stdout.flush()


""""
Función del consumidor; solo se usa cuando se usa la cola de imágenes (opcional).

Extrae imágenes de una cola de bloqueo y las procesa.
"""
def consumer_func(q, return_queue, model_file, confidence_threshold):
    if verbose:
        print('Empieza consumo'); sys.stdout.flush()
    start_time = time.time()
    tf_detector = TFDetector(model_file)
    elapsed = time.time() - start_time
    print('Cargado modelo (antes de hacer cola) en {}'.format(humanfriendly.format_timespan(elapsed)))
    sys.stdout.flush()

    results = []

    while True:
        r = q.get()
        if r is None:
            q.task_done()
            return_queue.put(results)
            return
        im_file = r[0]
        image = r[1]
        if verbose:
            print('Encolada imagen {}'.format(im_file)); sys.stdout.flush()
        results.append(process_image(im_file, tf_detector, confidence_threshold, image))
        if verbose:
            print('Procesada imagen {}'.format(im_file)); sys.stdout.flush()
        q.task_done()


""""
Función de controlador para la cola de imágenes basada en multiprocesamiento (opcional);
solo se usa cuando --use_image_queue está especificado.

Inicia un proceso de lectura para leer imágenes del disco, pero procesa imágenes en el 
proceso desde el cual se llama a esta función (actualmente no genera un consumidor separado
proceso).
"""
def run_detector_with_image_queue(image_files,model_file,confidence_threshold):
    q = multiprocessing.JoinableQueue(max_queue_size)
    return_queue = multiprocessing.Queue(1)
    
    if use_threads_for_queue:
        producer = Thread(target=producer_func,args=(q,image_files,))
    else:
        producer = Process(target=producer_func,args=(q,image_files,))
    producer.daemon = False
    producer.start()

    run_separate_consumer_process = False

    if run_separate_consumer_process:
        if use_threads_for_queue:
            consumer = Thread(target=consumer_func,args=(q,return_queue,model_file,confidence_threshold,))
        else:
            consumer = Process(target=consumer_func,args=(q,return_queue,model_file,confidence_threshold,))
        consumer.daemon = True
        consumer.start()
    else:
        consumer_func(q,return_queue,model_file,confidence_threshold)

    producer.join()
    print('Productor finalizado')
   
    if run_separate_consumer_process:
        consumer.join()
        print('Consumidor finalizado')
    
    q.join()
    print('Se unió a la cola')

    results = return_queue.get()
    
    return results



#%% Otras funciones auxiliares

""""
Divide una lista en n partes pares.

Args
    - ls: list
    - n: int, # de partes
"""
def chunks_by_number_of_chunks(ls, n):
    for i in range(0, n):
        yield ls[i::n]



#%% Funciones de procesamiento de imagen

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
def process_images(im_files, tf_detector, confidence_threshold, use_image_queue=False):
    if isinstance(tf_detector, str):
        start_time = time.time()
        tf_detector = TFDetector(tf_detector)
        elapsed = time.time() - start_time
        print('Cargado modelo (batch level) en {}'.format(humanfriendly.format_timespan(elapsed)))

    if use_image_queue:
        run_detector_with_image_queue(im_files, tf_detector, confidence_threshold)
    else:
        results = []
        for im_file in im_files:
            results.append(process_image(im_file, tf_detector, confidence_threshold))
        return results


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
def process_image(im_file, tf_detector, confidence_threshold, image=None):
    print('Procesando imagen {}'.format(im_file))
    
    if image is None:
        try:
            image = viz_utils.load_image(im_file)
        except Exception as e:
            print('Imagen {} no se puede cargar. Excepción: {}'.format(im_file, e))
            result = {
                'fichero': im_file,
                'error': TFDetector.FAILURE_IMAGE_OPEN
            }
            return result

    try:
        result = tf_detector.generate_detections_one_image(
            image, im_file, detection_threshold=confidence_threshold)
    except Exception as e:
        print('Imagen {} no puede ser procesada. Excepción: {}'.format(im_file, e))
        result = {
            'fichero': im_file,
            'error': TFDetector.FAILURE_TF_INFER
        }
        return result

    return result



""""
Args
- model_file: str, ruta al fichero .pb del modelo 
- image_file_names: list de str, ruta al directorio de imágenes
- checkpoint_path: str, ruta al fichero JSON (checkpoint)
- confidence_threshold: float, ssolo se devuelven las detecciones por encima de este umbral
- checkpoint_frequency: int, escribir los resultados en el fichero JSON cada N imágenes
- results: list de dict, resultados existentes cargados desde un checkpoint
- n_cores: int, # de núcleos de CPU para usar

Returns
- results: list de dict, cada dict representa detecciones en una imagen
    para ver el formato más completo: https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format
"""
def load_and_run_detector_batch(model_file, image_file_names, checkpoint_path=None,
                                confidence_threshold=0, checkpoint_frequency=-1,
                                results=None, n_cores=0, use_image_queue=False):
    if results is None:
        results = []

    already_processed = set([i['file'] for i in results])

    if n_cores > 1 and tf.test.is_gpu_available():
        print('Warning: se solicitaron varios núcleos, pero hay una GPU disponible.; la paralelización entre GPU no es compatible actualmente, por defecto es una GPU')
        n_cores = 1

    if n_cores > 1 and use_image_queue:
        print('Warning: múltiples núcleos solicitados, pero la cola de imágenes está habilitada; la paralelización con la cola de imágenes no se admite actualmente, por defecto a un núcleo')
        n_cores = 1
        
    if use_image_queue:
        
        assert n_cores <= 1
        results = run_detector_with_image_queue(image_file_names, model_file, confidence_threshold)        
        
    elif n_cores <= 1:
        # Cargando el detector
        start_time = time.time()
        tf_detector = TFDetector(model_file)
        elapsed = time.time() - start_time
        print('Cargado modelo en {}'.format(humanfriendly.format_timespan(elapsed)))

        # No cuenta los ya procesados
        count = 0

        for im_file in tqdm(image_file_names):

            # No agregará entradas adicionales que no estén en el punto de control inicial
            if im_file in already_processed:
                print('Bypassing imagen {}'.format(im_file))
                continue

            count += 1

            result = process_image(im_file, tf_detector, confidence_threshold)
            results.append(result)

            # Escriba un punto de control si es necesario
            if checkpoint_frequency != -1 and count % checkpoint_frequency == 0:
                
                print('Escribir un nuevo punto de control después de haber procesado {} imágenes desde el último reinicio'.format(count))
                
                assert checkpoint_path is not None
                
                # Copia de seguridad de los puntos de control anteriores
                checkpoint_tmp_path = None
                if os.path.isfile(checkpoint_path):
                    checkpoint_tmp_path = checkpoint_path + '_tmp'
                    shutil.copyfile(checkpoint_path,checkpoint_tmp_path)
                    
                # Escribe el nuevo punto de control
                with open(checkpoint_path, 'w') as f:
                    json.dump({'images': results}, f, indent=1)
                    
                # Eliminar el punto de control de copia de seguridad si existe
                if checkpoint_tmp_path is not None:
                    os.remove(checkpoint_tmp_path)
                    
            # ...si es hora de hacer un punto de control
            
    else:
        
        # Al usar multiprocesamiento, deje que los cores carguen el modelo.
        tf_detector = model_file

        print('Creando pool con {} cores'.format(n_cores))

        if len(already_processed) > 0:
            print('Warning: cuando se usa el multiprocesamiento, todas las imágenes se reprocesan')

        pool = workerpool(n_cores)

        image_batches = list(chunks_by_number_of_chunks(image_file_names, n_cores))
        results = pool.map(partial(process_images, tf_detector=tf_detector,
                                   confidence_threshold=confidence_threshold), image_batches)

        results = list(itertools.chain.from_iterable(results))

    # Es posible que los resultados se hayan modificado en su lugar, pero también lo 
    # devolvemos por compatibilidad con versiones anteriores.
    return results



""""
Escribe la lista de resultados de detección en el archivo de salida JSON. Formato:
https://github.com/microsoft/CameraTraps/tree/master/api/batch_processing#batch-processing-api-output-format

Args
- results: list de dict, cada dict representa detecciones en una imagen
- output_file: str, ruta al fichero JSON de salida, debe terminar en '.json'
- relative_path_base: str, ruta a un directorio como base para rutas relativas
"""
def write_results_to_file(results, output_file, relative_path_base=None):
    if relative_path_base is not None:
        results_relative = []
        for r in results:
            r_relative = copy.copy(r)
            r_relative['file'] = os.path.relpath(r_relative['file'], start=relative_path_base)
            results_relative.append(r_relative)
        results = results_relative

    final_output = {
        'images': results,
        'detection_categories': TFDetector.DEFAULT_DETECTOR_LABEL_MAP,
        'info': {
            'detection_completion_time': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
            'format_version': '1.0'
        }
    }
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=1)
    print('Fichero de salida guardado en {}'.format(output_file))



#%% Driver interactivo

if False:
    
    pass

    #%%
    
    checkpoint_path = None
    model_file = r'G:\temp\models\md_v4.1.0.pb'
    confidence_threshold = 0.1
    checkpoint_frequency = -1
    results = None
    ncores = 1
    use_image_queue = True
    image_dir = r'G:\temp\demo_images\ssmini'
    image_file_names = image_file_names = ImagePathUtils.find_images(image_dir, recursive=False)
    # image_file_names = image_file_names[0:2]
    
    start_time = time.time()
    
    # python run_tf_detector_batch.py "g:\temp\models\md_v4.1.0.pb" "g:\temp\demo_images\ssmini" "g:\temp\ssmini.json" --recursive --output_relative_filenames --use_image_queue
    
    results = load_and_run_detector_batch(model_file=model_file,
                                          image_file_names=image_file_names,
                                          checkpoint_path=checkpoint_path,
                                          confidence_threshold=confidence_threshold,
                                          checkpoint_frequency=checkpoint_frequency,
                                          results=results,
                                          n_cores=ncores,
                                          use_image_queue=use_image_queue)
    
    elapsed = time.time() - start_time
    
    print('Inferencia terminada en {}'.format(humanfriendly.format_timespan(elapsed)))



#%% Command-line driver


def main():
    
    parser = argparse.ArgumentParser(
        description='Módulo para ejecutar un modelo de detección de animales TF en muchas imágenes')
    parser.add_argument(
        'detector_file',
        help='Ruta al archivo de modelo del detector TensorFlow .pb')
    parser.add_argument(
        'image_file',
        help='Ruta a un solo archivo de imagen, un archivo JSON que contiene una lista de rutas a imágenes o un directorio')
    parser.add_argument(
        'output_file',
        help='La ruta al archivo de resultados JSON de salida debe terminar con una extensión .json')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recurre a directorios, solo se ejecuta si image_file apunta a un directorio')
    parser.add_argument(
        '--output_relative_filenames',
        action='store_true',
        help='Salida de nombres de archivos relativos, solo significativos si image_file apunta a un directorio')
    parser.add_argument(
        '--use_image_queue',
        action='store_true',
        help='Precargar las imágenes puede ayudar a mantener su GPU ocupada; actualmente no es compatible con los puntos de control. Útil si tienes una GPU muy rápida y un disco muy lento.')
    parser.add_argument(
        '--threshold',
        type=float,
        default=TFDetector.DEFAULT_OUTPUT_CONFIDENCE_THRESHOLD,
        help="Umbral de confianza entre 0 y 1,0, no incluya casillas debajo de esta confianza en el archivo de salida. El valor predeterminado es 0,1")
    parser.add_argument(
        '--checkpoint_frequency',
        type=int,
        default=-1,
        help='Escriba los resultados en un archivo temporal cada N imágenes; el valor predeterminado es -1, lo que deshabilita esta característica')
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default=None,
        help='Nombre de archivo en el que se escribirán los puntos de control si checkpoint_frequency es > 0')
    parser.add_argument(
        '--resume_from_checkpoint',
        help='La ruta a un archivo de punto de control JSON para reanudar, debe estar en el mismo directorio que el archivo de salida')
    parser.add_argument(
        '--ncores',
        type=int,
        default=0,
        help='Número de núcleos a utilizar; solo se aplica a la inferencia basada en CPU, no admite puntos de control cuando ncores > 1')

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    assert os.path.exists(args.detector_file), 'El detector_file especificado no existe'
    assert 0.0 < args.threshold <= 1.0, 'El umbral de confianza debe estar entre 0 y 1'  # Python chained comparison
    assert args.output_file.endswith('.json'), 'El archivo de salida especificado debe terminar con .json'
    if args.checkpoint_frequency != -1:
        assert args.checkpoint_frequency > 0, 'Checkpoint_frequency debe ser > 0 o == -1'
    if args.output_relative_filenames:
        assert os.path.isdir(args.image_file), 'image_file debe ser un directorio cuando se establece --output_relative_filenames'

    if os.path.exists(args.output_file):
        print('Warning: output_file {} ya existe y será sobrescrito.'.format(args.output_file))

    # Load the checkpoint if available
    #
    # Relative file names are only output at the end; all file paths in the checkpoint are
    # still full paths.
    if args.resume_from_checkpoint:
        assert os.path.exists(args.resume_from_checkpoint), 'El archivo en resume_from_checkpoint especificado no existe'
        with open(args.resume_from_checkpoint) as f:
            saved = json.load(f)
        assert 'images' in saved, \
            'El archivo guardado como punto de control no tiene los campos correctos; no se puede restaurar'
        results = saved['images']
        print('Restauradas {} entradas desde el punto de control'.format(len(results)))
    else:
        results = []

    # Find the images to score; images can be a directory, may need to recurse
    if os.path.isdir(args.image_file):
        image_file_names = ImagePathUtils.find_images(args.image_file, args.recursive)
        print('{} archivos de imagen encontrados en el directorio de entrada'.format(len(image_file_names)))
    # A json list of image paths
    elif os.path.isfile(args.image_file) and args.image_file.endswith('.json'):
        with open(args.image_file) as f:
            image_file_names = json.load(f)
        print('{} archivos de imagen encontrados en la lista json'.format(len(image_file_names)))
    # A single image file
    elif os.path.isfile(args.image_file) and ImagePathUtils.is_image_file(args.image_file):
        image_file_names = [args.image_file]
        print('Una sola imagen en {} es el archivo de entrada'.format(args.image_file))
    else:
        raise ValueError('image_file especificado no es un directorio, una lista json o un archivo de imagen, '
                         '(o no tiene extensiones reconocibles).')

    assert len(image_file_names) > 0, 'El archivo de imagen especificado no apunta a archivos de imagen válidos'
    assert os.path.exists(image_file_names[0]), 'La primera imagen a puntuar no existe en {}'.format(image_file_names[0])

    output_dir = os.path.dirname(args.output_file)

    if len(output_dir) > 0:
        os.makedirs(output_dir,exist_ok=True)
        
    assert not os.path.isdir(args.output_file), 'El archivo de salida especificado es un directorio'

    # Test that we can write to the output_file's dir if checkpointing requested
    if args.checkpoint_frequency != -1:
        
        if args.checkpoint_path is not None:
            checkpoint_path = args.checkpoint_path
        else:
            checkpoint_path = os.path.join(output_dir, 'checkpoint_{}.json'.format(datetime.utcnow().strftime("%Y%m%d%H%M%S")))
        
        # Confirm that we can write to the checkpoint path, rather than failing after 10000 images
        with open(checkpoint_path, 'w') as f:
            json.dump({'images': []}, f)
        print('El archivo de punto de control se escribirá en {}'.format(checkpoint_path))
        
    else:
        
        checkpoint_path = None

    start_time = time.time()

    results = load_and_run_detector_batch(model_file=args.detector_file,
                                          image_file_names=image_file_names,
                                          checkpoint_path=checkpoint_path,
                                          confidence_threshold=args.threshold,
                                          checkpoint_frequency=args.checkpoint_frequency,
                                          results=results,
                                          n_cores=args.ncores,
                                          use_image_queue=args.use_image_queue)

    elapsed = time.time() - start_time
    print('Inferencia terminada en {}'.format(humanfriendly.format_timespan(elapsed)))

    relative_path_base = None
    if args.output_relative_filenames:
        relative_path_base = args.image_file
    write_results_to_file(results, args.output_file, relative_path_base=relative_path_base)

    if checkpoint_path:
        os.remove(checkpoint_path)
        print('Archivo de punto de control eliminado {}'.format(checkpoint_path))

    print('Hecho!')


if __name__ == '__main__':
    main()