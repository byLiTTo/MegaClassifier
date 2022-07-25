"""
Módulo que genera imágenes con los bboxes de las detecciones dibujados sobre ella. Además también
indicará a qué clase pertenece la detección. Estas clases son las que contiene el modelo 
de MegaDetector. En este proyecto usamos las nuestras propias, pero nos parece interesante mostrar
esta información de cara a futuras comparaciones.

Hacemos uso de funciones de módulos de los repositorios CameraTraps.visualization.visualization_utils

Para mostrar las detecciones usamos un umbral distinto al umbral de confianza para considerar 
deteccones. Para diferenciarlo lo hemos denominada umbral de renderizado, por defecto posee valor 
0.7, si se desea usar otro valor, habría que modificar la variable CONFIDENCE.

Para mayor fluidez a la hora de renderizar, hacemos un redimensionado de las imágenes resultantes,
por defecto redimensinamos con ancho 700px, si se desea otra dimensión, habría que modificar la 
variable OUTPUT_IMAGE_WIDTH.

En el caso de no querer hacer redimensionado y mantener la escala original de cada imagen, habría 
que descomentar la línea: 
    image = v_utils.open_image(image_file)
y comentar la línea:
    image = v_utils.resize_image(v_utils.open_image(image_file), OUTPUT_IMAGE_WIDTH)  
"""

###################################################################################################
# IMPORTs

import os
import sys
import time
import json
import platform
import argparse
import traceback
import statistics
import humanfriendly
import visualization.visualization_utils as v_utils

from tqdm import tqdm
from path_utils import PathUtils as p_utils
from data_management.annotations.annotation_constants import (
    detector_bbox_category_id_to_name)  # here id is int



###################################################################################################

# convert category ID from int to str
DEFAULT_DETECTOR_LABEL_MAP = {
    str(k): v for k, v in detector_bbox_category_id_to_name.items()
}

CONFIDENCE: float = 0.7
OUTPUT_IMAGE_WIDTH: int = 700



###################################################################################################
# FUNCION PRINCIPAL

def run(input_file_names, output_dir):
    """
    Carga las detecciones y la imagen original desde un fichero JSON y genera tanto recortes como
    detecciones haya en cada imagen. Estos recortes corresponden a los diferentes bboxes 
    encontrados.

    Args:
        - input_file_names: Lista de fichros JSON de los que se tomarán los datos de detecciones
            y de imagen original.
        - output_dir: Ruta al directorio donde se guardarán las imágenes recortadas resultantes.
    """
    if platform.system() == 'Windows':
        windows = True
    else:
        windows = False
        
    if len(input_file_names) == 0:
        print("WARNING: No hay ficheros disponibles")
        return

    time_infer = []

    for input_path in tqdm(input_file_names):
        try:
            with open(input_path) as f:
                input_file = json.load(f)
        except Exception as e:
            print('Error al cargar el fichero JSON en la ruta {}, EXCEPTION: {}'
                .format(input_path, e))
            print('------------------------------------------------------------------------------------------')
            print(traceback.format_exc())
            continue

        image_file = input_file['file']

        name, ext = os.path.splitext(os.path.basename(image_file).lower())
        if windows:
            output_file = (output_dir + '\\' + name + '.png')
        else:
            output_file = (output_dir + '/' + name + '.png')

        detector_label_map = DEFAULT_DETECTOR_LABEL_MAP
        if 'detection_categories' in input_file:
            print('detection_categories provided')
            detector_label_map = input_file['detection_categories']

        print('')
        start_time = time.time()
        
        #image = v_utils.open_image(image_file)
        image = v_utils.resize_image(v_utils.open_image(image_file), OUTPUT_IMAGE_WIDTH)

        v_utils.render_detection_bounding_boxes(
            input_file['detections'], image, label_map=detector_label_map,
            confidence_threshold=CONFIDENCE)

        image.save(output_file)

        elapsed_time = time.time() - start_time
        print('')
        print('Renderizadas detecciones de imagen {} en {}.'
            .format(image_file, humanfriendly.format_timespan(elapsed_time)))
        time_infer.append(elapsed_time)

    average_time_infer = statistics.mean(time_infer)

    if len(time_infer) > 1:
        std_dev_time_infer = humanfriendly.format_timespan(statistics.stdev(time_infer))
    else:
        std_dev_time_infer = 'NO DISPONIBLE'

    print('')
    print('==========================================================================================')
    print('De media, por cada imagen: ')
    print('Ha tomado {} en renderizar las detecciones, con desviación de {}'
        .format(humanfriendly.format_timespan(average_time_infer), std_dev_time_infer))
    print('==========================================================================================')




###################################################################################################
# Command-line driver

def main():
    parser = argparse.ArgumentParser(
        description = 'Módulo para renderizar los bounding boxes de las detecciones de las imágenes '
            'indicadas')
    group = parser.add_mutually_exclusive_group(required = True)
    group.add_argument(
        '--input_file',
        help = 'Fichero JSON del que se tomarán los datos para realizar el recorte'
    )
    group.add_argument(
        '--input_dir',
        help = 'Ruta al directorio donde se encuentran los ficheros JSON de los cuales se tomarán '
            'los datos para realizar los recortes a las diferentes imágenes, hace uso de la '
            'opción --recursive'
    )
    parser.add_argument(
        '--recursive',
        action = 'store_true',
        help = 'Maneja directorios de forma recursiva, solo tiene sentido usarlo con --input_file'
    )
    parser.add_argument(
        '--output_dir',
        help = 'Ruta al directorio de donde se guardaran las imágenes con los bounding boxes '
            'renderizados'
    )

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    if args.input_file:
        input_file_names = [args.input_file]
    else:
        input_file_names = p_utils.find_detections(args.input_dir, args.recursive)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        if args.input_dir:
            args.output_dir = args.input_dir
        else:
            args.output_dir = os.path.dirname(args.input_file)

    print('')
    print('==========================================================================================')
    print('Renderizando detecciones de {} imágenes...'
        .format(len(input_file_names)))
    print('')
    
    run(input_file_names=input_file_names, output_dir=args.output_dir)

    print('')
    print('Resultados guardados en: {}'.format(args.output_dir))
    print('')
    print('==========================================================================================')



if __name__ == '__main__':
    main()    



###################################################################################################