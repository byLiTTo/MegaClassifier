"""
It takes a list of JSON files, each containing a list of detections, and renders the detections on the corresponding
image
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
import traceback

import humanfriendly
from tqdm import tqdm

import repos.CameraTraps.visualization.visualization_utils as v_utils
from path_utils import PathUtils
from repos.CameraTraps.data_management.annotations.annotation_constants import (
    detector_bbox_category_id_to_name)  # here id is int

########################################################################################################################

# convert category ID from int to str
DEFAULT_DETECTOR_LABEL_MAP = {
    str(k): v for k, v in detector_bbox_category_id_to_name.items()
}

CONFIDENCE: float = 0.7
OUTPUT_IMAGE_WIDTH: int = 700


########################################################################################################################
# FUNCTION PRINCIPAL

def run(input_file_names, output_dir):
    """
    It takes a list of JSON files, each containing a list of detections, and renders the detections on the corresponding
    image

    :param input_file_names: A list of paths to the JSON files that contain the detections
    :param output_dir: The directory where the output images will be saved
    :return: the average time it takes to render the detections of the images.
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
            print('Error al cargar el fichero JSON en la ruta {}, EXCEPTION: {}'.format(input_path, e))
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

        # image = v_utils.open_image(image_file)
        image = v_utils.resize_image(v_utils.open_image(image_file), OUTPUT_IMAGE_WIDTH)

        v_utils.render_detection_bounding_boxes(
            input_file['detections'], image, label_map=detector_label_map,
            confidence_threshold=CONFIDENCE)

        image.save(output_file)

        elapsed_time = time.time() - start_time
        print('')
        print('Renderizadas detecciones de imagen {} en {}.'.format(image_file,
                                                                    humanfriendly.format_timespan(elapsed_time)))
        time_infer.append(elapsed_time)

    average_time_infer = statistics.mean(time_infer)

    if len(time_infer) > 1:
        std_dev_time_infer = humanfriendly.format_timespan(statistics.stdev(time_infer))
    else:
        std_dev_time_infer = 'NO DISPONIBLE'

    print('')
    print('==========================================================================================')
    print('De media, por cada imagen: ')
    print('Ha tomado {} en renderizar las detecciones, con desviación de {}'.format(
        humanfriendly.format_timespan(average_time_infer), std_dev_time_infer))
    print('==========================================================================================')


########################################################################################################################
# Command-line driver

def main():
    """
    It takes a list of JSON files containing detection data, and renders the bounding boxes of the detections on the
    images
    """
    parser = argparse.ArgumentParser(
        description='Módulo para renderizar los bounding boxes de las detecciones de las imágenes indicadas')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--input_file',
        help='Fichero JSON del que se tomarán los datos para realizar el recorte'
    )
    group.add_argument(
        '--input_dir',
        help='Ruta al directorio donde se encuentran los ficheros JSON de los cuales se tomarán los datos para realizar '
             'los recortes a las diferentes imágenes, hace uso de la opción --recursive'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Maneja directorios de forma recursiva, solo tiene sentido usarlo con --input_file'
    )
    parser.add_argument(
        '--output_dir',
        help='Ruta al directorio de donde se guardaran las imágenes con los bounding boxes renderizados'
    )

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    if args.input_file:
        input_file_names = [args.input_file]
    else:
        input_file_names = PathUtils.find_detections(args.input_dir, args.recursive)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        if args.input_dir:
            args.output_dir = args.input_dir
        else:
            args.output_dir = os.path.dirname(args.input_file)

    print('')
    print('==========================================================================================')
    print('Renderizando detecciones de {} imágenes...'.format(len(input_file_names)))
    print('')

    run(input_file_names=input_file_names, output_dir=args.output_dir)

    print('')
    print('Resultados guardados en: {}'.format(args.output_dir))
    print('')
    print('==========================================================================================')


if __name__ == '__main__':
    main()
