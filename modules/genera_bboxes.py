"""

"""

###################################################################################################
# IMPORTs

import os
import sys
import time
import json
import platform
import argparse
import visualization.visualization_utils as v_utils

from tqdm import tqdm
from pathutils import PathUtils as p_utils
from data_management.annotations.annotation_constants import (
    detector_bbox_category_id_to_name)  # here id is int



###################################################################################################

# convert category ID from int to str
DEFAULT_DETECTOR_LABEL_MAP = {
    str(k): v for k, v in detector_bbox_category_id_to_name.items()
}

CONFIDENCE: float = 0.8
OUTPUT_IMAGE_WIDTH: int = 700



###################################################################################################
# FUNCION PRINCIPAL

def run(input_file_names, output_dir):
    """
    
    """
    if platform.system() == 'Windows':
        windows = True
    else:
        windows = False
    if len(input_file_names) == 0:
        print("WARNING: No hay ficheros disponibles")
        return

    time_load = []
    time_infer = []

    for input_path in tqdm(input_file_names):
        with open(input_path) as f:
            input_file = json.load(f)

        image_file = input_file['file']

        name, ext = os.path.splitext(os.path.basename(image_file).lower())
        if windows:
            output_file = (output_dir + '\\' + name + '.jpg')
        else:
            output_file = (output_dir + '/' + name + '.jpg')

        detector_label_map = DEFAULT_DETECTOR_LABEL_MAP
        if 'detection_categories' in input_file:
            print('detection_categories provided')
            detector_label_map = input_file['detection_categories']
        
        image = v_utils.resize_image(v_utils.open_image(image_file), OUTPUT_IMAGE_WIDTH)

        v_utils.render_detection_bounding_boxes(
            input_file['detections'], image, label_map=detector_label_map,
            confidence_threshold=CONFIDENCE)

        image.save(output_file)




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
    
    run(input_file_names=input_file_names, output_dir=args.output_dir)
    print('==========================================================================================')



if __name__ == '__main__':
    main()    



###################################################################################################