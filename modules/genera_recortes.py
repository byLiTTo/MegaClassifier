"""

"""
###################################################################################################
# IMPORTs

import os
import sys
import time
import json
import argparse
import visualization.visualization_utils as v_utils

from tqdm import tqdm
from pathutils import PathUtils as p_utils




###################################################################################################
# FUNCION PRINCIPAL

def run(input_file_names, output_dir):
    """
    
    """
    if len(input_file_names) == 0:
        print("WARNING: No hay ficheros disponibles")
        return

    time_load = []
    time_infer = []

    output_filename_collision_counts = {}

    def generate_crop(fn, crop_index=-1):
        fn = os.path.basename(fn).lower()
        name, ext = os.path.splitext(fn)
        if crop_index >= 0:
            name += '_crop{:0>2d}'.format(crop_index)
        fn = '{}_{}'.format(name, '.png')
        if fn in output_filename_collision_counts:
            n_collisions = output_filename_collision_counts[fn]
            fn = '{:0>4d}'.format(n_collisions) + '_' + fn
            output_filename_collision_counts[fn] += 1
        else:
            output_filename_collision_counts[fn] = 0
        fn = os.path.join(output_dir, fn)
        return fn


    for input_path in tqdm(input_file_names):

        with open(input_path) as f:
            input_file = json.load(f)

        image_file = input_file['file']

        try:
            start_time = time.time()

            image_obj = v_utils.load_image(image_file)

            elapsed = time.time() - start_time
            time_load.append(elapsed)

        except Exception as e:
            print('Image {} cannot be loaded. Exception: {}'.format(image_file, e))
            continue

        images_cropped = v_utils.crop_image(input_file['detections'], image_obj)

        for i_crop, cropped_image in enumerate(images_cropped):
            output_full_path = generate_crop(image_file, i_crop)
            cropped_image.save(output_full_path)
        



###################################################################################################
# Command-line driver

def main():
    parser = argparse.ArgumentParser(
        description = 'Módulo para generar recortes de las detecciones de las imágenes indicadas')
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
        help = 'Ruta al directorio de donde se guardaran los recortes generados'
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