"""
It takes a list of JSON files, loads the images, crops them, and saves the crops to disk
"""

import argparse
import json
import os
import statistics
import sys
import time
import traceback

import humanfriendly
from tqdm import tqdm

import repos.CameraTraps.visualization.visualization_utils as v_utils
from path_utils import PathUtils


########################################################################################################################
# FUNCTION PRINCIPAL

def run(input_file_names, output_dir):
    """
    It takes a list of JSON files, loads the images, crops them, and saves the crops to disk
    :param input_file_names: A list of paths to the JSON files that contain the detections
    :param output_dir: The directory where the cropped images will be saved
    :return: the average time it takes to load the image and the average time it takes to process the image.
    """
    if len(input_file_names) == 0:
        print("WARNING: No hay ficheros disponibles")
        return

    time_load = []
    time_infer = []

    output_filename_collision_counts = {}

    def generate_crop(fn, crop_index=-1):
        """
        It takes a filename, and returns a filename that is guaranteed to be unique
        :param fn: the filename of the image to be cropped
        :param crop_index: the index of the crop to generate. If -1, then generate all crops
        :return: the file name of the image.
        """
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

        try:
            start_time = time.time()

            image_obj = v_utils.load_image(image_file)

            elapsed = time.time() - start_time
            time_load.append(elapsed)

        except Exception as e:
            print('La imagen {} no ha podido ser cargada. EXCEPTION: {}'.format(image_file, e))
            continue

        start_time = time.time()

        images_cropped = v_utils.crop_image(input_file['detections'], image_obj)

        for i_crop, cropped_image in enumerate(images_cropped):
            output_full_path = generate_crop(image_file, i_crop)
            cropped_image.save(output_full_path)

        elapsed = time.time() - start_time
        print('')
        print('Generados recortes de imagen {} en {}.'
              .format(image_file, humanfriendly.format_timespan(elapsed)))
        time_infer.append(elapsed)

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
        description='Módulo para generar recortes de las detecciones de las imágenes indicadas')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--input_file',
        help='Fichero JSON del que se tomarán los datos para realizar el recorte'
    )
    group.add_argument(
        '--input_dir',
        help='Ruta al directorio donde se encuentran los ficheros JSON de los cuales se tomarán '
             'los datos para realizar los recortes a las diferentes imágenes, hace uso de la '
             'opción --recursive'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Maneja directorios de forma recursiva, solo tiene sentido usarlo con --input_dir'
    )
    parser.add_argument(
        '--output_dir',
        help='Ruta al directorio de donde se guardaran los recortes generados'
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
    print('Generando recortes de {} imágenes...'.format(len(input_file_names)))
    print('')

    run(input_file_names=input_file_names, output_dir=args.output_dir)

    print('')
    print('Resultados guardados en: {}'.format(args.output_dir))
    print('')
    print('==========================================================================================')


if __name__ == '__main__':
    main()
