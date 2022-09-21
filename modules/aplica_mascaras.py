"""
It takes a list of JSON files, applies the masks to the images
and saves the masked images in the output directory
"""

import argparse
import json
import os
import platform
import statistics
import sys
import time
import traceback

import cv2
import humanfriendly
import numpy as np
from PIL import Image
from tqdm import tqdm

from path_utils import PathUtils


########################################################################################################################
# FUNCIONES

def generate_masked_image(image_path, mask_path):
    """
    It takes an image and a mask, and returns the masked image

    :param image_path: The path to the image you want to mask
    :param mask_path: path to the mask image
    :return: A masked image.
    """
    try:
        image = cv2.imread(image_path)
    except Exception as e:
        print('')
        print('Ha ocurrido un error. No se puede cargar la imagen {}. EXCEPTION: {}'.format(image_path, e))
        print('------------------------------------------------------------------------------------------')
        print(traceback.format_exc())
        return
    try:
        bin_mask = np.array(Image.open(mask_path))
    except Exception as e:
        print('')
        print('Ha ocurrido un error. No se puede cargar la máscara {}. EXCEPTION: {}'.format(mask_path, e))
        print('------------------------------------------------------------------------------------------')
        print(traceback.format_exc())
        return

    masked = cv2.bitwise_and(image, image, mask=bin_mask)

    return masked


########################################################################################################################
# FUNCIÓN PRINCIPAL

def run(input_file_names, output_mask, output_masked):
    """
    It takes a list of input files, and for each one, it loads the image, applies the mask, and saves the masked image

    :param input_file_names: A list of paths to the JSON files that contain the image paths
    :param output_mask: The path to the folder where the masks will be saved
    :param output_masked: The directory where the masked images will be saved
    :return: the masked image.
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
            print('')
            print('Ha ocurrido un error. No se ha podido abrir el fichero {}. EXCEPTION: {}'.format(input_path, e))
            print('------------------------------------------------------------------------------------------')
            print(traceback.format_exc())
            continue

        image_file = input_file['file']

        name, ext = os.path.splitext(os.path.basename(image_file).lower())
        if windows:
            mask_path = (output_mask + '\\' + name + '_mask.png')
            output_path = (output_masked + '\\' + name + '.png')
        else:
            mask_path = (output_mask + '/' + name + '_mask.png')
            output_path = (output_masked + '/' + name + '.png')

        print('')
        start_time = time.time()
        masked = generate_masked_image(image_file, mask_path)
        elapsed_time = time.time() - start_time
        print('')
        print('Aplicada máscara en imagen {} en {}.'.format(image_file, humanfriendly.format_timespan(elapsed_time)))
        time_infer.append(elapsed_time)

        cv2.imwrite(output_path, masked)

    average_time_infer = statistics.mean(time_infer)

    if len(time_infer) > 1:
        std_dev_time_infer = humanfriendly.format_timespan(statistics.stdev(time_infer))
    else:
        std_dev_time_infer = 'NO DISPONIBLE'

    print('')
    print('==========================================================================================')
    print('De media, por cada imagen: ')
    print('Ha tomado {} en aplicar la máscara, con desviación de {}'.format(
        humanfriendly.format_timespan(average_time_infer), std_dev_time_infer))
    print('==========================================================================================')


########################################################################################################################
# Command-line driver

def main():
    """
    It takes a list of JSON files, applies the masks to the images
    and saves the masked images in the output directory
    """
    parser = argparse.ArgumentParser(
        description='Módulo para aplicar las máscaras anteriormente generadas a las imágenes indicadas'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--json_file',
        help='Fichero JSON del que se tomará la ruta de la imagen original a la que aplicar la máscara'
    )
    group.add_argument(
        '--json_dir',
        help='Ruta al directorio donde se encuentran los ficheros JSON de los cuales se tomarán las rutas de '
             'las imágenes a las que aplicar sus correspondientes máscaras, hace uso de la opción --recursive'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Maneja directorios de forma recursiva, solo tiene sentido usarlo con --json_file'
    )
    parser.add_argument(
        '--output_mask',
        help='Ruta al directorio de donde se encuentran los ficheros de las máscaras '
    )
    parser.add_argument(
        '--output_masked',
        help='Ruta al directorio de donde se guardaran las imágenes una vez aplicadas sus máscaras'
             'correspondientes'
    )

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    if args.json_file:
        input_file_names = [args.json_file]
    else:
        input_file_names = PathUtils.find_detections(args.json_dir, args.recursive)

    if args.output_mask:
        os.makedirs(args.output_mask, exist_ok=True)
    else:
        if args.json_dir:
            args.output_mask = args.json_dir
        else:
            args.output_mask = os.path.dirname(args.json_file)

    if args.output_masked:
        os.makedirs(args.output_masked, exist_ok=True)
    else:
        if args.json_dir:
            args.output_masked = args.json_dir
        else:
            args.output_masked = os.path.dirname(args.json_file)

    print('')
    print('==========================================================================================')
    print('Aplicando máscaras en {} imágenes...'.format(len(input_file_names)))
    print('')

    run(input_file_names=input_file_names, output_mask=args.output_mask,
        output_masked=args.output_masked)

    print('')
    print('Resultados guardados en: {}'.format(args.output_masked))
    print('')
    print('==========================================================================================')


if __name__ == '__main__':
    main()
