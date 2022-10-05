import argparse
import json
import os
import platform
import statistics
import sys
import time
import traceback

import humanfriendly
import numpy as np
from PIL import Image
from tqdm import tqdm

from path_utils import PathUtils


########################################################################################################################
# FUNCIONES


def remove_border(detections, np_image):
    height, width = np_image.shape[0], np_image.shape[1]

    if not detections:
        if height <= width:
            np_empty = np.zeros((height, height, 3), dtype='uint8')
            return np_empty
        else:
            np_empty = np.zeros((width, width, 3), dtype='uint8')
            return np_empty

    tops = []
    bottoms = []
    lefts = []
    rights = []

    for detection in detections:
        x1, y1, w_box, h_box = detection['bbox']
        y_min, x_min, y_max, x_max = y1, x1, y1 + h_box, x1 + w_box

        # Convert to pixels, so we can use the PIL crop() function
        (left, right, top, bottom) = (x_min * width, x_max * width, y_min * height, y_max * height)

        lefts.append(round(left))
        rights.append(round(right))

        tops.append(round(top))
        bottoms.append(round(bottom))

    lefts.sort()
    rights.sort(reverse=True)
    tops.sort()
    bottoms.sort(reverse=True)

    left = lefts[0]
    right = rights[0]
    top = tops[0]
    bottom = bottoms[0]

    new_height = bottom - top
    new_width = right - left

    np_cropped = np.zeros((new_height, new_width, 3), dtype='uint8')

    i_old = top
    i_new = 0

    j_old = left
    j_new = 0

    while i_old < bottom:
        while j_old < right:
            np_cropped[i_new][j_new] = np_image[i_old][j_old]
            j_old = j_old + 1
            j_new = j_new + 1
        i_old = i_old + 1
        i_new = i_new + 1

        j_old = lefts[0]
        j_new = 0

    return np_cropped


def make_squared(np_image):
    height, width = np_image.shape[0], np_image.shape[1]
    im = Image.fromarray(np_image, 'RGB')
    desired_size = 500
    if height < width:
        desired_size = width
    if height > width:
        desired_size = width
    if height == width:
        return im

    old_size = im.size  # old_size[0] is in (width, height) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    # use thumbnail() or resize() method to resize the input image

    # thumbnail is an in-place operation

    # im.thumbnail(new_size, Image.ANTIALIAS)

    im = im.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it

    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2,
                      (desired_size - new_size[1]) // 2))

    return new_im


########################################################################################################################
# FUNCIÓN PRINCIPAL

def run(input_file_names, output_masked, output_edited):
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
        try:
            image_file = input_file['file']
            name, ext = os.path.splitext(os.path.basename(image_file).lower())
            if windows:
                image_path = (output_masked + '\\' + name + '.png')
                output_file = (output_edited + '\\' + name + '.png')
            else:
                image_path = (output_masked + '/' + name + '.png')
                output_file = (output_edited + '/' + name + '.png')
            np_image = np.asarray(Image.open(image_path), dtype='uint8')
        except Exception as e:
            print('Error al cargar imagen desde la ruta {}, EXCEPTION: {}'.format(image_path, e))
            print('------------------------------------------------------------------------------------------')
            print(traceback.format_exc())
            continue
        try:
            print('')
            start_time = time.time()
            np_crop = remove_border(input_file['detections'], np_image)
            im = make_squared(np_crop)
            elapsed_time = time.time() - start_time
            print('')
            print(
                'Ajustada imagen {} en {}.'.format(image_path, humanfriendly.format_timespan(elapsed_time)))
            time_infer.append(elapsed_time)
        except Exception as e:
            print('')
            print('Ha ocurrido un error mientras se ajustaba la imagen {}. EXCEPTION: {}'
                  .format(image_path, e))
            print('------------------------------------------------------------------------------------------')
            print(traceback.format_exc())
            continue

        try:
            im.save(output_file)
        except Exception as e:
            print('')
            print('Ha ocurrido un error. No puede guardarse la imagen en la ruta {}. EXCEPTION: {}'.format(output_file,
                                                                                                           e))
            print('------------------------------------------------------------------------------------------')
            print(traceback.format_exc())
            continue

    average_time_infer = statistics.mean(time_infer)

    if len(time_infer) > 1:
        std_dev_time_infer = humanfriendly.format_timespan(statistics.stdev(time_infer))
    else:
        std_dev_time_infer = 'NO DISPONIBLE'

    print('')
    print('==========================================================================================')
    print('De media, por cada imagen: ')
    print('Ha tomado {} en ajustar la imagen, con desviación de {}'.format(
        humanfriendly.format_timespan(average_time_infer), std_dev_time_infer))
    print('==========================================================================================')


########################################################################################################################
# Command-line driver

def main():
    parser = argparse.ArgumentParser(
        description='Módulo para ajustar las imágenes enmascaradas convirtiéndolas en cuadradas')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--json_file',
        help='Fichero JSON del que se tomarán los datos para delimitar la zona de la máscara'
    )
    group.add_argument(
        '--json_dir',
        help='Ruta al directorio donde se encuentran los ficheros JSON de los cuales se tomarán los datos para generar '
             'las diferentes máscaras, hace uso de la opción --recursive'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Maneja directorios de forma recursiva, solo tiene sentido usarlo con --json_dir'
    )
    parser.add_argument(
        '--output_masked',
        help='Ruta al directorio de donde se encuentran guardadas las imágenes enmascaradas'
    )
    parser.add_argument(
        '--output_edited',
        help='Ruta al directorio de donde se guardarán las imágenes ajustadas'
    )

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    if args.json_file:
        input_file_names = [args.json_file]
    else:
        input_file_names = PathUtils.find_detections(args.json_dir, args.recursive)

    if args.output_masked:
        os.makedirs(args.output_masked, exist_ok=True)
    else:
        if args.json_dir:
            args.output_masked = args.json_dir
        else:
            args.output_masked = os.path.dirname(args.json_file)

    if args.output_edited:
        os.makedirs(args.output_edited, exist_ok=True)
    else:
        if args.json_dir:
            args.output_edited = args.json_dir
        else:
            args.output_edited = os.path.dirname(args.json_file)

    print('')
    print('==========================================================================================')
    print('Ajustando {} imágenes...'.format(len(input_file_names)))
    print('')

    run(input_file_names=input_file_names, output_masked=args.output_masked, output_edited=args.output_edited)

    print('')
    print('Resultados guardados en: {}'.format(args.output_edited))
    print('')
    print('==========================================================================================')


if __name__ == '__main__':
    main()
