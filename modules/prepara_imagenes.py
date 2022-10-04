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

def remove_borderV1(image):
    img_np = np.array(image)
    im_height, im_width = img_np.shape[0], img_np.shape[1]

    image.show()

    border_up = im_height
    for j in range(im_width):
        for i in range(im_height):
            if img_np[i][j][0] != 0 or img_np[i][j][1] != 0 or img_np[i][j][2] != 0:
                if i < border_up:
                    border_up = i
                    break

    edited = np.zeros((im_height - border_up, im_width, 3), dtype='uint8')
    print(edited.dtype)

    for i in range(im_height - border_up):
        for j in range(im_width):
            edited[i][j][0] = img_np[i + border_up][j][0]
            edited[i][j][1] = img_np[i + border_up][j][1]
            edited[i][j][2] = img_np[i + border_up][j][2]

    im = Image.fromarray(edited, mode='RGB')
    im.show()


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
            np_without_border = remove_border(input_file['detections'], np_image)
            elapsed_time = time.time() - start_time
            print('')
            print(
                'Generada máscara de imagen {} en {}.'.format(image_path, humanfriendly.format_timespan(elapsed_time)))
            time_infer.append(elapsed_time)
        except Exception as e:
            print('')
            print('Ha ocurrido un error mientras se generaba la máscara en la imagen {}. EXCEPTION: {}'
                  .format(image_path, e))
            print('------------------------------------------------------------------------------------------')
            print(traceback.format_exc())
            continue

        try:
            im = Image.fromarray(np_without_border, 'RGB')
            im.save(output_file)
        except Exception as e:
            print('')
            print('Ha ocurrido un error. No puede guardarse la máscara en la ruta {}. EXCEPTION: {}'.format(output_file,
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
    print('Ha tomado {} en generar la máscara, con desviación de {}'.format(
        humanfriendly.format_timespan(average_time_infer), std_dev_time_infer))
    print('==========================================================================================')


########################################################################################################################
# Command-line driver

def main():
    parser = argparse.ArgumentParser(
        description='Módulo para generar máscaras abarcando únicamente la zona de la detección')
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
        help='Ruta al directorio de donde se guardarán las máscaras como fichero de imagen'
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
    print('Generando máscaras de {} imágenes...'.format(len(input_file_names)))
    print('')

    run(input_file_names=input_file_names, output_masked=args.output_masked, output_edited=args.output_edited)

    print('')
    print('Resultados guardados en: {}'.format(args.output_edited))
    print('')
    print('==========================================================================================')


if __name__ == '__main__':
    main()
