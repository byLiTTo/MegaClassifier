"""
Módulo que genera imágenes binarias que podrán ser usadas como máscaras.

A partir de un fichero JSON con las coordenadas de los bboxes de las detecciones de una imagen de
origen, crea una máscara con fondo negro y pixeles con valor True (ó 255) que se encuentran en la
región que abarcan los bboxes.
"""
###################################################################################################
# IMPORTs

import os
import sys
import cv2
import time
import json
import numpy as np
import platform
import argparse
import traceback
import statistics
import humanfriendly
import visualization.visualization_utils as v_utils

from PIL import Image
from tqdm import tqdm
from path_utils import PathUtils as p_utils



###################################################################################################
# FUNCIONES

def generate_binary_image(detections, image):
    """
    Genera una imagen binaria donde las posiciones de los pixeles que se encuentran dentro de las 
    detecciones, tiene valor True.

    Args:
        - detections: Lista de detecciones  de la imagen en cuestión.
        - image: np.array del objeto imagen
        
    Returns:
        - mask: np.array correspondiente a la imagen binaria como resultado (máscara)
    """
    im_height, im_width = image.shape[0], image.shape[1] 

    mask = np.zeros((im_height, im_width))

    for detection in detections:
        x1, y1, w_box, h_box = detection['bbox']
        ymin,xmin,ymax,xmax = y1, x1, y1 + h_box, x1 + w_box

        # Convert to pixels so we can use the PIL crop() function
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                        ymin * im_height, ymax * im_height)

        left = round(left)
        right = round(right)
        top = round(top)
        bottom = round(bottom)
        
        # Representación: img[y0:y1, x0:x1]
        mask[top:bottom,left:right] = 1

    return mask



###################################################################################################
# FUNCIÓN PRINCIPAL

def run(input_file_names, output_mask):
    """
    Carga el fichero JSON especificado, a partir de las detecciones, genera una imagen binaria 
    (máscara) y la guarda en la ruta especificada.

    Args:
        - input_file_names: Lista de las rutas de los ficheros JSON de los que se tomará la 
            información para generar las máscaras.
        - output_mask: Ruta donde se guardarán las máscaras generadas.
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
        try:
            image_file = input_file['file']
            
            image = np.array(Image.open(image_file)) 
        except Exception as e:
            print('Error al cargar imagen desde la ruta {}, EXCEPTION: {}'
                .format(image_file, e))
            print('------------------------------------------------------------------------------------------')
            print(traceback.format_exc())
            continue
        try:
            print('')
            start_time = time.time()
            mask = generate_binary_image(input_file['detections'],image)
            elapsed_time = time.time() - start_time
            print('')
            print('Generada máscara de imagen {} en {}.'
                .format(image_file, humanfriendly.format_timespan(elapsed_time)))
            time_infer.append(elapsed_time)
        except Exception as e:
            print('')
            print('Ha ocurrido un error mientras se generaba la máscara en la imagen {}. EXCEPTION: {}'
                .format(image_file, e))
            print('------------------------------------------------------------------------------------------')
            print(traceback.format_exc())
            continue

        try:
            name, ext = os.path.splitext(os.path.basename(image_file).lower())
            if windows:
                output_file = (output_mask + '\\' + name + '_mask.png')
            else:
                output_file = (output_mask + '/' + name + '_mask.png')
            
            cv2.imwrite(output_file, mask * 255)

        except Exception as e:
            print('')
            print('Ha ocurrido un error. No puede guardarse la máscara en la ruta {}. EXCEPTION: {}'
                .format(output_file, e))
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
    print('Ha tomado {} en generar la máscara, con desviación de {}'
        .format(humanfriendly.format_timespan(average_time_infer), std_dev_time_infer))
    print('==========================================================================================')



###################################################################################################
# Command-line driver

def main():
    parser = argparse.ArgumentParser(
        description = 'Módulo para generar máscaras abarcando únicamente la zona de la detección')
    group = parser.add_mutually_exclusive_group(required = True)
    group.add_argument(
        '--json_file',
        help = 'Fichero JSON del que se tomarán los datos para delimitar la zona de la máscara'
    )
    group.add_argument(
        '--json_dir',
        help = 'Ruta al directorio donde se encuentran los ficheros JSON de los cuales se tomarán '
            'los datos para generar las diferentes máscaras, hace uso de la opción --recursive'
    )
    parser.add_argument(
        '--recursive',
        action = 'store_true',
        help = 'Maneja directorios de forma recursiva, solo tiene sentido usarlo con --json_dir'
    )
    parser.add_argument(
        '--output_mask',
        help = 'Ruta al directorio de donde se guardarán las máscaras como fichero de imagen'
    )

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    if args.json_file:
        input_file_names = [args.json_file]
    else:
        input_file_names = p_utils.find_detections(args.json_dir, args.recursive)

    if args.output_mask:
        os.makedirs(args.output_mask, exist_ok=True)
    else:
        if args.json_dir:
            args.output_mask = args.json_dir
        else:
            args.output_mask = os.path.dirname(args.json_file)

    print('')
    print('==========================================================================================')
    print('Generando máscaras de {} imágenes...'
        .format(len(input_file_names)))
    print('')
    
    run(input_file_names=input_file_names, output_mask=args.output_mask)

    print('')
    print('Resultados guardados en: {}'.format(args.output_mask))
    print('')
    print('==========================================================================================')



if __name__ == '__main__':
    main()    



###################################################################################################