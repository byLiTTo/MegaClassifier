"""

"""
###################################################################################################
# IMPORTs

import os
import sys
import cv2
import json
import numpy as np
import scipy.io as sio
import platform
import argparse



from PIL import Image
from tqdm import tqdm
from pathutils import PathUtils as p_utils
from matplotlib import pyplot as plt



###################################################################################################
# FUNCIONES

def genera_imagen(image_path, mask_path, output_path):
    image_np = np.array(Image.open(image_path))
    bin_mask = np.array(Image.open(mask_path))

    masked = cv2.bitwise_and(image_np, image_np, mask=bin_mask)

    cv2.imwrite(output_path, masked)



###################################################################################################
# FUNCIÓN PRINCIPAL

def run(input_file_names, output_mask, output_dir):
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
            mask_path = (output_mask + '\\' + name + '_mask.png')
            output_path = (output_dir + '\\' + name + '.png')
        else:
            mask_path = (output_mask + '/' + name + '_mask.png')
            output_path = (output_dir + '/' + name + '.png')
        genera_imagen(image_file,mask_path, output_path)




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
        '--output_mask',
        help = 'Ruta al directorio de donde se guardaran las imágenes con los bounding boxes '
            'renderizados'
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

    if args.output_mask:
        os.makedirs(args.output_mask, exist_ok=True)
    else:
        if args.input_dir:
            args.output_mask = args.input_dir
        else:
            args.output_mask = os.path.dirname(args.input_file)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    else:
        if args.input_dir:
            args.output_dir = args.input_dir
        else:
            args.output_dir = os.path.dirname(args.input_file)
    
    run(input_file_names=input_file_names, output_mask=args.output_mask, output_dir=args.output_dir)
    print('==========================================================================================')



if __name__ == '__main__':
    main()    



###################################################################################################    