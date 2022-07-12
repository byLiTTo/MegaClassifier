"""

"""
###################################################################################################
# IMPORTs

import os
import sys
import cv2
import json
import numpy as np
import platform
import argparse



from PIL import Image
from tqdm import tqdm
from pathutils import PathUtils as p_utils



###################################################################################################
# FUNCIONES

def generate_masked_image(image_path, mask_path):
    """
    Crea una imagen a la que se le ha aplicado una máscara. En este caso el resultado será una 
    imagen con el fondo en negro salvo la zona que abarca la detección.

    Args:
        - image_path: Ruta del fichero de la imagen a la que aplicar la máscara.
        - mask_path: Ruta donde se encuentra el fichero de la máscara a aplicar.
        
    Returns:
        - masked: Imagen con la máscara aplicada.
    """
    image = cv2.imread(image_path)
    bin_mask = np.array(Image.open(mask_path))

    masked = cv2.bitwise_and(image, image, mask=bin_mask)

    return masked



###################################################################################################
# FUNCIÓN PRINCIPAL

def run(input_file_names, output_mask, output_masked):
    """
    
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
        with open(input_path) as f:
            input_file = json.load(f)

        image_file = input_file['file']
        name, ext = os.path.splitext(os.path.basename(image_file).lower())
        if windows:
            mask_path = (output_mask + '\\' + name + '_mask.png')
            output_path = (output_masked + '\\' + name + '.png')
        else:
            mask_path = (output_mask + '/' + name + '_mask.png')
            output_path = (output_masked + '/' + name + '.png')
        masked = generate_masked_image(image_file, mask_path)
        
        cv2.imwrite(output_path, masked)




###################################################################################################
# Command-line driver

def main():
    parser = argparse.ArgumentParser(
        description = 'Módulo para renderizar los bounding boxes de las detecciones de las imágenes '
            'indicadas')
    group = parser.add_mutually_exclusive_group(required = True)
    group.add_argument(
        '--json_file',
        help = 'Fichero JSON del que se tomarán los datos para realizar el recorte'
    )
    group.add_argument(
        '--json_dir',
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
        '--output_masked',
        help = 'Ruta al directorio de donde se guardaran las imágenes con los bounding boxes '
            'renderizados'
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

    if args.output_masked:
        os.makedirs(args.output_masked, exist_ok=True)
    else:
        if args.json_dir:
            args.output_masked = args.json_dir
        else:
            args.output_masked = os.path.dirname(args.json_file)
    
    run(input_file_names=input_file_names, output_mask=args.output_mask, output_masked=args.output_masked)
    print('==========================================================================================')



if __name__ == '__main__':
    main()    



###################################################################################################    