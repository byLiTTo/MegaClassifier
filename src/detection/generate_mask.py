import argparse
import json
import os
import statistics
import sys
from time import time

import cv2
import humanfriendly
import numpy as np
from PIL import Image
from tqdm import tqdm
import traceback


# ----------------------------------------------------------------------------------------------------------------------
def generate_binary_image(detections, image):
    im_height, im_width = image.shape[0], image.shape[1]

    mask = np.zeros((im_height, im_width))

    for detection in detections:
        x1, y1, w_box, h_box = detection["bbox"]
        y_min, x_min, y_max, x_max = y1, x1, y1 + h_box, x1 + w_box

        # Convert to pixels, so we can use the PIL crop() function
        (left, right, top, bottom) = (
            x_min * im_width,
            x_max * im_width,
            y_min * im_height,
            y_max * im_height,
        )

        left = round(left)
        right = round(right)
        top = round(top)
        bottom = round(bottom)

        # Representation: img[y0:y1, x0:x1]
        mask[top:bottom, left:right] = 1

    return mask


# ----------------------------------------------------------------------------------------------------------------------
def generate_binary_imageV2(detections, image):
    im_height, im_width = image.shape[:2]

    mask = np.zeros((im_height, im_width))

    for detection in detections:
        x1, y1, w_box, h_box = detection["bbox"]
        y_min, x_min, y_max, x_max = y1, x1, y1 + h_box, x1 + w_box

        # Convert to pixels, so we can use the PIL crop() function
        left, right, top, bottom = (
            int(x_min * im_width),
            int(x_max * im_width),
            int(y_min * im_height),
            int(y_max * im_height),
        )

        # Representation: img[y0:y1, x0:x1]
        mask[top:bottom, left:right] = 1

    return mask


# ----------------------------------------------------------------------------------------------------------------------
def run(file_names, output_path):
    assert len(file_names) > 0, "No input files provided"

    time_infer = []

    for sample in tqdm(file_names):
        image_path = sample["file"]
        image = np.array(Image.open(image_path))

        start_time = time()
        mask = generate_binary_imageV2(sample["detections"], image)
        elapsed_time = time() - start_time

        time_infer.append(elapsed_time)

        pos = 0
        for i in range(len(image_path.split("/"))):
            if image_path.split("/")[i] == "emptyNonEmptyDataset":
                pos = i
                break

        new_path = output_path
        count = pos
        while count < len(image_path.split("/"))-1:
            new_path = os.path.join(new_path, image_path.split("/")[count])
            count += 1
        
        os.makedirs(new_path, exist_ok=True)
        name = os.path.basename(image_path).split(".")[0]
        output_file = os.path.join(new_path, name + "_mask.png")

        cv2.imwrite(output_file, mask * 255)

    average_time_infer = statistics.mean(time_infer)
    if len(time_infer) > 1:
        std_dev_time_infer = humanfriendly.format_timespan(statistics.stdev(time_infer))
    else:
        std_dev_time_infer = "NO DISPONIBLE"

    print("On average, for each image: ")
    print(
        "It took {} to generate the mask, with a deviation of {}".format(
            humanfriendly.format_timespan(average_time_infer), std_dev_time_infer
        )
    )


# ----------------------------------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Módulo para generar máscaras abarcando únicamente la zona de la detección"
    )
    parser.add_argument(
        "--json_file",
        help="Fichero JSON del que se tomarán los datos para delimitar la zona de la máscara",
    )
    parser.add_argument(
        "--output_path",
        help="Ruta al directorio de donde se guardarán las máscaras como fichero de imagen",
    )

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    args = parser.parse_args()

    if args.json_file:
        json_file = args.json_file
    else:
        parser.print_help()
        parser.exit()

    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
    else:
        parser.print_help()
        parser.exit()

    with open(json_file, "r") as file:
        data = json.load(file)

    print("Generating masks for {} images...".format(len(data["images"])))

    run(data["images"], args.output_path)

    print("Results saved in: {}".format(args.output_path))


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
