# %%
import os

import numpy as np
from PIL import Image


# %%
COLUMNS_AXIS = 0
ROWS_AXIS = 1


def add_margin_to_detections(shadow, margin=0):
    indices = np.where(shadow == 255)[0]
    new_shadow = np.zeros_like(shadow)

    for index in indices:
        start = max(0, index - margin)
        end = min(len(shadow), index + margin + 1)
        new_shadow[start:end] = 255

    return new_shadow


def fit_image(image_file, mask_file, fit_image_file, margin=0):
    image = Image.open(str(image_file))
    image_array = np.array(image)

    if np.all(image_array == 0):
        os.makedirs(
            fit_image_file[: -len(os.path.basename(fit_image_file))], exist_ok=True
        )
        image.save(fit_image_file)
        return

    mask = Image.open(mask_file)
    mask_array = np.array(mask.convert("L"))
    mask_binary_array = mask_array > 0

    shadow_row = np.sum(mask_binary_array, axis=COLUMNS_AXIS)
    shadow_row = np.where(shadow_row > 0, 255, 0)

    shadow_column = np.sum(mask_binary_array, axis=ROWS_AXIS)
    shadow_column = np.where(shadow_column > 0, 255, 0)

    shadow_row_with_margin = add_margin_to_detections(shadow_row, margin)
    shadow_column_with_margin = add_margin_to_detections(shadow_column, margin)

    image_array = np.delete(
        image_array, np.where(shadow_row_with_margin == 0), axis=ROWS_AXIS
    )
    image_array = np.delete(
        image_array, np.where(shadow_column_with_margin == 0), axis=COLUMNS_AXIS
    )

    modified_image = Image.fromarray(image_array).convert("RGB")

    os.makedirs(fit_image_file[: -len(os.path.basename(fit_image_file))], exist_ok=True)

    modified_image.save(fit_image_file)


def fit_image_with_padding(image_file, mask_file, fit_image_file, margin=0):
    image = Image.open(image_file)
    image_array = np.array(image)

    if np.all(image_array == 0):
        os.makedirs(
            fit_image_file[: -len(os.path.basename(fit_image_file))], exist_ok=True
        )
        image.save(fit_image_file)
        return

    mask = Image.open(mask_file)
    mask_array = np.array(mask.convert("L"))
    mask_binary_array = mask_array > 0

    shadow_row = np.sum(mask_binary_array, axis=COLUMNS_AXIS)
    shadow_row = np.where(shadow_row > 0, 255, 0)

    shadow_column = np.sum(mask_binary_array, axis=ROWS_AXIS)
    shadow_column = np.where(shadow_column > 0, 255, 0)

    shadow_row_with_margin = add_margin_to_detections(shadow_row, margin)
    shadow_column_with_margin = add_margin_to_detections(shadow_column, margin)

    image_array = np.delete(
        image_array, np.where(shadow_row_with_margin == 0), axis=ROWS_AXIS
    )
    image_array = np.delete(
        image_array, np.where(shadow_column_with_margin == 0), axis=COLUMNS_AXIS
    )

    modified_image = Image.fromarray(image_array).convert("RGB")

    height, width, _ = image_array.shape
    if height > width:
        padding = (height - width) // 2
        image_array = np.pad(
            image_array,
            ((0, 0), (padding, padding), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    elif width > height:
        padding = (width - height) // 2
        image_array = np.pad(
            image_array,
            ((padding, padding), (0, 0), (0, 0)),
            mode="constant",
            constant_values=0,
        )

    modified_image = Image.fromarray(image_array).convert("RGB")
    os.makedirs(fit_image_file[: -len(os.path.basename(fit_image_file))], exist_ok=True)
    modified_image.save(fit_image_file)
