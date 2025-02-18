# %%
import os

import numpy as np
from PIL import Image

# %%
COLUMNS_AXIS = 0
ROWS_AXIS = 1


def add_margin_to_detections(shadow, margin=0):
    """
    Adds margin around detected indices in a shadow array.

    This function processes a 1D binary array `shadow`, where indices with a value
    of 255 represent detected features. For each detected index, a margin specified
    by `margin` is applied, resulting in a wider region around the indices. The
    output is a new 1D array with the modified shadow, where the margin has been
    applied to detected regions.

    Args:
        shadow (np.ndarray): A 1D binary array where the value of 255 represents
            detected features, and other values are treated as non-detected.
        margin (int, optional): An integer specifying the number of positions to
            extend on each side of detected indices. Defaults to 0.

    Returns:
        np.ndarray: A new 1D binary array of the same length as the input `shadow`,
        with the margin applied to detected regions.
    """
    indices = np.where(shadow == 255)[0]
    new_shadow = np.zeros_like(shadow)

    for index in indices:
        start = max(0, index - margin)
        end = min(len(shadow), index + margin + 1)
        new_shadow[start:end] = 255

    return new_shadow


def fit_image(image_file, mask_file, fit_image_file, margin=0):
    """
    Fits an image to a mask by removing rows and columns that are outside the
    mask's region of interest, optionally applying a margin around the detected
    area. The modified image is then saved to the specified location.

    Args:
        image_file (Path): Path to the source image file.
        mask_file (Path): Path to the mask image file.
        fit_image_file (Path): Path where the modified image will be saved.
        margin (int, optional): Additional margin to apply around the mask's
            detected region. Defaults to 0.

    Raises:
        FileNotFoundError: If the image_file or mask_file paths do not exist.
        ValueError: If the provided image or mask files are invalid or unreadable.
    """
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
    """
    Processes an input image and mask to output a cropped and padded image. The function removes
    the regions of the image where the mask has no non-zero pixels, crops the image accordingly,
    and applies padding to ensure the resulting image is square. The processed image is then saved
    at the specified output location.

    Args:
        image_file (str): Path to the input image file.
        mask_file (str): Path to the input mask file. The mask determines the regions to keep.
        fit_image_file (str): Path where the processed image will be saved.
        margin (int, optional): Additional margin to apply to the regions defined by the mask.
            Defaults to 0.

    Raises:
        OSError: Raised if an error occurs while opening, saving, or manipulating the image files.
    """
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
