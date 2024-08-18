import math
import os
import platform

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


# ----------------------------------------------------------------------------------------------------------------------
def load_from_csv(path, sep=";", encoding="utf-8"):
    """
    Load data from a CSV file.

    Args:
        path (str): The path to the CSV file.
        sep (str, optional): The delimiter used in the CSV file. Defaults to ';'.
        encoding (str, optional): The encoding of the CSV file. Defaults to 'utf-8'.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the CSV file.
    """

    try:
        df = pd.read_csv(path, sep=sep)
        print(f"El archivo {path} se ha abierto con éxito.")
    except FileNotFoundError:
        print("El archivo no existe.")
    except PermissionError:
        print("No tienes permisos para leer el archivo.")
    except pd.errors.EmptyDataError:
        print("El archivo está vacío.")
    except pd.errors.ParserError:
        print("Hubo un error al analizar el archivo.")

    return df


def crop_dataset(dataset, number_of_samples):
    """
    Crop a number of samples from the dataset.

    Args:
        dataset (pd.DataFrame): The dataset to drop samples from.
        number_of_samples (int): The number of samples to drop.

    Returns:
        pd.DataFrame: The dataset without the dropped samples.
    """

    empty = shuffle(dataset[dataset["label"] == 0], random_state=42)
    animals = shuffle(dataset[dataset["label"] == 1], random_state=42)

    empty_crop = empty[:number_of_samples]

    percentage = len(empty) / (len(dataset["file_name"].values))

    number_of_samples = math.ceil(percentage * number_of_samples)
    animals_crop = animals[:number_of_samples]

    dataset = pd.concat([empty_crop, animals_crop])
    dataset = shuffle(dataset, random_state=42)

    return dataset


# ----------------------------------------------------------------------------------------------------------------------
def convert_to_binary(dataset, negative_nomenclature, positive_nomenclature):
    """
    Convert the labels of the dataset to binary.

    Args:
        dataset (pd.DataFrame): The dataset to convert.
        negative_nomenclature (list): The nomenclature of the negative class.
        positive_nomenclature (list): The nomenclature of the positive class.

    Returns:
        pd.DataFrame: The dataset with binary labels.
    """
    try:
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("dataset debe ser un DataFrame de pandas.")
        if not isinstance(negative_nomenclature, list) or not isinstance(
            positive_nomenclature, list
        ):
            raise TypeError(
                "negative_nomenclature y positive_nomenclature deben ser listas."
            )
        if "label" not in dataset.columns:
            raise ValueError("El conjunto de datos debe contener una columna 'label'.")

        dataset = dataset[
            dataset["label"].isin(negative_nomenclature + positive_nomenclature)
        ]
        dataset.loc[:, "label"] = dataset["label"].apply(
            lambda x: 0 if x in negative_nomenclature else 1
        )
    except Exception as e:
        print(f"Ocurrió un error: {e}")
        return None

    return dataset


# ----------------------------------------------------------------------------------------------------------------------
def dataset_to_csv(dataset, csv_path, index=False, sep=";"):
    """
    Save the dataset to a CSV file.

    Args:
        dataset (pd.DataFrame): The dataset to be saved.
        csv_path (str): The path to save the CSV file.
        index (bool, optional): Whether to include the index in the CSV file. Defaults to False.
        sep (str, optional): The delimiter to use in the CSV file. Defaults to ';'.
    """
    try:
        dataset.to_csv(csv_path, index=index, sep=sep)
        print(f"El conjunto de datos se ha guardado correctamente en {csv_path}.")
    except PermissionError:
        print("No tienes permisos para guardar el archivo.")
    except Exception as e:
        print(f"Ocurrió un error al guardar el conjunto de datos: {e}")


# ----------------------------------------------------------------------------------------------------------------------
def split_dataset(dataset, percentaje_train, percentaje_val, percentaje_test):
    """
    Split the dataset into training, validation, and test sets.

    Args:
        dataset (pd.DataFrame): The dataset to be split.
        percentaje_train (float): The percentage of data to be used for training.
        percentaje_val (float): The percentage of data to be used for validation.
        percentaje_test (float): The percentage of data to be used for testing.

    Raises:
        TypeError: If the dataset is not a pandas DataFrame.
        ValueError: If the dataset does not contain the columns 'label' and 'file_name'.

    Returns:
        pd.DataFrame: The training set.
        pd.DataFrame: The validation set.
        pd.DataFrame: The test set.
    """
    try:
        assert (
            percentaje_train + percentaje_val + percentaje_test == 1.0
        ), "La suma de los tres porcentajes debe ser igual a 1.0. Ahora es {}".format(
            percentaje_train + percentaje_val + percentaje_test
        )

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("dataset debe ser un DataFrame de pandas.")
        if "label" not in dataset.columns or "file_name" not in dataset.columns:
            raise ValueError(
                "El conjunto de datos debe contener las columnas 'label' y 'file_name'."
            )

        empty = shuffle(dataset[dataset["label"] == 0], random_state=42)
        animals = shuffle(dataset[dataset["label"] == 1], random_state=42)

        num_samples = len(empty["file_name"].values)
        num_train = math.floor(num_samples * percentaje_train)
        num_val = math.floor(num_samples * percentaje_val)

        train_empty_set = empty[:num_train]
        val_empty_set = empty[num_train : num_train + num_val]
        test_empty_set = empty[num_train + num_val :]

        num_samples = len(animals["file_name"].values)
        num_train = math.floor(num_samples * percentaje_train)
        num_val = math.floor(num_samples * percentaje_val)

        train_animals_set = animals[:num_train]
        val_animals_set = animals[num_train : num_train + num_val]
        test_animals_set = animals[num_train + num_val :]

        train_set = pd.concat([train_empty_set, train_animals_set])
        train_set = shuffle(train_set, random_state=42)

        val_set = pd.concat([val_empty_set, val_animals_set])
        val_set = shuffle(val_set, random_state=42)

        test_set = pd.concat([test_empty_set, test_animals_set])
        test_set = shuffle(test_set, random_state=42)

    except AssertionError as e:
        print(f"Error de aserción: {e}")
        return None, None, None
    except Exception as e:
        print(f"Ocurrió un error: {e}")
        return None, None, None

    return train_set, val_set, test_set


# ----------------------------------------------------------------------------------------------------------------------
def convert_csv_to_abstract(dataset, dataset_dir):
    """
    Converts the file paths in the dataset to absolute paths by joining them with the dataset directory.

    Args:
        dataset (pandas.DataFrame): The dataset containing the file paths.
        dataset_dir (str): The directory where the dataset is located.

    Returns:
        pandas.DataFrame: The dataset with the updated file paths.
    """
    try:
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("dataset debe ser un DataFrame de pandas.")
        if not isinstance(dataset_dir, str):
            raise TypeError("dataset_dir debe ser una cadena de texto.")

        dataset.loc[:, "file_name"] = dataset["file_name"].apply(
            lambda x: dataset_dir + x.replace("/", "\\")
            if platform.system() == "Windows"
            else dataset_dir + x.replace("\\", "/")
        )

    except Exception as e:
        print(f"Ocurrió un error: {e}")
        return None

    return dataset
