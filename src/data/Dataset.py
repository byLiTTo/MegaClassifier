import math
import os
import platform
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import os


def load_from_csv(path: str, sep: str = ";", encoding: str = "utf-8") -> pd.DataFrame:
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
        df = pd.read_csv(path, sep=sep, encoding=encoding)
        print(f"The file {path} has been successfully opened.")
    except FileNotFoundError:
        print("The file does not exist.")
    except PermissionError:
        print("You do not have permission to read the file.")
    except pd.errors.EmptyDataError:
        print("The file is empty.")
    except pd.errors.ParserError:
        print("There was an error parsing the file.")

    return df


def convert_to_binary(
    dataset: pd.DataFrame, negative_nomenclature: list, positive_nomenclature: list
) -> pd.DataFrame:
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
            raise TypeError("dataset must be a pandas DataFrame.")
        if not isinstance(negative_nomenclature, list) or not isinstance(
            positive_nomenclature, list
        ):
            raise TypeError(
                "negative_nomenclature and positive_nomenclature must be lists."
            )
        if "label" not in dataset.columns:
            raise ValueError("The dataset must contain a 'label' column.")

        dataset = dataset[
            dataset["label"].isin(negative_nomenclature + positive_nomenclature)
        ]
        dataset.loc[:, "label"] = dataset["label"].apply(
            lambda x: 0 if x in negative_nomenclature else 1
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return dataset


def dataset_to_csv(
    dataset: pd.DataFrame, csv_path: str, index: bool = False, sep: str = ";"
) -> None:
    """
    Save the dataset to a CSV file.

    Args:
        dataset (pd.DataFrame): The dataset to be saved.
        csv_path (str): The path to save the CSV file.
        index (bool, optional): Whether to include the index in the CSV file. Defaults to False.
        sep (str, optional): The delimiter to use in the CSV file. Defaults to ';'.
    """
    try:
        if not os.path.exists(os.path.dirname(csv_path)):
            os.makedirs(os.path.dirname(csv_path))

        dataset.to_csv(csv_path, index=index, sep=sep)
        print(f"The dataset has been successfully saved to {csv_path}.")
    except PermissionError:
        print("You do not have permission to save the file.")
    except Exception as e:
        print(f"An error occurred while saving the dataset: {e}")


def crop_dataset(dataset: pd.DataFrame, number_of_samples: int) -> pd.DataFrame:
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

    percentage = number_of_samples / len(empty) * 100

    number_of_samples = math.ceil(len(animals) * percentage / 100)
    animals_crop = animals[:number_of_samples]

    dataset = pd.concat([empty_crop, animals_crop])
    dataset = shuffle(dataset, random_state=42)

    return dataset


def split_dataset(
    dataset: pd.DataFrame,
    percentaje_train: float,
    percentaje_val: float,
    percentaje_test: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        ), "The sum of the three percentages must be equal to 1.0. It is currently {}".format(
            percentaje_train + percentaje_val + percentaje_test
        )

        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("dataset must be a pandas DataFrame.")
        if "label" not in dataset.columns or "file_name" not in dataset.columns:
            raise ValueError(
                "The dataset must contain the 'label' and 'file_name' columns."
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
        print(f"Assertion error: {e}")
        return None, None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None, None

    return train_set, val_set, test_set
