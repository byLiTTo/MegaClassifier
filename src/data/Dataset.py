import math
import os
import platform

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


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
        dataset["label"] = dataset["label"].apply(
            lambda x: 0 if x in negative_nomenclature else 1
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return dataset
