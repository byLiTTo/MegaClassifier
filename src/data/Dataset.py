import math
import os
import platform

import numpy as np
import pandas as pd
from sklearn.utils import shuffle


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
        # dataset["label"] = dataset["label"].apply(
        #     lambda x: 0 if x in negative_nomenclature else 1
        # )
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
        dataset.to_csv(csv_path, index=index, sep=sep)
        print(f"The dataset has been successfully saved to {csv_path}.")
    except PermissionError:
        print("You do not have permission to save the file.")
    except Exception as e:
        print(f"An error occurred while saving the dataset: {e}")
