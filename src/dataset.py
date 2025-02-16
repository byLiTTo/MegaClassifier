from typing import Dict

import pandas as pd


def drop_class_from_dataset(dataset_original: pd.DataFrame, class_name: str) -> pd.DataFrame:
    """
    Removes all rows belonging to a specified class from the given dataset.

    This function creates a copy of the input dataset and removes all rows where
    the 'label' column matches the specified class name. It returns a new dataset
    excluding rows with the given class label, leaving the original dataset
    unchanged.

    Args:
        dataset_original (pd.DataFrame): The original dataset to process. This
            input is expected to have a column named 'label', which denotes the
            class of each row.
        class_name (str): The name of the class to remove from the dataset. Rows
            having this class name in the 'label' column will be excluded.

    Returns:
        pd.DataFrame: A new DataFrame containing all rows from the original
            dataset except those with the specified class label.

    """
    dataset = dataset_original.copy()
    return dataset[dataset['label'] != class_name]


def parse_dataset(dataset_original: pd.DataFrame) -> pd.DataFrame:
    """
    Parses and transforms the original dataset by updating column names and generating
    new labels for binary classification.

    The function creates a deep copy of the provided dataset to avoid modifying the
    original DataFrame. It renames the "label" column to "original_label" for clarity,
    and generates two new columns:
    - "label": A categorical column where any value except "vacia" is replaced with
      "animal".
    - "binary_label": A binary column where "vacia" is assigned "0" and all other
      values are assigned "1".

    Args:
        dataset_original (pd.DataFrame): The original dataset to be parsed. It must
            contain a column named "label".

    Returns:
        pd.DataFrame: A new DataFrame containing the modified and newly created columns.
    """
    dataset_parsed = dataset_original.copy()
    dataset_parsed.rename(columns={"label": "original_label"}, inplace=True)

    dataset_parsed["label"] = dataset_parsed["original_label"].apply(lambda x: "vacia" if x == "vacia" else "animal")
    dataset_parsed["binary_label"] = dataset_parsed["original_label"].apply(lambda x: "0" if x == "vacia" else "1")

    return dataset_parsed


def remove_character(string: str, custom_map: Dict[str, str]) -> str:
    """
    Removes characters from a string based on a custom mapping. The function processes
    each character in the input string, and if the character exists in the custom
    mapping, it replaces it with the corresponding value. If a character does not
    exist in the mapping, it remains unchanged.

    Args:
        string: The input string to process, where characters will be replaced
            based on the provided mapping.
        custom_map: A dictionary where the keys represent characters to be replaced
            and the corresponding values represent their replacements.

    Returns:
        A new string with characters replaced as defined by the custom mapping.
    """
    return "".join(custom_map.get(char, char) for char in string)


def parse_from_windows_path(path: str) -> str:
    """
    Parses a given Windows file path and converts it to a Unix-like file path by replacing
    backslashes with forward slashes.

    Args:
        path (str): A string representing a Windows file path.

    Returns:
        str: A string representing the converted Unix-like file path.
    """
    custom_map = {"\\": "/"}
    return remove_character(path, custom_map)


def parse_from_unix_path(path: str) -> str:
    """
    Parses a Unix-style file path and converts it to a Windows-style file path format by replacing
    Unix path separators with Windows path separators.

    Args:
        path (str): The Unix-style file path to be converted.

    Returns:
        str: The converted Windows-style file path.
    """
    custom_map = {"/": "\\"}
    return remove_character(path, custom_map)


def remove_spaces(file_name: str) -> str:
    """
    Replaces spaces in the given file name with underscores.

    This function takes a `file_name` string and substitutes all spaces within it
    with underscores ('_') using a predefined mapping. The transformed file name
    is then returned. It utilizes an auxiliary helper function
    `remove_character` for the actual substitution process.

    Args:
        file_name (str): The name of the file that will have spaces replaced with
            underscores.

    Returns:
        str: The updated file name, where all spaces have been replaced by
            underscores.
    """
    custom_map = {" ": "_"}
    return remove_character(file_name, custom_map)


def remove_parenthesis(file_name: str) -> str:
    """
    Removes parenthesis from the input file name and replaces them with underscores.

    This function utilizes a predefined mapping to replace parenthesis characters
    in the input string with underscores. It leverages a helper function `remove_character`
    to execute the replacement based on the mapping.

    Args:
        file_name: The name of the file as a string that contains characters to
            be replaced.

    Returns:
        A string that is derived from the input file name with parenthesis
        characters replaced by underscores.
    """
    custom_map = {"(": "_", ")": "_"}
    return remove_character(file_name, custom_map)


def remove_upper_accent(file_name: str) -> str:
    """
    Removes uppercase accented characters from a given string and replaces
    them with their non-accented equivalents.

    Args:
        file_name (str): The input string potentially containing uppercase
            accented characters.

    Returns:
        str: A new string with uppercase accented characters replaced by
        their non-accented counterparts.
    """
    custom_map = {"Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U"}
    return remove_character(file_name, custom_map)


def remove_lower_accent(file_name: str) -> str:
    """
    Removes specific accented lowercase characters from a given string using
    a predefined mapping.

    This function takes an input string and replaces accented lowercase
    characters (á, é, í, ó, ú) with their unaccented equivalents ('a', 'e',
    'i', 'o', 'u') based on the provided mapping. It utilizes a helper
    function `remove_character` to perform the replacement.

    Args:
        file_name: The string from which accented lowercase characters
            should be removed.

    Returns:
        str: A new string with the accented lowercase characters replaced
            by their unaccented equivalents.
    """
    custom_map = {"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u"}
    return remove_character(file_name, custom_map)


def remove_spanish_n(file_name: str) -> str:
    """
    Replaces all occurrences of the Spanish character 'ñ' with 'n' and 'Ñ' with
    'N' in the given string.

    This function utilizes a predefined mapping dictionary to ensure consistent
    replacement of the specified characters. The modified string is then returned.

    Args:
        file_name: The input string containing potential occurrences of the
            Spanish 'ñ' or 'Ñ' characters.

    Returns:
        The modified string where all occurrences of 'ñ' and 'Ñ' are replaced
        with 'n' and 'N', respectively.
    """
    custom_map = {"ñ": "n", "Ñ": "N"}
    return remove_character(file_name, custom_map)
