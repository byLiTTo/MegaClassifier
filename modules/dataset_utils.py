import math
import os
import platform

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.utils import shuffle

########################################################################################################################
# IMPORT TENSORFLOW
# Useful hack to force CPU inference
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print('')
print('===============================================================================================================')
print('TensorFlow version:', tf.__version__)
# print('Is GPU available? tf.test.is_gpu_available:', tf.test.is_gpu_available())
print('Is GPU available? tf.test.is_gpu_available:', tf.config.list_physical_devices('GPU'))
print('===============================================================================================================')
print('')


########################################################################################################################
# CLASES:

class DatasetUtils:
    """
    This class contains a set of utility functions for working with datasets
    """

    @staticmethod
    def generate_train_validation_test(empty, animals, samples, per_train, per_test):
        """
        It takes in the empty and animal dataframes, the number of samples to use, the percentage of the data to use for
        training, and the percentage of the data to use for testing. It then shuffles the data, splits it into training,
        validation, and test sets, and returns the three dataframes and the full set.

        :param empty: the list of empty images
        :param animals: the list of all the animal images
        :param samples: the number of samples to use from the empty and animal datasets
        :param per_train: percentage of the data that will be used for training
        :param per_test: The percentage of the data that will be used for testing
        :return: a tuple of four dataframes, train, validation and test.
        """

        empty_cropped = shuffle(empty, random_state=42)[:samples]
        empty_train_len = int(per_train * len(empty_cropped))
        empty_validation_len = int((per_train + per_test) * len(empty_cropped))

        empty_train, empty_validation, empty_test = np.split(
            empty_cropped,
            [empty_train_len, empty_validation_len]
        )

        animals_len = math.trunc(len(animals) * (samples / len(empty)))

        animals_cropped = shuffle(animals, random_state=42)[:animals_len]
        animals_train_len = int(per_train * len(animals_cropped))
        animals_validation_len = int((per_train + per_test) * len(animals_cropped))

        animals_train, animals_validation, animals_test = np.split(
            animals_cropped,
            [animals_train_len, animals_validation_len]
        )

        train = pd.DataFrame(
            shuffle(np.concatenate((empty_train, animals_train), axis=0), random_state=42),
            columns=['file_name', 'label']
        )

        validation = pd.DataFrame(
            shuffle(np.concatenate((empty_validation, animals_validation), axis=0), random_state=42),
            columns=['file_name', 'label']
        )

        test = pd.DataFrame(
            shuffle(np.concatenate((empty_test, animals_test), axis=0), random_state=42),
            columns=['file_name', 'label']
        )

        full = np.concatenate((train.to_numpy(), validation.to_numpy(), test.to_numpy()), axis=0)

        return full, train, validation, test

    @staticmethod
    def load_dataset(csv_file, location, encoding):
        """
        It takes a csv file and a location and returns the file names and labels

        :param csv_file: The path to the csv file containing the file names and labels
        :param location: The location of the dataset
        :param encoding: The encoding of the csv file
        :return: The file names and labels are being returned.
        """
        file_names, labels = DatasetUtils.load_csv(csv_file, encoding)
        file_names, labels = DatasetUtils.reset_path(file_names, labels)
        file_names, labels = DatasetUtils.convert_to_abspath(location, file_names, labels)
        return file_names, labels

    @staticmethod
    def load_csv(csv_file, encoding):
        """
        It reads a csv file and returns the file names and labels

        :param csv_file: the path to the csv file
        :param encoding: The encoding of the csv file
        :return: The file name and the label
        """
        df = pd.read_csv(csv_file, sep=';', encoding=encoding)
        return df['file_name'].values, df['label'].values

    @staticmethod
    def reset_path(file_names, labels):
        """
        It takes a list of file names and a list of labels, and returns a list of file names and a list of labels in the
        correct format of the current platform system

        :param file_names: a list of file names
        :param labels: a list of labels for each image
        :return: The new_file_names and labels are being returned.
        """
        new_file_names = []
        if platform.system() == 'Windows':
            for fn in file_names:
                fn = os.path.basename(fn.replace('/', '\\')).lower()
                name, ext = os.path.splitext(fn)
                new_file_names.append('\\' + name)
        else:
            for fn in file_names:
                fn = os.path.basename(fn.replace('\\', '/')).lower()
                name, ext = os.path.splitext(fn)
                new_file_names.append('/' + name)
        return new_file_names, labels

    @staticmethod
    def convert_to_abspath(location, file_names, labels):
        """
        It converts the file names to absolute paths

        :param location: the location of the dataset
        :param file_names: a list of file names
        :param labels: a list of labels for each image
        :return: The new_file_names and labels are being returned.
        """
        new_file_names = []
        if platform.system() == 'Windows':
            for fn in file_names:
                new_file_names.append(location + fn.replace('/', '\\'))
        else:
            print("macOS")
            for fn in file_names:
                new_file_names.append(location + fn.replace('\\', '/'))
        return new_file_names, labels

    @staticmethod
    def save_dataset(file_path, labels, output_path):
        for image_path, label in zip(file_path, labels):
            image_path = image_path + '.png'
            name, ext = os.path.splitext(os.path.basename(image_path).lower())
            if label == 0:
                if platform.system() == 'Windows':
                    output_fn = (output_path + '\\' + 'VACIA' + '\\' + name + ext)
                else:
                    print("macOS")
                    output_fn = (output_path + '/' + 'VACIA' + '/' + name + ext)
                image = Image.open(image_path)
                cv2.imwrite(output_fn, np.array(image))
            else:
                if platform.system() == 'Windows':
                    output_fn = (output_path + '\\' + 'ANIMAL' + '\\' + name + ext)
                else:
                    print("macOS")
                    output_fn = (output_path + '/' + 'ANIMAL' + '/' + name + ext)
                image = Image.open(image_path)
                cv2.imwrite(output_fn, np.array(image))

    @staticmethod
    def load_image(image_file, labels):
        """
        It takes an image file and a label, reads the image file, decodes it, and converts it to a float32

        :param image_file: The path to the image file
        :param labels: The labels for the images
        :return: The image and the labels
        """
        image = tf.io.read_file((image_file + '.png'))
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image, labels

    @staticmethod
    def resize_image(image_file, labels):
        """
        The function takes in an image file and its labels, resizes the image to 500x500 pixels, and returns the resized
        image and its labels.

        :param image_file: The image file to be resized
        :param labels: The labels for the images
        :return: The image is being resized to 500x500 pixels and the labels are being returned.
        """
        return tf.image.resize(image_file, (227, 227)), labels

    @staticmethod
    def resize_224(image_file, labels):
        return tf.image.resize(image_file, (224, 224)), labels

    @staticmethod
    def resize_448(image_file, labels):
        return tf.image.resize(image_file, (448, 448)), labels

    @staticmethod
    def normalize_images(image, label):
        """
        It takes an image and a label, and returns the image normalized to the range [0, 1] and the label unchanged

        :param image: The image to be normalized
        :param label: The label of the image
        :return: The image normalized to the range [0, 1].
        """
        return tf.cast(image, tf.float32) / 255.0, label
