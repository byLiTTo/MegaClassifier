import os
import math
import numpy as np
import pandas as pd
import platform

from sklearn.utils import shuffle



###################################################################################################
# IMPORTACION TENSORFLOW

# Useful hack to force CPU inference
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf

print('')
print('==========================================================================================')
print('TensorFlow version:', tf.__version__)
#print('Is GPU available? tf.test.is_gpu_available:', tf.test.is_gpu_available())
print('Is GPU available? tf.test.is_gpu_available:',tf.config.list_physical_devices('GPU'))
print('==========================================================================================')
print('')



###################################################################################################
# CLASES:
class DatasetUtils:
    """
    Colección de funciones para manejar datasets con csv.
    """

    @staticmethod
    def generate_train_validation_test(empty, animals, samples, per_train, per_test):

        df = pd.DataFrame(empty[:samples])
        train_empty, validation_empty, test_empty = np.split(df.sample(frac=1, random_state=42),[int(per_train*len(df)), int((per_train+per_test)*len(df))])
    
        df = pd.DataFrame(animals[:(math.trunc(len(animals)*(samples/len(empty))))])
        train_animals, validation_animals, test_animals = np.split(df.sample(frac=1, random_state=42),[int(per_train*len(df)), int((per_train+per_test)*len(df))])

        aux = [["file_name","label"]]

        train = pd.concat([pd.DataFrame(aux), shuffle(pd.concat([train_empty, train_animals]), random_state=42)])
        validation = pd.concat([pd.DataFrame(aux), shuffle(pd.concat([validation_empty, validation_animals]), random_state=42)])
        test = pd.concat([pd.DataFrame(aux), shuffle(pd.concat([test_empty, test_animals]), random_state=42)])

        return train, validation, test
    
    @staticmethod
    def load_csv(csv_file):
        """
        Lee un fichero CSV que contiene las rutas relativas de las imágenes que forman un dataset.
        Además de estas rutas, contiene la clase a la que pertenece cada imagen. Esta función
        devuelve una tupla con la ruta absoluta de la imagen y su correspondiente clase.

        Args:
            - location: Directorio raiz del dataset de imágenes.
            - csv_file: Fichero CSV que contiene la información para cargar correctamente el 
                dataset.
                
        Return:
            - file_names: Lista de rutas absolutas de las imágenes.
            - labels: Clase a la que pertenece cada imagen.
        """
        df = pd.read_csv(csv_file, sep=';')
        return df['file_name'].values, df['label'].values

    @staticmethod
    def convert_to_abspath(location,file_names,labels):
        new_file_names = []
        if platform.system() == 'Windows':
            for fn in file_names:
                new_file_names.append(location+fn.replace('/','\\'))
        else:
            for fn in file_names:
                new_file_names.append(location+fn.replace('\\','/'))
        return new_file_names, labels

    @staticmethod
    def load_image(image_file, labels):
        image = tf.io.read_file(image_file)
        image = tf.image.decode_image(image, channels=3)
        return image, labels