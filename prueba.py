import tensorflow as tf
print('Version de TENSORFLOW: ' + tf.__version__)
print('=======================================================================================================================================')
print('Configuracion de GPU:')
tf.config.list_physical_devices('GPU')