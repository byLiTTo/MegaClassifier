import os
import platform

from datetime import datetime
now = datetime.now()

#########################################################################################################################################################
# PARAMETROS DEFINIDOS EN FUNCION DEL USUARIO

# Ruta relativa hacia la librería ai4eutils
ai4eutils_relative = "./repos/ai4eutils"

# Ruta relativa hacia la librería CameraTraps
CameraTraps_relative = "./repos/CameraTraps"

# Ruta relativa hacia carpeta de imagenes de entrada
images_dir_relative = './input'

# Ruta relativa y nombre del fichero JSON de salida
output_file_path_relative = ''
# Si se desea generar de forma automatica en cada ejecución descomentar:
output_file_path_relative = ('./output_tmp/' + str(now.date()) + '_' + str(now.hour) + '-' + str(now.minute) + '.json')

# Ruta relativa a carpeta de imagenes de salida
visualization_dir_relative = './output'

# Ruta relativa al modelo
model_relative = './models/megadetector_v4_1_0.pb'

#########################################################################################################################################################

home = os.path.expanduser("~")

ai4utils = os.path.abspath(ai4eutils_relative)
CameraTraps = os.path.abspath(CameraTraps_relative)
images_dir = os.path.abspath(images_dir_relative)
output_file_path = os.path.abspath(output_file_path_relative)
visualization_dir = os.path.abspath(visualization_dir_relative)
model = os.path.abspath(model_relative)

try:
    os.environ['PYTHONPATH']
except KeyError:
    os.environ['PYTHONPATH'] = ""
if platform.system() == 'Windows':
    os.environ['PYTHONPATH'] += (";" + ai4utils)
    os.environ['PYTHONPATH'] += (";" + CameraTraps)
else:
    os.environ['PYTHONPATH'] += (":" + ai4utils)
    os.environ['PYTHONPATH'] += (":" + CameraTraps)

print('=======================================================================================================================================')
print('PYTHONPATH: ' + os.environ['PYTHONPATH'])
print('ai4eutils PATH: ' + ai4utils)
print('CameraTraps PATH: ' + CameraTraps)
print('input PATH: ' + images_dir)
print('output_file_path: ' + output_file_path)
print('output PATH: ' + visualization_dir)
print('modelo: ' + model)
print('=======================================================================================================================================')
# print('Version de TENSORFLOW: ' + tf.__version__)
# print('Configuracion de GPU:')
# tf.config.list_physical_devices('GPU')

os.system('d:\\WORKSPACE\\TFG-DeteccionFototrampeo\\modules\\ejecutar_tf_detector.py d:\\WORKSPACE\\TFG-DeteccionFototrampeo\\models\\megadetector_v4_1_0.pb d:\\WORKSPACE\\TFG-DeteccionFototrampeo\\input d:\\WORKSPACE\\TFG-DeteccionFototrampeo\\output_tmp')

print('TERMINO====================================')