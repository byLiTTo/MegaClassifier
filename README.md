# Aplicación de estrategias de deep-learning para la detección de animales en imágenes de fototrampeo
**Alumno:** Carlos García Silva

**Centro:** Universidad de Huelva (UHU)   
**Titulación:** Grado Ingeniería Informática   
**Departamento:** Ingeniería Electrónica, de Sistemas Informáticos y Automática   
**Área de conocimiento:** Ingeniería de Sistemas y Automática   

**Tutor 1:** Diego Marín Santos   
**Tutor 2:** Manuel Emilio Gegúndez Arias   

___

# Introducción:
La aplicación de estrategias de Deep Learning (DL) está mostrando gran potencialidad en el análisis de imágenes digitales de distinta naturaleza. En este TFG se aplican a imágenes adquiridas mediante cámaras de fototrampeo para la detección de la presencia de animales en las imágenes.

# Objetivos:
Aprendizaje de los fundamentos de las redes neuronales convolucionales (CNNs) y su aplicación mediante la librería TensorFlow de Python. Revisión del estado del arte, diseño e implementación de CNN en el problema de experimentación planteado. Análisis y evaluación de resultados.

# Requisitos previos:
- Python: v3.8
- TensorFlow:
    - v2.3 GPU (En nuestro entorno Windows)
    - v2.4 GPU (En nuestro entorno Linux)
    - v2.8 GPU (En nuestro entorno macOS)
- Repositorio [CameraTraps](https://github.com/microsoft/CameraTraps) de Microsoft, que hemos incluido en la carpeta [repos](https://github.com/byLiTTo/TFG-DeteccionFototrampeo/tree/main/repos) de nuestro repositorio
- Repositorio [ai4eutils](https://github.com/microsoft/ai4eutils) de Microsoft, que hemos incluido en la carpeta [repos](https://github.com/byLiTTo/TFG-DeteccionFototrampeo/tree/main/repos) de nuestro repositorio
- Modelo MegaDetector: [megadetector_v4_1_0.pb](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb)

# Librerías de Python utilizadas:
Si al ejecutar los notebooks, sucede algún tipo de error, asegurese de tener instaladas las siguientes librerías en su entorno:

_os, sys, json, time, umpy, argparse, platform, traceback, statistics, humandfriendly_

___

# Ejecutar aplicación
Para poder ejecutar los módulos de python, hemos creado un entorno conda en nuestro dispositivo. Una vez instalados todos los requisitos, dependencias y librerías en nuestro entorno, lo usaremos como kernel para ejecutar los diferentes notebooks que hemos implementado para poder ejecutar de forma clara y ordenada, las diferentes fases de la aplicación.

## Notebook: a01_GeneraDetecciones
Se le indica un directorio de entrada de donde tomará las imágenes a las que le aplicará el modelo de detecciones entrenado de MegaDetector. (Por defecto hemos creado la carpeta [input](https://github.com/byLiTTo/TFG-DeteccionFototrampeo/tree/main/input) para que tome las imágenes).

Una vez obtenidos los resultados, genera un fichero JSON con los datos de cada detección encontrada en cada una de las imágenes. Se generan tanto un JSON global que contiene todos los resultados de la ejecución, así como un fichero JSON por cada imagen de forma individual. 

Todos estos resultados son guardados en la carpeta [output_json](https://github.com/byLiTTo/TFG-DeteccionFototrampeo/tree/main/output_json), para los ficheros de resultados globales, hemos creado una carpeta dentro de la anteriormente mencionada, a modo de historial, se trata de [registry](https://github.com/byLiTTo/TFG-DeteccionFototrampeo/tree/main/output_json/registry).

___