# Aplicación de estrategias de deep-learning para la detección de animales en imágenes de foto trampeo

**Alumno:** Carlos García Silva

**Centro:** Universidad de Huelva (UHU)   
**Titulación:** Grado Ingeniería Informática   
**Departamento:** Ingeniería Electrónica, de Sistemas Informáticos y Automática   
**Área de conocimiento:** Ingeniería de Sistemas y Automática

**Tutor 1:** Diego Marín Santos   
**Tutor 2:** Manuel Emilio Gegúndez Arias

___

# Introducción:

La aplicación de estrategias de Deep Learning (DL) está mostrando gran potencialidad en el análisis de imágenes
digitales de distinta naturaleza. En este TFG se aplican a imágenes adquiridas mediante cámaras de foto trampeo para la
detección de la presencia de animales en las imágenes.

# Objetivos:

Aprendizaje de los fundamentos de las redes neuronales convolucionales (CNN) y su aplicación mediante la librería
TensorFlow de Python. Revisión del estado del arte, diseño e implementación de CNN en el problema de experimentación
planteado. Análisis y evaluación de resultados.

# Requisitos previos:

- Python: v3.8
- TensorFlow: v2.10.0 GPU (En nuestro entorno macOS con Apple Silicon)
- Repositorio [CameraTraps](https://github.com/microsoft/CameraTraps) de Microsoft, que hemos incluido en la
  carpeta [repos](https://github.com/byLiTTo/TFG-DeteccionFototrampeo/tree/main/repos) de nuestro repositorio
- Repositorio [ai4eutils](https://github.com/microsoft/ai4eutils) de Microsoft, que hemos incluido en la
  carpeta [repos](https://github.com/byLiTTo/TFG-DeteccionFototrampeo/tree/main/repos) de nuestro repositorio
- Modelo
  MegaDetector: [megadetector_v4_1_0.pb](https://lilablobssc.blob.core.windows.net/models/camera_traps/megadetector/md_v4.1.0/md_v4.1.0.pb)

# Librerías de Python utilizadas:

Si al ejecutar los notebooks, sucede algún tipo de error, asegúrese de tener instaladas las siguientes librerías en su
entorno:

| Nombre | Versión |     | Nombre | Versión |     | Nombre | Versión |
|---------|---------|-----|---------|---------|-----|---------|---------|
| humanfriendly | 10.0 |     |jsonpickle | 2.2.0 |     | jupyterlab | 3.4.4 |
| matplotlib | 3.5.2 |     | numpy | 1.23.1 |     | opencv | 4.6.0 |
| pandas | 1.4.4 |     | pillow | 9.2.0 |     | python | 3.8.13 |
| scikit-learn | 1.1.1 |     | tensorflow-macos | 2.10.0 |     | tensorflow-metal | 0.6.0 |

<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAU8AAACWCAMAAABpVfqTAAAAkFBMVEX///9Nd89Nq889bcxHc85Ecc05a8yktuNEqM3s8Pn09vxJdc6gz+Pb4vM/bsxujta7yOrK1O87pczQ2vGSqd+Gn9za4fNWfdHL5O/6+/5dstOo0+Wqu+bt9vqIxN232unD4O2dseJ+mtrk6ffY6/NottWxweh4vdnL1e92lNhbgdKVyeDe7vUsZMqgtONqi9VG0v0AAAAKmklEQVR4nO2d6XqiMBSGoSFUTYtrW+nYGZeOY+3i/d/dAFnIihDQLp7vR59HFpO8JDlLgg0CEAgEAoFAIBAIBAKBQCAQCAQCgUAgEAgEAoE89fTZFfhRevo9ufrz2ZX4Mcpo9q6ugGg3ojRzAdH2KmkC0fZSaQLRdjJpAlF/2WkCUT+9/XPRZETTz67h91L6q1fBsze5AZ5N5STam7wCTQ/Z+2hv8huCT09ZiE7+vX12rb6P3v4ZplslOvn7rJ7+8wi23qXCppvOUEl08jhXTy2zy3vgPVklPCQX0cnVUj08f5zQGRWIGlL8TYNoNo8ax57/TkobBUQVPeveu0lU+5zxV60+EBUyaNpHvaQnyx1AlMpK88o2X3K5InsgWsyC7rjSSvTp1X3HpROtpEmJ6rHQ22RSdUOvd8FEn68q2WRwbsyb0puqTEkWjv46f0O+jLgL6aBpT3tUEO31fl14qsRFtDd5dac9HESBZi4bUSOJlC7Vm9IbY+Z19ueLk07UTMll8dHjUj2kEc3mzUumOVUbLxPtGSm5P/b4XSKajfTT1veraxUvHERNmldlfmmpnmLzqEnz6bXzGn9tXWOMbUSNBOdSmQtsRCeGFcpd/u6r/KV1jUOCyUI9OH/UE5ymrTKIPulLnUUAdYE8w5zotuoiOSUnE51X3MPC0YvkmRFN3ETfrDSriYpUyYXyLIiObBcc2dBgJSolni6WZ0Y0NoFW06REdXdTSeMZPHfTTPdWF3VKtRt00zRZ6W6q6rbffSG5JJ4hCY3Ty4q4vsBl2PQ3NSmq8xy8J5lQvLPUJUb5ueT9vrPWCfWLYiUhFB1W0+4LknmGiWXEV2dKDA9J788Gz5iNhVuzKEKKc8hyqq36KNRFSITCRdexnMKTPNgucWZKLFmPpX6pg2c2FsyWnJdnSB0b20BpIYVniMbWi6yZEmtkOa/LM4yGxs3n55kPlFWnBak8yd5xmZEpccTp9XmGyUy/+TN4ZsV1ClTlGSLnFK1kSgyaz2x1owHP0HAnzsKTZAE2jlhRtB5dDnmNp2UUCs2ZX2/SzLwqlvdowjOMN+q5c/Akd7Nc6wcUCcKmW+MvjWeIqpyVnKilb2Y2nS8wNeKp26Rz8ET8GW7WiZh5rIGMn3Se5EU5fau1bf6o23S6bO/FUx8NZ+UZBIvEXo1W0nlqg3D8vt64bs3FF5r9eIaJYgvOzDN44ZMo7s4LNXhGd/LpMYpiN9Fy2b7nM3/mT0920FrwHPQH2mc1nrTynPIO2uETNHiGsVyTcVaki+hc3ljn1z9DIrdP4bm4oxJe1WBND6ypC0I/FDPfdI0QikOexB1shyg/sP8ov9vKM+B1SXbBjBW3UrrqiB2tH5iaPLHsF46LR2gjOleyeL48FZuk8LxLolxYGN9+XByImJsV4fzD+zjoH6ipJslD0RM+EKamm+BYjDU7z4gN+Mwgjdi3Jwfp/JYexHH9+cDkGcbS0BmzIREhlehcy4k25ynmrtIAqjwZFBECcyR4JF0c3W2wcCWLZ3MQZju/dp9W8eRtT7JZh3+NbD8eiNnDmvPEkpEYi9rJfVSn6cHzYc8xJB8teOZpDanmd8FBbQ8eVvBMpfEefLAbJXi37HzcILdn4UkkczeWnnb0zqbtG0s0b+fZm/x28MSbhIMQNsmLp6JopjcHjdw8S3uUHRwg3npxfsWqoLqQjXmGuFyek3kKX//GTDFbedreUOI843QsRj7qt+JJcJLwmbC4KUoSMQmw4MfKc81CJEKkAovBT8XmJHcMXpMnibrgaX/fa1COoQ/+3ZyZH89kv53uPspZNER34+lIjHuaaLXxHIsYdJ1/3LCPEe+O90irgS/PEIu1udo8df/T3Fyi8sz75Asvm9kkL55sMPUFUDbCeXejbTF5pguRImFZymEkHnUhNnfgyqXfWjzLDIEnTxdNlWcgjHxy7csz4uaDz5zRWr2cxieCJ14tcq1eItFwXgTvr3y6oyWQZsGTlSd/xp483TQ1nhskbNLOk6cwxyNWUzH9MWeHWpMyX4exnrCLp9odtMgpUgtow1O0woNnFU2NZ6DapFY8ubFOeILswDrwUOVpqMwnb7HcTlYBVJm/qMlTLHw053nkPVmVZ7CSbVLYBU+RcByy3lbNk8Rl/0uZC0enCCQ9Dk+e5RjgCx9NeS7/ajT17Q4aT2EEQrzm4+2cPCMk5z65w5mkYjZFDZP3Ek98LQFlTldTntrcnQVSR3imoky0fTkvT0ISNFPSUn1WuTzBvCZq8R48t9vyA+vnTXnqNHuueJPzFG5fXvfwDDypPcpCgDhcj/SdKMzDJwceizZzljSeCzmIi+9b8qRB/lGewUjLOJ2WJ/WXttvR1BaUMxc+q92IckFNM80qz22JjwYNNXmaLxvxLY7HeQYzuYxT8zxirlmWBo9eIuXba0vlGUgzaJG4qsXTjCzLDaM1eAaHKJT1mTxZewmrkr4Ce1waz1HJr/AaavCsolmPZyo9xU/myb+W/T1UX2yRxjPYSya+X4NnNc16PCWbZPIs8/Nn4blQ9sfZtx9VSec5LpuWV/UIz2M0a/JUbRLlyXzBEsB5eA6kBvtsdNB5yh00GVTztLyKWDs/r/IM5Dww5clnHr7eunngCaWT8pRrghdHrrXI4DmVOuhK48nz8z0HTX+esk1CasYyuesH6f1MpE5OzHNTDpXEY6O0wVNpWbBz8XS8JuvNMy1dX1aMGCgRQnG52+jUPMtdDupOhJoyed6XDwhvp3aeJk0WtnvzFItfopixI+Q+NU8xQJs7S4GNZ/mAsgl5LJs7zvPV+FXF+WPT/SEGz9Im8WKGqlvKE8Cn5skHRtTcWQqsPKUZJBrKniFv6Fzrm/PHSeP9NiZPYQl4MYNQAkrQHSN2cp4LWqyHsxRYeYp1vzC0Odq6io22DXiSQrY17UNSnBLFDA7cCBGcbPMtB7kSzrOQWO+YInpA4knFeLJPNXhSR8JzV6iN58Yxc9l4shdq6vN8Z7tmbOmIWVigLregjvaoeLclXOWmdlecRgpPrPOM7TzZY7S+V6KKOWo+zlJg5ynWBo/yFK8n1eYZpEw169ffjUa7qj7V+Y8XsOkz9nurzMrTkc3WeUove9Xn+dXF3AovZylw8Axm1g6q8lRenfs5PHn39NwS+mHlObB2UJmn9iLij+HJvE+PzBLVOMEWnsHKtuxZ8jRe6/wxPA9tnKVc6Qphk2eaEDdPy8+JOXnq63FfXHzPEmnxHYMZjY4VD2Fh6aBqPqQOT+NHMb66WBCAr1t9S38dR7rHhc0O2pSn84cuv6xu+dvPbV/B3wxjovLcJi15fru+mWn4HudSNjl4anp4VyOC0OigTXh+R5rZ1EfVzZeN1bX7kdFB6/P8njRPrAe9g9blWf1DQhcrI6FbjyfQdGlPPHgCTad2qDHP5+Wn1fYb6ECa8gRV6T4Gnp1qjevy1N+DA9kkrYpX8jR+LRRk1620kcDJE2jW12af6Bs3NJ5As5m2ootaePaAZmMN1jGx8wSafpo+YAtPoOmv63zQKzyBZiv1XxCRePYm/4BmO+3C8vcugGYXmrGtQa/6S5qgVoJ/WwwCgUAgEAgEAoFAIBAIBAKBQCAQCAQCgUCgC9Z/qx3nis8RseIAAAAASUVORK5CYII="/>

Para mayor facilidad hemos creado una carpeta donde guardaremos copias de seguridad de los entornos que vayamos creando
a medida que completemos proyecto. Cada vez que instalemos una nueva librería o actualicemos la versión de las ya
existentes subiremos una copia de seguridad. La carpeta en cuestión
es [TFG-DeteccionFototrampeo/env](https://github.com/byLiTTo/TFG-DeteccionFototrampeo/tree/main/env).

___

# Ejecutar aplicación

En la creación de la aplicación hemos implementado varios notebooks, cada uno de ellos ejecuta cada una de las fases
que forman el proyecto global. Lo hemos realizado de esta manera para una mejor gestión de las nuevas funcionalidades
que se añadían a medida que se avanzaba el desarrollo.

Para realizar este proyecto, se nos ha facilitado un dataset con imágenes de foto trampeo de diferentes especies de
animales, humanos y vehículos, capturadas en los parajes
del [Parque natural de Doñana](https://es.wikipedia.org/wiki/Parque_nacional_y_natural_de_Doñana). La forma de manejar
los datos es a través de un fichero CSV, donde en una columna se nos indica la ruta del fichero correspondiente a la
imagen y en su segunda columna se nos dice a qué clase pertenece dicha imagen.

Como hemos mencionado anteriormente, para cada funcionalidad se ha intentado crear un notebook que la ejecute, por
algunos de ellos no necesitan ser ejecutados para obtener los resultados. A continuación, vamos a explicar la
funcionalidad de cada uno, así como el orden de ejecución.

___

# Notebooks principales:

## a01_GeneraDataset

En el dataset existen varias especies de animales, en nuestro proyecto, por el momento solo nos
interesa identificar la presencia de animales, por lo que solo nos serán necesarias dos clases _Animal_ o _Vacía_.
Además en nuestro caso queríamos trabajar con un dataset de valores binarios para las clases, lo que tuvimos que
realizar una conversión previa, en la que:

- Indicamos la nomenclatura de las clases originales (variable _ORIGINAL_CLASSES_).
- Nomenclaturas que formarán la nueva clase _ANIMAL_ (variable _CLASS_ANIMAL_).
- Nomenclaturas que formarán la nueva clase _VACÍA_ (variable _CLASS_EMPTY_).

Para hacer funcionar el notebook de forma correcta, deberemos indicar:

- Localización del fichero CSV de origen, indicada en la variable _dataset_csv_relative_.
- Localización donde se guardarán los ficheros CSV fraccionados, indicada en la variable _custom_csv_relative_.
- Número de muestras de la clase _VACÍA_, indicada en la variable _NUMBER_IMAGES_.
- Porcentaje de muestras de la clase _VACÍA_ para el CSV de entrenamiento, indicada en la variable _TRAIN_PERCENTAGE_.
- Porcentaje de muestras de la clase _VACÍA_ para el CSV de validación, indicada en la variable _VALIDATION_PERCENTAGE_.
- Porcentaje de muestras de la clase _VACÍA_ para el CSV de test, indicada en la variable _TEST_PERCENTAGE_.

Si lo deseamos podemos hacer una partición del número de muestras de la clase vacía, en nuestro caso hemos hecho una de
700 muestras y otra para el tamaño original de 10000 muestras.

El dataset quedará dividido en tres partes, _Train_, _Validation_ y _Test_. Por cada uno generamos un CSV con el mismo
formato que el original. Estos ficheros que guardarán por defecto en la
carpeta [TFG-DeteccionFototrampeo/data](https://github.com/byLiTTo/TFG-DeteccionFototrampeo/tree/main/data).

___
SIN ACTUALIZAR
___

## Notebook: a02_GeneraDetecciones

Para este notebook, tenemos dos modos de aportarle los datos de entrada.

La primera es indicarle la ruta de un fichero CSV, del que tomará las rutas de las imágenes de un dataset y la ruta raíz
de dicho dataset, en nuestro caso los ficheros CSV, por defecto se encuentran
en [data](https://github.com/byLiTTo/TFG-DeteccionFototrampeo/tree/main/data).

La segunda forma es indicando un directorio de entrada de donde tomará las imágenes a las que le aplicará el modelo de
detecciones entrenado de MegaDetector. (Por defecto hemos creado la
carpeta [input](https://github.com/byLiTTo/TFG-DeteccionFototrampeo/tree/main/input) para que tome las imágenes).

Una vez obtenidos los resultados, genera un fichero JSON con los datos de cada detección encontrada en cada una de las
imágenes. Se generan tanto un JSON global que contiene todos los resultados de la ejecución, así como un fichero JSON
por cada imagen de forma individual.

Todos estos resultados son guardados en la
carpeta [output_json](https://github.com/byLiTTo/TFG-DeteccionFototrampeo/tree/main/output_json), para los ficheros de
resultados globales, hemos creado una carpeta dentro de la anteriormente mencionada, a modo de historial, se trata
de [registry](https://github.com/byLiTTo/TFG-DeteccionFototrampeo/tree/main/output_json/registry).

### Notebook: b01_GeneraMascaras

Módulo creado para generar máscaras con las regiones de interés de las imágenes dadas. En este caso tomaremos como
regiones de interés los bounding boxes de las detecciones, por tanto, partiremos de los ficheros JSON generados en el
anterior notebook, que se encuentran en la
carpeta [output_json](https://github.com/byLiTTo/TFG-DeteccionFototrampeo/tree/main/output_json), por defecto.

Como salida tendremos imágenes binarias, las cuales han sido guardadas con valores 0 y 255. Estas imágenes las usaremos
como máscara, por defecto hemos asignado que se guarden en la
carpeta [output_mask](https://github.com/byLiTTo/TFG-DeteccionFototrampeo/tree/main/output_mask), con el nombre de la
imagen original añadiendo la terminación __mask_

#### Notebook: c01_AplicaMascaras

En este módulo combinaremos los resultados de los dos notebooks anteriores. Es decir, aplicaremos las máscaras a las
imágenes originales, con el fin de obtener imágenes con el fondo negro y la zona correspondiente a las detecciones,
manteniendo el valor de los píxeles originales.

Tomaremos como entrada un fichero JSON, por defecto tomamos como origen la
carpeta [output_json](https://github.com/byLiTTo/TFG-DeteccionFototrampeo/tree/main/output_json). De estos ficheros
obtenemos tanto la ruta al fichero de la imagen original, para tomar el valor de los píxeles, como los datos de las
coordenadas de los bounding boxes de las detecciones.

También necesitaremos una segunda entrada, que será la ruta al fichero de la máscara generada con el notebook
b01_GeneraMascaras, por lo que antes de ejecutar este módulo se ha debido de lanzar el anterior.

Por último, deberemos indicar la ruta destino donde serán guardados los resultados. En este caso por defecto hemos
asignado la carpeta [output_masked](https://github.com/byLiTTo/TFG-DeteccionFototrampeo/tree/main/output_masked), donde
se guardarán ficheros de imagen con las características anteriormente mencionadas.

### Notebook: b02_GeneraRecortes

En este módulo partiremos de los resultados de ejecutar el notebook _a01_GeneraDetecciones_. A partir de los ficheros
JSON, situaremos los bounding boxes en las fotos correspondientes. Generaremos nuevas imágenes con únicamente el
contenido de dichos recuadros, serán creados tantos ficheros como detecciones haya por imagen original.

La carpeta de origen será [output_json](https://github.com/byLiTTo/TFG-DeteccionFototrampeo/tree/main/output_json) y
como carpeta destino, por defecto hemos
asignado [output_crop](https://github.com/byLiTTo/TFG-DeteccionFototrampeo/tree/main/output_crop)

### Notebook: b03_GeneraBoundingBoxes

Módulo muy similar al anterior. En este caso, una vez calculados los bounding boxes, crearemos una copia de la imagen
original, a la que le hemos renderizado los BBoxes sobre ellas (estas copias han sido previamente redimensionadas).

El resultado final es una imagen con los bounding boxes de las detecciones, dibujados.

Una vez más la carpeta de origen de donde se tomarán los datos,
será [output_json](https://github.com/byLiTTo/TFG-DeteccionFototrampeo/tree/main/output_json) y como carpeta
destino [output_img](https://github.com/byLiTTo/TFG-DeteccionFototrampeo/tree/main/output_img)

## Notebook: a03_EntrenaClasificador

En este notebook cargamos los dataset generados hasta ahora, así como las imágenes con las máscaras aplicadas.

Nuestro clasificador de imágenes se tratará de un modelo basado en una Red Neuronal Convolucional (CNN), la cual
diseñaremos y entrenaremos en este fichero.



___