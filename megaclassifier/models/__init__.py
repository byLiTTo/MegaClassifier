import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(project_root, "yolov5"))

from .detection import *
