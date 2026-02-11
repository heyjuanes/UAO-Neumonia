"""
Módulo para cargar el modelo de neumonía entrenado (.h5).
"""
#----
import tensorflow as tf
from keras import backend as K
from keras import load_model
#----
_MODEL = None

def load_cnn_model(model_path: str):
    """
    Carga el modelo CNN desde un archivo .h5.
    El modelo se carga una sola vez (singleton).
    
    :param model_path: Ruta al archivo .h5
    :return: Modelo Keras cargado
    """
    global _MODEL

    if _MODEL is None:
        _MODEL = load_model(model_path)
    return _MODEL