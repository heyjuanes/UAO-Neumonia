"""
Módulo para cargar el modelo de neumonía entrenado (.h5).
"""

from tensorflow.keras.models import load_model # type: ignore

_MODEL = None


def load_cnn_model(model_path: str):
    """
    Carga el modelo CNN desde un archivo .h5.
    El modelo se carga una sola vez (singleton).
    """
    global _MODEL

    if _MODEL is None:
        _MODEL = load_model(model_path, compile=False)

    return _MODEL
