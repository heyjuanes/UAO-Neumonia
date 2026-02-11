"""
Módulo para preprocesamiento de imágenes médicas.
"""

import cv2
import numpy as np


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Realiza el preprocesamiento requerido:
    - Resize a 512x512
    - Conversión a escala de grises
    - Ecualización con CLAHE
    - Normalización entre 0 y 1
    - Conversión a batch (tensor 4D)

    :param img: Imagen en formato numpy array
    :return: Imagen preprocesada lista para el modelo
    """

    # 1️⃣ Resize
    img_resized = cv2.resize(img, (512, 512))

    # 2️⃣ Convertir a escala de grises
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    # 3️⃣ CLAHE (ecualización adaptativa)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_gray)

    # 4️⃣ Normalizar entre 0 y 1
    img_normalized = img_clahe / 255.0

    # 5️⃣ Expandir dimensiones (batch y canal)
    img_batch = np.expand_dims(img_normalized, axis=-1)  # canal
    img_batch = np.expand_dims(img_batch, axis=0)        # batch

    return img_batch
