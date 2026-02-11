"""
Generación del mapa de calor Grad-CAM.
"""

import cv2
import numpy as np
import tensorflow as tf
from keras import backend as K



def generate_grad_cam(model, image_array, class_index):
    """
    Genera una imagen Grad-CAM superpuesta.

    :param model: Modelo CNN cargado
    :param image_array: Imagen original (RGB)
    :param class_index: Índice de la clase predicha
    :return: Imagen RGB con Grad-CAM
    """

    img_resized = cv2.resize(image_array, (512, 512))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_gray = img_gray / 255.0

    img_input = np.expand_dims(img_gray, axis=(0, -1))

    # Última capa convolucional (ajusta el nombre si cambia)
    last_conv_layer = model.get_layer("conv10_thisone")

    # Asegurar que class_index sea int normal
    class_index = int(class_index)

    # Obtener salida correcta del modelo
    if isinstance(model.output, list):
        output_tensor = model.output[0]
    else:
        output_tensor = model.output

    output = output_tensor[:, class_index]


    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    iterate = K.function(
        [model.input],
        [pooled_grads, last_conv_layer.output[0]]
    )

    pooled_grads_value, conv_output = iterate(img_input)

    for i in range(conv_output.shape[-1]):
        conv_output[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    heatmap = cv2.resize(heatmap, (512, 512))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed = cv2.addWeighted(
        img_resized, 0.6, heatmap, 0.4, 0
    )

    return superimposed
