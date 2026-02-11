"""
Módulo integrador del sistema de detección de neumonía.
Integra lectura de imagen, preprocesamiento, modelo y Grad-CAM.
"""

from src.read_img import read_dicom_file, read_jpg_file
from src.preprocess_img import preprocess_image
from src.load_model import load_cnn_model
from src.grad_cam import generate_grad_cam


def run_inference(image_path: str, model_path: str):
    """
    Ejecuta el pipeline completo de inferencia.

    :param image_path: Ruta de la imagen (DICOM o JPG)
    :param model_path: Ruta del modelo .h5
    :return: clase_predicha, probabilidad, imagen_grad_cam
    """

    # 1. Leer imagen
    if image_path.lower().endswith(".dcm"):
        img_array, _ = read_dicom_file(image_path)
    else:
        img_array, _ = read_jpg_file(image_path)

    # 2. Preprocesar
    input_tensor = preprocess_image(img_array)

    # 3. Cargar modelo
    model = load_cnn_model(model_path)

    # 4. Predicción
    predictions = model.predict(input_tensor)
    class_index = predictions.argmax()
    probability = predictions.max() * 100

    classes = ["bacteriana", "normal", "viral"]
    predicted_class = classes[class_index]

    # 5. Grad-CAM
    heatmap_img = generate_grad_cam(model, img_array, class_index)

    return predicted_class, probability, heatmap_img
