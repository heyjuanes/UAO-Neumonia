import cv2
import numpy as np
from PIL import Image
import pydicom


def read_dicom_file(path: str):
    """
    Lee imagen DICOM y la convierte a formato RGB numpy.
    """
    dicom_data = pydicom.dcmread(path)
    img_array = dicom_data.pixel_array

    img2show = Image.fromarray(img_array)

    img_array = img_array.astype(float)
    img_array = (np.maximum(img_array, 0) / img_array.max()) * 255.0
    img_array = np.uint8(img_array)

    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

    return img_rgb, img2show


def read_jpg_file(path: str):
    """
    Lee imagen JPG/PNG y la convierte a numpy array.
    """
    img = cv2.imread(path)
    img_array = np.asarray(img)

    img2show = Image.fromarray(img_array)

    img_array = img_array.astype(float)
    img_array = (np.maximum(img_array, 0) / img_array.max()) * 255.0
    img_array = np.uint8(img_array)

    return img_array, img2show
