from geopy.geocoders import Nominatim
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps, ExifTags
import numpy as np
import json
from tensorflow.keras.models import model_from_json
from roboflow import Roboflow
import uuid
import cv2
from collections import Counter
import shutil
from ultralytics import YOLO
import os

# YOLO model path
YOLO_MODEL_PATH = 'models/best.pt'

# Load the trained YOLOv8 model
model = YOLO(YOLO_MODEL_PATH)

# Ensure the result and process directories exist
result_folder = 'results'
results_process_folder = 'results_process'
output_folder = 'output'
os.makedirs(result_folder, exist_ok=True)
os.makedirs(results_process_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Mango subcategory A
MANGO_SUBCATEGORY_Mango_A_COLOUR_MODEL_PATH = 'models/mango_colour_model.h5'
MANGO_SUBCATEGORY_Mango_A_COLOUR_MODEL_LABELS_PATH = 'models/mango_colour_model_labels.txt'
MANGO_SUBCATEGORY_Mango_A_SIZE_MODEL_PATH = 'models/mango_size_model.h5'
MANGO_SUBCATEGORY_Mango_A_SIZE_MODEL_LABELS_PATH = 'models/mango_size_model_labels.txt'
MANGO_SUBCATEGORY_Mango_A_SHAPE_MODEL_PATH = 'models/mango_shape_model.h5'
MANGO_SUBCATEGORY_Mango_A_SHAPE_MODEL_LABELS_PATH = 'models/mango_shape_model_labels.txt'
MANGO_SUBCATEGORY_Mango_A_SURFACE_MODEL_PATH = 'models/mango_surface_model.h5'
MANGO_SUBCATEGORY_Mango_A_SURFACE_MODEL_LABELS_PATH = 'models/mango_surface_model_labels.txt'

# Mango subcategory B
MANGO_SUBCATEGORY_Mango_B_COLOUR_MODEL_PATH = 'models/mango_colour_model.h5'
MANGO_SUBCATEGORY_Mango_B_COLOUR_MODEL_LABELS_PATH = 'models/mango_colour_model_labels.txt'
MANGO_SUBCATEGORY_Mango_B_SIZE_MODEL_PATH = 'models/mango_size_model.h5'
MANGO_SUBCATEGORY_Mango_B_SIZE_MODEL_LABELS_PATH = 'models/mango_size_model_labels.txt'
MANGO_SUBCATEGORY_Mango_B_SHAPE_MODEL_PATH = 'models/mango_shape_model.h5'
MANGO_SUBCATEGORY_Mango_B_SHAPE_MODEL_LABELS_PATH = 'models/mango_shape_model_labels.txt'
MANGO_SUBCATEGORY_Mango_B_SURFACE_MODEL_PATH = 'models/mango_surface_model.h5'
MANGO_SUBCATEGORY_Mango_B_SURFACE_MODEL_LABELS_PATH = 'models/mango_surface_model_labels.txt'

# Mango subcategory C
MANGO_SUBCATEGORY_Mango_C_COLOUR_MODEL_PATH = 'models/mango_colour_model.h5'
MANGO_SUBCATEGORY_Mango_C_COLOUR_MODEL_LABELS_PATH = 'models/mango_colour_model_labels.txt'
MANGO_SUBCATEGORY_Mango_C_SIZE_MODEL_PATH = 'models/mango_size_model.h5'
MANGO_SUBCATEGORY_Mango_C_SIZE_MODEL_LABELS_PATH = 'models/mango_size_model_labels.txt'
MANGO_SUBCATEGORY_Mango_C_SHAPE_MODEL_PATH = 'models/mango_shape_model.h5'
MANGO_SUBCATEGORY_Mango_C_SHAPE_MODEL_LABELS_PATH = 'models/mango_shape_model_labels.txt'
MANGO_SUBCATEGORY_Mango_C_SURFACE_MODEL_PATH = 'models/mango_surface_model.h5'
MANGO_SUBCATEGORY_Mango_C_SURFACE_MODEL_LABELS_PATH = 'models/mango_surface_model_labels.txt'

# Apple subcategory A
APPLE_SUBCATEGORY_Apple_A_COLOUR_MODEL_PATH = 'models/apple_colour_model.h5'
APPLE_SUBCATEGORY_Apple_A_COLOUR_MODEL_LABELS_PATH = 'models/apple_colour_model_labels.txt'
APPLE_SUBCATEGORY_Apple_A_SIZE_MODEL_PATH = 'models/apple_size_model.h5'
APPLE_SUBCATEGORY_Apple_A_SIZE_MODEL_LABELS_PATH = 'models/apple_size_model_labels.txt'
APPLE_SUBCATEGORY_Apple_A_SHAPE_MODEL_PATH = 'models/apple_shape_model.h5'
APPLE_SUBCATEGORY_Apple_A_SHAPE_MODEL_LABELS_PATH = 'models/apple_shape_model_labels.txt'

# Apple subcategory B
APPLE_SUBCATEGORY_Apple_B_COLOUR_MODEL_PATH = 'models/apple_colour_model.h5'
APPLE_SUBCATEGORY_Apple_B_COLOUR_MODEL_LABELS_PATH = 'models/apple_colour_model_labels.txt'
APPLE_SUBCATEGORY_Apple_B_SIZE_MODEL_PATH = 'models/apple_size_model.h5'
APPLE_SUBCATEGORY_Apple_B_SIZE_MODEL_LABELS_PATH = 'models/apple_size_model_labels.txt'
APPLE_SUBCATEGORY_Apple_B_SHAPE_MODEL_PATH = 'models/apple_shape_model.h5'
APPLE_SUBCATEGORY_Apple_B_SHAPE_MODEL_LABELS_PATH = 'models/apple_shape_model_labels.txt'

# Apple subcategory C
APPLE_SUBCATEGORY_Apple_C_COLOUR_MODEL_PATH = 'models/apple_colour_model.h5'
APPLE_SUBCATEGORY_Apple_C_COLOUR_MODEL_LABELS_PATH = 'models/apple_colour_model_labels.txt'
APPLE_SUBCATEGORY_Apple_C_SIZE_MODEL_PATH = 'models/apple_size_model.h5'
APPLE_SUBCATEGORY_Apple_C_SIZE_MODEL_LABELS_PATH = 'models/apple_size_model_labels.txt'
APPLE_SUBCATEGORY_Apple_C_SHAPE_MODEL_PATH = 'models/apple_shape_model.h5'
APPLE_SUBCATEGORY_Apple_C_SHAPE_MODEL_LABELS_PATH = 'models/apple_shape_model_labels.txt'

# Strawberry subcategory A
STRAWBERRY_SUBCATEGORY_Strawberry_A_COLOUR_MODEL_PATH = 'models/strawberry_colour_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_A_COLOUR_MODEL_LABELS_PATH = 'models/strawberry_colour_model_labels.txt'
STRAWBERRY_SUBCATEGORY_Strawberry_A_SIZE_MODEL_PATH = 'models/strawberry_size_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_A_SIZE_MODEL_LABELS_PATH = 'models/strawberry_size_model_labels.txt'
STRAWBERRY_SUBCATEGORY_Strawberry_A_SHAPE_MODEL_PATH = 'models/strawberry_shape_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_A_SHAPE_MODEL_LABELS_PATH = 'models/strawberry_shape_model_labels.txt'
STRAWBERRY_SUBCATEGORY_Strawberry_A_SURFACE_MODEL_PATH = 'models/strawberry_surface_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_A_SURFACE_MODEL_LABELS_PATH = 'models/strawberry_surface_model_labels.txt'

# Strawberry subcategory B
STRAWBERRY_SUBCATEGORY_Strawberry_B_COLOUR_MODEL_PATH = 'models/strawberry_colour_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_B_COLOUR_MODEL_LABELS_PATH = 'models/strawberry_colour_model_labels.txt'
STRAWBERRY_SUBCATEGORY_Strawberry_B_SIZE_MODEL_PATH = 'models/strawberry_size_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_B_SIZE_MODEL_LABELS_PATH = 'models/strawberry_size_model_labels.txt'
STRAWBERRY_SUBCATEGORY_Strawberry_B_SHAPE_MODEL_PATH = 'models/strawberry_shape_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_B_SHAPE_MODEL_LABELS_PATH = 'models/strawberry_shape_model_labels.txt'
STRAWBERRY_SUBCATEGORY_Strawberry_B_SURFACE_MODEL_PATH = 'models/strawberry_surface_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_B_SURFACE_MODEL_LABELS_PATH = 'models/strawberry_surface_model_labels.txt'

# Strawberry subcategory C
STRAWBERRY_SUBCATEGORY_Strawberry_C_COLOUR_MODEL_PATH = 'models/strawberry_colour_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_C_COLOUR_MODEL_LABELS_PATH = 'models/strawberry_colour_model_labels.txt'
STRAWBERRY_SUBCATEGORY_Strawberry_C_SIZE_MODEL_PATH = 'models/strawberry_size_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_C_SIZE_MODEL_LABELS_PATH = 'models/strawberry_size_model_labels.txt'
STRAWBERRY_SUBCATEGORY_Strawberry_C_SHAPE_MODEL_PATH = 'models/strawberry_shape_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_C_SHAPE_MODEL_LABELS_PATH = 'models/strawberry_shape_model_labels.txt'
STRAWBERRY_SUBCATEGORY_Strawberry_C_SURFACE_MODEL_PATH = 'models/strawberry_surface_model.h5'
STRAWBERRY_SUBCATEGORY_Strawberry_C_SURFACE_MODEL_LABELS_PATH = 'models/strawberry_surface_model_labels.txt'

# Anthurium subcategory A
ANTHURIUM_SUBCATEGORY_Anthurium_A_HEALTHY_UNHEALTHY_MODEL_PATH = 'models/healthyAnthurium.h5'
ANTHURIUM_SUBCATEGORY_Anthurium_A_HEALTHY_UNHEALTHY_MODEL_LABELS_PATH = 'models/healthyAnthurium_labels.txt'
ANTHURIUM_SUBCATEGORY_Anthurium_A_COLOUR_MODEL_PATH = 'models/colourAnthurium.h5'
ANTHURIUM_SUBCATEGORY_Anthurium_A_COLOUR_MODEL_LABELS_PATH = 'models/colourAnthurium_labels.txt'
ANTHURIUM_SUBCATEGORY_Anthurium_A_SIZE_MODEL_PATH = 'models/sizeAnthurium.h5'
ANTHURIUM_SUBCATEGORY_Anthurium_A_SIZE_MODEL_LABELS_PATH = 'models/sizeAnthurium_labels.txt'

# Anthurium subcategory B
ANTHURIUM_SUBCATEGORY_Anthurium_B_HEALTHY_UNHEALTHY_MODEL_PATH = 'models/healthyAnthurium.h5'
ANTHURIUM_SUBCATEGORY_Anthurium_B_HEALTHY_UNHEALTHY_MODEL_LABELS_PATH = 'models/healthyAnthurium_labels.txt'
ANTHURIUM_SUBCATEGORY_Anthurium_B_COLOUR_MODEL_PATH = 'models/colourAnthurium.h5'
ANTHURIUM_SUBCATEGORY_Anthurium_B_COLOUR_MODEL_LABELS_PATH = 'models/colourAnthurium_labels.txt'
ANTHURIUM_SUBCATEGORY_Anthurium_B_SIZE_MODEL_PATH = 'models/sizeAnthurium.h5'
ANTHURIUM_SUBCATEGORY_Anthurium_B_SIZE_MODEL_LABELS_PATH = 'models/sizeAnthurium_labels.txt'

# Anthurium subcategory C
ANTHURIUM_SUBCATEGORY_Anthurium_C_HEALTHY_UNHEALTHY_MODEL_PATH = 'models/healthyAnthurium.h5'
ANTHURIUM_SUBCATEGORY_Anthurium_C_HEALTHY_UNHEALTHY_MODEL_LABELS_PATH = 'models/healthyAnthurium_labels.txt'
ANTHURIUM_SUBCATEGORY_Anthurium_C_COLOUR_MODEL_PATH = 'models/colourAnthurium.h5'
ANTHURIUM_SUBCATEGORY_Anthurium_C_COLOUR_MODEL_LABELS_PATH = 'models/colourAnthurium_labels.txt'
ANTHURIUM_SUBCATEGORY_Anthurium_C_SIZE_MODEL_PATH = 'models/sizeAnthurium.h5'
ANTHURIUM_SUBCATEGORY_Anthurium_C_SIZE_MODEL_LABELS_PATH = 'models/sizeAnthurium_labels.txt'

# BellPepper
BELL_PEPPER_DISEASE_MODEL_PATH = 'models/bellpepper_disease_model.h5'
BELL_PEPPER_DISEASE_MODEL_LABELS_PATH = 'models/bellpepper_disease_model_labels.txt'
BELL_PEPPER_HEALTHY_UNHEALTHY_MODEL_PATH = 'models/bellpepper_healthy_unhealthy_model.h5'
BELL_PEPPER_HEALTHY_UNHEALTHY_MODEL_LABELS_PATH = 'models/bellpepper_healthy_unhealthy_labels.txt'
BELL_PEPPER_MAGNESIUM_MODEL_PATH = 'models/bellpepper_initial_model.h5'
BELL_PEPPER_MAGNESIUM_MODEL_LABELS_PATH = 'models/bellpepper_initial_model_labels.txt'
BELL_PEPPER_POWDERY_MODEL_PATH = 'models/bellpepper_initial_severe_model.h5'
BELL_PEPPER_POWDERY_MODEL_LABELS_PATH = 'models/bellpepper_initial_severe_model_labels.txt'

rf = Roboflow(api_key="zpJpqip0iN5TpMWJPaM2")
project = rf.workspace().project("pepper-segmentation")
model = project.version(1).model


# Utility function to sanitize subcategory
def sanitize_subcategory(subcategory):
    return subcategory.replace(" ", "_")


# General prediction method
def getPrediction(filename, model_path, labels_path):
    try:
        np.set_printoptions(suppress=True)
        model = load_model(model_path, compile=False)
        class_names = open(labels_path, "r").readlines()

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image_path = 'uploads/' + filename
        openImage = Image.open(image_path).convert("RGB")
        size = (224, 224)
        image = ImageOps.fit(openImage, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # Convert numpy float32 to Python float
        confidence_score = round(float(confidence_score), 2)

        print("Class:", class_name[2:], end="")
        print("Confidence Score:", confidence_score)
        return class_name.strip(), confidence_score
    except Exception as e:
        print(f"Error in getPrediction: {e}")
        raise


# Anthurium methods


def getAnthuriumHealthPrediction(filename, subcategory):
    try:
        print(f"Received request for Anthurium Health Prediction for {subcategory}")
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'ANTHURIUM_SUBCATEGORY_{subcategory}_HEALTHY_UNHEALTHY_MODEL_PATH')
        labels_path = eval(f'ANTHURIUM_SUBCATEGORY_{subcategory}_HEALTHY_UNHEALTHY_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


def getAnthuriumClPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'ANTHURIUM_SUBCATEGORY_{subcategory}_COLOUR_MODEL_PATH')
        labels_path = eval(f'ANTHURIUM_SUBCATEGORY_{subcategory}_COLOUR_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


def getAnthuriumSiPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'ANTHURIUM_SUBCATEGORY_{subcategory}_SIZE_MODEL_PATH')
        labels_path = eval(f'ANTHURIUM_SUBCATEGORY_{subcategory}_SIZE_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


# Apple methods
def getAppleClPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'APPLE_SUBCATEGORY_{subcategory}_COLOUR_MODEL_PATH')
        labels_path = eval(f'APPLE_SUBCATEGORY_{subcategory}_COLOUR_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


def getAppleShPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'APPLE_SUBCATEGORY_{subcategory}_SHAPE_MODEL_PATH')
        labels_path = eval(f'APPLE_SUBCATEGORY_{subcategory}_SHAPE_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


def getAppleSiPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'APPLE_SUBCATEGORY_{subcategory}_SIZE_MODEL_PATH')
        labels_path = eval(f'APPLE_SUBCATEGORY_{subcategory}_SIZE_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


# Strawberry methods
def getStrawberryClPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'STRAWBERRY_SUBCATEGORY_{subcategory}_COLOUR_MODEL_PATH')
        labels_path = eval(f'STRAWBERRY_SUBCATEGORY_{subcategory}_COLOUR_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


def getStrawberryShPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'STRAWBERRY_SUBCATEGORY_{subcategory}_SHAPE_MODEL_PATH')
        labels_path = eval(f'STRAWBERRY_SUBCATEGORY_{subcategory}_SHAPE_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


def getStrawberrySiPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'STRAWBERRY_SUBCATEGORY_{subcategory}_SIZE_MODEL_PATH')
        labels_path = eval(f'STRAWBERRY_SUBCATEGORY_{subcategory}_SIZE_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


def getStrawberrySurPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'STRAWBERRY_SUBCATEGORY_{subcategory}_SURFACE_MODEL_PATH')
        labels_path = eval(f'STRAWBERRY_SUBCATEGORY_{subcategory}_SURFACE_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


# Mango methods
def getMangoClPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'MANGO_SUBCATEGORY_{subcategory}_COLOUR_MODEL_PATH')
        labels_path = eval(f'MANGO_SUBCATEGORY_{subcategory}_COLOUR_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


def getMangoShPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'MANGO_SUBCATEGORY_{subcategory}_SHAPE_MODEL_PATH')
        labels_path = eval(f'MANGO_SUBCATEGORY_{subcategory}_SHAPE_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


def getMangoSiPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'MANGO_SUBCATEGORY_{subcategory}_SIZE_MODEL_PATH')
        labels_path = eval(f'MANGO_SUBCATEGORY_{subcategory}_SIZE_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


def getMangoSurPrediction(filename, subcategory):
    try:
        subcategory = sanitize_subcategory(subcategory)
        model_path = eval(f'MANGO_SUBCATEGORY_{subcategory}_SURFACE_MODEL_PATH')
        labels_path = eval(f'MANGO_SUBCATEGORY_{subcategory}_SURFACE_MODEL_LABELS_PATH')
        return getPrediction(filename, model_path, labels_path)
    except NameError:
        print(f"Model path for subcategory {subcategory} not found.")
        raise


# BellPepper methods
def getBellPepperDisPrediction(filename):
    try:
        return getPrediction(filename, BELL_PEPPER_DISEASE_MODEL_PATH, BELL_PEPPER_DISEASE_MODEL_LABELS_PATH)
    except NameError:
        print("Bell Pepper disease model path not found.")
        raise


def getBellPepperHealthPrediction(filename):
    try:
        return getPrediction(filename, BELL_PEPPER_HEALTHY_UNHEALTHY_MODEL_PATH,
                             BELL_PEPPER_HEALTHY_UNHEALTHY_MODEL_LABELS_PATH)
    except NameError:
        print("Bell Pepper health model path not found.")
        raise


def getBellPepperMagPrediction(filename):
    try:
        return getPrediction(filename, BELL_PEPPER_MAGNESIUM_MODEL_PATH, BELL_PEPPER_MAGNESIUM_MODEL_LABELS_PATH)
    except NameError:
        print("Bell Pepper magnesium model path not found.")
        raise


def getBellPepperPowPrediction(filename):
    try:
        return getPrediction(filename, BELL_PEPPER_POWDERY_MODEL_PATH, BELL_PEPPER_POWDERY_MODEL_LABELS_PATH)
    except NameError:
        print("Bell Pepper powdery model path not found.")
        raise


# ExifData class and metadata extraction functions
class ExifData:
    def __init__(self, data):
        self.GPSInfo = data.get('GPSInfo', 'unknown')
        self.Make = data.get('Make', 'unknown')
        self.Model = data.get('Model', 'unknown')
        self.DateTime = data.get('DateTime', 'unknown')
        self.XResolution = data.get('XResolution', 'unknown')
        self.YResolution = data.get('YResolution', 'unknown')
        self.ExifVersion = data.get('ExifVersion', 'unknown')
        self.ApertureValue = data.get('ApertureValue', 'unknown')
        self.BrightnessValue = data.get('BrightnessValue', 'unknown')
        self.FocalLength = data.get('FocalLength', 'unknown')
        self.DigitalZoomRatio = data.get('DigitalZoomRatio', 'unknown')
        self.ExposureTime = data.get('ExposureTime', 'unknown')
        self.Contrast = data.get('Contrast', 'unknown')
        self.ISOSpeedRatings = data.get('ISOSpeedRatings', 'unknown')
        self.Saturation = data.get('Saturation', 'unknown')
        self.LensSpecification = data.get('LensSpecification', 'unknown')
        self.Sharpness = data.get('Sharpness', 'unknown')

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"


def get_image_metadata(image_path):
    img = Image.open(image_path)
    exif_data = img._getexif() or {}

    def get_tag_value(tag_id):
        return exif_data.get(tag_id, 'unknown')

    tags = {ExifTags.TAGS[k]: k for k in ExifTags.TAGS.keys()}
    extracted_data = {}
    for tag in tags:
        if tag in [
            'GPSInfo', 'Make', 'Model', 'DateTime', 'XResolution', 'YResolution',
            'ExifVersion', 'ApertureValue', 'BrightnessValue', 'FocalLength',
            'DigitalZoomRatio', 'ExposureTime', 'Contrast',
            'ISOSpeedRatings', 'Saturation', 'LensSpecification', 'Sharpness'
        ]:
            value = get_tag_value(tags[tag])
            if value is None:
                value = 'unknown'
            extracted_data[tag] = value

    # Handle GPS Info
    lat, lon = None, None
    if isinstance(extracted_data.get('GPSInfo', 'unknown'), dict):
        gps_info = extracted_data['GPSInfo']
        if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
            lat = gps_info['GPSLatitude']
            lon = gps_info['GPSLongitude']
            lat_ref = gps_info.get('GPSLatitudeRef', 'N')
            lon_ref = gps_info.get('GPSLongitudeRef', 'W')
            lat = (lat[0] + lat[1] / 60 + lat[2] / 3600) * (-1 if lat_ref == 'S' else 1)
            lon = (lon[0] + lon[1] / 60 + lon[2] / 3600) * (-1 if lon_ref == 'W' else 1)

    # Handle Lens Specification
    if isinstance(extracted_data.get('LensSpecification', 'unknown'), tuple):
        extracted_data['LensSpecification'] = tuple(
            x if x is not None else 'unknown' for x in extracted_data['LensSpecification'])
    else:
        extracted_data['LensSpecification'] = 'unknown'

    # Extract DateTime and Area information
    exif_record = ExifData(extracted_data)
    Date = extracted_data.get('DateTime', 'unknown')

    if Date != 'unknown':
        try:
            DateWork, Time = Date.split(' ')
        except ValueError:
            DateWork = 'unknown'
            Time = 'unknown'
    else:
        DateWork = 'unknown'
        Time = 'unknown'

    if lat is not None and lon is not None:
        geoLoc = Nominatim(user_agent="GetLoc")
        locname = geoLoc.reverse((lat, lon))
        Area = locname.address
    else:
        Area = "Unknown"

    attributes = [
        'Make', 'Model', 'XResolution', 'YResolution',
        'ExifVersion', 'ApertureValue', 'BrightnessValue', 'FocalLength',
        'DigitalZoomRatio', 'ExposureTime', 'Contrast',
        'ISOSpeedRatings', 'Saturation', 'LensSpecification', 'Sharpness'
    ]

    # Prepare the output data
    output_data = []
    output_data.append(f"Date: {DateWork.replace(':', '-')}")
    output_data.append(f"Time: {Time}")
    output_data.append(f"Latitude: {lat:.6f}" if lat is not None else "Latitude: unknown")
    output_data.append(f"Longitude: {lon:.6f}" if lon is not None else "Longitude: unknown")
    output_data.append(f"Area: {Area}")

    for attr in attributes:
        value = getattr(exif_record, attr)
        if isinstance(value, tuple) or isinstance(value, dict):
            value = str(value)
        output_data.append(f"{attr}: {value}")

    return output_data


# Methods for Roboflow Predictions
def get_bellpepper_count(filename):
    try:
        image_path = 'uploads/' + filename
        response = model.predict(image_path, confidence=38, overlap=30).json()
        pepper_count = sum(1 for pred in response['predictions'] if pred['class'] == 'pepper')
        print("Number of Pepper Seeds \t" + str(pepper_count))
        return pepper_count
    except Exception as e:
        print(f"Error in get_pepper_count: {e}")
        raise


def get_papaw_count(filename):
    try:
        image_path = 'uploads/' + filename
        response = model.predict(image_path, confidence=38, overlap=30).json()
        papaw_count = sum(1 for pred in response['predictions'] if pred['class'] == 'papaw')
        print("Number of Papaw Seeds \t" + str(papaw_count))
        return papaw_count
    except Exception as e:
        print(f"Error in get_papaw_count: {e}")
        raise


# Function to process image for brightness and sharpness
def process_image(file_path, result_folder='results'):
    image = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(result_folder, 'original.png'), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(image_rgb, -1, sharpening_kernel)

    hsv_image = cv2.cvtColor(sharpened, cv2.COLOR_RGB2HSV)
    hsv_image[:, :, 2] = cv2.add(hsv_image[:, :, 2], 50)
    brightened = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    cv2.imwrite(os.path.join(result_folder, 'brightened.png'), cv2.cvtColor(brightened, cv2.COLOR_RGB2BGR))

    return os.path.join(result_folder, 'brightened.png')


# Function to zoom into an image and save in results_process
def zoom_image(file_path, result_folder='results_process'):
    image = cv2.imread(file_path)
    h, w, _ = image.shape
    x_center, y_center = w // 2, h // 2
    zoom_factor = 2

    x_start = x_center - (w // (2 * zoom_factor))
    x_end = x_center + (w // (2 * zoom_factor))
    y_start = y_center - (h // (2 * zoom_factor))
    y_end = y_center + (h // (2 * zoom_factor))

    zoomed_image = image[y_start:y_end, x_start:x_end]
    zoomed_image = cv2.resize(zoomed_image, (w, h), interpolation=cv2.INTER_LINEAR)

    zoomed_image_path = os.path.join(result_folder, 'zoomed_image.jpg')
    cv2.imwrite(zoomed_image_path, zoomed_image)

    return zoomed_image_path


# Function to generate unique colors for each class
def generate_colors(num_classes):
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    return colors


# Function to display image with bounding boxes and calculate pepper seed purity
def display_image_with_boxes(image, results, pepper_seed_class_id):
    num_classes = len(model.names)
    colors = generate_colors(num_classes)

    pepper_seed_count = 0
    total_count = 0

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for box, cls_id, confidence in zip(boxes, class_ids, confidences):
            x_min, y_min, x_max, y_max = box
            color = colors[cls_id].tolist()
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            label = f"{model.names[cls_id]}: {confidence:.2f}"
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            total_count += 1
            if cls_id == pepper_seed_class_id:
                pepper_seed_count += 1

    purity_percentage = (pepper_seed_count / total_count) * 100 if total_count > 0 else 0
    return image, purity_percentage, pepper_seed_count, total_count


# Helper function to find the pepper seed class id
def find_pepper_seed_class_id():
    pepper_seed_class_id = None
    for idx, class_name in model.names.items():
        if "pepper" in class_name.lower():
            pepper_seed_class_id = idx
            break
    return pepper_seed_class_id


# New function to process an image with object detection, adapted from additional logic
def process_image_with_object_detection(input_folder, output_folder, input_image, output_image, model):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the input image
    image_path = os.path.join(input_folder, input_image)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not read the image at {image_path}")
        return False

    # Perform object detection
    results = model(image)

    # Define output path
    output_path = os.path.join(output_folder, output_image)

    # Check if any objects were detected
    if len(results[0].boxes) > 0:
        # Generate image with bounding boxes
        output_image_with_boxes = display_image_with_boxes(image.copy(), results, find_pepper_seed_class_id())[0]

        # Save the output image with bounding boxes
        success = cv2.imwrite(output_path, output_image_with_boxes)
        if success:
            print(f"Image with object detection boxes saved as {output_path}")
        else:
            print(f"Error: Failed to save image to {output_path}")
            return False
    else:
        # If no objects detected, copy the original image to the output folder
        try:
            shutil.copy(image_path, output_path)
            print(f"No objects detected. Original image copied to {output_path}")
        except Exception as e:
            print(f"Error copying file: {e}")
            return False

    return True


# Function to process the uploaded image sequentially and save final image to output folder
def process_uploaded_image(file_path, result_folder='results'):
    # First process and brighten the image
    brightened_image_path = process_image(file_path, result_folder)

    # Zoom into the image and save it to the results_process folder
    zoomed_image_path = zoom_image(brightened_image_path, results_process_folder)

    # Process images in the results folder
    total_boxes_results, pepper_boxes_results, max_boxes_results, max_box_image_data_results = process_images_in_folder(
        result_folder, find_pepper_seed_class_id())

    # Find the image with the highest bounding boxes
    if max_box_image_data_results:
        filename, image, results = max_box_image_data_results
        image_with_boxes, purity, pepper_seed_count, total_count = display_image_with_boxes(image, results,
                                                                                            find_pepper_seed_class_id())
        # Save the final image with bounding boxes to the output folder
        output_path = os.path.join(output_folder, f'highest_boxes_{filename}')
        cv2.imwrite(output_path, image_with_boxes)
        return output_path, purity, pepper_seed_count, total_count
    else:
        # Use the new logic to process an image without changing the original flow
        success = process_image_with_object_detection(result_folder, output_folder, 'brightened.png',
                                                      'highest_boxes_image.png', model)
        return output_folder if success else None, 0, 0, 0


# Function to count number of objects in each image
def count_objects(results):
    class_ids = []
    for result in results:
        class_ids.extend(result.boxes.cls.cpu().numpy().astype(int))
    count = Counter(class_ids)
    return sum(count.values()), count


# Process images in a folder and return detailed stats
def process_images_in_folder(folder_path, pepper_seed_class_id):
    total_boxes = 0
    total_pepper_boxes = 0
    max_boxes_in_single_image = 0
    max_box_image_data = None

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        results = model(image)
        num_boxes, class_counts = count_objects(results)

        pepper_count = class_counts.get(pepper_seed_class_id, 0)

        total_boxes += num_boxes
        total_pepper_boxes += pepper_count

        if num_boxes > max_boxes_in_single_image:
            max_boxes_in_single_image = num_boxes
            max_box_image_data = (filename, image.copy(), results)

    return total_boxes, total_pepper_boxes, max_boxes_in_single_image, max_box_image_data
