# coding: utf-8
"""Module that does all the ML trained model prediction heavy lifting."""
from logging import Logger, getLogger
from datetime import datetime, date
from os.path import normpath, join, dirname
from typing import Any, Iterable, Dict

import numpy as np
import traceback
import pandas as pd
import os
import tensorflow as tf
from keras import layers
from tensorflow.keras.models  import load_model

# from tensorflow.keras.models  import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

import ast
# import joblib

log: Logger = getLogger(__name__)


def full_path(filename: str) -> str:
    """Returns the full normalised path of a file in the same folder as this module."""
    return normpath(join(dirname(__file__), filename))


MODEL: Any = None
# MODEL_SERVING_FUNCTION: Any = None

def init() -> None:
    """Loads the ML trained model (plus ancillary files) from file."""
    global MODEL

    from time import sleep  # pylint: disable=import-outside-toplevel

    model_path = full_path("garbage_model")
    log.debug("Initialise model from file %s", model_path)
    sleep(5)  # Fake delay to emulate a large model that takes a long time to load
    
    if not os.path.exists(model_path):
        log.error(f"Model folder not found: {model_path}")
        raise FileNotFoundError(f"Model folder not found: {model_path}")

    try:
        MODEL = layers.TFSMLayer(model_path, call_endpoint="serving_default")  # Load from folder
        log.info("Model loaded successfully.")
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        raise

def run(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Makes a prediction using the trained ML model."""
    log.info("Received input_data: %s", input_data)

    if MODEL is None:
        raise ValueError("Model is not loaded. Please call init() first.")

    # Load and preprocess the image
    img_path = input_data['image']
    img = load_img(img_path, target_size=(224, 224))  # Adjust size if needed
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Use EfficientNetV2's preprocessing

   
    # Run inference using TFSMLayer
    predictions_dict = MODEL(img_array)

    # 🔍 Extract predictions from the correct key
    if "output_0" in predictions_dict:
        predictions = predictions_dict["output_0"].numpy()[0]
    else:
        raise KeyError(f"Unexpected model output keys: {predictions_dict.keys()}")

    print(f'predictions: {predictions}')

    # Extract label and accuracy
    waste_types = ast.literal_eval(input_data['classifiers'][0]) # Expecting a list of class names
    index = np.argmax(predictions)
    waste_label = waste_types[index]
    accuracy = "{0:.2f}".format(predictions[index] * 100)

    return {"accuracy": accuracy, "label": waste_label}


def sample() -> Dict:
    """Returns a sample input vector as a dictionary."""
    return {
        "int_param": 10,
        "string_param": "foobar",
        "float_param": 0.1,
        "bool_param": True,
        "datetime_param": datetime.now().isoformat() + "Z",
        "date_param": date.today().isoformat(),
    }


if __name__ == "__main__":
    init()
    print(sample())
    print(run(sample()))
