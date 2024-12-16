# coding: utf-8
"""Module that does all the ML trained model prediction heavy lifting."""
from logging import Logger, getLogger
from datetime import datetime, date
from os.path import normpath, join, dirname
from typing import Any, Iterable, Dict
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models  import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.mobilenet import preprocess_input
import ast
# import joblib

log: Logger = getLogger(__name__)


def full_path(filename: str) -> str:
    """Returns the full normalised path of a file in the same folder as this module."""
    return normpath(join(dirname(__file__), filename))


MODEL: Any = None
MODEL_SERVING_FUNCTION: Any = None

def init() -> None:
    """Loads the ML trained model (plus ancillary files) from file."""
    from time import sleep  # pylint: disable=import-outside-toplevel

    model_path = full_path("mobilenetv2_skin_classification_model.keras")
    log.debug("Initialise model from file %s", model_path)
    sleep(5)  # Fake delay to emulate a large model that takes a long time to load
    
    # feature_selector) from pickle file(s):
    global MODEL, MODEL_SERVING_FUNCTION
    MODEL = load_model(model_path)
    try:
        MODEL = load_model(model_path)
        # MODEL_SERVING_FUNCTION = MODEL.signatures.get('serving_default', None)
        log.info("Model loaded successfully.")
    except FileNotFoundError:
        log.error("Model file not found at %s", model_path)
        raise
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        raise

def run(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Makes a prediction using the trained ML model."""
    log.info("input_data:%s", input_data)

    # Ensure MODEL is loaded
    if MODEL is None:
        raise ValueError("Model is not loaded. Please call init() first.")

    # Load and preprocess the image
    img_path = input_data['image']
    img = load_img(img_path, target_size=(224, 224))
    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)

    # Use the predict method of the Keras model
    model_pred = MODEL.predict(img_array)[0]

    classification_types = ast.literal_eval(input_data['classifiers'][0])
    index = np.argmax(model_pred)
    label = classification_types[index]
    accuracy = "{0:.2f}".format(model_pred[index] * 100)
    
    return {"accuracy": accuracy, "label": label}


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
