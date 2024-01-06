# coding: utf-8
"""Module that does all the ML trained model prediction heavy lifting."""
from logging import Logger, getLogger
from datetime import datetime, date
from os.path import normpath, join, dirname
from typing import Any, Iterable, Dict
import numpy as np
import pandas as pd
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


def init() -> None:
    """Loads the ML trained model (plus ancillary files) from file."""
    from time import sleep  # pylint: disable=import-outside-toplevel

    log.debug("Initialise model from file %s", full_path("model.pkl"))
    sleep(5)  # Fake delay to emulate a large model that takes a long time to load

    # deserialise the ML model (and possibly other objects such as feature_list,
    # feature_selector) from pickle file(s):
    global MODEL
    # MODEL = load_model(full_path('hack_mobilenet.h5'))

def run(input_data: Iterable) -> Dict:
    """Makes a prediction using the trained ML model."""
    log.info("input_data:%s", input_data)

    MODEL = load_model(full_path('model.h5'))
    img = load_img(input_data['image'], target_size=(224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    waste_types = ast.literal_eval(input_data['classifiers'][0])
    waste_pred = MODEL.predict(img)[0]
    index = np.argmax(waste_pred)
    waste_label = waste_types[index]
    accuracy = "{0:.2f}".format(waste_pred[index] * 100)
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
