"""This module implements the ModelPredict class."""
import json
from typing import Type, Dict
from aniso8601 import parse_date, parse_datetime
from flask import request
from flask_restx import Resource, Model, fields, reqparse
from ml_rest_api.api.restx import api, FlaskApiReturnType, MLRestAPINotReadyException
from ml_rest_api.ml_trained_model.wrapper import trained_model_wrapper
from ml_rest_api.ml_trained_model.ml_trained_model import full_path
from werkzeug.datastructures import FileStorage

"""
This will be used to validate input and automatically generate the Swagger prototype.
"""
upload_parser = reqparse.RequestParser()
upload_parser.add_argument('file', location='files',
                           type=FileStorage, required=True)
upload_parser.add_argument('classifiers',
                           required=True, action='append', help="['battery', 'biological', 'cardboard', 'clothes', 'glass', 'metal', 'paper','plastic','shoes','trash']")


ns = api.namespace(  # pylint: disable=invalid-name
    "model",
    description="Methods supported by our ML model",
    validate=bool(trained_model_wrapper.sample()),
)


@ns.route("/predict")
class ModelPredict(Resource):
    """Implements the /model/predict POST method."""

    # @staticmethod
    @api.expect(upload_parser)
    @api.doc(
        responses={
            200: "Success",
            400: "Input Validation Error",
            500: "Internal Server Error",
            503: "Server Not Ready",
        }
    )
    def post(self):
        """
        Returns a prediction using the model.
        """
        # if not trained_model_wrapper.ready():
        #     raise MLRestAPINotReadyException()
        if 'file' not in request.files:
            return { 'error': 'No file part in request' }, 400

        if request.files['file'].filename == '':
            return { "message": 'No selected file'}, 400

        temp_file = full_path("temp/upload.jpg")
        file = request.files['file']
        if file.filename == '':
            return { 'error': 'No selected file' }, 400
        img_bytes = file.read()
        with open(temp_file, mode="wb") as jpg:
            jpg.write(img_bytes)
        
        args = upload_parser.parse_args()
        model_dict: Dict = {
            "image": temp_file,
            "classifiers": args['classifiers']
        }
        ret = trained_model_wrapper.run(model_dict)
        return ret, 200
