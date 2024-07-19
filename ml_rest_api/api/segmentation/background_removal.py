"""This module implements the SegmentationMasking class."""
from flask import request, send_file
from flask_restx import Resource,  reqparse
import PIL
from PIL import Image
import io
import os
from ml_rest_api.api.restx import api
from ml_rest_api.ml_trained_model.ml_trained_model import full_path
# from ml_rest_api.ml_trained_model.image_segmentation import init, run
from werkzeug.datastructures import FileStorage
import matplotlib.pyplot as plt
import tempfile
import cv2
import numpy as np

from rembg import remove

# Initialize the model
from ml_rest_api.ml_trained_model.ml_trained_model import init as model_init
model_init()


upload_parser = reqparse.RequestParser()
upload_parser.add_argument('file', location='files', type=FileStorage, required=True)

ns = api.namespace(
    "segmentation",
    description="Segmentation methods to detect objects in images.",
)

@ns.route("/background_removal")
class SegmentationBackgroundRemoval(Resource):
    """Implements the /segmentation/background_removal POST method."""

    @ns.expect(upload_parser)
    @ns.doc(
        responses={
            200: "Success",
            400: "Input Validation Error",
            500: "Internal Server Error",
        }
    )
    def post(self):
        """
        Returns image with background removed.
        """
        if 'file' not in request.files:
            return {'error': 'No file part in request'}, 400

        file = request.files['file']
        if file.filename == '':
            return {'error': 'No selected file'}, 400
        
        img_bytes = file.read()
        input_image = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

        # Use rembg to remove the background
        output_image = remove(input_image)

        # Save the result image temporarily
        result_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        result_file.close()

        output_image.save(result_file.name, format="PNG")

        # Ensure the correct filename with .png extension
        original_filename = file.filename
        base_filename = os.path.splitext(original_filename)[0]
        download_filename = f"bg_removed_{base_filename}.png"

        return send_file(result_file.name, mimetype='image/png', as_attachment=True, download_name=download_filename)
