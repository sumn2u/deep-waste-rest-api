from flask_restx import Resource
from ml_rest_api.api.restx import api, FlaskApiReturnType
import json
import os
from logging import getLogger

log = getLogger(__name__)

# Function to load questions from the JSON file
def load_questions():
    """Load questions from the JSON file."""
    file_path = os.path.join(os.path.dirname(__file__), 'questions.json')
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        log.exception("Error loading questions")
        raise RuntimeError("Failed to load questions") from e

@api.default_namespace.route("/questions")
class Questions(Resource):
    """Implements the /questions GET method."""

    @staticmethod
    @api.doc(
        responses={
            200: "Success",
            500: "Failed to load questions",
        }
    )
    def get() -> FlaskApiReturnType:
        """
        Returns the list of questions from the JSON file
        """
        try:
            questions_data = load_questions()
            return questions_data, 200
        except RuntimeError:
            return {"message": "Failed to load questions"}, 500

@api.default_namespace.route("/questions/<string:question_id>")
class Question(Resource):
    """Implements the /questions/<id> GET method for a specific question."""

    @staticmethod
    @api.doc(
        responses={
            200: "Success",
            404: "Question Not Found",
            500: "Failed to load questions",
        }
    )
    def get(question_id: str) -> FlaskApiReturnType:
        """
        Returns a specific question by its ID.
        """
        try:
            questions_data = load_questions()
            # Find the question by id
            question = next((q for q in questions_data["questions"] if q["id"] == question_id), None)
            
            if question is not None:
                return question, 200
            else:
                return {"message": "Question not found"}, 404
        except RuntimeError:
            return {"message": "Failed to load questions"}, 500