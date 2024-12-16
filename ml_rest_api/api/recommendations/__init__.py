from flask_restx import Resource
from ml_rest_api.api.restx import api, FlaskApiReturnType
import json
import os
from logging import getLogger

log = getLogger(__name__)

# Function to load recommendations from the JSON file
def load_recommendations():
    """Load recommendations from the JSON file."""
    file_path = os.path.join(os.path.dirname(__file__), 'recommendations.json')  # Adjust path if needed
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        log.exception("Error loading recommendations")
        raise RuntimeError("Failed to load recommendations") from e

@api.default_namespace.route("/recommendations")
class Recommendations(Resource):
    """Implements the /recommendations GET method for all recommendations."""

    @staticmethod
    @api.doc(
        responses={
            200: "Success",
            500: "Failed to load recommendations",
        }
    )
    def get() -> FlaskApiReturnType:
        """
        Returns the list of all recommendations from the JSON file
        """
        try:
            recommendations_data = load_recommendations()
            return recommendations_data, 200
        except RuntimeError:
            return {"message": "Failed to load recommendations"}, 500


@api.default_namespace.route("/recommendations/score/<int:score>")
class RecommendationByScore(Resource):
    """Implements the /recommendations/score/<score> GET method for recommendations based on score."""

    @staticmethod
    @api.doc(
        responses={
            200: "Success",
            404: "No recommendations found for this score range",
            500: "Failed to load recommendations",
        }
    )
    def get(score: int) -> FlaskApiReturnType:
        """
        Returns recommendations based on the score.
        """
        try:
            recommendations_data = load_recommendations()
            # Find recommendations that match the score range
            matched_recommendations = [
                rec for rec in recommendations_data["recommendations"]
                if rec["scoreRange"][0] <= score <= rec["scoreRange"][1]
            ]
            
            if matched_recommendations:
                return {"recommendations": matched_recommendations}, 200
            else:
                return {"message": "No recommendations found for this score range"}, 404
        except RuntimeError:
            return {"message": "Failed to load recommendations"}, 500
