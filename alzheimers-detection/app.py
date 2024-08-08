from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route("/predict", methods=["POST"])
def predict():
    ethnicity = request.form["ethnicity"]
    diabetes = request.form["diabetes"]
    cholesterol_HDL = request.form["cholesterol_HDL"]
    MMSE = request.form["MMSE"]
    functional_assessment = request.form["functional_assessment"]
    memory_complaints = request.form["memory_complaints"]
    behavioral_problems = request.form["behavioral_problems"]
    ADL = request.form["ADL"]
    file = request.files["mri_image"]

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Here you would call your AI model with the clinical info and file path
        # For example:
        # result = your_ai_model(education_level, sleep_quality, cholesterol_HDL, MMSE, functional_assessment, memory_complaints, behavioral_problems, ADL, file_path)

        # Placeholder response
        result = {"prediction": "positive", "confidence": 0.85}

        return jsonify(result)
    else:
        return jsonify({"error": "No file uploaded"}), 400


if __name__ == "__main__":
    app.run(debug=True)
