from flask import Flask, request, jsonify
from flask_cors import CORS
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import necessary functions and models
# from model2 import predict_alzheimers  # Import the predict_alzheimers function

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route("/predict", methods=["POST"])
def predict():
    # Collect clinical data from the form
    user_input = {
        "Ethnicity": int(request.form["ethnicity"]),
        "Diabetes": int(request.form["diabetes"]),
        "CholesterolHDL": float(request.form["cholesterol_HDL"]),
        "MMSE": int(request.form["MMSE"]),
        "FunctionalAssessment": int(request.form["functional_assessment"]),
        "MemoryComplaints": int(request.form["memory_complaints"]),
        "BehavioralProblems": int(request.form["behavioral_problems"]),
        "ADL": int(request.form["ADL"]),
    }

    # Get the MRI image from the form
    file = request.files["mri_image"]

    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        # Call the AI model to get the prediction and confidence score
        # prediction, confidence = predict_alzheimers(user_input, file_path)

        result = {
            "prediction": "positive",
            "confidence": 0.75,
        }

        return jsonify(result)
    else:
        return jsonify({"error": "No file uploaded"}), 400


if __name__ == "__main__":
    app.run(debug=True)
