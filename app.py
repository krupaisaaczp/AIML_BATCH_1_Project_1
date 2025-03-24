from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
import joblib
import numpy as np
from extract_pdf_features import extract_features
from flask_cors import CORS

app = Flask(__name__, static_folder="../aiml_project_webpage/build", static_url_path="")
CORS(app)

# Load models
model = joblib.load("models/best_model.pkl")
scaler = joblib.load("models/scaler.pkl")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Extract features
    features = extract_features(filepath)
    features = scaler.transform([features])

    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0].tolist()

    return jsonify({"prediction": "Malicious" if prediction == 1 else "Benign", "probability": probability})

# Serve React Frontend
@app.route("/")
def serve():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

if __name__ == "__main__":
    app.run(debug=True)
