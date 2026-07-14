from flask import Flask, render_template
import os

from flask import (
    Flask,
    render_template,
    request
)
from datetime import datetime
timestamp = datetime.now().strftime("%d %b %Y %H:%M")

from werkzeug.utils import secure_filename
from utils.predict import predict
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "uploads"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/upload")
def upload():
    return render_template("upload.html")


@app.route("/map")
def map_view():
    return render_template("map.html")

@app.route("/predict", methods=["POST"])
def predict_route():

    if "image" not in request.files:
        return "No image uploaded."

    file = request.files["image"]

    if file.filename == "":
        return "No file selected."

    filename = secure_filename(file.filename)

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(image_path)

    prediction, confidence = predict(image_path)

    return render_template(
    "map.html",
    prediction=prediction,
    confidence=confidence,
    uploaded_image=filename,
    timestamp=timestamp
)


if __name__ == "__main__":
    app.run(debug=True)