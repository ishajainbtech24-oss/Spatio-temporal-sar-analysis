from flask import (
    Flask,
    render_template,
    request,
    send_from_directory,
    redirect,
    url_for
)
import os
from datetime import datetime
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
    # No prediction data yet — send the user back to the upload page
    # instead of rendering map.html with missing variables.
    return redirect(url_for("upload"))


# NEW: serves the actual uploaded image file back to the browser.
# This is the endpoint map.html's url_for('uploaded_file', ...) needs.
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/predict", methods=["POST"])
def predict_flood():

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

    # Compute timestamp per-request (not once at import time)
    timestamp = datetime.now().strftime("%d %b %Y %H:%M")

    return render_template(
        "map.html",
        prediction=prediction,
        confidence=confidence,
        image=filename,       # renamed from uploaded_image to match map.html's {{ image }}
        timestamp=timestamp
    )


if __name__ == "__main__":
    app.run(debug=True)