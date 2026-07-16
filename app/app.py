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


# ==========================================================
# Flask App
# ==========================================================

app = Flask(__name__)


# ==========================================================
# Upload Folder Configuration
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================================
# Home Page
# ==========================================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================================
# Dashboard
# ==========================================================

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ==========================================================
# Upload Page
# ==========================================================

@app.route("/upload")
def upload():
    return render_template("upload.html")


# ==========================================================
# Flood Map Page
# ==========================================================

@app.route("/map")
def map_view():
    """
    Displays the Flood Map page.

    At the moment this page only contains the UI.
    In future versions it can display
    GIS flood overlays and prediction history.
    """
    return render_template("map.html")


# ==========================================================
# Serve Uploaded Images
# ==========================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# ==========================================================
# Prediction Route
# ==========================================================

@app.route("/predict", methods=["POST"])
def predict_route():

    # --------------------------------------------
    # Check image exists
    # --------------------------------------------

    if "image" not in request.files:
        print("No image uploaded.")
        return redirect(url_for("upload"))

    file = request.files["image"]

    if file.filename == "":
        print("No file selected.")
        return redirect(url_for("upload"))

    # --------------------------------------------
    # Save uploaded image
    # --------------------------------------------

    filename = secure_filename(file.filename)

    image_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    file.save(image_path)

    print("\n==============================")
    print("Uploaded Image :", filename)
    print("Saved To       :", image_path)
    print("==============================")

    # --------------------------------------------
    # Run Prediction
    # --------------------------------------------

    try:

        prediction, confidence = predict(image_path)

    except Exception as e:

        print("\nPrediction Error")
        print(e)

        return f"""
        <h2>Prediction Failed</h2>
        <p>{e}</p>
        <a href="/upload">Go Back</a>
        """

    # --------------------------------------------
    # Timestamp
    # --------------------------------------------

    timestamp = datetime.now().strftime(
        "%d %b %Y %H:%M"
    )

    # --------------------------------------------
    # Display Result Page
    # --------------------------------------------

    return render_template(

        "result.html",

        prediction=prediction,

        confidence=confidence,

        image=filename,

        timestamp=timestamp

    )


# ==========================================================
# Error Pages
# ==========================================================

@app.errorhandler(404)
def page_not_found(error):

    return redirect(url_for("home"))


@app.errorhandler(500)
def internal_server_error(error):

    return """
    <h2>Internal Server Error</h2>
    <p>Please try again.</p>
    """, 500


# ==========================================================
# Run Flask
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="0.0.0.0",
        port=5000
    )