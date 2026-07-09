from flask import Flask, render_template, request
import os

# Create Flask application
app = Flask(__name__)

# Folder to store uploaded images
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Home Page
@app.route("/")
def home():
    return render_template("index.html")

# Upload Route
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No image uploaded!"

    image = request.files["image"]

    if image.filename == "":
        return "No image selected!"

    image_path = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)

    image.save(image_path)

    return f"Image '{image.filename}' uploaded successfully!"

# Run Application
if __name__ == "__main__":
    app.run(debug=True)