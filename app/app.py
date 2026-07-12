from utils.predict import predict_flood
from flask import Flask, render_template, request, send_from_directory
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html")

        upload_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )

        file.save(upload_path)

        result = predict_flood(upload_path)

        return render_template(
            "result.html",
            image=file.filename,
            prediction=result["label"],
            confidence=result["confidence"]
        )

    return render_template("index.html")


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


if __name__ == "__main__":
    app.run(debug=True)