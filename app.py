from flask import Flask, jsonify, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import shutil
import requests
import os

app = Flask(__name__)

# Define the API endpoint
API_URL = "https://brain-mri-function-kbdckfaerq-uc.a.run.app/predict"
UPLOAD_FOLDER = "uploads"  # Folder where uploaded files will be stored
PREDICTED_FOLDER = "predicted"  # Folder where predicted files will be stored

# Ensure the upload and predicted folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTED_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PREDICTED_FOLDER"] = PREDICTED_FOLDER


# Home route to render index.html
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"})

    if file and file.filename.endswith((".tif")):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        # Call the API
        files = {"file": (filename, open(file_path, "rb"))}

        response = requests.post(API_URL, files=files)

        # Close the file handle explicitly
        files["file"][1].close()

        # Save the predicted image as .png
        predicted_filename = f"{filename[:-4]}_predicted.png"
        predicted_file_path = os.path.join(
            app.config["PREDICTED_FOLDER"], predicted_filename
        )
        with open(predicted_file_path, "wb") as f:
            f.write(response.content)

        # Convert original .tif file to .png
        img = Image.open(file_path)
        png_filename = f"{filename[:-4]}.png"
        png_file_path = os.path.join(app.config["UPLOAD_FOLDER"], png_filename)
        img.save(png_file_path)

        # Remove the original .tif file
        os.remove(file_path)

        # Return JSON response with filenames
        return jsonify(
            {"uploaded_image": png_filename, "predicted_image": predicted_filename}
        )

    return jsonify({"error": "File format not supported (must be .tif)"})


# Route to serve uploaded images
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# Route to serve predicted images
@app.route("/predicted/<filename>")
def predicted_file(filename):
    return send_from_directory(app.config["PREDICTED_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
