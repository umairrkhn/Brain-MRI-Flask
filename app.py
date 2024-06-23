from flask import Flask, jsonify, request, render_template, send_file
from werkzeug.utils import secure_filename
import numpy as np
from keras.models import load_model
from keras import backend as K
from PIL import Image
import os

app = Flask(__name__)

# Configure upload and result directories
UPLOAD_FOLDER = "static/uploads/"
RESULT_FOLDER = "static/results/"
MODEL_FOLDER = "models/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER


# function to create dice coefficient
def dice_coef(y_true, y_pred, smooth=100):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)

    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)


# function to create dice loss
def dice_loss(y_true, y_pred, smooth=100):
    return -dice_coef(y_true, y_pred, smooth)


# function to create iou coefficient
def iou_coef(y_true, y_pred, smooth=100):
    intersection = K.sum(y_true * y_pred)
    sum = K.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum - intersection + smooth)
    return iou


custom_objects = {"dice_coef": dice_coef, "dice_loss": dice_loss, "iou_coef": iou_coef}

# Load the pre-trained model
model = load_model(
    os.path.join(MODEL_FOLDER, "brain_mri_seg.h5"), custom_objects=custom_objects
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file part"

    file = request.files["file"]
    if file.filename == "":
        return "No selected file"

    if file and file.filename.endswith(".tif"):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # Process the uploaded image
        uploaded_img = Image.open(filepath).convert("RGB")  # Convert to RGB
        uploaded_img.save(
            os.path.join(app.config["UPLOAD_FOLDER"], filename.replace(".tif", ".png"))
        )

        # Resize and preprocess the image for prediction
        image = uploaded_img.resize((256, 256))
        image = np.array(image)
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)

        # Predict the mask
        pred_mask = model.predict(image)
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

        # Save the predicted mask as .png
        result_filename = "pred_" + filename.replace(".tif", ".png")
        result_filepath = os.path.join(app.config["RESULT_FOLDER"], result_filename)
        pred_mask_img = Image.fromarray(pred_mask.squeeze(), mode="L")
        pred_mask_img.save(result_filepath)

        response = {
            "uploaded_image": f"/{UPLOAD_FOLDER}{filename.replace('.tif', '.png')}",
            "predicted_image": f"/{result_filepath}",
        }

        return jsonify(response)

    return "Invalid file format. Please upload a TIFF image."


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))