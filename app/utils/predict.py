import cv2
import numpy as np
import tensorflow as tf
import os

# --------------------------------------------------
# Load models only once
# --------------------------------------------------

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "model"
)

encoder = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "encoder.keras"),
    compile=False
)

transformer = tf.keras.models.load_model(
    os.path.join(MODEL_DIR, "simple_transformer_final.keras"),
    compile=False
)

# --------------------------------------------------
# Constants
# --------------------------------------------------

IMG_SIZE = 128

NUM_TOKENS = 32 * 32

FEATURE_DIM = 16

# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------

def preprocess_image(image_path):

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = image.astype(np.float32)

    image /= 255.0

    image = np.expand_dims(
        image,
        axis=-1
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    image = np.nan_to_num(image)

    return image

# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict(image_path):

    image = preprocess_image(image_path)

    features = encoder.predict(
        image,
        verbose=0
    )

    tokens = features.reshape(
        features.shape[0],
        NUM_TOKENS,
        FEATURE_DIM
    )

    probability = transformer.predict(
        tokens,
        verbose=0
    )[0][0]

    prediction = "Flood"

    if probability < 0.5:
        prediction = "No Flood"

    confidence = probability

    if probability < 0.5:
        confidence = 1 - probability

    confidence = round(
        float(confidence) * 100,
        2
    )

    return prediction, confidence