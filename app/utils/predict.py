import os
import tensorflow as tf
from tensorflow.keras import layers
import cv2
import numpy as np


# -------------------------------------------------
# Custom Position Embedding Layer
# -------------------------------------------------
@tf.keras.utils.register_keras_serializable()
class PositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, embed_dim, **kwargs):
        super().__init__(**kwargs)

        self.sequence_length = sequence_length
        self.embed_dim = embed_dim

        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=sequence_length,
            output_dim=embed_dim
        )

    def call(self, inputs):
        positions = tf.range(
            start=0,
            limit=tf.shape(inputs)[1],
            delta=1
        )

        embedded_positions = self.position_embedding(positions)
        return inputs + embedded_positions

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "embed_dim": self.embed_dim,
        })
        return config

# ---------------------------------------------------
# Path to the trained Transformer model
# ---------------------------------------------------

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "model",
    "simple_transformer_final.keras"
)

# ---------------------------------------------------
# Load the model only once
# ---------------------------------------------------

try:
    model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "PositionEmbedding": PositionEmbedding
    }
)
    print("✅ Transformer model loaded successfully!")

except Exception as e:
    print("Error loading model:")
    print(e)
    model = None
# ==========================================================
# Image Preprocessing
# ==========================================================

IMG_SIZE = 128

def preprocess_image(image_path):
    """
    Preprocess uploaded SAR image for Transformer prediction.

    Returns:
        image (numpy array)
        Shape -> (1, 128, 128, 1)
    """

    # Read image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Unable to read image.")

    # Resize
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # Normalize
    image = image.astype(np.float32) / 255.0

    # Add channel dimension
    image = np.expand_dims(image, axis=-1)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    return image
# ==========================================================
# Prediction Function
# ==========================================================

def predict_flood(image_path):
    """
    Predict Flood / Non-Flood from uploaded image.
    """

    image = preprocess_image(image_path)

    prediction = model.predict(image, verbose=0)

    probability = float(prediction[0][0])

    if probability >= 0.5:
        label = "Flood"
    else:
        label = "Non-Flood"

    confidence = probability if probability >= 0.5 else (1 - probability)

    return {
        "label": label,
        "confidence": round(confidence * 100, 2)
    }
# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    test_image = os.path.join(
        os.path.dirname(__file__),
        "..",
        "uploads",
        "test.png"
    )

    if os.path.exists(test_image):

        result = predict_flood(test_image)

        print("\nPrediction:", result["label"])
        print("Confidence:", result["confidence"], "%")

    else:
        print("Place a test image in app/uploads/test.png")