import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model


# ==========================================================
# Paths
# ==========================================================

# app/utils/predict.py -> BASE_DIR is app/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "simple_transformer_pretrained_final.keras"
)

# Tuned during training/validation (see notebook 07) - NOT the default 0.5.
# The dataset is heavily imbalanced (~87% No-Flood), so the model was tuned to flag
# flood at a lower probability cutoff than 0.5 - using 0.5 here would under-detect floods.
DECISION_THRESHOLD = 0.20


# ==========================================================
# Custom Layer (required to load the saved model)
# ==========================================================
#
# PositionEmbedding is a custom layer defined during training (notebook 07). Keras
# needs this exact class available whenever the saved model is loaded - it can't
# infer custom layer code from the saved file alone.
#
# __init__ accepts **kwargs and forwards them to super().__init__() because Keras
# also passes standard layer args (trainable, dtype, name, ...) when rebuilding this
# layer from the saved config - without **kwargs there's nowhere for them to go.

class PositionEmbedding(layers.Layer):
    def __init__(self, sequence_length, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.position_embedding = layers.Embedding(
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


# ==========================================================
# Model (loaded once, reused across requests)
# ==========================================================

_model = None


def _get_model():
    global _model

    if _model is None:

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Make sure simple_transformer_pretrained_final.keras has been "
                "downloaded from Drive and placed in the app/models/ folder."
            )

        print("Loading model from:", MODEL_PATH)

        _model = load_model(
            MODEL_PATH,
            # required so Keras can resolve the PositionEmbedding layer above
            custom_objects={"PositionEmbedding": PositionEmbedding},
            # required because the model also contains a raw Python Lambda layer
            # (the channel-tiling/rescaling step before MobileNetV2) - Keras blocks
            # deserializing arbitrary lambdas by default for security
            safe_mode=False
        )

        print("Model loaded successfully.")

    return _model


# ==========================================================
# Preprocessing
# ==========================================================

def _preprocess(image_path):

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    img = cv2.resize(img, (128, 128))

    img = img.astype(np.float32)
    img = img / 255.0

    # add channel dimension - model input is (128,128,1); it tiles this into
    # 3 fake-RGB channels internally before MobileNetV2, so nothing more is
    # needed here
    img = np.expand_dims(img, axis=-1)

    # add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


# ==========================================================
# Prediction
# ==========================================================

def predict(image_path):
    """
    Runs flood detection on a single image.

    Returns:
        prediction (str): "Flood" or "No Flood"
        confidence (float): confidence in the PREDICTED label, as a percentage
                             (e.g. predicting "No Flood" with a raw flood
                             probability of 0.12 returns confidence=88.0)
    """

    model = _get_model()

    img = _preprocess(image_path)

    flood_probability = float(model.predict(img, verbose=0)[0][0])

    if flood_probability >= DECISION_THRESHOLD:
        prediction = "Flood"
        confidence = flood_probability * 100
    else:
        prediction = "No Flood"
        confidence = (1 - flood_probability) * 100

    confidence = round(confidence, 2)

    print(
        f"Raw flood probability: {flood_probability:.4f}  "
        f"| Threshold: {DECISION_THRESHOLD}  "
        f"| Prediction: {prediction}  "
        f"| Confidence: {confidence}%"
    )

    return prediction, confidence
