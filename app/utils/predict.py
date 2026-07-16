import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


# ==========================================================
# Custom Position Embedding Layer
# ==========================================================

@tf.keras.utils.register_keras_serializable()
class PositionEmbedding(tf.keras.layers.Layer):

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
            "embed_dim": self.embed_dim
        })

        return config


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "model",
    "simple_transformer_final.keras"
)

ENCODER_PATH = os.path.join(
    BASE_DIR,
    "..",
    "model",
    "encoder.keras"
)


# ==========================================================
# Verify Model Files
# ==========================================================

print("=" * 60)
print("Loading Models...")
print("Transformer :", MODEL_PATH)
print("Encoder     :", ENCODER_PATH)
print("=" * 60)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Transformer model not found:\n{MODEL_PATH}")

if not os.path.exists(ENCODER_PATH):
    raise FileNotFoundError(f"Encoder model not found:\n{ENCODER_PATH}")


# ==========================================================
# Load Encoder
# ==========================================================

encoder = tf.keras.models.load_model(
    ENCODER_PATH,
    compile=False
)

print("✅ Encoder Loaded")


# ==========================================================
# Load Transformer
# ==========================================================

model = tf.keras.models.load_model(

    MODEL_PATH,

    compile=False,

    custom_objects={
        "PositionEmbedding": PositionEmbedding
    }

)

print("✅ Transformer Loaded")


# ==========================================================
# Image Size
# ==========================================================

IMG_SIZE = 128


# ==========================================================
# Image Preprocessing
# ==========================================================

def preprocess_image(image_path):

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            f"Unable to read image:\n{image_path}"
        )

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = image.astype(np.float32)

    image = image / 255.0

    image = np.expand_dims(
        image,
        axis=-1
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ==========================================================
# Prediction Function
# ==========================================================

def predict(image_path):

    image = preprocess_image(image_path)

    # ------------------------------------------
    # Encoder
    # ------------------------------------------

    features = encoder.predict(
        image,
        verbose=0
    )

    # Shape:
    # (1,32,32,16)
    # →
    # (1,1024,16)

    tokens = features.reshape(

        features.shape[0],

        32 * 32,

        16

    )

    # ------------------------------------------
    # Transformer
    # ------------------------------------------

    prediction = model.predict(
        tokens,
        verbose=0
    )

    probability = float(prediction[0][0])

    # ------------------------------------------
    # Label
    # ------------------------------------------

    if probability >= 0.5:

        label = "Flood"

        confidence = probability

    else:

        label = "No Flood"

        confidence = 1 - probability

    confidence = round(confidence * 100, 2)

    # ------------------------------------------
    # Console Logs
    # ------------------------------------------

    print("\n==============================")
    print("Prediction Completed")
    print("==============================")
    print("Image      :", image_path)
    print("Prediction :", label)
    print("Confidence :", confidence, "%")
    print("==============================\n")

    return label, confidence


# ==========================================================
# Local Testing
# ==========================================================

if __name__ == "__main__":

    TEST_IMAGE = os.path.join(

        BASE_DIR,

        "..",

        "uploads",

        "test.png"

    )

    if os.path.exists(TEST_IMAGE):

        label, confidence = predict(TEST_IMAGE)

        print("Prediction :", label)
        print("Confidence :", confidence)

    else:

        print("Test image not found:")
        print(TEST_IMAGE)