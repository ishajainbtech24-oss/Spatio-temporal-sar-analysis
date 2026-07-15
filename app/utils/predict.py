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

BASE_DIR = os.path.dirname(__file__)

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
# Check files
# ==========================================================

print("=" * 60)
print("Transformer:", MODEL_PATH)
print("Encoder:", ENCODER_PATH)

print("Transformer exists:", os.path.exists(MODEL_PATH))
print("Encoder exists:", os.path.exists(ENCODER_PATH))
print("=" * 60)


# ==========================================================
# Load Encoder
# ==========================================================

try:

    encoder = tf.keras.models.load_model(ENCODER_PATH)

    print("✅ Encoder loaded successfully!")

    print("\n============================")
    print("ENCODER MODEL")
    print("============================")

    encoder.summary()

    print("\nInput Shape :", encoder.input_shape)
    print("Output Shape:", encoder.output_shape)
    print("============================\n")
except Exception as e:

    print("❌ Error loading encoder")
    print(e)

    encoder = None


# ==========================================================
# Load Transformer
# ==========================================================

try:

    model = tf.keras.models.load_model(

        MODEL_PATH,

        custom_objects={
            "PositionEmbedding": PositionEmbedding
        }

    )

    print("✅ Transformer model loaded successfully!")
    print("\n============================")
    print("TRANSFORMER MODEL")
    print("============================")

    model.summary()
    print("\nInput Shape :", model.input_shape)
    print("Output Shape:", model.output_shape)

    print("============================\n")

except Exception as e:

    print("❌ Error loading Transformer")
    print(e)

    model = None


# ==========================================================
# Image Preprocessing
# ==========================================================

IMG_SIZE = 128


def preprocess_image(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Unable to read image.")

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = image.astype(np.float32) / 255.0

    image = np.expand_dims(image, axis=-1)

    image = np.expand_dims(image, axis=0)

    return image


# ==========================================================
# Prediction
# ==========================================================
def predict_flood(image_path):

    image = preprocess_image(image_path)

    features = encoder.predict(image, verbose=0)

    tokens = features.reshape(
        features.shape[0],
        32 * 32,
        16
    )

    print("\n========== PIPELINE ==========")
    print("Input Image Shape      :", image.shape)
    print("Encoder Output Shape   :", features.shape)
    print("Transformer Input Shape:", tokens.shape)

    prediction = model.predict(tokens, verbose=0)

    print("Raw Transformer Output :", prediction)
    print("==============================\n")

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
        BASE_DIR,
        "..",
        "uploads",
        "test.png"
    )

    if os.path.exists(test_image):

        result = predict_flood(test_image)

        print("\n==============================")
        print("Prediction :", result["label"])
        print("Confidence :", result["confidence"], "%")
        print("==============================")

    else:

        print("Place a test image inside:")
        print("app/uploads/test.png")