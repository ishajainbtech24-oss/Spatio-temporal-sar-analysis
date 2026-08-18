import cv2
import numpy as np
from sklearn.cluster import KMeans


# ==========================================================
# Terrain Classification (unsupervised, heuristic-based)
# ==========================================================
#
# This is NOT a trained model - there is no labeled terrain dataset in this
# project (Sen1Floods11 only labels flood/no-flood, never land-cover type).
# Instead, this clusters each image into 4 groups using two well-established
# SAR interpretation cues, then labels each cluster by its characteristics:
#
#   Water         : very low backscatter (intensity), low texture
#   Bare Soil     : low-moderate backscatter, low texture
#   Vegetation    : moderate backscatter, higher texture (canopy scattering)
#   Urban/Built-up: high backscatter, high texture (double-bounce off structures)
#
# Because this is a heuristic, not a validated classifier, treat its output
# as an estimate for visualization - not with the same confidence as the
# flood prediction.

CLASS_ORDER = ["Water", "Vegetation", "Urban", "Bare Soil"]

CLASS_COLORS = {
    "Water": "#00dbe9",
    "Vegetation": "#4fdbc8",
    "Urban": "#ffb690",
    "Bare Soil": "#c9a66b",
}


def _compute_texture(img, window=5):
    """Local standard deviation as a simple, fast texture proxy."""
    img_f = img.astype(np.float32)
    mean = cv2.blur(img_f, (window, window))
    mean_sq = cv2.blur(img_f ** 2, (window, window))
    variance = np.maximum(mean_sq - mean ** 2, 0)
    return np.sqrt(variance)


def classify_terrain(image_path, target_size=(128, 128)):
    """
    Runs unsupervised terrain classification on a single image.

    Returns:
        dict: {"Water": pct, "Vegetation": pct, "Urban": pct, "Bare Soil": pct}
              percentages always sum to 100 (rounded), keys always in
              CLASS_ORDER order for consistent chart coloring.
    """

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    img = cv2.resize(img, target_size)

    intensity = img.astype(np.float32) / 255.0

    texture = _compute_texture(img, window=5)
    texture = texture / (texture.max() + 1e-8)

    features = np.stack(
        [intensity.flatten(), texture.flatten()],
        axis=1
    )

    kmeans = KMeans(
        n_clusters=4,
        n_init=10,
        random_state=42
    )
    cluster_labels = kmeans.fit_predict(features)
    centers = kmeans.cluster_centers_  # shape (4, 2) -> [intensity, texture]

    # --- assign semantic labels to clusters based on centroid characteristics ---
    combined_score = centers[:, 0] + centers[:, 1]  # intensity + texture

    water_cluster = int(np.argmin(combined_score))
    urban_cluster = int(np.argmax(combined_score))

    remaining = [c for c in range(4) if c not in (water_cluster, urban_cluster)]
    # among the two remaining clusters, higher texture -> Vegetation, lower -> Bare Soil
    remaining_sorted = sorted(remaining, key=lambda c: centers[c, 1])
    bare_soil_cluster = remaining_sorted[0]
    vegetation_cluster = remaining_sorted[-1]

    cluster_to_label = {
        water_cluster: "Water",
        vegetation_cluster: "Vegetation",
        urban_cluster: "Urban",
        bare_soil_cluster: "Bare Soil",
    }

    # --- compute percentages, always in a fixed key order ---
    total_pixels = cluster_labels.size
    percentages = {name: 0.0 for name in CLASS_ORDER}

    for cluster_id, label_name in cluster_to_label.items():
        count = int(np.sum(cluster_labels == cluster_id))
        percentages[label_name] = round((count / total_pixels) * 100, 2)

    return percentages
