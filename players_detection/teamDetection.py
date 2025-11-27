from typing import List
import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamClassifier:
    """
    A lightweight classifier that uses HSV Color Histograms and KMeans
    to cluster players into teams.
    """

    def __init__(self):
        self.cluster_model = KMeans(n_clusters=2, n_init=10, random_state=42)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extracts HSV Hue-Saturation histograms from a list of player crops.

        Args:
            crops (List[np.ndarray]): List of image crops (BGR format).

        Returns:
            np.ndarray: Extracted features.
        """
        features = []

        for crop in crops:
            # Convert BGR to HSV
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            h_crop, w_crop = hsv.shape[:2]

            # Create a mask for the jersey area (upper torso)
            # We skip the head (top 10%) and legs (bottom 50%)
            # We skip the arms (left/right 20%)
            jersey_mask = np.zeros((h_crop, w_crop), np.uint8)
            jersey_mask[
                int(0.1 * h_crop) : int(0.5 * h_crop),
                int(0.2 * w_crop) : int(0.8 * w_crop),
            ] = 255

            # Compute 2D Histogram: Hue (Color) + Saturation (Whiteness/Grayness)
            # Channels: [0, 1] -> Hue, Saturation
            # Bins: [8, 4] -> 8 Hue bins (enough for main colors), 4 Saturation bins (enough for white vs color)
            # Ranges: Hue [0, 180], Saturation [0, 256]
            hist = cv2.calcHist([hsv], [0, 1], jersey_mask, [8, 4], [0, 180, 0, 256])

            # Normalize the histogram so it is scale-invariant (size of crop doesn't matter)
            hist = cv2.normalize(hist, None).flatten()
            features.append(hist)

        return np.vstack(features)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the K-Means clustering model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
        """
        if not crops:
            return

        data = self.extract_features(crops)
        self.cluster_model.fit(data)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the cluster labels for a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted cluster labels (0 or 1).
        """
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        return self.cluster_model.predict(data)
