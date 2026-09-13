"""SIFT feature extraction helpers."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def load_image_bgr(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def to_gray(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def extract_sift(image_bgr: np.ndarray) -> Tuple[List[cv2.KeyPoint], Optional[np.ndarray]]:
    """Extract SIFT keypoints and descriptors.

    Keypoint properties used in this project:
    - `kp.pt`: (x, y) location in image pixels
    - `kp.size`: keypoint scale (diameter-like measure)
    - `kp.angle`: keypoint orientation in degrees
    """
    gray = to_gray(image_bgr)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors
