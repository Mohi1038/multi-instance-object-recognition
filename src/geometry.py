"""Geometry utilities: sanity checks, corner projection, box utilities."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def triangle_area2(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    return float(abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])))


def nearly_collinear(points: np.ndarray, eps: float) -> bool:
    if points.shape[0] < 3:
        return True
    max_area2 = 0.0
    n = points.shape[0]
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            for k in range(j + 1, n):
                max_area2 = max(max_area2, triangle_area2(points[i], points[j], points[k]))
    return max_area2 < eps


def affine_det(affine_2x3: np.ndarray) -> float:
    a, b = float(affine_2x3[0, 0]), float(affine_2x3[0, 1])
    d, e = float(affine_2x3[1, 0]), float(affine_2x3[1, 1])
    return a * e - b * d


def affine_scale_estimate(affine_2x3: np.ndarray) -> float:
    m = affine_2x3[:, :2]
    det = float(np.linalg.det(m))
    return float(np.sqrt(abs(det)))


def affine_rotation_estimate_rad(affine_2x3: np.ndarray) -> float:
    """Estimate the dominant rotation angle from the linear part of an affine transform."""
    a = float(affine_2x3[0, 0])
    d = float(affine_2x3[1, 0])
    return float(np.arctan2(d, a))


def project_template_corners(template_shape: Tuple[int, int, int], affine_2x3: np.ndarray) -> np.ndarray:
    h, w = template_shape[:2]
    corners = np.array(
        [[0.0, 0.0], [float(w), 0.0], [float(w), float(h)], [0.0, float(h)]], dtype=np.float64
    )
    ones = np.ones((4, 1), dtype=np.float64)
    homo = np.hstack([corners, ones])
    projected = (affine_2x3 @ homo.T).T
    return projected


def polygon_to_aabb(poly: np.ndarray) -> np.ndarray:
    x_min = float(np.min(poly[:, 0]))
    y_min = float(np.min(poly[:, 1]))
    x_max = float(np.max(poly[:, 0]))
    y_max = float(np.max(poly[:, 1]))
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float64)


def polygon_is_valid(poly: np.ndarray, area_epsilon: float = 1e-6) -> bool:
    if poly.shape != (4, 2):
        return False

    edge_vectors = np.roll(poly, -1, axis=0) - poly
    cross_products = np.array(
        [
            edge_vectors[i, 0] * edge_vectors[(i + 1) % 4, 1]
            - edge_vectors[i, 1] * edge_vectors[(i + 1) % 4, 0]
            for i in range(4)
        ],
        dtype=np.float64,
    )
    if np.any(np.abs(cross_products) <= area_epsilon):
        return False
    if not (np.all(cross_products > 0.0) or np.all(cross_products < 0.0)):
        return False

    area = 0.5 * abs(
        float(
            np.sum(poly[:, 0] * np.roll(poly[:, 1], -1))
            - np.sum(poly[:, 1] * np.roll(poly[:, 0], -1))
        )
    )
    return area > area_epsilon


def aabb_area(aabb: np.ndarray) -> float:
    w = max(0.0, float(aabb[2] - aabb[0]))
    h = max(0.0, float(aabb[3] - aabb[1]))
    return w * h


def bbox_sanity(
    aabb: np.ndarray,
    scene_shape: Tuple[int, int, int],
    min_area: float,
    max_outside_fraction: float,
) -> bool:
    area = aabb_area(aabb)
    if area < min_area:
        return False

    h, w = scene_shape[:2]
    sx1, sy1, sx2, sy2 = 0.0, 0.0, float(w), float(h)
    x1, y1, x2, y2 = [float(v) for v in aabb]

    ix1 = max(x1, sx1)
    iy1 = max(y1, sy1)
    ix2 = min(x2, sx2)
    iy2 = min(y2, sy2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    outside = max(0.0, area - inter)
    outside_frac = outside / max(area, 1e-12)
    return outside_frac <= max_outside_fraction
