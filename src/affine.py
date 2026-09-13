"""Affine transformation estimation from scratch via least squares."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def estimate_affine_lstsq(template_points: np.ndarray, scene_points: np.ndarray) -> np.ndarray:
    """Estimate affine transform parameters with least squares.

    Model:
    x_s = a*x_t + b*y_t + c
    y_s = d*x_t + e*y_t + f

    Unknown vector: [a, b, c, d, e, f]

    This function builds A x = B and solves it with np.linalg.lstsq,
    which is the least-squares solution required by the assignment.
    """
    if template_points.shape[0] != scene_points.shape[0]:
        raise ValueError("Point count mismatch")
    if template_points.shape[0] < 3:
        raise ValueError("At least 3 points are required for affine estimation")

    n = template_points.shape[0]
    A = np.zeros((2 * n, 6), dtype=np.float64)
    B = np.zeros((2 * n,), dtype=np.float64)

    for i, (tp, sp) in enumerate(zip(template_points, scene_points)):
        xt, yt = float(tp[0]), float(tp[1])
        xs, ys = float(sp[0]), float(sp[1])

        A[2 * i] = [xt, yt, 1.0, 0.0, 0.0, 0.0]
        A[2 * i + 1] = [0.0, 0.0, 0.0, xt, yt, 1.0]
        B[2 * i] = xs
        B[2 * i + 1] = ys

    x, _, _, _ = np.linalg.lstsq(A, B, rcond=None)

    affine_2x3 = np.array(
        [[x[0], x[1], x[2]], [x[3], x[4], x[5]]], dtype=np.float64
    )
    return affine_2x3


def apply_affine(points: np.ndarray, affine_2x3: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    homo = np.hstack([points.astype(np.float64), ones])
    transformed = (affine_2x3 @ homo.T).T
    return transformed


def reprojection_errors(template_points: np.ndarray, scene_points: np.ndarray, affine_2x3: np.ndarray) -> np.ndarray:
    projected = apply_affine(template_points, affine_2x3)
    diffs = projected - scene_points
    return np.sqrt(np.sum(diffs * diffs, axis=1))
