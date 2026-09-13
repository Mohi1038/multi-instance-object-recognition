"""Manual IoU and Non-Maximum Suppression."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np


def iou_xyxy(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 1e-12:
        return 0.0
    return inter / union


def non_max_suppression_indices(
    boxes: Sequence[np.ndarray],
    scores: Sequence[float],
    iou_threshold: float,
) -> List[int]:
    if len(boxes) == 0:
        return []

    order = np.argsort(np.asarray(scores))[::-1].tolist()
    keep: List[int] = []

    while order:
        current = order.pop(0)
        keep.append(current)

        remaining = []
        for idx in order:
            overlap = iou_xyxy(np.asarray(boxes[current]), np.asarray(boxes[idx]))
            if overlap <= iou_threshold:
                remaining.append(idx)
        order = remaining

    return keep
