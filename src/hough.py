"""Custom 4D generalized Hough voting implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import cv2

from .matching import FeatureMatch


@dataclass
class HoughHypothesis:
    center_x: float
    center_y: float
    scale: float
    rotation_rad: float
    match_index: int


@dataclass
class HoughCluster:
    key: Tuple[int, int, int, int]
    vote_count: int
    match_indices: List[int]
    mean_center_x: float
    mean_center_y: float
    mean_scale: float
    mean_rotation_rad: float


def normalize_angle_rad(angle_rad: float) -> float:
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def build_hough_hypotheses(
    template_keypoints: Sequence[cv2.KeyPoint],
    scene_keypoints: Sequence[cv2.KeyPoint],
    matches: Sequence[FeatureMatch],
    template_shape: Tuple[int, int, int],
) -> List[HoughHypothesis]:
    h, w = template_shape[:2]
    cx = 0.5 * w
    cy = 0.5 * h

    hypotheses: List[HoughHypothesis] = []

    for idx, m in enumerate(matches):
        tkp = template_keypoints[m.template_idx]
        skp = scene_keypoints[m.scene_idx]

        xt, yt = tkp.pt
        xs, ys = skp.pt
        st = float(tkp.size)
        ss = float(skp.size)

        if st <= 1e-12 or ss <= 1e-12:
            continue

        dx = xt - cx
        dy = yt - cy

        scale = ss / st

        theta_t = np.deg2rad(float(tkp.angle))
        theta_s = np.deg2rad(float(skp.angle))
        rotation = normalize_angle_rad(theta_s - theta_t)

        # SIFT orientations can differ by 180 degrees for symmetric local
        # structures. Fold that ambiguity before predicting the object center.
        if rotation > 0.5 * np.pi:
            rotation -= np.pi
        elif rotation < -0.5 * np.pi:
            rotation += np.pi

        cos_r = float(np.cos(rotation))
        sin_r = float(np.sin(rotation))

        # v' = scale * R(rotation) * v
        vx_p = scale * (cos_r * dx - sin_r * dy)
        vy_p = scale * (sin_r * dx + cos_r * dy)

        center_x = xs - vx_p
        center_y = ys - vy_p

        hypotheses.append(
            HoughHypothesis(
                center_x=float(center_x),
                center_y=float(center_y),
                scale=float(scale),
                rotation_rad=float(rotation),
                match_index=idx,
            )
        )

    return hypotheses


def quantize_hypothesis(
    h: HoughHypothesis,
    x_bin: float,
    y_bin: float,
    scale_bin: float,
    angle_bin_rad: float,
) -> Tuple[int, int, int, int]:
    xb = int(np.floor(h.center_x / x_bin))
    yb = int(np.floor(h.center_y / y_bin))
    sb = int(np.floor(h.scale / scale_bin))
    rb = int(np.floor((normalize_angle_rad(h.rotation_rad) + np.pi) / angle_bin_rad))
    return xb, yb, sb, rb


def hough_vote_and_cluster(
    hypotheses: Sequence[HoughHypothesis],
    x_bin: float,
    y_bin: float,
    scale_bin: float,
    angle_bin_rad: float,
    min_votes: int,
) -> List[HoughCluster]:
    space: Dict[Tuple[int, int, int, int], List[int]] = {}
    for i, hyp in enumerate(hypotheses):
        key = quantize_hypothesis(hyp, x_bin, y_bin, scale_bin, angle_bin_rad)
        if key not in space:
            space[key] = []
        space[key].append(i)

    clusters: List[HoughCluster] = []
    for key, hyp_indices in space.items():
        if len(hyp_indices) < min_votes:
            continue

        centers_x = [hypotheses[i].center_x for i in hyp_indices]
        centers_y = [hypotheses[i].center_y for i in hyp_indices]
        scales = [hypotheses[i].scale for i in hyp_indices]
        rots = [hypotheses[i].rotation_rad for i in hyp_indices]

        # mean of circular variable via unit vectors
        c = float(np.mean(np.cos(rots)))
        s = float(np.mean(np.sin(rots)))
        mean_rot = float(np.arctan2(s, c))

        clusters.append(
            HoughCluster(
                key=key,
                vote_count=len(hyp_indices),
                match_indices=[hypotheses[i].match_index for i in hyp_indices],
                mean_center_x=float(np.mean(centers_x)),
                mean_center_y=float(np.mean(centers_y)),
                mean_scale=float(np.mean(scales)),
                mean_rotation_rad=mean_rot,
            )
        )

    clusters.sort(key=lambda c_: c_.vote_count, reverse=True)
    return clusters
