"""Custom descriptor matching with Euclidean distance and Lowe ratio test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class FeatureMatch:
    scene_idx: int
    template_idx: int
    distance: float
    ratio: float


@dataclass
class RawNearestMatch:
    scene_idx: int
    template_idx: int
    distance: float


@dataclass
class MatcherDiagnostics:
    candidate_count: int
    accepted_count: int
    rejected_count: int
    ratios: List[float]
    best_distances: List[float]
    second_best_distances: List[float]


def euclidean_distance(desc_a: np.ndarray, desc_b: np.ndarray) -> float:
    diff = desc_a - desc_b
    return float(np.sqrt(np.sum(diff * diff)))


def match_scene_to_template_with_diagnostics(
    scene_descriptors: np.ndarray,
    template_descriptors: np.ndarray,
    ratio_threshold: float,
) -> Tuple[List[RawNearestMatch], List[FeatureMatch], MatcherDiagnostics]:
    """Match every scene descriptor against all template descriptors.

    Returns:
    - raw nearest-neighbor matches (without ratio filtering) for visualization
    - ratio-test validated matches
    - matching diagnostics for plots/statistics
    """
    if scene_descriptors is None or template_descriptors is None:
        empty = MatcherDiagnostics(0, 0, 0, [], [], [])
        return [], [], empty
    if len(scene_descriptors) == 0 or len(template_descriptors) < 2:
        empty = MatcherDiagnostics(0, 0, 0, [], [], [])
        return [], [], empty

    raw_matches: List[RawNearestMatch] = []
    good_matches: List[FeatureMatch] = []
    ratios: List[float] = []
    best_distances: List[float] = []
    second_best_distances: List[float] = []

    for s_idx, s_desc in enumerate(scene_descriptors):
        best_dist = float("inf")
        second_dist = float("inf")
        best_t_idx = -1

        for t_idx, t_desc in enumerate(template_descriptors):
            dist = euclidean_distance(s_desc, t_desc)
            if dist < best_dist:
                second_dist = best_dist
                best_dist = dist
                best_t_idx = t_idx
            elif dist < second_dist:
                second_dist = dist

        if best_t_idx < 0:
            continue

        raw_matches.append(RawNearestMatch(scene_idx=s_idx, template_idx=best_t_idx, distance=best_dist))
        best_distances.append(best_dist)
        second_best_distances.append(second_dist)

        if not np.isfinite(second_dist) or second_dist <= 1e-12:
            continue

        ratio = best_dist / second_dist
        ratios.append(float(ratio))
        if ratio < ratio_threshold:
            good_matches.append(
                FeatureMatch(
                    scene_idx=s_idx,
                    template_idx=best_t_idx,
                    distance=best_dist,
                    ratio=ratio,
                )
            )

    diagnostics = MatcherDiagnostics(
        candidate_count=len(best_distances),
        accepted_count=len(good_matches),
        rejected_count=max(0, len(best_distances) - len(good_matches)),
        ratios=ratios,
        best_distances=best_distances,
        second_best_distances=second_best_distances,
    )
    return raw_matches, good_matches, diagnostics


def match_scene_to_template(
    scene_descriptors: np.ndarray,
    template_descriptors: np.ndarray,
    ratio_threshold: float,
) -> Tuple[List[RawNearestMatch], List[FeatureMatch]]:
    raw_matches, good_matches, _ = match_scene_to_template_with_diagnostics(
        scene_descriptors=scene_descriptors,
        template_descriptors=template_descriptors,
        ratio_threshold=ratio_threshold,
    )
    return raw_matches, good_matches
