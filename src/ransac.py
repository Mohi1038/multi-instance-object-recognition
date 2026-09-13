"""Custom RANSAC for affine model fitting and geometric verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np

from .affine import estimate_affine_lstsq, reprojection_errors
from .geometry import affine_det, affine_scale_estimate, nearly_collinear
from .matching import FeatureMatch


@dataclass
class RansacResult:
    success: bool
    affine_2x3: np.ndarray | None
    inlier_indices: List[int]
    mean_error: float
    max_error: float
    determinant: float
    scale: float
    models_tested: int
    all_errors: np.ndarray | None


def _gather_points(
    matches_subset: Sequence[FeatureMatch],
    template_keypoints: Sequence[cv2.KeyPoint],
    scene_keypoints: Sequence[cv2.KeyPoint],
) -> tuple[np.ndarray, np.ndarray]:
    tp = []
    sp = []
    for m in matches_subset:
        t = template_keypoints[m.template_idx].pt
        s = scene_keypoints[m.scene_idx].pt
        tp.append([float(t[0]), float(t[1])])
        sp.append([float(s[0]), float(s[1])])
    return np.asarray(tp, dtype=np.float64), np.asarray(sp, dtype=np.float64)


def run_affine_ransac(
    candidate_matches: Sequence[FeatureMatch],
    template_keypoints: Sequence[cv2.KeyPoint],
    scene_keypoints: Sequence[cv2.KeyPoint],
    iterations: int,
    error_threshold: float,
    min_inliers: int,
    random_seed: int,
    collinearity_eps: float,
    det_epsilon: float,
    min_scale: float,
    max_scale: float,
) -> RansacResult:
    if len(candidate_matches) < 3:
        return RansacResult(False, None, [], float("inf"), float("inf"), 0.0, 0.0, 0, None)

    tp_all, sp_all = _gather_points(candidate_matches, template_keypoints, scene_keypoints)

    rng = np.random.default_rng(random_seed)

    best_inliers: List[int] = []
    best_model: np.ndarray | None = None
    best_mean_error = float("inf")
    models_tested = 0

    n = len(candidate_matches)

    for _ in range(iterations):
        sample_ids = rng.choice(n, size=3, replace=False)
        tp_s = tp_all[sample_ids]
        sp_s = sp_all[sample_ids]

        # Geometric sanity: reject nearly collinear samples
        if nearly_collinear(tp_s, collinearity_eps) or nearly_collinear(sp_s, collinearity_eps):
            continue

        try:
            model = estimate_affine_lstsq(tp_s, sp_s)
        except np.linalg.LinAlgError:
            continue
        except ValueError:
            continue

        models_tested += 1

        det = affine_det(model)
        if abs(det) < det_epsilon:
            continue

        scale = affine_scale_estimate(model)
        if scale < min_scale or scale > max_scale:
            continue

        errs = reprojection_errors(tp_all, sp_all, model)
        inliers = np.where(errs < error_threshold)[0].tolist()

        if len(inliers) < min_inliers:
            continue

        mean_err = float(np.mean(errs[inliers]))

        if (len(inliers) > len(best_inliers)) or (
            len(inliers) == len(best_inliers) and mean_err < best_mean_error
        ):
            best_inliers = inliers
            best_model = model
            best_mean_error = mean_err

    if best_model is None or len(best_inliers) < min_inliers:
        return RansacResult(False, None, [], float("inf"), float("inf"), 0.0, 0.0, models_tested, None)

    # Re-estimate final affine transform using all inliers
    tp_in = tp_all[best_inliers]
    sp_in = sp_all[best_inliers]

    final_model = estimate_affine_lstsq(tp_in, sp_in)
    final_det = affine_det(final_model)
    final_scale = affine_scale_estimate(final_model)

    if abs(final_det) < det_epsilon:
        return RansacResult(False, None, [], float("inf"), float("inf"), final_det, final_scale, models_tested, None)
    if final_scale < min_scale or final_scale > max_scale:
        return RansacResult(False, None, [], float("inf"), float("inf"), final_det, final_scale, models_tested, None)

    final_errs = reprojection_errors(tp_all, sp_all, final_model)
    final_inlier_errs = final_errs[best_inliers]
    final_mean_error = float(np.mean(final_inlier_errs)) if final_inlier_errs.size else float("inf")
    final_max_error = float(np.max(final_errs)) if final_errs.size else float("inf")

    return RansacResult(
        success=True,
        affine_2x3=final_model,
        inlier_indices=best_inliers,
        mean_error=final_mean_error,
        max_error=final_max_error,
        determinant=final_det,
        scale=final_scale,
        models_tested=models_tested,
        all_errors=final_errs,
    )
