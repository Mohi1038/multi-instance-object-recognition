"""Greedy multi-instance detection using Hough + custom RANSAC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np

from .config import Config
from .geometry import bbox_sanity, polygon_is_valid, polygon_to_aabb, project_template_corners
from .hough import HoughCluster, HoughHypothesis, build_hough_hypotheses, hough_vote_and_cluster
from .matching import FeatureMatch
from .nms import iou_xyxy, non_max_suppression_indices
from .ransac import RansacResult, run_affine_ransac


@dataclass
class Detection:
    polygon: np.ndarray
    bbox_xyxy: np.ndarray
    score: float
    inliers: int
    hough_votes: int
    mean_error: float
    determinant: float
    scale: float
    affine_2x3: np.ndarray


@dataclass
class DetectionTrace:
    cluster: HoughCluster
    candidate_matches: List[FeatureMatch]
    ransac_result: RansacResult
    inlier_indices: List[int]
    outlier_indices: List[int]
    inlier_ratio: float


@dataclass
class DetectionReport:
    detections_before_nms: List[Detection]
    detections_after_nms: List[Detection]
    detection_traces: List[DetectionTrace]
    ransac_models_tested: int


def _log(config: Config, message: str) -> None:
    if config.verbose:
        print(message)


def _verify_false_positive(config: Config, det: Detection) -> bool:
    if det.inliers < config.min_inliers:
        return False
    if det.hough_votes < config.min_hough_votes:
        return False
    if abs(det.determinant) < config.det_epsilon:
        return False
    if det.scale < config.min_affine_scale or det.scale > config.max_affine_scale:
        return False
    if det.mean_error > config.max_mean_reprojection_error:
        return False
    return True


def _build_iou_matrix(boxes: List[np.ndarray]) -> np.ndarray:
    n = len(boxes)
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            matrix[i, j] = iou_xyxy(boxes[i], boxes[j])
    return matrix


def _detect_with_traces(
    template_image: np.ndarray,
    scene_image: np.ndarray,
    template_keypoints: Sequence[cv2.KeyPoint],
    scene_keypoints: Sequence[cv2.KeyPoint],
    ratio_matches: Sequence[FeatureMatch],
    config: Config,
) -> tuple[List[Detection], List[DetectionTrace], int]:
    remaining_matches: List[FeatureMatch] = list(ratio_matches)
    detections: List[Detection] = []
    traces: List[DetectionTrace] = []
    total_models_tested = 0

    _log(config, f"Raw ratio-test matches available for detection: {len(remaining_matches)}")

    for det_iter in range(config.max_detections):
        if len(remaining_matches) < config.min_remaining_matches:
            _log(config, "Stopping: too few matches remain.")
            break

        hypotheses = build_hough_hypotheses(
            template_keypoints=template_keypoints,
            scene_keypoints=scene_keypoints,
            matches=remaining_matches,
            template_shape=template_image.shape,
        )

        _log(config, f"Iteration {det_iter}: Hough hypotheses = {len(hypotheses)}")

        if not hypotheses:
            _log(config, "Stopping: no Hough hypotheses.")
            break

        clusters = hough_vote_and_cluster(
            hypotheses=hypotheses,
            x_bin=config.hough_x_bin_size,
            y_bin=config.hough_y_bin_size,
            scale_bin=config.hough_scale_bin_size,
            angle_bin_rad=float(np.deg2rad(config.hough_angle_bin_size_deg)),
            min_votes=config.min_hough_votes,
        )

        _log(config, f"Candidate Hough clusters: {len(clusters)}")

        if not clusters:
            _log(config, "Stopping: no cluster reached minimum Hough votes.")
            break

        accepted_detection: Detection | None = None
        accepted_trace: DetectionTrace | None = None
        accepted_inlier_remaining_ids: List[int] = []

        for c_idx, cluster in enumerate(clusters):
            cand_matches = [remaining_matches[i] for i in cluster.match_indices]

            ransac_result = run_affine_ransac(
                candidate_matches=cand_matches,
                template_keypoints=template_keypoints,
                scene_keypoints=scene_keypoints,
                iterations=config.ransac_iterations,
                error_threshold=config.ransac_error_threshold,
                min_inliers=config.min_inliers,
                random_seed=config.random_seed + det_iter * 31 + c_idx,
                collinearity_eps=config.collinearity_eps,
                det_epsilon=config.det_epsilon,
                min_scale=config.min_affine_scale,
                max_scale=config.max_affine_scale,
            )

            total_models_tested += ransac_result.models_tested

            if not ransac_result.success or ransac_result.affine_2x3 is None:
                continue

            if ransac_result.mean_error > config.max_mean_reprojection_error:
                continue

            polygon = project_template_corners(template_image.shape, ransac_result.affine_2x3)
            if not polygon_is_valid(polygon):
                continue
            bbox = polygon_to_aabb(polygon)
            if not bbox_sanity(
                bbox,
                scene_image.shape,
                min_area=config.min_bbox_area,
                max_outside_fraction=config.max_outside_fraction,
            ):
                continue

            inlier_count = len(ransac_result.inlier_indices)
            score = float(inlier_count + 0.1 * cluster.vote_count - 0.05 * ransac_result.mean_error)

            detection = Detection(
                polygon=polygon,
                bbox_xyxy=bbox,
                score=score,
                inliers=inlier_count,
                hough_votes=cluster.vote_count,
                mean_error=ransac_result.mean_error,
                determinant=ransac_result.determinant,
                scale=ransac_result.scale,
                affine_2x3=ransac_result.affine_2x3,
            )

            if not _verify_false_positive(config, detection):
                continue

            outlier_indices = [i for i in range(len(cand_matches)) if i not in set(ransac_result.inlier_indices)]
            inlier_ratio = float(len(ransac_result.inlier_indices) / max(1, len(cand_matches)))
            accepted_trace = DetectionTrace(
                cluster=cluster,
                candidate_matches=cand_matches,
                ransac_result=ransac_result,
                inlier_indices=list(ransac_result.inlier_indices),
                outlier_indices=outlier_indices,
                inlier_ratio=inlier_ratio,
            )
            accepted_detection = detection
            accepted_inlier_remaining_ids = [cluster.match_indices[i] for i in ransac_result.inlier_indices]
            break

        if accepted_detection is None or accepted_trace is None:
            _log(config, "Stopping: no geometrically valid cluster in this iteration.")
            break

        detections.append(accepted_detection)
        traces.append(accepted_trace)
        _log(
            config,
            (
                f"Accepted detection {len(detections)}: inliers={accepted_detection.inliers}, "
                f"votes={accepted_detection.hough_votes}, score={accepted_detection.score:.2f}"
            ),
        )

        inlier_set = set(accepted_inlier_remaining_ids)
        remaining_matches = [m for i, m in enumerate(remaining_matches) if i not in inlier_set]
        _log(config, f"Remaining matches after inlier subtraction: {len(remaining_matches)}")

    return detections, traces, total_models_tested


def run_multi_instance_detection(
    template_image: np.ndarray,
    scene_image: np.ndarray,
    template_keypoints: Sequence[cv2.KeyPoint],
    scene_keypoints: Sequence[cv2.KeyPoint],
    ratio_matches: Sequence[FeatureMatch],
    config: Config,
) -> List[Detection]:
    report = run_multi_instance_detection_with_report(
        template_image=template_image,
        scene_image=scene_image,
        template_keypoints=template_keypoints,
        scene_keypoints=scene_keypoints,
        ratio_matches=ratio_matches,
        config=config,
    )
    return report.detections_after_nms


def run_multi_instance_detection_with_report(
    template_image: np.ndarray,
    scene_image: np.ndarray,
    template_keypoints: Sequence[cv2.KeyPoint],
    scene_keypoints: Sequence[cv2.KeyPoint],
    ratio_matches: Sequence[FeatureMatch],
    config: Config,
) -> DetectionReport:
    detections, traces, total_models_tested = _detect_with_traces(
        template_image=template_image,
        scene_image=scene_image,
        template_keypoints=template_keypoints,
        scene_keypoints=scene_keypoints,
        ratio_matches=ratio_matches,
        config=config,
    )

    boxes = [d.bbox_xyxy for d in detections]
    scores = [d.score for d in detections]
    keep_ids = non_max_suppression_indices(boxes, scores, config.nms_iou_threshold)
    final_detections = [detections[i] for i in keep_ids]

    _log(config, f"Detections before NMS: {len(detections)}")
    _log(config, f"Detections after NMS: {len(final_detections)}")

    return DetectionReport(
        detections_before_nms=detections,
        detections_after_nms=final_detections,
        detection_traces=traces,
        ransac_models_tested=total_models_tested,
    )
