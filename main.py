"""Entry point for CV Assignment 1: Multi-instance object recognition."""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

from src.config import default_config
from src.detection import DetectionTrace, run_multi_instance_detection_with_report
from src.geometry import affine_rotation_estimate_rad
from src.features import extract_sift, load_image_bgr
from src.hough import build_hough_hypotheses, hough_vote_and_cluster
from src.matching import match_scene_to_template_with_diagnostics
from src.nms import iou_xyxy, non_max_suppression_indices
from src.visualization import (
    save_affine_projection_visualization,
    save_detections_before_nms,
    save_final_detections,
    save_hough_center_heatmap,
    save_hough_rotation_distribution,
    save_hough_scale_distribution,
    save_hough_votes_visualization,
    save_input_overview,
    save_iou_matrix,
    save_keypoint_orientation_histogram,
    save_keypoint_scale_distribution,
    save_keypoint_visualization,
    save_lowe_ratio_histogram,
    save_match_distance_histogram,
    save_matches_after_ratio_test,
    save_naive_matches_visualization,
    save_pipeline_summary,
    save_ransac_inliers_outliers,
)


def _ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _p(*parts: str) -> str:
    return os.path.join(*parts)


def _read_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read generated image: {path}")
    return image


def _detection_stats(detection, trace: DetectionTrace) -> dict:
    all_errors = trace.ransac_result.all_errors
    candidate_count = len(trace.candidate_matches)
    inlier_count = len(trace.inlier_indices)
    outlier_count = candidate_count - inlier_count
    mean_error = float(trace.ransac_result.mean_error)
    max_error = float(trace.ransac_result.max_error)
    return {
        "hough_votes": int(trace.cluster.vote_count),
        "candidate_matches": int(candidate_count),
        "inlier_count": int(inlier_count),
        "outlier_count": int(outlier_count),
        "inlier_ratio": float(trace.inlier_ratio),
        "mean_reprojection_error": mean_error,
        "max_reprojection_error": max_error,
        "affine_determinant": float(detection.determinant),
        "estimated_scale": float(detection.scale),
        "estimated_rotation_degrees": float(np.degrees(affine_rotation_estimate_rad(detection.affine_2x3))),
        "bbox_xyxy": [float(v) for v in detection.bbox_xyxy.tolist()],
        "confidence_score": float(detection.score),
        "models_tested": int(trace.ransac_result.models_tested),
        "all_errors_available": bool(all_errors is not None),
    }


def _save_statistics_json(path: str, stats: dict) -> None:
    _ensure_output_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def _save_statistics_txt(path: str, stats: dict) -> None:
    lines = []
    lines.append("CV Assignment 1 Statistics")
    lines.append("")
    lines.append(f"Template dimensions: {stats['template_image_dimensions']}")
    lines.append(f"Scene dimensions: {stats['scene_image_dimensions']}")
    lines.append(f"Template keypoints: {stats['template_keypoint_count']}")
    lines.append(f"Scene keypoints: {stats['scene_keypoint_count']}")
    lines.append(f"Raw candidate matches: {stats['raw_candidate_match_count']}")
    lines.append(f"Lowe accepted matches: {stats['lowe_ratio_accepted_match_count']}")
    lines.append(f"Lowe rejected matches: {stats['lowe_ratio_rejected_count']}")
    lines.append(f"Hough hypotheses: {stats['hough_hypothesis_count']}")
    lines.append(f"Hough clusters: {stats['hough_cluster_count']}")
    lines.append(f"RANSAC models tested: {stats['ransac_models_tested']}")
    lines.append(f"Detections before NMS: {stats['detections_before_nms']}")
    lines.append(f"Detections after NMS: {stats['detections_after_nms']}")
    lines.append("")
    for idx, det in enumerate(stats.get("detections", []), start=1):
        lines.append(f"Detection {idx}: {det}")
    _ensure_output_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    config = default_config()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = _p(root_dir, config.template_path)
    query_path = _p(root_dir, config.query_path)
    output_dir = _p(root_dir, config.output_dir)
    _ensure_output_dir(output_dir)

    template_img = load_image_bgr(template_path)
    query_img = load_image_bgr(query_path)

    template_kps, template_desc = extract_sift(template_img)
    scene_kps, scene_desc = extract_sift(query_img)

    if config.verbose:
        print(f"Template keypoints: {len(template_kps)}")
        print(f"Scene keypoints: {len(scene_kps)}")

    raw_matches, ratio_matches, matcher_diag = match_scene_to_template_with_diagnostics(
        scene_descriptors=scene_desc,
        template_descriptors=template_desc,
        ratio_threshold=config.ratio_threshold,
    )

    if config.verbose:
        print(f"Raw correspondences (nearest only): {len(raw_matches)}")
        print(f"Ratio-test matches: {len(ratio_matches)}")

    initial_hypotheses = build_hough_hypotheses(
        template_keypoints=template_kps,
        scene_keypoints=scene_kps,
        matches=ratio_matches,
        template_shape=template_img.shape,
    )
    clusters = hough_vote_and_cluster(
        hypotheses=initial_hypotheses,
        x_bin=config.hough_x_bin_size,
        y_bin=config.hough_y_bin_size,
        scale_bin=config.hough_scale_bin_size,
        angle_bin_rad=float(np.deg2rad(config.hough_angle_bin_size_deg)),
        min_votes=config.min_hough_votes,
    )

    if config.verbose:
        print(f"Hough hypotheses: {len(initial_hypotheses)}")
        print(f"Candidate clusters: {len(clusters)}")

    if config.visualization_enabled:
        save_input_overview(template_img, query_img, _p(output_dir, "01_input_images.jpg"), dpi=config.plot_dpi)
        save_keypoint_visualization(
            template_img,
            template_kps,
            _p(output_dir, "02_template_keypoints.jpg"),
            title="Template SIFT keypoints",
            max_points=config.max_keypoints_visualized,
            dpi=config.plot_dpi,
        )
        save_keypoint_visualization(
            query_img,
            scene_kps,
            _p(output_dir, "03_scene_keypoints.jpg"),
            title="Scene SIFT keypoints",
            max_points=config.max_keypoints_visualized,
            dpi=config.plot_dpi,
        )
        save_keypoint_scale_distribution(template_kps, scene_kps, _p(output_dir, "04_keypoint_scale_distribution.png"), dpi=config.plot_dpi)
        save_keypoint_orientation_histogram(template_kps, scene_kps, _p(output_dir, "05_keypoint_orientation_histogram.png"), dpi=config.plot_dpi)

        save_naive_matches_visualization(
            template_image=template_img,
            query_image=query_img,
            template_keypoints=template_kps,
            scene_keypoints=scene_kps,
            raw_matches=raw_matches,
            output_path=_p(output_dir, "06_naive_matches.jpg"),
            dpi=config.plot_dpi,
        )
        save_match_distance_histogram(raw_matches, _p(output_dir, "07_match_distance_histogram.png"), dpi=config.plot_dpi)
        save_lowe_ratio_histogram(matcher_diag, config.ratio_threshold, _p(output_dir, "08_lowe_ratio_histogram.png"), dpi=config.plot_dpi)
        save_matches_after_ratio_test(
            template_image=template_img,
            query_image=query_img,
            template_keypoints=template_kps,
            scene_keypoints=scene_kps,
            ratio_matches=ratio_matches,
            output_path=_p(output_dir, "09_matches_after_ratio_test.jpg"),
            dpi=config.plot_dpi,
        )

        save_hough_center_heatmap(initial_hypotheses, query_img, _p(output_dir, "10_hough_center_heatmap.png"), dpi=config.plot_dpi)
        save_hough_scale_distribution(initial_hypotheses, clusters, _p(output_dir, "11_hough_scale_distribution.png"), dpi=config.plot_dpi)
        save_hough_rotation_distribution(initial_hypotheses, clusters, _p(output_dir, "12_hough_rotation_distribution.png"), dpi=config.plot_dpi)
        save_hough_votes_visualization(query_img, initial_hypotheses, clusters, _p(output_dir, "13_hough_votes_visualization.jpg"), dpi=config.plot_dpi)

    detection_report = run_multi_instance_detection_with_report(
        template_image=template_img,
        scene_image=query_img,
        template_keypoints=template_kps,
        scene_keypoints=scene_kps,
        ratio_matches=ratio_matches,
        config=config,
    )

    detections_before_nms = detection_report.detections_before_nms
    final_detections = detection_report.detections_after_nms

    keep_ids = non_max_suppression_indices(
        [d.bbox_xyxy for d in detections_before_nms],
        [d.score for d in detections_before_nms],
        config.nms_iou_threshold,
    )
    final_traces = [detection_report.detection_traces[i] for i in keep_ids] if keep_ids else []

    if config.visualization_enabled:
        save_ransac_inliers_outliers(
            template_image=template_img,
            scene_image=query_img,
            template_keypoints=template_kps,
            scene_keypoints=scene_kps,
            detection_traces=detection_report.detection_traces,
            output_path=_p(output_dir, "14_ransac_inliers_outliers.jpg"),
            dpi=config.plot_dpi,
        )
        save_affine_projection_visualization(
            scene_image=query_img,
            detections=detections_before_nms,
            detection_traces=detection_report.detection_traces,
            template_keypoints=template_kps,
            scene_keypoints=scene_kps,
            output_path=_p(output_dir, "15_affine_projection.jpg"),
            dpi=config.plot_dpi,
        )
        save_detections_before_nms(query_img, detections_before_nms, _p(output_dir, "16_detections_before_nms.jpg"), dpi=config.plot_dpi)

    iou_matrix = np.zeros((len(detections_before_nms), len(detections_before_nms)), dtype=np.float64)
    for i, det_i in enumerate(detections_before_nms):
        for j, det_j in enumerate(detections_before_nms):
            iou_matrix[i, j] = iou_xyxy(det_i.bbox_xyxy, det_j.bbox_xyxy)

    if config.visualization_enabled:
        save_iou_matrix(detections_before_nms, iou_matrix, _p(output_dir, "17_iou_matrix.png"), dpi=config.plot_dpi)
        save_final_detections(query_img, final_detections, _p(output_dir, "18_final_detection.jpg"), dpi=config.plot_dpi)

    template_keypoint_vis = _read_image(_p(output_dir, "02_template_keypoints.jpg")) if config.visualization_enabled else template_img
    scene_keypoint_vis = _read_image(_p(output_dir, "03_scene_keypoints.jpg")) if config.visualization_enabled else query_img
    naive_matches_vis = _read_image(_p(output_dir, "06_naive_matches.jpg")) if config.visualization_enabled else query_img
    ratio_matches_vis = _read_image(_p(output_dir, "09_matches_after_ratio_test.jpg")) if config.visualization_enabled else query_img
    hough_votes_vis = _read_image(_p(output_dir, "13_hough_votes_visualization.jpg")) if config.visualization_enabled else query_img
    affine_projection_vis = _read_image(_p(output_dir, "15_affine_projection.jpg")) if config.visualization_enabled else query_img
    final_detections_vis = _read_image(_p(output_dir, "18_final_detection.jpg")) if config.visualization_enabled else query_img

    if config.visualization_enabled and config.debug_visualization:
        save_pipeline_summary(
            template_image=template_img,
            query_image=query_img,
            template_keypoint_vis=template_keypoint_vis,
            scene_keypoint_vis=scene_keypoint_vis,
            naive_matches_vis=naive_matches_vis,
            ratio_matches_vis=ratio_matches_vis,
            hough_votes_vis=hough_votes_vis,
            affine_projection_vis=affine_projection_vis,
            final_detections_vis=final_detections_vis,
            output_path=_p(output_dir, "19_pipeline_summary.jpg"),
            dpi=config.plot_dpi,
        )

    stats = {
        "template_image_dimensions": list(template_img.shape),
        "scene_image_dimensions": list(query_img.shape),
        "template_keypoint_count": len(template_kps),
        "scene_keypoint_count": len(scene_kps),
        "raw_candidate_match_count": matcher_diag.candidate_count,
        "lowe_ratio_accepted_match_count": matcher_diag.accepted_count,
        "lowe_ratio_rejected_count": matcher_diag.rejected_count,
        "hough_hypothesis_count": len(initial_hypotheses),
        "hough_cluster_count": len(clusters),
        "ransac_models_tested": detection_report.ransac_models_tested,
        "detections_before_nms": len(detections_before_nms),
        "detections_after_nms": len(final_detections),
        "detections_before_nms_details": [
            _detection_stats(det, trace) for det, trace in zip(detections_before_nms, detection_report.detection_traces)
        ],
        "detections_after_nms_details": [
            _detection_stats(det, trace) for det, trace in zip(final_detections, final_traces)
        ],
        "detections": [
            _detection_stats(det, trace) for det, trace in zip(final_detections, final_traces)
        ],
        "iou_matrix": iou_matrix.tolist(),
    }

    _save_statistics_json(_p(output_dir, "statistics.json"), stats)
    if config.save_statistics_txt:
        _save_statistics_txt(_p(output_dir, "statistics.txt"), stats)

    print("\n=== Detection Summary ===")
    print(f"Final detections: {len(final_detections)}")
    for i, det in enumerate(final_detections, start=1):
        x1, y1, x2, y2 = det.bbox_xyxy.tolist()
        print(
            f"{i}: bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), "
            f"inliers={det.inliers}, votes={det.hough_votes}, err={det.mean_error:.2f}"
        )


if __name__ == "__main__":
    main()
