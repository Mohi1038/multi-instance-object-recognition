"""Visualization utilities for assignment outputs."""

from __future__ import annotations

import math
import os
from typing import List, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .detection import Detection, DetectionTrace
from .geometry import affine_rotation_estimate_rad
from .hough import HoughCluster, HoughHypothesis
from .matching import FeatureMatch, MatcherDiagnostics, RawNearestMatch


def _to_int_pt(pt: Tuple[float, float]) -> Tuple[int, int]:
    return int(round(pt[0])), int(round(pt[1]))


def draw_naive_matches(
    template_image: np.ndarray,
    scene_image: np.ndarray,
    template_keypoints: Sequence[cv2.KeyPoint],
    scene_keypoints: Sequence[cv2.KeyPoint],
    raw_matches: Sequence[RawNearestMatch],
    max_draw: int = 400,
) -> np.ndarray:
    """Draw naive nearest-neighbor correspondences without robust filtering."""
    h1, w1 = template_image.shape[:2]
    h2, w2 = scene_image.shape[:2]

    out_h = max(h1, h2)
    out_w = w1 + w2
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[:h1, :w1] = template_image
    canvas[:h2, w1 : w1 + w2] = scene_image

    draw_count = min(len(raw_matches), max_draw)
    rng = np.random.default_rng(123)

    for i in range(draw_count):
        m = raw_matches[i]
        t_pt = template_keypoints[m.template_idx].pt
        s_pt = scene_keypoints[m.scene_idx].pt

        p1 = _to_int_pt((t_pt[0], t_pt[1]))
        p2 = _to_int_pt((s_pt[0] + w1, s_pt[1]))

        color = tuple(int(c) for c in rng.integers(40, 255, size=3))
        cv2.circle(canvas, p1, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 2, color, -1, cv2.LINE_AA)
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)

    return canvas


def draw_hough_centers(scene_image: np.ndarray, clusters: Sequence[HoughCluster], top_k: int = 30) -> np.ndarray:
    vis = scene_image.copy()
    rng = np.random.default_rng(456)

    for i, c in enumerate(clusters[:top_k]):
        color = tuple(int(v) for v in rng.integers(20, 255, size=3))
        center = _to_int_pt((c.mean_center_x, c.mean_center_y))
        radius = int(max(4, min(30, 2 + c.vote_count)))
        cv2.circle(vis, center, radius, color, 2, cv2.LINE_AA)
        cv2.putText(
            vis,
            f"v={c.vote_count}",
            (center[0] + 4, center[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    return vis


def draw_final_detections(scene_image: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    vis = scene_image.copy()
    for i, det in enumerate(detections):
        poly_i = np.round(det.polygon).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [poly_i], isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)

        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 180, 0), 2, cv2.LINE_AA)
        cv2.putText(
            vis,
            f"ID {i+1} | inliers={det.inliers} | score={det.score:.1f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return vis


def ensure_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def _bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _save_figure(fig: plt.Figure, output_path: str, dpi: int = 220) -> None:
    _ensure_parent_dir(output_path)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _save_image_canvas(canvas_bgr: np.ndarray, output_path: str, title: str | None = None, figsize: Tuple[float, float] | None = None, dpi: int = 220) -> None:
    if figsize is None:
        figsize = (12, max(4.0, 12.0 * canvas_bgr.shape[0] / max(1, canvas_bgr.shape[1])))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(_bgr_to_rgb(canvas_bgr))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14)
    _save_figure(fig, output_path, dpi=dpi)


def _draw_text_box(image: np.ndarray, lines: Sequence[str], origin: Tuple[int, int] = (12, 28), color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    out = image.copy()
    x, y = origin
    line_height = 18
    max_box_width = 260
    box_width = 0
    for line in lines:
        (w, _), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        box_width = max(box_width, w)
    pad = 8
    box_width = min(box_width + 2 * pad, max_box_width)
    box_height = line_height * len(lines) + 2 * pad
    cv2.rectangle(out, (x - pad, y - 20), (x + box_width + pad, y - 20 + box_height), (0, 0, 0), -1)
    cv2.rectangle(out, (x - pad, y - 20), (x + box_width + pad, y - 20 + box_height), (255, 255, 255), 1)
    for i, line in enumerate(lines):
        yy = y + i * line_height
        cv2.putText(out, line, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def _select_keypoints(keypoints: Sequence[cv2.KeyPoint], max_points: int) -> List[cv2.KeyPoint]:
    if len(keypoints) <= max_points:
        return list(keypoints)
    ordered = sorted(keypoints, key=lambda kp: float(getattr(kp, "response", 0.0)), reverse=True)
    indices = np.linspace(0, len(ordered) - 1, num=max_points, dtype=int)
    return [ordered[i] for i in indices]


def _draw_keypoints_overlay(image_bgr: np.ndarray, keypoints: Sequence[cv2.KeyPoint], max_points: int, title: str) -> np.ndarray:
    vis = image_bgr.copy()
    selected = _select_keypoints(keypoints, max_points)

    for kp in selected:
        x, y = kp.pt
        radius = int(max(2.0, min(18.0, kp.size * 0.5)))
        center = (int(round(x)), int(round(y)))
        cv2.circle(vis, center, radius, (0, 255, 255), 1, cv2.LINE_AA)

        angle_deg = float(kp.angle)
        if angle_deg >= 0.0:
            angle_rad = math.radians(angle_deg)
            length = max(8.0, radius * 1.4)
            tip = (
                int(round(x + length * math.cos(angle_rad))),
                int(round(y + length * math.sin(angle_rad))),
            )
            cv2.arrowedLine(vis, center, tip, (255, 80, 0), 1, cv2.LINE_AA, tipLength=0.25)

    stats = [
        title,
        f"Total keypoints: {len(keypoints)}",
        f"Displayed keypoints: {len(selected)}",
        "Yellow circles = position / scale",
        "Orange arrows = orientation",
    ]
    return _draw_text_box(vis, stats)


def _get_match_fields(match: object) -> tuple[int, int, float]:
    scene_idx = int(getattr(match, "scene_idx"))
    template_idx = int(getattr(match, "template_idx"))
    distance = float(getattr(match, "distance", 0.0))
    return scene_idx, template_idx, distance


def _draw_matches_canvas(
    template_image: np.ndarray,
    scene_image: np.ndarray,
    template_keypoints: Sequence[cv2.KeyPoint],
    scene_keypoints: Sequence[cv2.KeyPoint],
    matches: Sequence[object],
    title: str,
    max_draw: int = 450,
    line_color: Tuple[int, int, int] | None = None,
    draw_random_colors: bool = True,
) -> np.ndarray:
    h1, w1 = template_image.shape[:2]
    h2, w2 = scene_image.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = template_image
    canvas[:h2, w1 : w1 + w2] = scene_image

    draw_count = min(len(matches), max_draw)
    rng = np.random.default_rng(123)
    for i in range(draw_count):
        scene_idx, template_idx, _ = _get_match_fields(matches[i])
        t_pt = template_keypoints[template_idx].pt
        s_pt = scene_keypoints[scene_idx].pt
        p1 = (int(round(t_pt[0])), int(round(t_pt[1])))
        p2 = (int(round(s_pt[0] + w1)), int(round(s_pt[1])))
        if draw_random_colors:
            color = tuple(int(c) for c in rng.integers(50, 255, size=3))
        else:
            color = line_color if line_color is not None else (0, 255, 0)
        cv2.circle(canvas, p1, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p2, 2, color, -1, cv2.LINE_AA)
        cv2.line(canvas, p1, p2, color, 1, cv2.LINE_AA)

    stats = [title, f"Displayed correspondences: {draw_count}"]
    if len(matches) > draw_count:
        stats.append(f"Additional correspondences omitted: {len(matches) - draw_count}")
    return _draw_text_box(canvas, stats)


def save_input_overview(template_image: np.ndarray, query_image: np.ndarray, output_path: str, dpi: int = 220) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, image, title in [
        (axes[0], template_image, "Template image"),
        (axes[1], query_image, "Query image"),
    ]:
        ax.imshow(_bgr_to_rgb(image))
        ax.set_title(title, fontsize=14)
        ax.axis("off")
    _save_figure(fig, output_path, dpi=dpi)


def save_keypoint_visualization(
    image: np.ndarray,
    keypoints: Sequence[cv2.KeyPoint],
    output_path: str,
    title: str,
    max_points: int,
    dpi: int = 220,
) -> None:
    canvas = _draw_keypoints_overlay(image, keypoints, max_points=max_points, title=title)
    _save_image_canvas(canvas, output_path, title=title, figsize=(12, 9), dpi=dpi)


def save_keypoint_scale_distribution(
    template_keypoints: Sequence[cv2.KeyPoint],
    scene_keypoints: Sequence[cv2.KeyPoint],
    output_path: str,
    dpi: int = 220,
) -> None:
    template_sizes = np.asarray([kp.size for kp in template_keypoints], dtype=np.float64)
    scene_sizes = np.asarray([kp.size for kp in scene_keypoints], dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    if template_sizes.size:
        ax.hist(template_sizes, bins=30, alpha=0.65, label=f"Template ({len(template_sizes)})", color="#1f77b4")
    if scene_sizes.size:
        ax.hist(scene_sizes, bins=30, alpha=0.55, label=f"Scene ({len(scene_sizes)})", color="#ff7f0e")
    ax.set_title("Keypoint scale distribution", fontsize=14)
    ax.set_xlabel("Keypoint scale (OpenCV SIFT size)")
    ax.set_ylabel("Number of keypoints")
    ax.legend()
    ax.grid(True, alpha=0.25)
    _save_figure(fig, output_path, dpi=dpi)


def save_keypoint_orientation_histogram(
    template_keypoints: Sequence[cv2.KeyPoint],
    scene_keypoints: Sequence[cv2.KeyPoint],
    output_path: str,
    dpi: int = 220,
) -> None:
    template_angles = np.asarray([kp.angle for kp in template_keypoints if kp.angle >= 0], dtype=np.float64)
    scene_angles = np.asarray([kp.angle for kp in scene_keypoints if kp.angle >= 0], dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bins = np.linspace(0.0, 360.0, 37)
    if template_angles.size:
        ax.hist(template_angles, bins=bins, alpha=0.65, label=f"Template ({len(template_angles)})", color="#2ca02c")
    if scene_angles.size:
        ax.hist(scene_angles, bins=bins, alpha=0.55, label=f"Scene ({len(scene_angles)})", color="#d62728")
    ax.set_title("Keypoint orientation distribution", fontsize=14)
    ax.set_xlabel("Keypoint orientation (degrees)")
    ax.set_ylabel("Number of keypoints")
    ax.set_xlim(0, 360)
    ax.legend()
    ax.grid(True, alpha=0.25)
    _save_figure(fig, output_path, dpi=dpi)


def save_match_distance_histogram(
    raw_matches: Sequence[RawNearestMatch],
    output_path: str,
    dpi: int = 220,
) -> None:
    distances = np.asarray([m.distance for m in raw_matches], dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    if distances.size:
        ax.hist(distances, bins=35, color="#9467bd", alpha=0.8)
    ax.set_title("Nearest-neighbor descriptor distance histogram", fontsize=14)
    ax.set_xlabel("Descriptor distance")
    ax.set_ylabel("Number of matches")
    ax.grid(True, alpha=0.25)
    ax.text(0.98, 0.95, f"Matches: {len(distances)}", transform=ax.transAxes, ha="right", va="top", fontsize=11, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    _save_figure(fig, output_path, dpi=dpi)


def save_lowe_ratio_histogram(
    diagnostics: MatcherDiagnostics,
    ratio_threshold: float,
    output_path: str,
    dpi: int = 220,
) -> None:
    ratios = np.asarray(diagnostics.ratios, dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    if ratios.size:
        ax.hist(ratios, bins=35, color="#17becf", alpha=0.85)
    ax.axvline(ratio_threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold = {ratio_threshold:.2f}")
    ax.set_title("Lowe ratio histogram", fontsize=14)
    ax.set_xlabel("Ratio d1 / d2")
    ax.set_ylabel("Number of candidate correspondences")
    ax.set_xlim(0, max(1.2, float(ratios.max() + 0.1) if ratios.size else 1.2))
    ax.legend()
    ax.grid(True, alpha=0.25)
    accepted = diagnostics.accepted_count
    rejected = diagnostics.rejected_count
    candidate = diagnostics.candidate_count
    acceptance_pct = 100.0 * accepted / max(1, candidate)
    text = (
        f"Candidate correspondences: {candidate}\n"
        f"Accepted matches: {accepted}\n"
        f"Rejected matches: {rejected}\n"
        f"Acceptance: {acceptance_pct:.1f}%"
    )
    ax.text(0.98, 0.95, text, transform=ax.transAxes, ha="right", va="top", fontsize=11, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
    _save_figure(fig, output_path, dpi=dpi)


def save_matches_after_ratio_test(
    template_image: np.ndarray,
    query_image: np.ndarray,
    template_keypoints: Sequence[cv2.KeyPoint],
    scene_keypoints: Sequence[cv2.KeyPoint],
    ratio_matches: Sequence[FeatureMatch],
    output_path: str,
    dpi: int = 220,
) -> None:
    canvas = _draw_matches_canvas(
        template_image=template_image,
        scene_image=query_image,
        template_keypoints=template_keypoints,
        scene_keypoints=scene_keypoints,
        matches=ratio_matches,
        title="Matches after Lowe ratio test",
        max_draw=180,
        line_color=(0, 180, 255),
        draw_random_colors=False,
    )
    _save_image_canvas(canvas, output_path, title="Matches after Lowe ratio test", figsize=(14, 8), dpi=dpi)


def save_naive_matches_visualization(
    template_image: np.ndarray,
    query_image: np.ndarray,
    template_keypoints: Sequence[cv2.KeyPoint],
    scene_keypoints: Sequence[cv2.KeyPoint],
    raw_matches: Sequence[RawNearestMatch],
    output_path: str,
    dpi: int = 220,
) -> None:
    canvas = _draw_matches_canvas(
        template_image=template_image,
        scene_image=query_image,
        template_keypoints=template_keypoints,
        scene_keypoints=scene_keypoints,
        matches=raw_matches,
        title="Naive nearest-neighbor matches",
        max_draw=120,
        draw_random_colors=True,
    )
    _save_image_canvas(canvas, output_path, title="Naive nearest-neighbor matches", figsize=(14, 8), dpi=dpi)


def save_hough_center_heatmap(
    hypotheses: Sequence[HoughHypothesis],
    scene_image: np.ndarray,
    output_path: str,
    dpi: int = 220,
) -> None:
    h, w = scene_image.shape[:2]
    centers_x = np.asarray([h.center_x for h in hypotheses], dtype=np.float64)
    centers_y = np.asarray([h.center_y for h in hypotheses], dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    if centers_x.size:
        heatmap, _, _ = np.histogram2d(centers_x, centers_y, bins=[50, 50], range=[[0, w], [0, h]])
        im = ax.imshow(
            heatmap.T,
            origin="lower",
            extent=[0, w, 0, h],
            cmap="inferno",
            aspect="auto",
        )
        fig.colorbar(im, ax=ax, label="Votes")
    else:
        ax.text(0.5, 0.5, "No Hough hypotheses", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("Hough center heatmap", fontsize=14)
    ax.set_xlabel("Predicted object center X")
    ax.set_ylabel("Predicted object center Y")
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    _save_figure(fig, output_path, dpi=dpi)


def save_hough_scale_distribution(
    hypotheses: Sequence[HoughHypothesis],
    clusters: Sequence[HoughCluster],
    output_path: str,
    dpi: int = 220,
) -> None:
    scales = np.asarray([h.scale for h in hypotheses], dtype=np.float64)
    cluster_scales = np.asarray([c.mean_scale for c in clusters], dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    if scales.size:
        ax.hist(scales, bins=35, color="#8c564b", alpha=0.8, label=f"All hypotheses ({len(scales)})")
    if cluster_scales.size:
        ax.hist(cluster_scales, bins=20, color="#e377c2", alpha=0.5, label=f"Cluster means ({len(cluster_scales)})")
    ax.set_title("Hough scale distribution", fontsize=14)
    ax.set_xlabel("Predicted object scale")
    ax.set_ylabel("Number of hypotheses")
    ax.legend()
    ax.grid(True, alpha=0.25)
    _save_figure(fig, output_path, dpi=dpi)


def save_hough_rotation_distribution(
    hypotheses: Sequence[HoughHypothesis],
    clusters: Sequence[HoughCluster],
    output_path: str,
    dpi: int = 220,
) -> None:
    rotations = np.asarray([np.degrees(h.rotation_rad) for h in hypotheses], dtype=np.float64)
    cluster_rotations = np.asarray([np.degrees(c.mean_rotation_rad) for c in clusters], dtype=np.float64)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bins = np.linspace(-180.0, 180.0, 37)
    if rotations.size:
        ax.hist(rotations, bins=bins, color="#7f7f7f", alpha=0.8, label=f"All hypotheses ({len(rotations)})")
    if cluster_rotations.size:
        ax.hist(cluster_rotations, bins=bins, color="#bcbd22", alpha=0.55, label=f"Cluster means ({len(cluster_rotations)})")
    ax.set_title("Hough rotation distribution", fontsize=14)
    ax.set_xlabel("Predicted object rotation (degrees)")
    ax.set_ylabel("Number of hypotheses")
    ax.set_xlim(-180, 180)
    ax.legend()
    ax.grid(True, alpha=0.25)
    _save_figure(fig, output_path, dpi=dpi)


def save_hough_votes_visualization(
    scene_image: np.ndarray,
    hypotheses: Sequence[HoughHypothesis],
    clusters: Sequence[HoughCluster],
    output_path: str,
    dpi: int = 220,
) -> None:
    rgb = _bgr_to_rgb(scene_image)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(rgb)
    if hypotheses:
        xs = np.asarray([h.center_x for h in hypotheses], dtype=np.float64)
        ys = np.asarray([h.center_y for h in hypotheses], dtype=np.float64)
        ax.scatter(xs, ys, s=18, c="#4c72b0", alpha=0.20, label="Individual hypotheses")
    if clusters:
        cx = np.asarray([c.mean_center_x for c in clusters], dtype=np.float64)
        cy = np.asarray([c.mean_center_y for c in clusters], dtype=np.float64)
        votes = np.asarray([c.vote_count for c in clusters], dtype=np.float64)
        sizes = 90.0 + 35.0 * votes
        sc = ax.scatter(cx, cy, s=sizes, c=votes, cmap="plasma", edgecolors="white", linewidths=1.0, label="Cluster centers")
        fig.colorbar(sc, ax=ax, label="Vote count")
        for cluster in clusters[:10]:
            ax.text(cluster.mean_center_x + 5, cluster.mean_center_y - 5, f"{cluster.vote_count}", color="white", fontsize=9, weight="bold", bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"))
    ax.set_title("Hough votes on the query image", fontsize=14)
    ax.axis("off")
    ax.legend(loc="lower left")
    _save_figure(fig, output_path, dpi=dpi)


def save_ransac_inliers_outliers(
    template_image: np.ndarray,
    scene_image: np.ndarray,
    template_keypoints: Sequence[cv2.KeyPoint],
    scene_keypoints: Sequence[cv2.KeyPoint],
    detection_traces: Sequence[DetectionTrace],
    output_path: str,
    dpi: int = 220,
    max_outliers_draw: int = 60,
) -> None:
    if not detection_traces:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, "No accepted detections to visualize", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        _save_figure(fig, output_path, dpi=dpi)
        return

    usable = min(len(detection_traces), 9)
    nrows, ncols = 3, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 18))
    fig.subplots_adjust(wspace=0.08, hspace=0.10, top=0.92, bottom=0.04, left=0.03, right=0.97)
    axes = axes.flatten()

    rng = np.random.default_rng(2026)
    for idx in range(nrows * ncols):
        ax = axes[idx]
        if idx >= usable:
            ax.axis("off")
            continue

        trace = detection_traces[idx]
        h1, w1 = template_image.shape[:2]
        h2, w2 = scene_image.shape[:2]
        canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        canvas[:h1, :w1] = template_image
        canvas[:h2, w1 : w1 + w2] = scene_image

        candidate_matches = trace.candidate_matches
        outliers = list(trace.outlier_indices)
        if len(outliers) > max_outliers_draw:
            outliers = list(rng.choice(outliers, size=max_outliers_draw, replace=False))

        for idx_match in outliers:
            m = candidate_matches[idx_match]
            t_pt = template_keypoints[m.template_idx].pt
            s_pt = scene_keypoints[m.scene_idx].pt
            p1 = (int(round(t_pt[0])), int(round(t_pt[1])))
            p2 = (int(round(s_pt[0] + w1)), int(round(s_pt[1])))
            cv2.line(canvas, p1, p2, (40, 40, 255), 1, cv2.LINE_AA)

        for idx_match in trace.inlier_indices:
            m = candidate_matches[idx_match]
            t_pt = template_keypoints[m.template_idx].pt
            s_pt = scene_keypoints[m.scene_idx].pt
            p1 = (int(round(t_pt[0])), int(round(t_pt[1])))
            p2 = (int(round(s_pt[0] + w1)), int(round(s_pt[1])))
            cv2.line(canvas, p1, p2, (0, 200, 0), 2, cv2.LINE_AA)
            cv2.circle(canvas, p1, 3, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, p2, 3, (0, 255, 0), -1, cv2.LINE_AA)

        stats = [
            f"Detection {idx + 1}",
            f"Hough votes: {trace.cluster.vote_count}",
            f"Inliers: {len(trace.inlier_indices)}",
            f"Outliers: {len(candidate_matches) - len(trace.inlier_indices)}",
            f"Inlier ratio: {trace.inlier_ratio:.2f}",
            f"Mean error: {trace.ransac_result.mean_error:.2f}",
        ]
        canvas = _draw_text_box(canvas, stats, origin=(16, 24))
        ax.imshow(_bgr_to_rgb(canvas))
        ax.axis("off")
        ax.set_title(f"RANSAC correspondence check {idx + 1}", fontsize=12)

    fig.suptitle("RANSAC inlier/outlier filtering", fontsize=16, y=0.98)
    _save_figure(fig, output_path, dpi=dpi)


def save_affine_projection_visualization(
    scene_image: np.ndarray,
    detections: Sequence[Detection],
    detection_traces: Sequence[DetectionTrace],
    template_keypoints: Sequence[cv2.KeyPoint],
    scene_keypoints: Sequence[cv2.KeyPoint],
    output_path: str,
    dpi: int = 220,
) -> None:
    if not detections:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.text(0.5, 0.5, "No detections to project", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        _save_figure(fig, output_path, dpi=dpi)
        return

    usable = min(len(detections), 9)
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    fig.subplots_adjust(wspace=0.08, hspace=0.10, top=0.92, bottom=0.04, left=0.03, right=0.97)
    axes = axes.flatten()

    for idx in range(9):
        ax = axes[idx]
        if idx >= usable:
            ax.axis("off")
            continue

        det = detections[idx]
        trace = detection_traces[idx]
        vis = scene_image.copy()
        poly = np.round(det.polygon).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis, [poly], True, (0, 255, 0), 3, cv2.LINE_AA)
        for match_idx in trace.inlier_indices:
            match = trace.candidate_matches[match_idx]
            pt = scene_keypoints[match.scene_idx].pt
            center = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(vis, center, 4, (0, 255, 255), -1, cv2.LINE_AA)

        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 180, 0), 2, cv2.LINE_AA)
        vis = _draw_text_box(
            vis,
            [
                f"Detection {idx + 1}",
                f"Determinant: {det.determinant:.3f}",
                f"Estimated scale: {det.scale:.3f}",
                f"Rotation: {np.degrees(affine_rotation_estimate_rad(det.affine_2x3)):.1f} deg",
            ],
            origin=(18, 28),
        )
        ax.imshow(_bgr_to_rgb(vis))
        ax.axis("off")
        ax.set_title(f"Affine projection {idx + 1}", fontsize=12)

    fig.suptitle("Projected template alignment on the scene", fontsize=16, y=0.98)
    _save_figure(fig, output_path, dpi=dpi)


def save_detections_before_nms(
    scene_image: np.ndarray,
    detections: Sequence[Detection],
    output_path: str,
    dpi: int = 220,
) -> None:
    vis = scene_image.copy()
    for idx, det in enumerate(detections, start=1):
        poly = np.round(det.polygon).astype(np.int32).reshape((-1, 1, 2))
        color = (0, 255, 0) if idx % 2 else (255, 180, 0)
        cv2.polylines(vis, [poly], True, color, 3, cv2.LINE_AA)
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.putText(
            vis,
            f"ID {idx} | score={det.score:.1f} | inliers={det.inliers}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    _save_image_canvas(vis, output_path, title="Detections before NMS", figsize=(12, 9), dpi=dpi)


def save_iou_matrix(
    detections: Sequence[Detection],
    iou_matrix: np.ndarray,
    output_path: str,
    dpi: int = 220,
) -> None:
    n = len(detections)
    display_limit = 12
    display_indices = sorted(range(n), key=lambda index: detections[index].score, reverse=True)[:display_limit]
    display_matrix = iou_matrix[np.ix_(display_indices, display_indices)] if display_indices else iou_matrix
    display_count = len(display_indices)
    fig, ax = plt.subplots(1, 1, figsize=(max(7, 0.72 * max(1, display_count)), max(6, 0.72 * max(1, display_count))))
    if n == 0:
        ax.text(0.5, 0.5, "No candidate detections available", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        _save_figure(fig, output_path, dpi=dpi)
        return

    im = ax.imshow(display_matrix, vmin=0.0, vmax=1.0, cmap="viridis")
    fig.colorbar(im, ax=ax, label="IoU")
    suffix = f" (top {display_count} by score)" if n > display_count else ""
    ax.set_title(f"IoU matrix for candidate detections{suffix}", fontsize=14)
    labels = [f"D{i + 1}" for i in display_indices]
    ax.set_xticks(range(display_count), labels)
    ax.set_yticks(range(display_count), labels)
    ax.set_xlabel("Detection ID")
    ax.set_ylabel("Detection ID")

    for i in range(display_count):
        for j in range(display_count):
            value = display_matrix[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.55 else "black", fontsize=8)

    _save_figure(fig, output_path, dpi=dpi)


def save_final_detections(
    scene_image: np.ndarray,
    detections: Sequence[Detection],
    output_path: str,
    dpi: int = 220,
) -> None:
    vis = scene_image.copy()
    colors = [(0, 235, 255), (255, 170, 0), (80, 220, 120), (220, 100, 255)]
    for idx, det in enumerate(detections, start=1):
        poly = np.round(det.polygon).astype(np.int32).reshape((-1, 1, 2))
        color = colors[(idx - 1) % len(colors)]
        cv2.polylines(vis, [poly], True, color, 4, cv2.LINE_AA)
        x1, y1, x2, y2 = [int(round(v)) for v in det.bbox_xyxy]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        label = f"Object {idx}  |  {det.inliers} inliers"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        label_x = max(4, min(x1, vis.shape[1] - label_width - 8))
        label_y = max(label_height + baseline + 4, y1 - 8)
        cv2.rectangle(
            vis,
            (label_x - 4, label_y - label_height - baseline - 4),
            (label_x + label_width + 4, label_y + 4),
            (20, 20, 20),
            -1,
        )
        cv2.putText(vis, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    _save_image_canvas(vis, output_path, title=f"Final detections ({len(detections)} objects)", figsize=(14, 9), dpi=dpi)


def save_hough_pipeline_visualization(
    scene_image: np.ndarray,
    hypotheses: Sequence[HoughHypothesis],
    clusters: Sequence[HoughCluster],
    output_path: str,
    dpi: int = 220,
) -> None:
    save_hough_votes_visualization(scene_image, hypotheses, clusters, output_path, dpi=dpi)


def save_pipeline_summary(
    template_image: np.ndarray,
    query_image: np.ndarray,
    template_keypoint_vis: np.ndarray,
    scene_keypoint_vis: np.ndarray,
    naive_matches_vis: np.ndarray,
    ratio_matches_vis: np.ndarray,
    hough_votes_vis: np.ndarray,
    affine_projection_vis: np.ndarray,
    final_detections_vis: np.ndarray,
    output_path: str,
    dpi: int = 220,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    panels = [
        (_bgr_to_rgb(template_image), "Input: template"),
        (_bgr_to_rgb(query_image), "Input: query"),
        (_bgr_to_rgb(ratio_matches_vis), "Ratio-test matches"),
        (_bgr_to_rgb(hough_votes_vis), "Hough clusters"),
        (_bgr_to_rgb(affine_projection_vis), "Affine projection"),
        (_bgr_to_rgb(final_detections_vis), "Final detections"),
    ]
    for ax, (img, title) in zip(axes.flat, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle("CV Assignment 1 | Recognition pipeline", fontsize=18, fontweight="bold")
    _save_figure(fig, output_path, dpi=dpi)
