"""Configuration parameters for CV Assignment 1 pipeline."""

from dataclasses import dataclass


VISUALIZATION_ENABLED = True
DEBUG_VISUALIZATION = True
MAX_KEYPOINTS_VISUALIZED = 250
PLOT_DPI = 220


@dataclass(frozen=True)
class Config:
    # Paths
    template_path: str = "data/template.jpeg"
    query_path: str = "data/query.jpeg"
    output_dir: str = "outputs"

    # Debug
    verbose: bool = True
    random_seed: int = 42
    visualization_enabled: bool = VISUALIZATION_ENABLED
    debug_visualization: bool = DEBUG_VISUALIZATION
    max_keypoints_visualized: int = MAX_KEYPOINTS_VISUALIZED
    plot_dpi: int = PLOT_DPI

    # Matching
    ratio_threshold: float = 0.85
    min_descriptor_norm: float = 1e-12

    # Hough (4D) binning
    hough_x_bin_size: float = 30.0
    hough_y_bin_size: float = 30.0
    hough_scale_bin_size: float = 0.3
    hough_angle_bin_size_deg: float = 20.0
    min_hough_votes: int = 3

    # RANSAC
    ransac_iterations: int = 3000
    ransac_error_threshold: float = 8.0
    min_inliers: int = 5

    # Geometric sanity
    collinearity_eps: float = 1.5
    det_epsilon: float = 1e-3
    min_affine_scale: float = 0.25
    max_affine_scale: float = 4.0
    max_mean_reprojection_error: float = 10.0

    # Detection loop
    min_remaining_matches: int = 8
    max_detections: int = 25

    # Bounding-box sanity
    min_bbox_area: float = 150.0
    max_outside_fraction: float = 0.4

    # NMS
    nms_iou_threshold: float = 0.5

    # Reporting / outputs
    save_statistics_txt: bool = True


def default_config() -> Config:
    return Config()
