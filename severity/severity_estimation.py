"""
Severity estimation module using OpenCV.
Estimates fracture severity using DISPLACEMENT ANGLE (clinically meaningful).
Uses PCA-based orientation analysis (robust for X-rays).
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Dict


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

def preprocess_roi(roi: np.ndarray) -> np.ndarray:
    """Convert ROI to smoothed grayscale image."""
    if roi.ndim == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    roi = cv2.GaussianBlur(roi, (5, 5), 0)
    return roi


# --------------------------------------------------
# Edge Detection
# --------------------------------------------------

def detect_edges(roi: np.ndarray) -> np.ndarray:
    """Detect fracture edges using Canny."""
    roi = preprocess_roi(roi)
    edges = cv2.Canny(roi, 60, 160)
    return edges


# --------------------------------------------------
# 🔑 DISPLACEMENT ANGLE (PCA – ROBUST)
# --------------------------------------------------

def estimate_displacement_angle(edges: np.ndarray) -> float:
    """
    Estimate angular displacement using PCA on edge points.
    Returns angle in degrees (0–90).
    """

    # Get edge coordinates
    points = np.column_stack(np.where(edges > 0))

    # Not enough data → no fracture orientation
    if points.shape[0] < 50:
        return 0.0

    # PCA
    mean = np.mean(points, axis=0)
    centered = points - mean
    cov = np.cov(centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eig(cov)
    dominant_vector = eigenvectors[:, np.argmax(eigenvalues)]

    angle_rad = np.arctan2(dominant_vector[0], dominant_vector[1])
    angle_deg = abs(angle_rad * 180 / np.pi)

    # Normalize to [0, 90]
    if angle_deg > 90:
        angle_deg = 180 - angle_deg

    return round(float(angle_deg), 1)


# --------------------------------------------------
# Severity Classification (ANGLE BASED)
# --------------------------------------------------

def classify_severity(angle: float) -> str:
    """
    Clinically interpretable severity classification.
    """
    if angle < 8:
        return "Low"
    elif angle < 20:
        return "Medium"
    else:
        return "High"


# --------------------------------------------------
# MAIN PIPELINE (USED BY predictions.py)
# --------------------------------------------------

def analyze_fracture_severity(
    original_image: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None
) -> Tuple[str, Dict[str, float], np.ndarray]:
    """
    Full severity analysis pipeline.

    Returns:
        severity_level (str)
        metrics dict { displacement_angle }
        edges image
    """

    # ---- Extract ROI ----
    if bbox is not None:
        x, y, w, h = bbox
        pad = 10
        roi = original_image[
            max(0, y - pad): y + h + pad,
            max(0, x - pad): x + w + pad
        ]
    else:
        roi = original_image.copy()

    # ---- Edge detection ----
    edges = detect_edges(roi)

    # ---- Angle estimation ----
    angle = estimate_displacement_angle(edges)

    # ---- Severity classification ----
    severity = classify_severity(angle)

    metrics = {
        "displacement_angle": angle
    }

    return severity, metrics, edges
