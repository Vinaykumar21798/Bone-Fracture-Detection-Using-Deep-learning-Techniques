"""
ROI (Region of Interest) extraction module for fracture localization.
Converts Grad-CAM heatmaps into binary masks and extracts fracture regions.
"""

import numpy as np
import cv2
from typing import Tuple, Optional


# -------------------------------------------------------------------
# Heatmap → Binary Mask
# -------------------------------------------------------------------

def heatmap_to_binary_mask(
    heatmap: np.ndarray,
    threshold_percentile: float = 70.0
) -> np.ndarray:
    """
    Convert Grad-CAM heatmap to binary mask using percentile thresholding.

    Args:
        heatmap: Grad-CAM heatmap (float or uint8)
        threshold_percentile: Percentile threshold (0–100)

    Returns:
        Binary mask (uint8, values {0,255})
    """

    if heatmap is None or heatmap.size == 0:
        raise ValueError("Heatmap is empty or None")

    # Ensure 2D
    if heatmap.ndim > 2:
        heatmap = np.squeeze(heatmap)

    # Normalize and convert to uint8 (CRITICAL FIX)
    if heatmap.dtype != np.uint8:
        heatmap = heatmap.astype(np.float32)
        heatmap = np.clip(heatmap, 0, 1) if heatmap.max() <= 1.0 else heatmap
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = heatmap.astype(np.uint8)

    # Percentile threshold
    threshold_value = np.percentile(heatmap, threshold_percentile)

    _, binary_mask = cv2.threshold(
        heatmap,
        threshold_value,
        255,
        cv2.THRESH_BINARY
    )

    return binary_mask.astype(np.uint8)


# -------------------------------------------------------------------
# Largest Connected Component
# -------------------------------------------------------------------

def get_largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    """
    Extract the largest connected component from a binary mask.

    Args:
        binary_mask: Binary mask (uint8, values {0,255})

    Returns:
        Binary mask with only the largest connected component
    """

    if binary_mask is None or binary_mask.size == 0:
        return binary_mask

    if binary_mask.dtype != np.uint8:
        binary_mask = binary_mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask, connectivity=8
    )

    # Only background found
    if num_labels <= 1:
        return binary_mask

    # Ignore background (label 0)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1

    largest_component = np.zeros_like(binary_mask, dtype=np.uint8)
    largest_component[labels == largest_label] = 255

    return largest_component


# -------------------------------------------------------------------
# Bounding Box Extraction
# -------------------------------------------------------------------

def extract_bounding_box(
    binary_mask: np.ndarray
) -> Optional[Tuple[int, int, int, int]]:
    """
    Extract bounding box from a binary mask.

    Args:
        binary_mask: Binary mask (uint8)

    Returns:
        (x, y, w, h) or None
    """

    if binary_mask is None or binary_mask.sum() == 0:
        return None

    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Reject tiny boxes (noise)
    if w < 5 or h < 5:
        return None

    return (x, y, w, h)


# -------------------------------------------------------------------
# Draw Bounding Box
# -------------------------------------------------------------------

def draw_bounding_box(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding box on image.

    Args:
        image: RGB image
        bbox: (x, y, w, h)

    Returns:
        Image with bounding box
    """

    if bbox is None:
        return image

    x, y, w, h = bbox
    output = image.copy()

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        color,
        thickness
    )

    return output


# -------------------------------------------------------------------
# Main ROI Extraction Pipeline
# -------------------------------------------------------------------

def extract_roi_from_heatmap(
    heatmap: np.ndarray,
    original_image: np.ndarray,
    threshold_percentile: float = 70.0
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]], np.ndarray]:
    """
    Extract ROI from Grad-CAM heatmap.

    Args:
        heatmap: Grad-CAM heatmap
        original_image: Original RGB image
        threshold_percentile: Threshold percentile

    Returns:
        (binary_mask, bbox, roi_image)
    """

    if original_image is None:
        raise ValueError("Original image is None")

    # Step 1: Binary mask
    binary_mask = heatmap_to_binary_mask(heatmap, threshold_percentile)

    # Step 2: Largest connected component
    largest_mask = get_largest_connected_component(binary_mask)

    # Step 3: Bounding box
    bbox = extract_bounding_box(largest_mask)

    # Step 4: Draw bounding box
    roi_image = original_image.copy()
    if bbox is not None:
        roi_image = draw_bounding_box(roi_image, bbox)

    return largest_mask, bbox, roi_image


# -------------------------------------------------------------------
# Optional: Crop ROI
# -------------------------------------------------------------------

def get_roi_crop(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: int = 10
) -> Optional[np.ndarray]:
    """
    Crop ROI from image using bounding box.

    Args:
        image: Original image
        bbox: (x, y, w, h)
        padding: Padding in pixels

    Returns:
        Cropped ROI or None
    """

    if bbox is None:
        return None

    x, y, w, h = bbox

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + w + padding)
    y2 = min(image.shape[0], y + h + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    return image[y1:y2, x1:x2]
