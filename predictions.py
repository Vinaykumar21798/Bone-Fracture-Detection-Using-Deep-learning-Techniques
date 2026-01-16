"""
Prediction module for bone fracture detection
Outputs:
- Bone type
- Fracture confidence
- Displacement angle
- Severity
- Recommendation
- ROI image
- Grad-CAM overlay with ROI
"""

from typing import Dict, Any, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import cv2

from config import (
    MODEL_BODY_PARTS,
    MODEL_ELBOW_FRAC,
    MODEL_HAND_FRAC,
    MODEL_SHOULDER_FRAC,
    BODY_PARTS,
    FRACTURE_STATUS,
    IMAGE_TARGET_SIZE
)

from explainability.gradcam import generate_gradcam
from localization.roi_extraction import extract_roi_from_heatmap
from severity.severity_estimation import analyze_fracture_severity
from rules.recommendation import get_recommendation


# ======================================================
# MODEL LOADER
# ======================================================

class ModelLoader:
    def __init__(self):
        self.parts = None
        self.elbow = None
        self.hand = None
        self.shoulder = None

    def _load(self, path):
        return tf.keras.models.load_model(str(path), compile=False)

    def load_parts(self):
        if self.parts is None:
            self.parts = self._load(MODEL_BODY_PARTS)
        return self.parts

    def load_fracture(self, body_part: str):
        if body_part == "Elbow":
            if self.elbow is None:
                self.elbow = self._load(MODEL_ELBOW_FRAC)
            return self.elbow
        if body_part == "Hand":
            if self.hand is None:
                self.hand = self._load(MODEL_HAND_FRAC)
            return self.hand
        if body_part == "Shoulder":
            if self.shoulder is None:
                self.shoulder = self._load(MODEL_SHOULDER_FRAC)
            return self.shoulder
        raise ValueError("Invalid body part")


_loader = ModelLoader()


# ======================================================
# IMAGE PREPROCESSING
# ======================================================

def _preprocess_image(img_path: str) -> np.ndarray:
    img = image.load_img(
        img_path,
        target_size=(IMAGE_TARGET_SIZE, IMAGE_TARGET_SIZE)
    )
    arr = image.img_to_array(img)
    return np.expand_dims(arr, axis=0).astype(np.float32)


# ======================================================
# BASIC PREDICTIONS
# ======================================================

def predict_body_part(img_path: str) -> Tuple[str, float]:
    x = _preprocess_image(img_path)
    preds = _loader.load_parts().predict(x, verbose=0)
    idx = int(np.argmax(preds[0]))
    return BODY_PARTS[idx], float(preds[0][idx])


def predict_fracture(img_path: str, body_part: str) -> Tuple[str, float]:
    model = _loader.load_fracture(body_part)
    x = _preprocess_image(img_path)
    preds = model.predict(x, verbose=0)
    idx = int(np.argmax(preds[0]))
    return FRACTURE_STATUS[idx], float(preds[0][idx])


# ======================================================
# FINAL PIPELINE
# ======================================================

def predict_full_enhanced(img_path: str) -> Dict[str, Any]:

    body_part, _ = predict_body_part(img_path)
    frac_status, frac_conf = predict_fracture(img_path, body_part)

    result: Dict[str, Any] = {
        "body_part": body_part,
        "fracture_status": frac_status,
        "fracture_confidence": frac_conf,
        "severity_level": None,
        "displacement_angle": None,
        "recommendation_text": "",
        "roi_image": None,
        "gradcam_heatmap_bbox": None
    }

    # ---------------- NORMAL CASE ----------------
    if frac_status != "fractured":
        result["recommendation_text"] = "No fracture detected."
        return result

    # ---------------- FRACTURED CASE ----------------
    rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    model = _loader.load_fracture(body_part)
    x = _preprocess_image(img_path)

    # ---------------- GRAD-CAM ----------------
    heatmap, _ = generate_gradcam(
        model=model,
        img_path=img_path,
        img_array=x,
        pred_index=0,
        original_img=rgb
    )

    # ---------------- ROI EXTRACTION ----------------
    roi_mask, roi_bbox, _ = extract_roi_from_heatmap(heatmap, rgb)

    # ROI IMAGE
    roi_image = None
    if roi_bbox:
        x0, y0, w, h = roi_bbox
        roi_image = rgb[y0:y0 + h, x0:x0 + w].copy()

    # ---------------- SEVERITY ----------------
    severity, metrics, _ = analyze_fracture_severity(
        original_image=rgb,
        roi_mask=roi_mask,
        bbox=roi_bbox
    )

    displacement_angle = metrics.get("displacement_angle")

    # ---------------- GRAD-CAM OVERLAY ----------------
    heatmap_norm = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    heatmap_color = cv2.resize(
        heatmap_color,
        (rgb.shape[1], rgb.shape[0])
    )

    overlay = cv2.addWeighted(
        rgb, 0.6,
        heatmap_color, 0.4,
        0
    )

    if roi_bbox:
        x0, y0, w, h = roi_bbox
        cv2.rectangle(
            overlay,
            (x0, y0),
            (x0 + w, y0 + h),
            (0, 255, 0),
            2
        )

    # ---------------- RECOMMENDATION ----------------
    rec = get_recommendation(severity)

    recommendation_text = (
        f"{rec.get('title', 'Clinical Recommendation')}\n\n"
        f"{rec.get('recommendation', '')}\n\n"
        f"Treatment Approach: {rec.get('treatment', 'Consult doctor')}\n"
        f"Surgery Required: {rec.get('surgery_required', 'To be evaluated')}\n\n"
        "Recommended Actions:\n"
        + "\n".join(rec.get("actions", []))
    )

    # ---------------- FINAL RESULT ----------------
    result.update({
        "severity_level": severity,
        "displacement_angle": displacement_angle,
        "recommendation_text": recommendation_text,
        "roi_image": roi_image,
        "gradcam_heatmap_bbox": overlay
    })

    return result