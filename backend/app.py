# backend/app.py

import uuid
import cv2
import base64
import shutil
import sys
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# =====================================================
# PATH FIX (for predictions.py in project root)
# =====================================================
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from predictions import predict_full_enhanced

# =====================================================
# APP INIT
# =====================================================
app = FastAPI(title="AI-Based Bone Fracture Detection API")

FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "backend" / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# STATIC FILES
# =====================================================
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# =====================================================
# SERVE FRONTEND
# =====================================================
@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")

# =====================================================
# UTIL: IMAGE → BASE64
# =====================================================
def image_to_base64(img):
    if img is None:
        return None
    success, buffer = cv2.imencode(".png", img)
    if not success:
        return None
    return base64.b64encode(buffer).decode("utf-8")

# =====================================================
# PREDICTION ENDPOINT (ENGLISH ONLY)
# =====================================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # -----------------------------
    # SAVE IMAGE
    # -----------------------------
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    img_path = UPLOAD_DIR / filename

    with img_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # -----------------------------
    # RUN MODEL PIPELINE
    # -----------------------------
    result = predict_full_enhanced(str(img_path))

    # -----------------------------
    # SAFE FALLBACKS
    # -----------------------------
    recommendation = result.get("recommendation_text", "").strip()
    if not recommendation:
        recommendation = "Clinical evaluation and orthopedic consultation recommended."

    # -----------------------------
    # RESPONSE (MATCHES script.js)
    # -----------------------------
    return {
        "bone_type": result.get("body_part") or "Unknown",
        "fracture": result.get("fracture_status") or "Unknown",
        "confidence": float(result.get("fracture_confidence") or 0.0),
        "displacement_angle": result.get("displacement_angle"),
        "severity": result.get("severity_level") or "N/A",
        "recommendation": recommendation,

        # Images for PDF & UI
        "roi_image": image_to_base64(result.get("roi_image")),
        "gradcam_heatmap": image_to_base64(
            result.get("gradcam_heatmap_bbox")
        ),
    }