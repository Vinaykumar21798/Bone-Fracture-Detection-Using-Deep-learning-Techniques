"""
Utility functions for the Bone Fracture Detection project.

Includes:
- Dataset loading (patient-wise structure)
- DataFrame creation
- Image path validation
- Medical-safe X-ray preprocessing (CLAHE)
"""

import os
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import pandas as pd


# =====================================================
# DATASET LOADER
# =====================================================

def load_path(path: Path, part: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Load X-ray dataset from directory structure.

    Expected structure:
        Dataset/
            train_valid/ or test/
                Elbow/
                    patient_001/
                        study_positive/
                        study_negative/
                Hand/
                Shoulder/

    Returns:
        List of dicts:
        {
            'body_part': str,
            'patient_id': str,
            'label': 'fractured' | 'normal',
            'image_path': str
        }
    """

    dataset = []

    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    for body_part in os.listdir(path):
        if part and body_part != part:
            continue

        body_path = path / body_part
        if not body_path.is_dir():
            continue

        for patient_id in os.listdir(body_path):
            patient_path = body_path / patient_id
            if not patient_path.is_dir():
                continue

            for study in os.listdir(patient_path):
                study_path = patient_path / study
                if not study_path.is_dir():
                    continue

                # Label extraction
                if study.endswith("positive"):
                    label = "fractured"
                elif study.endswith("negative"):
                    label = "normal"
                else:
                    continue

                for img_name in os.listdir(study_path):
                    img_path = study_path / img_name
                    if img_path.is_file():
                        dataset.append({
                            "body_part": body_part,
                            "patient_id": patient_id,
                            "label": label,
                            "image_path": str(img_path)
                        })

    return dataset


# =====================================================
# DATAFRAME UTILITIES
# =====================================================

def create_dataframe_from_dataset(dataset: List[Dict[str, str]]) -> pd.DataFrame:
    """
    Convert dataset list into a DataFrame compatible with Keras generators.
    """

    df = pd.DataFrame(dataset)
    df = df.rename(columns={
        "image_path": "Filepath",
        "label": "Label"
    })

    return df[["Filepath", "Label"]]


def validate_image_paths(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove invalid image paths.
    """

    valid_df = df[df["Filepath"].apply(lambda x: Path(x).exists())].copy()

    if len(valid_df) == 0:
        raise ValueError("No valid image paths found after validation")

    if len(valid_df) < len(df):
        print(f"⚠️ Removed {len(df) - len(valid_df)} invalid image paths")

    return valid_df.reset_index(drop=True)


# =====================================================
# X-RAY PREPROCESSING (MEDICAL SAFE)
# =====================================================

def preprocess_xray(img):
    """
    Medical-safe X-ray preprocessing using CLAHE.

    Why CLAHE?
    - Enhances bone edges
    - Improves fracture visibility
    - Standard in radiology workflows
    - Does NOT introduce artificial features
    """

    if img is None:
        raise ValueError("Input image is None")

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)
    return enhanced