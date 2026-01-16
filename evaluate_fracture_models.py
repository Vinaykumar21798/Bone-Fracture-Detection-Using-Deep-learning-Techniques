import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from config import DATASET_DIR, IMAGE_SIZE
from utils import preprocess_xray

# =====================================================
# CONFIG
# =====================================================
TEST_DIR = DATASET_DIR / "test"
WEIGHTS_DIR = Path("weights")

BODY_PARTS = ["Elbow", "Hand", "Shoulder"]

MODEL_PATHS = {
    "Elbow": WEIGHTS_DIR / "ResNet50_Elbow_frac.h5",
    "Hand": WEIGHTS_DIR / "ResNet50_Hand_frac.h5",
    "Shoulder": WEIGHTS_DIR / "ResNet50_Shoulder_frac.h5",
}

LABEL_MAP = {
    "normal": 0,
    "fractured": 1
}

# =====================================================
# SAME PREPROCESSING AS TRAINING
# =====================================================
def preprocess_image(img_path):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)

    img = preprocess_xray(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = tf.keras.applications.resnet50.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    return img


# =====================================================
# EVALUATION
# =====================================================
for part in BODY_PARTS:

    print("\n" + "=" * 60)
    print(f"{part} FRACTURE EVALUATION")
    print("=" * 60)

    model = tf.keras.models.load_model(MODEL_PATHS[part], compile=False)

    y_true = []
    y_pred = []

    part_dir = TEST_DIR / part

    for patient_id in os.listdir(part_dir):
        patient_path = part_dir / patient_id

        if not patient_path.is_dir():
            continue

        for study in os.listdir(patient_path):
            study_path = patient_path / study

            if not study_path.is_dir():
                continue

            # ---- LABEL LOGIC (FINAL & CORRECT) ----
            if study.endswith("positive"):
                true_label = LABEL_MAP["fractured"]
            elif study.endswith("negative"):
                true_label = LABEL_MAP["normal"]
            else:
                continue

            for img_name in os.listdir(study_path):
                img_path = study_path / img_name

                try:
                    x = preprocess_image(img_path)
                    pred = np.argmax(model.predict(x, verbose=0)[0])

                    y_true.append(true_label)
                    y_pred.append(pred)

                except Exception as e:
                    print(f"Skipped {img_path}: {e}")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred) * 100

    print(f"{part} Accuracy: {acc:.2f}%\n")

    print("Classification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["normal", "fractured"],
            digits=4
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))