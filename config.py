"""
Configuration file for Bone Fracture Detection project.
Contains all paths, settings, and constants.
"""

from pathlib import Path

# =====================================================
# BASE PROJECT DIRECTORY
# =====================================================
PROJECT_ROOT = Path(__file__).parent.absolute()

# =====================================================
# DIRECTORY PATHS
# =====================================================
DATASET_DIR = PROJECT_ROOT / "Dataset"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
IMAGES_DIR = PROJECT_ROOT / "images"
PLOTS_DIR = PROJECT_ROOT / "plots"
TEST_DIR = PROJECT_ROOT / "test"
PREDICT_RESULTS_DIR = PROJECT_ROOT / "PredictResults"

# Ensure directories exist
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# MODEL WEIGHTS (PRIMARY BACKBONE: ResNet50)
# =====================================================
MODEL_BODY_PARTS = WEIGHTS_DIR / "ResNet50_BodyParts.h5"
MODEL_ELBOW_FRAC = WEIGHTS_DIR / "ResNet50_Elbow_frac.h5"
MODEL_HAND_FRAC = WEIGHTS_DIR / "ResNet50_Hand_frac.h5"
MODEL_SHOULDER_FRAC = WEIGHTS_DIR / "ResNet50_Shoulder_frac.h5"

# =====================================================
# MODEL BACKBONE CONFIGURATION
# =====================================================
BACKBONE_NAME = "resnet50"   # Primary backbone used in this project

# =====================================================
# CATEGORIES
# =====================================================
BODY_PARTS = ["Elbow", "Hand", "Shoulder"]
FRACTURE_STATUS = ["fractured", "normal"]

# =====================================================
# IMAGE SETTINGS
# =====================================================
IMAGE_SIZE = (224, 224)
IMAGE_TARGET_SIZE = 224
COLOR_MODE = "rgb"

# =====================================================
# TRAINING SETTINGS
# =====================================================
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_VAL = 64
BATCH_SIZE_TEST = 32

EPOCHS = 25
EARLY_STOPPING_PATIENCE = 3

# =====================================================
# LEARNING RATE STRATEGY (IMPORTANT FOR 90%+ ACCURACY)
# =====================================================
LR_INITIAL = 1e-4       # Used when backbone is frozen
LR_FINE_TUNE = 1e-5    # Used during fine-tuning

# =====================================================
# FINE-TUNING SETTINGS
# =====================================================
FINE_TUNE_LAST_N_LAYERS = 30   # Unfreeze last N layers of backbone

# =====================================================
# REGULARIZATION (PREVENT OVERFITTING)
# =====================================================
DROPOUT_HEAD_1 = 0.4
DROPOUT_HEAD_2 = 0.3

# =====================================================
# GUI SETTINGS (IF USED)
# =====================================================
GUI_TITLE = "Bone Fracture Detection"
GUI_WIDTH = 500
GUI_HEIGHT = 740
GUI_MIN_WIDTH = 400
GUI_MIN_HEIGHT = 600

# =====================================================
# SUPPORTED IMAGE FORMATS
# =====================================================
SUPPORTED_IMAGE_FORMATS = [
    ("Image files", "*.png *.jpg *.jpeg *.jfif *.bmp *.tiff *.tif"),
    ("PNG files", "*.png"),
    ("JPEG files", "*.jpg *.jpeg *.jfif"),
    ("All files", "*.*"),
]