"""
Enhanced configuration for improved model performance.
Optimized hyperparameters for 85%+ accuracy.
"""

from pathlib import Path

# Base configuration
PROJECT_ROOT = Path(__file__).parent.absolute()
DATASET_DIR = PROJECT_ROOT / "Dataset"
WEIGHTS_DIR = PROJECT_ROOT / "weights"
PLOTS_DIR = PROJECT_ROOT / "plots"

# Ensure directories exist
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# ENHANCED MODEL SETTINGS FOR HIGHER ACCURACY
# =====================================================

# Body parts and labels
BODY_PARTS = ["Elbow", "Hand", "Shoulder"]
FRACTURE_STATUS = ["fractured", "normal"]

# Enhanced image settings
IMAGE_SIZE = (224, 224)  # Can experiment with (256, 256) for more detail
COLOR_MODE = "rgb"

# =====================================================
# OPTIMIZED TRAINING PARAMETERS
# =====================================================

# Batch sizes - balance between memory and gradient stability
BATCH_SIZE_TRAIN = 20    # Increased for better gradient estimates
BATCH_SIZE_VAL = 20
BATCH_SIZE_TEST = 16

# Enhanced epochs for better convergence
EPOCHS_STAGE1 = 15       # More epochs for head training
EPOCHS_STAGE2 = 25       # Extended fine-tuning
EARLY_STOPPING_PATIENCE = 6  # More patience

# =====================================================
# ADVANCED LEARNING RATE STRATEGY
# =====================================================

# Multi-step learning rate schedule
LR_INITIAL = 1e-3        # Higher initial LR for faster convergence
LR_STAGE1_MIN = 1e-5     # Minimum LR for stage 1
LR_FINE_TUNE = 5e-6      # Lower LR for fine-tuning
LR_FINE_TUNE_MIN = 1e-8  # Minimum LR for fine-tuning

# Learning rate scheduler parameters
LR_REDUCTION_FACTOR = 0.5
LR_REDUCTION_PATIENCE = 3

# =====================================================
# ENHANCED FINE-TUNING SETTINGS
# =====================================================

# Unfreeze more layers for better feature adaptation
FINE_TUNE_LAST_N_LAYERS = 60   # Unfreeze more layers (was 30)

# Progressive unfreezing strategy
PROGRESSIVE_UNFREEZING = True
UNFREEZE_STAGES = [40, 60, 80]  # Gradually unfreeze more layers

# =====================================================
# ADVANCED REGULARIZATION
# =====================================================

# Dropout rates optimized for medical images
DROPOUT_HEAD_1 = 0.5     # Increased dropout
DROPOUT_HEAD_2 = 0.4     # Increased dropout
DROPOUT_HEAD_3 = 0.3     # Additional dropout layer

# Weight decay for better generalization
WEIGHT_DECAY = 1e-4

# =====================================================
# DATA AUGMENTATION PARAMETERS
# =====================================================

# Geometric augmentations
ROTATION_RANGE = 15
WIDTH_SHIFT_RANGE = 0.12
HEIGHT_SHIFT_RANGE = 0.12
ZOOM_RANGE = 0.15
HORIZONTAL_FLIP = True

# Intensity augmentations (crucial for X-rays)
BRIGHTNESS_RANGE = [0.8, 1.2]
CONTRAST_RANGE = [0.9, 1.1]
CHANNEL_SHIFT_RANGE = 20.0

# =====================================================
# CLASS IMBALANCE HANDLING
# =====================================================

# Enhanced class weight settings
USE_CLASS_WEIGHTS = True
CLASS_WEIGHT_METHOD = 'balanced'
MAX_CLASS_WEIGHT = 3.0   # Limit extreme weights
MIN_CLASS_WEIGHT = 0.5

# Focal loss parameters (alternative to class weights)
USE_FOCAL_LOSS = False
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# =====================================================
# TEST TIME AUGMENTATION
# =====================================================

# TTA settings for improved test accuracy
USE_TTA = True
TTA_STEPS = 5           # Number of augmented predictions
TTA_AUGMENTATION = {
    'rotation_range': 10,
    'width_shift_range': 0.08,
    'height_shift_range': 0.08,
    'horizontal_flip': True
}

# =====================================================
# MODEL ARCHITECTURE ENHANCEMENTS
# =====================================================

# Backbone options (in order of preference)
BACKBONE_OPTIONS = [
    'efficientnetb3',    # Best accuracy-efficiency trade-off
    'efficientnetb4',    # Higher accuracy, more parameters
    'resnet50v2',        # Improved ResNet
    'resnet50'           # Fallback
]

# Head architecture
HEAD_UNITS = [512, 256, 128]  # Dense layer sizes
USE_ATTENTION = True          # Attention mechanism
USE_DUAL_POOLING = True       # Combine GAP and GMP

# =====================================================
# CALLBACKS AND MONITORING
# =====================================================

# Model checkpointing
SAVE_BEST_ONLY = True
MONITOR_METRIC = 'val_accuracy'
CHECKPOINT_MODE = 'max'

# Early stopping
EARLY_STOP_MONITOR = 'val_accuracy'
EARLY_STOP_MODE = 'max'
RESTORE_BEST_WEIGHTS = True

# =====================================================
# ENSEMBLE METHODS
# =====================================================

# Multi-model ensemble settings
USE_ENSEMBLE = False
ENSEMBLE_MODELS = ['efficientnetb3', 'resnet50', 'densenet121']
ENSEMBLE_WEIGHTS = [0.5, 0.3, 0.2]  # Weighted average

# =====================================================
# ADVANCED PREPROCESSING
# =====================================================

# Medical image preprocessing
USE_HISTOGRAM_EQUALIZATION = True
USE_GAUSSIAN_BLUR = True
GAUSSIAN_KERNEL_SIZE = (3, 3)
GAUSSIAN_SIGMA = 0

# Normalization strategy
NORMALIZATION_METHOD = 'imagenet'  # or 'custom', 'zscore'
CUSTOM_MEAN = [0.485, 0.456, 0.406]
CUSTOM_STD = [0.229, 0.224, 0.225]

# =====================================================
# HARDWARE OPTIMIZATIONS
# =====================================================

# CPU optimizations
USE_MIXED_PRECISION = False  # Disable for CPU
INTRA_OP_THREADS = 4
INTER_OP_THREADS = 4

# Memory optimizations
PREFETCH_BUFFER_SIZE = 2
BATCH_PREFETCH = True

# =====================================================
# VALIDATION AND EVALUATION
# =====================================================

# Cross-validation
USE_CROSS_VALIDATION = False
CV_FOLDS = 5

# Evaluation metrics
EVALUATION_METRICS = [
    'accuracy',
    'precision',
    'recall',
    'f1_score',
    'auc'
]

# =====================================================
# LOGGING AND VISUALIZATION
# =====================================================

# Training monitoring
PLOT_TRAINING_HISTORY = True
SAVE_CONFUSION_MATRIX = True
SAVE_CLASSIFICATION_REPORT = True

# Model interpretation
GENERATE_GRAD_CAM = False
SAVE_FEATURE_MAPS = False