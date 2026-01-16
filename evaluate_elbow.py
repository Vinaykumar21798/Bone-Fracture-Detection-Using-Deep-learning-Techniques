"""
Evaluate a saved model on test set with Test-Time Augmentation
"""

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from config import DATASET_DIR, WEIGHTS_DIR, PLOTS_DIR, IMAGE_SIZE
from utils import load_path, create_dataframe_from_dataset, validate_image_paths, preprocess_xray

# Disable warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

def medical_preprocess(img):
    """Same preprocessing as training"""
    img = img.astype(np.uint8)
    
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    
    return tf.keras.applications.resnet50.preprocess_input(rgb)


def predict_with_tta(model, generator, n_tta=5):
    """Test-Time Augmentation for robust predictions"""
    all_predictions = []
    
    for i in range(n_tta):
        generator.reset()
        predictions = model.predict(generator, verbose=0)
        all_predictions.append(predictions)
    
    final_predictions = np.mean(all_predictions, axis=0)
    return final_predictions


def evaluate_model(part_name, model_path):
    """
    Evaluate a saved model comprehensively
    
    Args:
        part_name: 'Elbow', 'Hand', or 'Shoulder'
        model_path: Path to saved .h5 model
    """
    
    print("\n" + "=" * 70)
    print(f"EVALUATING: {part_name}")
    print("=" * 70)
    
    # Load model
    print(f"\n[1/5] Loading model from: {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'FocalLoss': None},  # Handle custom loss
        compile=False
    )
    print("   ✓ Model loaded successfully")
    
    # Load data
    print(f"\n[2/5] Loading test data...")
    train_data = load_path(TRAIN_DIR, part_name)
    
    val_data = []
    if VAL_DIR.exists():
        val_data = load_path(VAL_DIR, part_name)
    
    # Create dataframe
    all_data = train_data + val_data
    df = create_dataframe_from_dataset(all_data)
    df = validate_image_paths(df)
    
    # Split to get test set (same as training)
    if val_data:
        train_df = create_dataframe_from_dataset(train_data)
        train_df = validate_image_paths(train_df)
        _, test_df = train_test_split(
            train_df, test_size=0.1, stratify=train_df['Label'], random_state=42
        )
    else:
        train_df, temp_df = train_test_split(
            df, test_size=0.25, stratify=df['Label'], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df['Label'], random_state=42
        )
    
    print(f"   ✓ Test set: {len(test_df)} images")
    print(f"\n   Class distribution:")
    print(test_df['Label'].value_counts())
    
    # Create test generator
    print(f"\n[3/5] Creating test generator...")
    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=medical_preprocess
    )
    
    test_images = test_gen.flow_from_dataframe(
        test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=16,
        shuffle=False
    )
    
    class_names = list(test_images.class_indices.keys())
    print(f"   ✓ Classes: {class_names}")
    
    # Standard evaluation
    print(f"\n[4/5] Standard Evaluation...")
    test_images.reset()
    standard_preds = model.predict(test_images, verbose=0)
    y_pred_standard = np.argmax(standard_preds, axis=1)
    y_true = test_images.classes
    
    standard_acc = accuracy_score(y_true, y_pred_standard)
    print(f"   Standard Accuracy: {standard_acc*100:.2f}%")
    
    # TTA evaluation
    print(f"\n[5/5] Test-Time Augmentation Evaluation...")
    print(f"   Running {5} augmented predictions...")
    tta_preds = predict_with_tta(model, test_images, n_tta=5)
    y_pred_tta = np.argmax(tta_preds, axis=1)
    
    tta_acc = accuracy_score(y_true, y_pred_tta)
    print(f"   TTA Accuracy: {tta_acc*100:.2f}%")
    
    # Results summary
    print(f"\n{'='*70}")
    print(f"{part_name} - FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Standard Test Accuracy:  {standard_acc*100:.2f}%")
    print(f"TTA Test Accuracy:       {tta_acc*100:.2f}%")
    print(f"Improvement from TTA:    +{(tta_acc - standard_acc)*100:.2f}%")
    print(f"{'='*70}\n")
    
    # Detailed metrics
    print("Classification Report (TTA):")
    print(classification_report(y_true, y_pred_tta, target_names=class_names, digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_tta)
    print("\nConfusion Matrix (TTA):")
    print(cm)
    
    # Calculate per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        class_acc = cm[i, i] / cm[i].sum()
        print(f"  {class_name}: {class_acc*100:.2f}%")
    
    # Plot confusion matrix
    plot_dir = PLOTS_DIR / "FractureDetection" / part_name
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'{part_name} - Confusion Matrix (TTA)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    output_path = plot_dir / 'confusion_matrix_final.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Confusion matrix saved: {output_path}")
    plt.close()
    
    return {
        'standard_acc': standard_acc,
        'tta_acc': tta_acc,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred_tta
    }


if __name__ == "__main__":
    
    print("\n" + "=" * 70)
    print("MODEL EVALUATION TOOL")
    print("=" * 70)
    
    # Evaluate Elbow model
    elbow_model = WEIGHTS_DIR / "ResNet50_Elbow_frac_best.h5"
    
    if elbow_model.exists():
        results = evaluate_model("Elbow", elbow_model)
        
        print("\n" + "=" * 70)
        print("EVALUATION COMPLETE")
        print("=" * 70)
        print(f"✓ Elbow Model Evaluated")
        print(f"  - Standard: {results['standard_acc']*100:.2f}%")
        print(f"  - TTA:      {results['tta_acc']*100:.2f}%")
        print("=" * 70)
    else:
        print(f"\n✗ Model not found: {elbow_model}")
        print("Available models:")
        for model_file in WEIGHTS_DIR.glob("*.h5"):
            print(f"  - {model_file.name}")