"""
Training script for body part classification model.
Classifies X-ray images into: Elbow, Hand, Shoulder.

Key Engineering Decisions:
- Medical-safe preprocessing (CLAHE + ResNet normalization)
- Controlled data augmentation (radiology-safe)
- Partial fine-tuning of ResNet50
- Regularized classification head
- Early stopping + checkpointing
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from config import (
    DATASET_DIR,
    WEIGHTS_DIR,
    PLOTS_DIR,
    BATCH_SIZE_TRAIN,
    BATCH_SIZE_VAL,
    BATCH_SIZE_TEST,
    IMAGE_SIZE,
    BODY_PARTS
)

from utils import (
    load_path,
    create_dataframe_from_dataset,
    validate_image_paths,
    preprocess_xray
)

# =====================================================
# MEDICAL PREPROCESSING PIPELINE
# =====================================================

def xray_preprocess_pipeline(img):
    """
    Medical-safe preprocessing pipeline:
    1. CLAHE for X-ray contrast enhancement
    2. Convert to RGB
    3. ResNet50 normalization
    """
    img = preprocess_xray(img)  # CLAHE (grayscale)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img


# =====================================================
# LOAD DATA
# =====================================================

def load_all_parts_data():
    """
    Load dataset across all body parts.
    Label = body part (Elbow / Hand / Shoulder)
    """
    dataset = []
    for part in BODY_PARTS:
        part_data = load_path(DATASET_DIR, part)
        for row in part_data:
            dataset.append({
                "label": row["body_part"],
                "image_path": row["image_path"]
            })
    return dataset


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("Training Body Part Classification Model (ResNet50)")
    print("=" * 70 + "\n")

    # ---------------- LOAD DATA ----------------
    data = load_all_parts_data()
    if not data:
        raise RuntimeError("Dataset is empty")

    images = create_dataframe_from_dataset(data)
    images = validate_image_paths(images)

    print(f"Total valid images: {len(images)}")
    print("Class distribution:")
    print(images["Label"].value_counts(), "\n")

    # ---------------- TRAIN / TEST SPLIT ----------------
    train_df, test_df = train_test_split(
        images,
        train_size=0.9,
        shuffle=True,
        stratify=images["Label"],
        random_state=42
    )

    # ---------------- DATA GENERATORS ----------------
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=xray_preprocess_pipeline,

        # ---- MEDICAL-SAFE AUGMENTATION ----
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        shear_range=0.05,
        horizontal_flip=True,
        fill_mode="nearest",

        validation_split=0.2
    )

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=xray_preprocess_pipeline
    )

    train_images = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="Filepath",
        y_col="Label",
        target_size=IMAGE_SIZE,
        class_mode="categorical",
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        seed=42,
        subset="training"
    )

    val_images = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col="Filepath",
        y_col="Label",
        target_size=IMAGE_SIZE,
        class_mode="categorical",
        batch_size=BATCH_SIZE_VAL,
        shuffle=True,
        seed=42,
        subset="validation"
    )

    test_images = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        x_col="Filepath",
        y_col="Label",
        target_size=IMAGE_SIZE,
        class_mode="categorical",
        batch_size=BATCH_SIZE_TEST,
        shuffle=False
    )

    # ---------------- MODEL ----------------
    base_model = tf.keras.applications.ResNet50(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )

    # ---- PARTIAL FINE-TUNING (LAST BLOCK ONLY) ----
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    x = base_model.output

    # ---- REGULARIZED CLASSIFICATION HEAD ----
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(
        len(BODY_PARTS),
        activation="softmax"
    )(x)

    model = tf.keras.Model(base_model.input, outputs)

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # ---------------- CALLBACKS ----------------
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(WEIGHTS_DIR / "ResNet50_BodyParts_best.h5"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    # ---------------- TRAIN ----------------
    history = model.fit(
        train_images,
        validation_data=val_images,
        epochs=25,
        callbacks=callbacks,
        verbose=1
    )

    # ---------------- SAVE FINAL MODEL ----------------
    final_model_path = WEIGHTS_DIR / "ResNet50_BodyParts.h5"
    model.save(str(final_model_path))
    print(f"\nModel saved at: {final_model_path}")

    # ---------------- EVALUATE ----------------
    test_loss, test_acc = model.evaluate(test_images, verbose=0)
    print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

    # ---------------- PLOTS ----------------
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.legend()
    plt.title("Body Part Classification Accuracy")
    plt.grid(True)
    plt.savefig(PLOTS_DIR / "BodyPart_Accuracy.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Body Part Classification Loss")
    plt.grid(True)
    plt.savefig(PLOTS_DIR / "BodyPart_Loss.png", dpi=150)
    plt.close()

    print("\nTraining completed successfully.")
    print("=" * 70)