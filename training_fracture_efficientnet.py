"""
Training script for fracture detection models (EfficientNetB0).
Trains separate models for Elbow, Hand, Shoulder.

Medical-safe, comparison-ready implementation.
"""

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
# DATASET ROOT
# =====================================================
TRAIN_DIR = DATASET_DIR / "train_valid"


# =====================================================
# MEDICAL-SAFE PREPROCESSING
# =====================================================
def keras_xray_preprocess(img):
    img = img.astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    enhanced = preprocess_xray(gray)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    return tf.keras.applications.efficientnet.preprocess_input(enhanced)


# =====================================================
# TRAIN SINGLE BODY PART
# =====================================================
def train_part(part: str):

    print("\n" + "=" * 70)
    print(f"Training EfficientNet fracture model → {part}")
    print("=" * 70)

    # -------- LOAD DATA --------
    data = load_path(TRAIN_DIR, part)
    if not data:
        raise RuntimeError(f"No data found for {part}")

    df = create_dataframe_from_dataset(data)
    df = validate_image_paths(df)

    print(f"Images found: {len(df)}")
    print(df["Label"].value_counts(), "\n")

    # -------- SPLIT --------
    train_df, test_df = train_test_split(
        df,
        train_size=0.9,
        stratify=df["Label"],
        random_state=42
    )

    # -------- DATA GENERATORS --------
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=keras_xray_preprocess,
        rotation_range=8,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2
    )

    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=keras_xray_preprocess
    )

    train_images = train_gen.flow_from_dataframe(
        train_df,
        x_col="Filepath",
        y_col="Label",
        target_size=IMAGE_SIZE,
        class_mode="categorical",
        batch_size=BATCH_SIZE_TRAIN,
        subset="training",
        shuffle=True
    )

    val_images = train_gen.flow_from_dataframe(
        train_df,
        x_col="Filepath",
        y_col="Label",
        target_size=IMAGE_SIZE,
        class_mode="categorical",
        batch_size=BATCH_SIZE_VAL,
        subset="validation"
    )

    test_images = test_gen.flow_from_dataframe(
        test_df,
        x_col="Filepath",
        y_col="Label",
        target_size=IMAGE_SIZE,
        class_mode="categorical",
        batch_size=BATCH_SIZE_TEST,
        shuffle=False
    )

    # -------- MODEL --------
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
        pooling="avg"
    )

    # Partial fine-tuning
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    x = base_model.output
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(base_model.input, outputs)

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    # -------- CALLBACKS --------
    WEIGHTS_DIR.mkdir(exist_ok=True)
    plot_dir = PLOTS_DIR / "FractureDetection_EfficientNet" / part
    plot_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(WEIGHTS_DIR / f"EfficientNet_{part}_frac_best.h5"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    # -------- TRAIN --------
    history = model.fit(
        train_images,
        validation_data=val_images,
        epochs=25,
        callbacks=callbacks,
        verbose=1
    )

    # -------- SAVE FINAL MODEL --------
    final_path = WEIGHTS_DIR / f"EfficientNet_{part}_frac.h5"
    model.save(str(final_path))
    print(f"Saved model → {final_path}")

    # -------- EVALUATE --------
    loss, acc = model.evaluate(test_images, verbose=0)
    print(f"{part} Test Accuracy: {acc * 100:.2f}%")

    # -------- PLOTS --------
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.legend()
    plt.title(f"{part} EfficientNet Accuracy")
    plt.savefig(plot_dir / "accuracy.png", dpi=150)
    plt.close()


# =====================================================
# RUN ALL BODY PARTS
# =====================================================
if __name__ == "__main__":
    for part in BODY_PARTS:
        train_part(part)

    print("\n" + "=" * 70)
    print("EfficientNet fracture training completed")
    print("=" * 70)