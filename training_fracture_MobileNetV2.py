"""
Training script for fracture detection using MobileNetV2.
Used as lightweight deployment baseline.
"""

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from config import (
    DATASET_DIR,
    WEIGHTS_DIR,
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

TRAIN_DIR = DATASET_DIR / "train_valid"


def mobilenet_preprocess(img):
    img = img.astype(np.uint8)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    enhanced = preprocess_xray(gray)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    return tf.keras.applications.mobilenet_v2.preprocess_input(enhanced)


def train_part(part):

    print("\n" + "=" * 60)
    print(f"Training MobileNetV2 → {part}")
    print("=" * 60)

    data = load_path(TRAIN_DIR, part)
    df = validate_image_paths(create_dataframe_from_dataset(data))

    train_df, test_df = train_test_split(
        df,
        train_size=0.9,
        stratify=df["Label"],
        random_state=42
    )

    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=mobilenet_preprocess,
        validation_split=0.2
    )

    test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=mobilenet_preprocess
    )

    train_images = train_gen.flow_from_dataframe(
        train_df,
        x_col="Filepath",
        y_col="Label",
        target_size=IMAGE_SIZE,
        class_mode="categorical",
        batch_size=BATCH_SIZE_TRAIN,
        subset="training"
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

    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
        pooling="avg"
    )

    base.trainable = False

    x = tf.keras.layers.Dense(128, activation="relu")(base.output)
    outputs = tf.keras.layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(base.input, outputs)

    model.compile(
        optimizer=Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(train_images, validation_data=val_images, epochs=20)

    model.save(WEIGHTS_DIR / f"MobileNetV2_{part}_frac.h5")

    loss, acc = model.evaluate(test_images, verbose=0)
    print(f"{part} MobileNet Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    for part in BODY_PARTS:
        train_part(part)

    print("\nMobileNetV2 training completed.")