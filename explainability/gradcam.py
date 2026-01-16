"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
Correct and stable implementation for ResNet50-based models.
"""

import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras
from typing import Tuple


def generate_gradcam(
    model: keras.Model,
    img_path: str,
    img_array: np.ndarray,
    pred_index: int,
    original_img: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Grad-CAM heatmap and overlay image.
    """

    # ✅ HARD-CODED correct layer for ResNet50
    conv_layer_name = "conv5_block3_out"

    # Build Grad-CAM model
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(conv_layer_name).output,
            model.output
        ]
    )

    # Forward & backward pass
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        loss = predictions[:, pred_index]   # ✅ scalar

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradients could not be computed")

    # Channel-wise importance
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (2048,)

    conv_outputs = conv_outputs[0]                         # (7,7,2048)
    pooled_grads = pooled_grads.numpy()

    # Weighted sum
    heatmap = np.sum(conv_outputs.numpy() * pooled_grads, axis=-1)

    # Normalize
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-8)

    # Resize to original image
    heatmap = cv2.resize(
        heatmap,
        (original_img.shape[1], original_img.shape[0])
    )

    # Color map
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap),
        cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = cv2.addWeighted(
        original_img.astype(np.uint8),
        0.6,
        heatmap_color,
        0.4,
        0
    )

    return heatmap, overlay
