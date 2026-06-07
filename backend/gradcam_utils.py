import tensorflow as tf
import numpy as np
import cv2

def get_last_conv_layer(model):
    # Backward sweep looking for explicit surface convolutional layers
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
            
    # Deep nested layer structure fallback search
    for layer in reversed(model.layers):
        if hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D) or 'conv' in sub_layer.name.lower():
                    if len(sub_layer.output_shape) == 4:
                        return sub_layer.name
    return None


def make_gradcam_heatmap(img_array, model):
    last_conv_layer_name = get_last_conv_layer(model)

    if last_conv_layer_name is None:
        raise ValueError("No viable Conv2D target layers located in configuration structure.")

    active_model = model
    try:
        target_layer = active_model.get_layer(last_conv_layer_name)
    except ValueError:
        for layer in active_model.layers:
            if hasattr(layer, 'layers'):
                try:
                    target_layer = layer.get_layer(last_conv_layer_name)
                    active_model = layer 
                    break
                except ValueError:
                    continue

    grad_model = tf.keras.models.Model(
        inputs=active_model.inputs,
        outputs=[target_layer.output, active_model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def save_gradcam_overlay(img_path, heatmap, output_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(output_path, overlay)
    return output_path