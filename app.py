import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import threading
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from PIL import Image
import gradio as gr
import cv2
import matplotlib.pyplot as plt
from io import BytesIO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSMAP_JSON = os.path.join(BASE_DIR, "class_names.json")
INTERMEDIATE_RESIZE = 256
DEFAULT_TOPK = 5

AUTO_FORCE_GRAY_FOR_ROBUST = False

MODEL_CONFIGS = {
    "standard": {
        "label": "EfficientNet-B3 (Color)",
        "path": os.path.join(BASE_DIR, "food_efficientnet_b3.tf"),
        "note": "Latest B3 color model."
    },
    "grayscale_robust": {
        "label": "EfficientNet-B3 Grayscale-Robust",
        "path": os.path.join(BASE_DIR, "food_efficientnet_b3_grayscale_robust_tf"),
        "note": "B3 model trained with grayscale augmentation."
    }
}

@keras.saving.register_keras_serializable(package="custom")
class RandomGrayscale(keras.layers.Layer):
    def __init__(self, p=0.2, prob=None, **kwargs):
        super().__init__(**kwargs)
        if prob is not None:
            p = prob
        self.p = p
    def call(self, inputs, training=None):
        return inputs
    def get_config(self):
        cfg = super().get_config()
        cfg["p"] = self.p
        return cfg

_model_cache = {}
_class_names = None
_load_lock = threading.Lock()

def load_class_names():
    global _class_names
    if _class_names is None:
        if not os.path.exists(CLASSMAP_JSON):
            raise FileNotFoundError(f"class_names.json not found at {CLASSMAP_JSON}")
        with open(CLASSMAP_JSON, "r") as f:
            _class_names = json.load(f)
    return _class_names

def load_model(key: str):
    if key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key '{key}'")
    if key in _model_cache:
        return _model_cache[key]

    with _load_lock:
        if key in _model_cache:
            return _model_cache[key]

        cfg = MODEL_CONFIGS[key]
        path = cfg["path"]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model folder '{path}' not found.")

        try:
            model = keras.models.load_model(path, compile=False)
            print(f"[INFO] Loaded Keras model '{key}' from {path}")
        except ValueError as e:
            print(f"[WARN] Keras metadata missing for '{key}', using tf.saved_model.load instead.")
            imported = tf.saved_model.load(path)

            class TFModelWrapper:
                def __init__(self, imported):
                    self.model = imported
                    self.inputs = [tf.TensorSpec(shape=[None] + list(imported.signatures['serving_default'].inputs[0].shape[1:]), dtype=tf.float32)]
                    self.outputs = [tf.TensorSpec(shape=[None] + list(imported.signatures['serving_default'].outputs[0].shape[1:]), dtype=tf.float32)]
                    self._func = imported.signatures['serving_default']

                def __call__(self, x, training=False):
                    if not isinstance(x, tf.Tensor):
                        x = tf.convert_to_tensor(x, dtype=tf.float32)
                    return self._func(x)['dense']

            model = TFModelWrapper(imported)
            print(f"[INFO] Loaded TF SavedModel '{key}' from {path} (no Keras metadata)")

        _model_cache[key] = model
        return model

def preprocess_image(pil_img: Image.Image, model, force_gray_effective: bool) -> np.ndarray:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    if force_gray_effective:
        g = pil_img.convert("L")
        pil_img = Image.merge("RGB", (g, g, g))

    model_h = model.inputs[0].shape[1]
    model_w = model.inputs[0].shape[2]
    if model_h is None or model_w is None:
        model_h = model_w = 224

    if (model_h, model_w) == (224, 224):
        if pil_img.size != (INTERMEDIATE_RESIZE, INTERMEDIATE_RESIZE):
            pil_img = pil_img.resize((INTERMEDIATE_RESIZE, INTERMEDIATE_RESIZE), Image.BILINEAR)
        arr = np.asarray(pil_img, dtype=np.float32)
        start = (INTERMEDIATE_RESIZE - 224) // 2
        arr = arr[start:start+224, start:start+224, :]
    else:
        if pil_img.size != (model_w, model_h):
            pil_img = pil_img.resize((model_w, model_h), Image.BILINEAR)
        arr = np.asarray(pil_img, dtype=np.float32)

    arr = np.expand_dims(arr, axis=0)
    arr = keras.applications.efficientnet.preprocess_input(arr)
    return arr


def generate_cam(model, image_array, class_idx, model_key):
    """
    Generate Class Activation Map (CAM) for the given image and class.
    
    Args:
        model: The loaded Keras model
        image_array: Preprocessed image array (1, H, W, 3)
        class_idx: Index of the class to generate CAM for
        model_key: Key identifying which model is being used
        
    Returns:
        cam_image: PIL Image with CAM overlay
    """
    try:
        # Check if model is a TFModelWrapper
        is_wrapper = hasattr(model, 'model')
        
        if is_wrapper:
            # For wrapped models, we'll use GradCAM approach
            print("[INFO] Using GradCAM for wrapped model")
            
            # Load the actual Keras model to get layer access
            cfg = MODEL_CONFIGS[model_key]
            path = cfg["path"]
            try:
                keras_model = keras.models.load_model(path, compile=False)
            except:
                print("[WARN] Cannot generate CAM for this model type")
                return None
            
            grad_model = keras_model
        else:
            grad_model = model
        
        # Find the last convolutional layer
        conv_layer = None
        for layer in reversed(grad_model.layers):
            if 'conv' in layer.name.lower() or isinstance(layer, keras.layers.Conv2D):
                conv_layer = layer
                break
        
        if conv_layer is None:
            # Try to find top_conv layer in EfficientNet
            for layer in reversed(grad_model.layers):
                if 'top_conv' in layer.name or 'block7' in layer.name or 'block6' in layer.name:
                    conv_layer = layer
                    break
        
        if conv_layer is None:
            print("[WARN] No convolutional layer found for CAM")
            return None
        
        print(f"[INFO] Using layer '{conv_layer.name}' for CAM")
        
        # Create a model that outputs both the conv layer output and predictions
        grad_model_cam = keras.models.Model(
            inputs=grad_model.inputs,
            outputs=[grad_model.get_layer(conv_layer.name).output, grad_model.output]
        )
        
        # Get the gradient of the predicted class with respect to the conv layer
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model_cam(image_array, training=False)
            loss = predictions[:, class_idx]
        
        # Calculate gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the conv outputs by the gradients
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        for i in range(pooled_grads.shape[0]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Create the heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)  # ReLU
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)  # Normalize
        
        # Resize heatmap to match input image size
        img_h = image_array.shape[1]
        img_w = image_array.shape[2]
        heatmap_resized = cv2.resize(heatmap, (img_w, img_h))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Denormalize the original image
        img_denorm = image_array[0].copy()
        # Reverse EfficientNet preprocessing
        img_denorm = img_denorm / 2.0 + 0.5  # Approximate denormalization
        img_denorm = np.clip(img_denorm * 255, 0, 255).astype(np.uint8)
        
        # Overlay heatmap on original image
        alpha = 0.4
        cam_image = cv2.addWeighted(heatmap_colored, alpha, img_denorm, 1 - alpha, 0)
        
        # Convert to PIL Image
        cam_pil = Image.fromarray(cam_image)
        
        return cam_pil
        
    except Exception as e:
        print(f"[ERROR] Failed to generate CAM: {e}")
        import traceback
        traceback.print_exc()
        return None


def predict(image: Image.Image, model_key: str, top_k: int, user_force_gray: bool, generate_cam_vis: bool = False):
    class_names = load_class_names()
    model = load_model(model_key)

    force_gray_effective = user_force_gray or (AUTO_FORCE_GRAY_FOR_ROBUST and model_key == "grayscale_robust")

    batch = preprocess_image(image, model, force_gray_effective)
    probs = model(batch, training=False)[0].numpy()

    top_k = max(1, min(top_k, len(class_names)))
    idxs = np.argsort(probs)[::-1][:top_k]

    rows = []
    label_map = {}
    for rank, idx in enumerate(idxs, start=1):
        p = float(probs[idx])
        cname = class_names[idx]
        rows.append([rank, cname, round(p, 6)])
        label_map[cname] = p

    meta = (
        f"Model: {MODEL_CONFIGS[model_key]['label']}  | "
        f"User forced grayscale: {'Yes' if user_force_gray else 'No'}  | "
        f"Effective grayscale applied: {'Yes' if force_gray_effective else 'No'}"
    )
    if AUTO_FORCE_GRAY_FOR_ROBUST and model_key == "grayscale_robust" and not user_force_gray:
        meta += " (Auto-applied due to robust model)"
    
    # Generate CAM if requested
    cam_image = None
    if generate_cam_vis and len(idxs) > 0:
        top_class_idx = idxs[0]  # Generate CAM for top prediction
        cam_image = generate_cam(model, batch, top_class_idx, model_key)
    
    return rows, label_map, probs, meta, cam_image

def gr_predict(image, model_choice, top_k, force_gray, show_cam):
    if image is None:
        return [], {}, None, "No image provided.", None
    rows, label_map, _, meta, cam_image = predict(image, model_choice, top_k, force_gray, show_cam)

    fig = None
    try:
        import plotly.express as px
        xs = [r[1] for r in rows]
        ys = [r[2] for r in rows]
        fig = px.bar(
            x=xs, y=ys,
            labels={"x": "Class", "y": "Probability"},
            title=f"Top-{len(rows)} Predictions",
            text=[f"{v:.3f}" for v in ys]
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis=dict(range=[0, 1]))
    except Exception:
        pass

    return rows, label_map, fig, meta, cam_image

def build_demo():
    load_class_names()
    with gr.Blocks(theme="soft") as demo:
        gr.Markdown(
            "# 🍜 Chinese Food Classifier (B3 Models)\n"
            "Switch between the standard and grayscale-robust EfficientNet-B3 models. "
            "You can manually force grayscale or rely on auto-application for the robust model.\n\n"
            "**New:** Enable CAM visualization to see which parts of the image the model focuses on!"
        )
        with gr.Row():
            with gr.Column(scale=1):
                model_choice = gr.Radio(
                    choices=list(MODEL_CONFIGS.keys()),
                    value="standard",
                    label="Model Variant",
                    info="Choose which model to use"
                )
                topk_in = gr.Slider(1, 10, value=5, step=1, label="Top-K")
                force_gray = gr.Checkbox(
                    label="Force convert input to grayscale",
                    value=False
                )
                show_cam = gr.Checkbox(
                    label="Show CAM Visualization",
                    value=True,
                    info="Display Class Activation Map for top prediction"
                )
                predict_btn = gr.Button("Predict", variant="primary")
                status_box = gr.Markdown("")
            image_in = gr.Image(
                label="Upload Image",
                type="pil",
                image_mode="RGB",
                height=320
            )

        gr.Markdown("### Results")
        
        with gr.Row():
            with gr.Column(scale=1):
                preds_df = gr.Dataframe(
                    headers=["rank", "class_name", "probability"],
                    datatype=["number", "str", "number"],
                    interactive=False,
                    label="Top-K Predictions"
                )
                label_out = gr.Label(label="(Mapping of Top-K)")
            
            with gr.Column(scale=1):
                cam_output = gr.Image(
                    label="CAM Visualization (Top Prediction)",
                    type="pil",
                    interactive=False
                )

        predict_btn.click(
            fn=gr_predict,
            inputs=[image_in, model_choice, topk_in, force_gray, show_cam],
            outputs=[preds_df, label_out, status_box, cam_output]
        )

    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch()