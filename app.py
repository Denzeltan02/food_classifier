import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Optional quieter logs:
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import threading
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer
from PIL import Image
import gradio as gr

# --------------------------------------------------
# Universal Keras register_keras_serializable import
# --------------------------------------------------
try:
    # Most TensorFlow 2.x installs
    from tensorflow.keras.utils import register_keras_serializable
except Exception:
    try:
        # Some Keras 3 builds
        from keras.utils import register_keras_serializable
    except Exception:
        # Last resort: no-op fallback
        def register_keras_serializable(*args, **kwargs):
            def deco(fn):
                return fn
            return deco

# --------------------------------------------------
# Configuration
# --------------------------------------------------
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

# --------------------------------------------------
# Stub for RandomGrayscale (identity)
# --------------------------------------------------
@register_keras_serializable(package="custom")
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

# --------------------------------------------------
# Caches
# --------------------------------------------------
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
            model = keras.models.load_model(
                path,
                compile=False,
                custom_objects={"RandomGrayscale": RandomGrayscale}
            )
            print(f"[INFO] Loaded Keras model '{key}' from {path}")
        except Exception:
            print(f"[WARN] Keras metadata missing for '{key}', using tf.saved_model.load instead.")
            imported = tf.saved_model.load(path)

            class TFModelWrapper:
                def __init__(self, imported):
                    sig = imported.signatures['serving_default']
                    self._func = sig
                    self._imported = imported  # Keep reference to prevent garbage collection
                    
                    # Get input name from signature
                    self.input_names = list(sig.structured_input_signature[1].keys())
                    self.input_name = self.input_names[0] if self.input_names else None
                    
                    self.inputs = [tf.TensorSpec(shape=[None] + list(sig.inputs[0].shape[1:]), dtype=tf.float32)]
                    self.outputs = [tf.TensorSpec(shape=[None] + list(sig.outputs[0].shape[1:]), dtype=tf.float32)]
                    
                    print(f"[INFO] SavedModel input name: {self.input_name}")

                def __call__(self, x, training=False):
                    if not isinstance(x, tf.Tensor):
                        x = tf.convert_to_tensor(x, dtype=tf.float32)
                    
                    # Call with keyword argument using the correct input name
                    if self.input_name:
                        out_dict = self._func(**{self.input_name: x})
                    else:
                        out_dict = self._func(x)
                    
                    first_key = next(iter(out_dict.keys()))
                    return out_dict[first_key]

            model = TFModelWrapper(imported)
            print(f"[INFO] Loaded TF SavedModel '{key}' from {path}")

        _model_cache[key] = model
        return model

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
def _ensure_pil(image_like):
    if image_like is None:
        return None
    if isinstance(image_like, Image.Image):
        pil_img = image_like
    else:
        arr = image_like
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return pil_img

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
def generate_gradcam(model, img_array, class_idx, last_conv_layer_name=None):
    """
    Generate Grad-CAM heatmap. Falls back to saliency/occlusion for SavedModels.
    """
    # Handle TF SavedModel wrapper
    if hasattr(model, '_func'):
        print("[INFO] Using saliency map for SavedModel (Grad-CAM not available)")
        return generate_saliency_map(model, img_array, class_idx)
    
    # Standard Keras model - use proper Grad-CAM
    try:
        # Auto-detect last conv layer
        if last_conv_layer_name is None:
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower() and len(layer.output_shape) == 4:
                    last_conv_layer_name = layer.name
                    break
        
        if last_conv_layer_name is None:
            print("[WARN] No conv layer found, falling back to saliency map")
            return generate_saliency_map(model, img_array, class_idx)
        
        print(f"[INFO] Using Grad-CAM with layer: {last_conv_layer_name}")
        
        # Create gradient model
        grad_model = keras.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]
        
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()
        
        for i in range(pooled_grads.shape[0]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
        
        return heatmap
        
    except Exception as e:
        print(f"[WARN] Grad-CAM failed: {e}, using saliency map")
        return generate_saliency_map(model, img_array, class_idx)


def generate_saliency_map(model, img_array, class_idx):
    """
    Generate saliency map using integrated gradients approach.
    Works reliably with SavedModels.
    """
    try:
        # Create a persistent tensor (not a variable)
        img_tensor = tf.constant(img_array, dtype=tf.float32)
        
        # For SavedModel wrapper, we need to use the correct input name
        if hasattr(model, 'input_name') and model.input_name:
            @tf.function
            def model_fn(x):
                out = model._func(**{model.input_name: x})
                first_key = next(iter(out.keys()))
                return out[first_key]
        else:
            @tf.function
            def model_fn(x):
                return model(x, training=False)
        
        # Compute gradients using tf.function for stability
        @tf.function
        def compute_gradients(img):
            with tf.GradientTape() as tape:
                tape.watch(img)
                predictions = model_fn(img)
                if isinstance(predictions, dict):
                    predictions = predictions[list(predictions.keys())[0]]
                target_score = predictions[:, class_idx]
            return tape.gradient(target_score, img)
        
        # Compute gradients
        grads = compute_gradients(img_tensor)
        
        if grads is None:
            print("[ERROR] Gradients are None")
            return generate_occlusion_map(model, img_array, class_idx)
        
        # Process gradients
        grads_np = grads.numpy()[0]
        
        # Take absolute value and average across channels
        saliency = np.abs(grads_np)
        saliency = np.mean(saliency, axis=-1)
        
        # Apply Gaussian smoothing
        try:
            from scipy.ndimage import gaussian_filter
            saliency = gaussian_filter(saliency, sigma=4)
        except ImportError:
            pass
        
        # Normalize
        if np.max(saliency) > 0:
            saliency = saliency / np.max(saliency)
        
        return saliency
        
    except Exception as e:
        print(f"[WARN] Saliency map failed: {e}")
        print("[INFO] Falling back to occlusion-based visualization...")
        return generate_occlusion_map(model, img_array, class_idx)


def generate_occlusion_map(model, img_array, class_idx, patch_size=32, stride=16):
    """
    Generate attribution map using occlusion sensitivity.
    This is slower but works reliably with any model.
    """
    try:
        print("[INFO] Generating occlusion map (this may take a few seconds)...")
        
        img = img_array[0]
        h, w, c = img.shape
        
        # Get baseline prediction
        if hasattr(model, 'input_name') and model.input_name:
            baseline_pred = model._func(**{model.input_name: img_array})
            baseline_pred = baseline_pred[next(iter(baseline_pred.keys()))]
        else:
            baseline_pred = model(img_array, training=False)
            if isinstance(baseline_pred, dict):
                baseline_pred = baseline_pred[list(baseline_pred.keys())[0]]
        
        baseline_score = baseline_pred[0, class_idx].numpy()
        
        # Create importance map
        importance_map = np.zeros((h // stride + 1, w // stride + 1))
        
        # Occlude patches and measure impact
        for i, y in enumerate(range(0, h - patch_size + 1, stride)):
            for j, x in enumerate(range(0, w - patch_size + 1, stride)):
                # Create occluded image (set patch to gray)
                occluded = img.copy()
                occluded[y:y+patch_size, x:x+patch_size, :] = 128.0  # Gray value
                occluded_batch = np.expand_dims(occluded, axis=0)
                
                # Get prediction
                if hasattr(model, 'input_name') and model.input_name:
                    pred = model._func(**{model.input_name: occluded_batch})
                    pred = pred[next(iter(pred.keys()))]
                else:
                    pred = model(occluded_batch, training=False)
                    if isinstance(pred, dict):
                        pred = pred[list(pred.keys())[0]]
                
                score = pred[0, class_idx].numpy()
                
                # Importance = drop in probability
                importance_map[i, j] = baseline_score - score
        
        # Resize to original image size
        importance_map = np.maximum(importance_map, 0)
        
        # Normalize
        if np.max(importance_map) > 0:
            importance_map = importance_map / np.max(importance_map)
        
        # Resize to match input image
        from PIL import Image
        importance_pil = Image.fromarray((importance_map * 255).astype(np.uint8))
        importance_resized = importance_pil.resize((w, h), Image.BILINEAR)
        importance_array = np.array(importance_resized) / 255.0
        
        # Apply smoothing
        try:
            from scipy.ndimage import gaussian_filter
            importance_array = gaussian_filter(importance_array, sigma=3)
        except:
            pass
        
        print("[INFO] ✓ Occlusion map completed")
        return importance_array
        
    except Exception as e:
        print(f"[ERROR] Occlusion map failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def overlay_heatmap_on_image(pil_img, heatmap, alpha=0.5, colormap='jet'):
    """
    Overlay heatmap on original image.
    """
    if heatmap is None:
        return None
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("[ERROR] matplotlib not available")
        return None
    
    try:
        # Resize heatmap to match image size
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        heatmap_pil = Image.fromarray(heatmap_uint8)
        heatmap_resized = heatmap_pil.resize(pil_img.size, Image.BILINEAR)
        heatmap_array = np.array(heatmap_resized) / 255.0
        
        # Apply colormap
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap_array)[:, :, :3]  # RGB only
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Convert to PIL
        heatmap_img = Image.fromarray(heatmap_colored)
        
        # Blend with original
        blended = Image.blend(pil_img, heatmap_img, alpha=alpha)
        
        return blended
        
    except Exception as e:
        print(f"[ERROR] Overlay failed: {e}")
        return None
# --------------------------------------------------
# Prediction
# --------------------------------------------------
def predict(image_like, model_key: str, top_k: int, user_force_gray: bool, show_cam: bool = True):
    class_names = load_class_names()
    model = load_model(model_key)
    force_gray_effective = user_force_gray or (AUTO_FORCE_GRAY_FOR_ROBUST and model_key == "grayscale_robust")

    pil_img = _ensure_pil(image_like)
    if pil_img is None:
        return [], {}, None, "No image provided.", None

    batch = preprocess_image(pil_img, model, force_gray_effective)
    
    # Get predictions
    probs = model(batch, training=False)
    if isinstance(probs, dict):
        probs = probs[list(probs.keys())[0]]
    probs = probs[0].numpy()

    top_k = max(1, min(top_k, len(class_names)))
    idxs = np.argsort(probs)[::-1][:top_k]

    rows = []
    label_map = {}
    for rank, idx in enumerate(idxs, start=1):
        p = float(probs[idx])
        cname = class_names[idx]
        rows.append([rank, cname, round(p, 6)])
        label_map[cname] = p

    # Build metadata string
    cam_status = "Enabled" if show_cam else "Disabled"
    meta = (
        f"Model: {MODEL_CONFIGS[model_key]['label']}  | "
        f"Grayscale: {'Yes' if force_gray_effective else 'No'}  | "
        f"CAM: {cam_status}"
    )

    # Generate CAM/Saliency visualization
    cam_image = None
    if show_cam:
        try:
            top_class_idx = idxs[0]
            top_class_name = class_names[top_class_idx]
            print(f"[INFO] Generating visualization for: {top_class_name} (prob: {probs[top_class_idx]:.4f})")
            
            heatmap = generate_gradcam(model, batch, top_class_idx)
            
            if heatmap is not None:
                cam_image = overlay_heatmap_on_image(pil_img, heatmap, alpha=0.5)
                if cam_image is not None:
                    print("[INFO] ✓ Visualization generated successfully")
                else:
                    print("[WARN] ✗ Overlay failed")
            else:
                print("[WARN] ✗ Heatmap generation failed")
                
        except Exception as e:
            print(f"[ERROR] Visualization failed: {e}")
            import traceback
            traceback.print_exc()

    # Generate bar plot
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
# --------------------------------------------------
# Gradio callback
# --------------------------------------------------
def gr_predict(image, model_choice, top_k, force_gray, show_cam):
    return predict(image, model_choice, top_k, force_gray, show_cam)

# --------------------------------------------------
# UI
# --------------------------------------------------
def build_demo():
    load_class_names()
    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("# 🍜 Chinese Food Classifier (B3 Models)\n")

        with gr.Tabs():
            # Upload or snapshot
            with gr.Tab("Upload or Snapshot"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_choice = gr.Radio(
                            choices=list(MODEL_CONFIGS.keys()),
                            value="standard",
                            label="Model Variant"
                        )
                        topk_in = gr.Slider(1, 10, value=5, step=1, label="Top-K")
                        force_gray = gr.Checkbox(label="Force convert to grayscale", value=False)
                        show_cam = gr.Checkbox(label="Show CAM visualization", value=True)  # NEW
                        predict_btn = gr.Button("Predict", variant="primary")
                        status_box = gr.Markdown("")
                    
                    with gr.Column(scale=1):
                        image_in = gr.Image(
                            label="Upload or Webcam Snapshot",
                            type="pil",
                            image_mode="RGB",
                            height=320,
                            sources=["upload", "webcam"]
                        )
                        cam_image_out = gr.Image(label="CAM Visualization", height=320)  # NEW

                preds_df = gr.Dataframe(
                    headers=["rank", "class_name", "probability"],
                    datatype=["number", "str", "number"],
                    interactive=False,
                    label="Top-K Predictions"
                )
                label_out = gr.Label(label="(Mapping of Top-K)")
                bar_plot = gr.Plot(label="Top-K Bar Chart")

                predict_btn.click(
                    fn=gr_predict,
                    inputs=[image_in, model_choice, topk_in, force_gray, show_cam],
                    outputs=[preds_df, label_out, bar_plot, status_box, cam_image_out]  # Added cam_image_out
                )

            # Live webcam
            with gr.Tab("Live (Webcam)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_choice_live = gr.Radio(
                            choices=list(MODEL_CONFIGS.keys()),
                            value="standard",
                            label="Model Variant (Live)"
                        )
                        topk_live = gr.Slider(1, 10, value=5, step=1, label="Top-K (Live)")
                        force_gray_live = gr.Checkbox(label="Force grayscale (Live)", value=False)
                        show_cam_live = gr.Checkbox(label="Show CAM (Live)", value=True)  # NEW
                        status_box_live = gr.Markdown("Start your camera to begin streaming predictions.")
                    
                    with gr.Column(scale=1):
                        image_live = gr.Image(
                            label="Webcam Stream",
                            streaming=True,
                            sources=["webcam"],
                            image_mode="RGB",
                            height=320
                        )
                        cam_image_live = gr.Image(label="CAM Visualization (Live)", height=320)  # NEW

                preds_df_live = gr.Dataframe(
                    headers=["rank", "class_name", "probability"],
                    datatype=["number", "str", "number"],
                    interactive=False,
                    label="Top-K Predictions (Live)"
                )
                label_out_live = gr.Label(label="(Mapping of Top-K, Live)")
                bar_plot_live = gr.Plot(label="Top-K Bar Chart (Live)")

                image_live.stream(
                    fn=gr_predict,
                    inputs=[image_live, model_choice_live, topk_live, force_gray_live, show_cam_live],
                    outputs=[preds_df_live, label_out_live, bar_plot_live, status_box_live, cam_image_live]
                )

        gr.Markdown("Tip: Use **Upload or Snapshot** for single image, or **Live** for continuous webcam predictions.")
    return demo
# --------------------------------------------------
# Launch
# --------------------------------------------------
demo = build_demo()

if __name__ == "__main__":
    demo.launch()
