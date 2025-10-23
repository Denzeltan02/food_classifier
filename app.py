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
                    self.inputs = [tf.TensorSpec(shape=[None] + list(sig.inputs[0].shape[1:]), dtype=tf.float32)]
                    self.outputs = [tf.TensorSpec(shape=[None] + list(sig.outputs[0].shape[1:]), dtype=tf.float32)]

                def __call__(self, x, training=False):
                    if not isinstance(x, tf.Tensor):
                        x = tf.convert_to_tensor(x, dtype=tf.float32)
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

# --------------------------------------------------
# Prediction
# --------------------------------------------------
def predict(image_like, model_key: str, top_k: int, user_force_gray: bool):
    class_names = load_class_names()
    model = load_model(model_key)
    force_gray_effective = user_force_gray or (AUTO_FORCE_GRAY_FOR_ROBUST and model_key == "grayscale_robust")

    pil_img = _ensure_pil(image_like)
    if pil_img is None:
        return [], {}, None, "No image provided."

    batch = preprocess_image(pil_img, model, force_gray_effective)
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

    return rows, label_map, fig, meta

# --------------------------------------------------
# Gradio callback
# --------------------------------------------------
def gr_predict(image, model_choice, top_k, force_gray):
    return predict(image, model_choice, top_k, force_gray)

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
                        predict_btn = gr.Button("Predict", variant="primary")
                        status_box = gr.Markdown("")
                    image_in = gr.Image(
                        label="Upload or Webcam Snapshot",
                        type="pil",
                        image_mode="RGB",
                        height=320,
                        sources=["upload", "webcam"]
                    )

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
                    inputs=[image_in, model_choice, topk_in, force_gray],
                    outputs=[preds_df, label_out, bar_plot, status_box]
                )

            # Live webcam (without time_interval for compatibility)
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
                        status_box_live = gr.Markdown("Start your camera to begin streaming predictions.")
                    image_live = gr.Image(
                        label="Webcam Stream",
                        streaming=True,
                        sources=["webcam"],
                        image_mode="RGB",
                        height=320
                    )

                preds_df_live = gr.Dataframe(
                    headers=["rank", "class_name", "probability"],
                    datatype=["number", "str", "number"],
                    interactive=False,
                    label="Top-K Predictions (Live)"
                )
                label_out_live = gr.Label(label="(Mapping of Top-K, Live)")
                bar_plot_live = gr.Plot(label="Top-K Bar Chart (Live)")

                # Older Gradio doesn't accept time_interval argument
                image_live.stream(
                    fn=gr_predict,
                    inputs=[image_live, model_choice_live, topk_live, force_gray_live],
                    outputs=[preds_df_live, label_out_live, bar_plot_live, status_box_live]
                )

        gr.Markdown("Tip: Use **Upload or Snapshot** for single image, or **Live** for continuous webcam predictions.")
    return demo

# --------------------------------------------------
# Launch
# --------------------------------------------------
demo = build_demo()

if __name__ == "__main__":
    demo.launch()
