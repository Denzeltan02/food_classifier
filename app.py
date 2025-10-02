import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# Optional quieter logs:
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import threading
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import gradio as gr

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSMAP_JSON = os.path.join(BASE_DIR, "class_names.json")
INTERMEDIATE_RESIZE = 256
DEFAULT_TOPK = 5

# Auto-force grayscale when using grayscale_robust model (even if user box unchecked)
AUTO_FORCE_GRAY_FOR_ROBUST = True

MODEL_CONFIGS = {
    "standard": {
        "label": "Standard EfficientNet (Color)",
        "path": os.path.join(BASE_DIR, "food_efficientnet_b0.keras"),
        "note": "Baseline color model."
    },
    "grayscale_robust": {
        "label": "Grayscale-Robust EfficientNet",
        "path": os.path.join(BASE_DIR, "food_efficientnet_b0_grayscale_robust.keras"),
        "note": "Trained with grayscale augmentation."
    }
}

# --------------------------------------------------
# Stub for RandomGrayscale (identity)
# --------------------------------------------------
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
            raise FileNotFoundError(f"Model file '{path}' not found.")
        model = keras.models.load_model(
            path,
            compile=False,
            custom_objects={"RandomGrayscale": RandomGrayscale}
        )
        _model_cache[key] = model
        print(f"[INFO] Loaded model '{key}' from {path} | In: {model.inputs[0].shape} Out: {model.outputs[0].shape}")
        return model

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
def preprocess_image(pil_img: Image.Image, model, force_gray_effective: bool) -> np.ndarray:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    if force_gray_effective:
        # Convert to grayscale then back to RGB triplet
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
def predict(image: Image.Image, model_key: str, top_k: int, user_force_gray: bool):
    class_names = load_class_names()
    model = load_model(model_key)

    # Decide effective grayscale conversion
    force_gray_effective = user_force_gray or (AUTO_FORCE_GRAY_FOR_ROBUST and model_key == "grayscale_robust")

    batch = preprocess_image(image, model, force_gray_effective)
    probs = model.predict(batch, verbose=0)[0]

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
    return rows, label_map, probs, meta

# --------------------------------------------------
# Gradio callback
# --------------------------------------------------
def gr_predict(image, model_choice, top_k, force_gray):
    if image is None:
        return [], {}, None, "No image provided."
    rows, label_map, _, meta = predict(image, model_choice, top_k, force_gray)

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
# UI
# --------------------------------------------------
def build_demo():
    load_class_names()
    with gr.Blocks(theme="soft") as demo:
        gr.Markdown(
            "# 🍜 Chinese Food Classifier (Dual Models)\n"
            "Switch between the standard and grayscale-robust models. "
            "You can manually force grayscale or rely on auto-application for the robust model."
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
                predict_btn = gr.Button("Predict", variant="primary")
                status_box = gr.Markdown("")
            image_in = gr.Image(
                label="Upload Image",
                type="pil",
                image_mode="RGB",
                height=320
            )

        gr.Markdown("### Results")
        preds_df = gr.Dataframe(
            headers=["rank", "class_name", "probability"],
            datatype=["number", "str", "number"],
            interactive=False,
            label="Top-K Predictions"
        )
        label_out = gr.Label(label="(Mapping of Top-K)")
        bar_out = gr.Plot(label="Probability Bar Chart")

        predict_btn.click(
            fn=gr_predict,
            inputs=[image_in, model_choice, topk_in, force_gray],
            outputs=[preds_df, label_out, bar_out, status_box]
        )

        gr.Markdown(
            "### Notes\n"
            f"* AUTO_FORCE_GRAY_FOR_ROBUST = {AUTO_FORCE_GRAY_FOR_ROBUST}\n"
            "* 'Effective grayscale applied' shows whether the actual tensor was gray.\n"
            "* Standard model: input kept color unless you check the box.\n"
            "* Robust model: may auto-apply grayscale if configured.\n"
        )
    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch()