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
import cv2
from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO
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
        "path": os.path.join(BASE_DIR, "food_efficientnet_b3"),
        "note": "Latest B3 color model."
    },
    "grayscale_robust": {
        "label": "EfficientNet-B3 Grayscale-Robust",
        "path": os.path.join(BASE_DIR, "food_efficientnet_b3_grayft"),
        "note": "B3 model trained with grayscale augmentation."
    }
}

_yolo_model = None
CONF_THRESHOLD = 0.7
imgsz = 768
FONT = ImageFont.load_default()  # fallback font
pt = "best.pt"
det_model = YOLO(str(pt))
print("[DEBUG] using model file:", det_model.ckpt_path)

def resize_to_342(img_rgb):
    return cv2.resize(img_rgb, (CLF_SIZE, CLF_SIZE), interpolation=cv2.INTER_LINEAR)

def preprocess_for_clf(img_rgb):
    x = resize_to_342(img_rgb).astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return tf.keras.applications.efficientnet.preprocess_input(x)

def crop_with_pad(img_rgb, xyxy, pad=0.10):
    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = map(float, xyxy)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = (x2-x1)*(1+pad), (y2-y1)*(1+pad)
    x1n, y1n = int(max(0, cx-bw/2)), int(max(0, cy-bh/2))
    x2n, y2n = int(min(W-1, cx+bw/2)), int(min(H-1, cy+bh/2))
    if y2n <= y1n or x2n <= x1n:
        return None, (x1n, y1n, x2n, y2n)
    return img_rgb[y1n:y2n, x1n:x2n].copy(), (x1n, y1n, x2n, y2n)

def draw_box(img_pil, box, label, color=(0, 140, 255)):
    draw = ImageDraw.Draw(img_pil)
    x1, y1, x2, y2 = [int(v) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    tw, th = draw.textbbox((0,0), label, font=FONT)[2:4]
    draw.rectangle([x1, y1-th-6, x1+tw+8, y1], fill=color)
    draw.text((x1+4, y1-th-4), label, font=FONT, fill=(255,255,255))
    return img_pil

def _as_probs(out):
    if isinstance(out, dict):
        return next(iter(out.values()))
    return out

CLF_SIZE = 342
DET_CONF = 0.35
DET_IOU  = 0.5

def resize_to_342(img_rgb):
    return cv2.resize(img_rgb, (CLF_SIZE, CLF_SIZE), interpolation=cv2.INTER_LINEAR)

def preprocess_for_clf(img_rgb):
    x = resize_to_342(img_rgb).astype(np.float32)
    return x[None, ...]

def crop_with_pad(img_rgb, xyxy, pad=0.10):
    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = map(float, xyxy)
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = (x2-x1), (y2-y1)
    bw *= (1+pad); bh *= (1+pad)
    x1n, y1n = int(max(0, cx-bw/2)), int(max(0, cy-bh/2))
    x2n, y2n = int(min(W-1, cx+bw/2)), int(min(H-1, cy+bh/2))
    if y2n <= y1n or x2n <= x1n:
        return None, (x1n, y1n, x2n, y2n)
    return img_rgb[y1n:y2n, x1n:x2n].copy(), (x1n, y1n, x2n, y2n)

def draw_box(img_pil, box, label, color=(0, 140, 255)):
    draw = ImageDraw.Draw(img_pil)
    x1, y1, x2, y2 = [int(v) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    tw, th = draw.textbbox((0,0), label, font=FONT)[2:]
    draw.rectangle([x1, y1-th-6, x1+tw+8, y1], fill=color)
    draw.text((x1+4, y1-th-4), label, font=FONT, fill=(255,255,255))
    return img_pil

def _as_probs(out):
    if isinstance(out, dict):
        return next(iter(out.values()))
    return out

def detect_and_classify(image_like, clf_model, min_box_area=4000, keep_topk=20):
    """
    Run YOLO detection on the image and classify each detected crop using clf_model.
    Returns:
        - detections: list of dicts with box, class, score
        - pil image with bounding boxes drawn
    """
    CLASS_NAMES = load_class_names()

    # -----------------------------
    # 1. Convert input to RGB NumPy
    # -----------------------------
    if isinstance(image_like, Image.Image):
        rgb = np.array(image_like.convert("RGB"))
    elif isinstance(image_like, np.ndarray):
        if image_like.shape[2] == 3:
            rgb = image_like
        else:
            raise ValueError("Expected 3-channel RGB array.")
    else:
        bgr = cv2.imread(str(image_like))
        if bgr is None:
            raise ValueError(f"Cannot read image: {image_like}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # -----------------------------
    # 2. YOLO detection
    # -----------------------------
    res = det_model.predict(
        source=rgb,
        conf=DET_CONF,
        iou=DET_IOU,
        verbose=False
    )[0]

    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return [], Image.fromarray(rgb)

    xyxy = boxes.xyxy.cpu().numpy()

    # -----------------------------
    # 3. Sort/filter by area
    # -----------------------------
    areas = (xyxy[:,2]-xyxy[:,0]) * (xyxy[:,3]-xyxy[:,1])
    order = np.argsort(-areas)
    keep = [i for i in order if areas[i] >= min_box_area][:keep_topk]

    pil = Image.fromarray(rgb)
    detections = []

    # -----------------------------
    # 4. Classification
    # -----------------------------
    for i in keep:

        # NEW → YOLO detection confidence
        yolo_conf = float(boxes.conf[i].cpu().numpy())

        crop, box = crop_with_pad(rgb, xyxy[i], pad=0.10)
        if crop is None:
            continue

        x = preprocess_for_clf(crop)
        out = clf_model.predict(x, verbose=0)
        probs = np.array(_as_probs(out))

        if probs.ndim == 1:
            probs = probs[None, ...]
        if probs.shape[-1] != len(CLASS_NAMES):
            probs = tf.nn.softmax(probs, axis=-1).numpy()

        p = probs[0]
        cls_id = int(np.argmax(p))
        score = float(p[cls_id])
        name = CLASS_NAMES[cls_id]

        if score < CONF_THRESHOLD:
            name = "Unknown Dish"
            cls_id = -1

        label = f"{name} {score:.2f}"
        color = (255, 0, 0) if name == "Unknown Dish" else (0, 140, 255)

        pil = draw_box(pil, box, label, color=color)

        detections.append({
            "box_xyxy": tuple(map(int, box)),
            "cls_id": cls_id,
            "cls_name": name,
            "score": score,         # classifier confidence
            "det_conf": yolo_conf   # NEW: detection confidence
        })

    return detections, pil







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
            # EXTRA safety check: ensure this is really a Keras model
            if not hasattr(model, "predict"):
                raise ValueError("Loaded model lacks Keras API, falling back.")
        except Exception as e:
            print(f"[WARN] Keras load failed ({e}), using tf.saved_model.load instead.")
            imported = tf.saved_model.load(path)
        
            class TFModelWrapper:
                def __init__(self, imported):
                    self.imported = imported
                    self.sig = imported.signatures["serving_default"]
        
                    self.is_saved_model = True
                    self._func = self.sig
        
                    self.input_name = list(self.sig.structured_input_signature[1].keys())[0]
        
                    # Detect real input shape from SavedModel signature
                    tensor_spec = list(self.sig.structured_input_signature[1].values())[0]
                    self.input_shape = tensor_spec.shape
                    self.inputs = [tensor_spec]
                    
                    # Extract H,W
                    self.model_h = tensor_spec.shape[1]
                    self.model_w = tensor_spec.shape[2]
        
                def __call__(self, x, training=False):
                    if not isinstance(x, tf.Tensor):
                        x = tf.convert_to_tensor(x, dtype=tf.float32)
                    out = self._func(**{self.input_name: x})
                    first_key = next(iter(out.keys()))
                    return out[first_key]

                def predict(self, x, verbose=0):
                    return self.__call__(x)
        
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

    # --- SAFE input size detection ---
    try:
        model_h = model.inputs[0].shape[1]
        model_w = model.inputs[0].shape[2]
        if model_h is None or model_w is None:
            raise ValueError("Invalid shape")
    except Exception:
        # TF SavedModel wrapper: no .inputs → default EfficientNet size
        model_h = model_w = 224

    # --- Resize logic ---
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
    if getattr(model, "is_saved_model", False):
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

    print("Model type:", type(model))
    print("Has layers:", hasattr(model, "layers"))
    if hasattr(model, "layers"):
        for l in model.layers:
            print(l.name, l.output_shape)

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
    cam_heatmap = None   # <- new: raw heatmap (normalized float array)
    if show_cam:
        try:
            top_class_idx = idxs[0]
            top_class_name = class_names[top_class_idx]
            print(f"[INFO] Generating visualization for: {top_class_name} (prob: {probs[top_class_idx]:.4f})")
        
            # Attempt Grad-CAM; fallback to saliency map automatically
            heatmap = None
            try:
                heatmap = generate_gradcam(model, batch, top_class_idx)
            except Exception as e:
                print(f"[WARN] Grad-CAM failed: {e}")
    
            if heatmap is None:
                print("[INFO] Falling back to saliency map...")
                heatmap = generate_saliency_map(model, batch, top_class_idx)
    
            if heatmap is not None:
                # keep a copy of the raw normalized heatmap
                cam_heatmap = heatmap.copy() if isinstance(heatmap, np.ndarray) else np.array(heatmap)
    
                # blended PIL for single-crop display (unchanged)
                cam_image = overlay_heatmap_on_image(pil_img, heatmap, alpha=0.5)
                if cam_image is not None:
                    print("[INFO] ✓ Visualization generated successfully")
                else:
                    print("[WARN] ✗ Overlay failed")
            else:
                print("[WARN] ✗ Heatmap generation failed completely")
    
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

    return rows, label_map, fig, meta, cam_image, cam_heatmap
# --------------------------------------------------
# Gradio callback
# --------------------------------------------------
def gr_predict_dynamic(image, model_choice, top_k, force_gray, show_cam):
    clf_model = load_model(model_choice)
    table_list = []

    try:
        detections, yolo_image = detect_and_classify(image, clf_model)
    except Exception as e:
        print(f"[WARN] YOLO detection failed: {e}")
        detections = []
        yolo_image = _ensure_pil(image)

    all_tables = []   # <-- collect each table separately
    
    combined_heatmap = np.zeros((yolo_image.height, yolo_image.width), dtype=np.float32)

    for i, det in enumerate(detections):
        cls_name = det.get("cls_name", f"Dish {i+1}")
        box = det.get("box_xyxy")
        det_conf = det.get("det_conf", None)  # <- get YOLO detection confidence

        if cls_name.lower() == "unknown dish" or box is None:
            continue

        x1, y1, x2, y2 = map(int, box)
        crop, _ = crop_with_pad(np.array(yolo_image), box, pad=0.10)
        if crop is None:
            continue
            
        rows, label_map, fig, meta, cam_crop, cam_crop_heatmap = predict(
            crop, model_choice, top_k, force_gray, show_cam
        )
        
        # Use the raw heatmap (cam_crop_heatmap) if available.
        if cam_crop_heatmap is not None:
            # cam_crop_heatmap is a small 2D array (normalized floats). Resize it to bbox size.
            from PIL import Image as PILImage
            heat_pil = PILImage.fromarray((cam_crop_heatmap * 255).astype(np.uint8))
            heat_resized = heat_pil.resize((x2 - x1, y2 - y1), PILImage.BILINEAR)
            cam_crop_array = np.array(heat_resized).astype(np.float32) / 255.0
        
            # Merge into the full heatmap (use max to preserve strongest response)
            combined_heatmap[y1:y2, x1:x2] = np.maximum(
                combined_heatmap[y1:y2, x1:x2], cam_crop_array
            )

        # Build markdown table for this single detection
        table_md = ""
        
        # Add YOLO detection confidence above the table
        if det_conf is not None:
            table_md += f"**YOLO Detection Confidence:** {det_conf:.2f}\n\n"
        
        # Add table header
        table_md += "| Rank | Class | Probability |\n|------|-------|------------|\n"
        
        # Add the classifier probabilities
        for r in rows:
            table_md += f"| {r[0]} | {r[1]} | {r[2]:.4f} |\n"
        
        full_md = f"### {cls_name} (Box {i+1})\n{table_md}\n"
        all_tables.append(full_md)

    # If nothing detected
    if not all_tables:
        all_tables = ["No recognized dishes detected."]

    # ---- NEW: SPLIT INTO TWO COLUMNS ----
    left_col = []
    right_col = []

    for idx, table in enumerate(all_tables):
        if idx % 2 == 0:
            left_col.append(table)
        else:
            right_col.append(table)

    left_md = "\n\n".join(left_col)
    right_md = "\n\n".join(right_col)

    clean_yolo = _ensure_pil(image).copy()
    clean_yolo = clean_yolo.resize(yolo_image.size)   # ensure same size
    combined_cam_image = overlay_heatmap_on_image(clean_yolo, combined_heatmap, alpha=0.5)

    # ---- UPDATED RETURN (must be 4 things) ----
    return left_md, right_md, combined_cam_image, yolo_image




def live_predict(image, model_choice, top_k, force_gray, show_cam):
    """
    Real-time frame prediction: YOLO + classification.
    Returns:
        - yolo_image with boxes and class labels
    """
    clf_model = load_model(model_choice)
    
    try:
        detections, yolo_image = detect_and_classify(image, clf_model)
    except Exception as e:
        print(f"[WARN] YOLO detection failed: {e}")
        yolo_image = _ensure_pil(image)
        return yolo_image

    # Overlay class names on boxes
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(yolo_image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = None  # fallback default
    
    for det in detections:
        cls_name = det.get("cls_name", "Dish")
        box = det.get("box_xyxy")
        if box is None:
            continue
        x1, y1, x2, y2 = map(int, box)
        
        # Choose color: blue if predicted class is known, red otherwise
        color = "blue" if cls_name.lower() != "unknown dish" else "red"
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, y1-16), cls_name, fill=color, font=font)
    
    return yolo_image





# --------------------------------------------------
# Build the Gradio Demo
# --------------------------------------------------
import gradio as gr
import numpy as np
from PIL import Image

# --------------------------------------------------
# Helper / prediction functions (simplified references)
# --------------------------------------------------
# Make sure you have your own implementations for:
# load_class_names(), load_model(), detect_and_classify(), crop_with_pad(), predict()
# --------------------------------------------------




# --------------------------------------------------
# Gradio Demo
# --------------------------------------------------
def build_demo():
    load_class_names()
    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("# 🍜 Chinese Food Classifier + YOLO Detection")

        with gr.Tabs():
            # ==========================================================
            # TAB 1 — UPLOAD OR SNAPSHOT
            # ==========================================================
            with gr.Tab("Upload or Snapshot"):
                # ---------------------------
                # ROW 1 — MODEL SELECTION
                # ---------------------------
                with gr.Row():
                    with gr.Column(scale=1):
                        model_choice = gr.Radio(
                            choices=list(MODEL_CONFIGS.keys()),
                            value="standard",
                            label="Classifier Model"
                        )
                        topk_in = gr.Slider(1, 10, value=5, step=1, label="Top-K")
                        force_gray = gr.Checkbox(label="Force grayscale", value=False)
                        show_cam = gr.Checkbox(label="Show CAM", value=True)
                        status_box = gr.Markdown("")

                # ---------------------------
                # ROW 2 — IMAGE UPLOAD
                # ---------------------------
                with gr.Row():
                    with gr.Column(scale=1):
                        image_in = gr.Image(
                            label="Upload or Webcam Snapshot",
                            type="pil",
                            image_mode="RGB",
                            height=350,
                            sources=["upload", "webcam"]
                        )
                        predict_btn = gr.Button("Predict", variant="primary")

                # ---------------------------
                # ROW 3 — YOLO + CAM OUTPUTS
                # Labels at bottom left
                # ---------------------------
                with gr.Row():
                    with gr.Column(scale=1):
                        cam_image_out = gr.Image(show_label=False, height=320)
                        gr.Markdown("**CAM Visualization**")

                    with gr.Column(scale=1):
                        yolo_image_out = gr.Image(show_label=False, height=320)
                        gr.Markdown("**YOLO Detection**")

                # ---------------------------
                # ROW 4 — Dynamic Top-K Tables (2 columns)
                # ---------------------------
                with gr.Row():
                    dynamic_md_left = gr.Markdown("")
                    dynamic_md_right = gr.Markdown("")

                # ---------------------------
                # Connect predict button
                # ---------------------------
                predict_btn.click(
                    fn=gr_predict_dynamic,
                    inputs=[image_in, model_choice, topk_in, force_gray, show_cam],
                    outputs=[dynamic_md_left, dynamic_md_right, cam_image_out, yolo_image_out]
                )

            # ==========================================================
            # TAB 2 — LIVE WEBCAM
            # ==========================================================
            with gr.Tab("Live (Webcam)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        model_choice_live = gr.Radio(
                            choices=list(MODEL_CONFIGS.keys()),
                            value="standard",
                            label="Model (Live)"
                        )
                        topk_live = gr.Slider(1, 10, value=5, step=1, label="Top-K (Live)")
                        force_gray_live = gr.Checkbox(label="Force grayscale (Live)", value=False)
                        show_cam_live = gr.Checkbox(label="Show CAM (Live)", value=False)
            
                with gr.Row():
                    with gr.Column(scale=1):
                        image_live = gr.Image(
                            label="Webcam Stream",
                            streaming=True,
                            sources=["webcam"],
                            image_mode="RGB",
                            height=320
                        )
            
                with gr.Row():
                    with gr.Column(scale=1):
                        yolo_image_live = gr.Image(show_label=False, height=320)
            
            # Connect live webcam to the prediction function
            image_live.stream(
                fn=live_predict,
                inputs=[image_live, model_choice_live, topk_live, force_gray_live, show_cam_live],
                outputs=[yolo_image_live],
            )

        gr.Markdown("Tip: **Upload or Snapshot** for single image, or **Live** for continuous webcam predictions.")

    return demo





# --- Launch demo ---
demo = build_demo()

if __name__ == "__main__":
    demo.launch()