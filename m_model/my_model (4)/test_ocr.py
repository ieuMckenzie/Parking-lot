"""Quick OCR test - runs YOLO + PaddleOCR on images, prints results.
Compares raw crop vs super-resolution upscaled crop, with optional preprocessing.

Usage:
  python test_ocr.py s3.png                        # single image, terminal output
  python test_ocr.py s3.png --show                  # single image, show comparison window
  python test_ocr.py *.png --save                   # batch mode, save results to ocr_results/
  python test_ocr.py *.png --save --out results/    # batch mode, custom output dir
  python test_ocr.py s3.png --no-preprocess          # skip preprocessing
"""
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import sys
import argparse
import datetime
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

SR_MODEL_DIR = os.path.join(os.path.dirname(__file__), "sr_models")

parser = argparse.ArgumentParser()
parser.add_argument("images", nargs="+", help="Image(s) to test")
parser.add_argument("--model", default="my_model.pt", help="YOLO model path")
parser.add_argument("--show", action="store_true", help="Show before/after images in a window")
parser.add_argument("--save", action="store_true", help="Save text log and comparison images to output dir")
parser.add_argument("--out", default="ocr_results", help="Output directory for --save (default: ocr_results)")
parser.add_argument("--sr", default="both", choices=["fsrcnn", "edsr", "both"],
                    help="Which SR model to test (default: both)")
parser.add_argument("--scale", default="all", choices=["2", "3", "4", "all"],
                    help="Upscale factor (default: all)")
parser.add_argument("--no-preprocess", action="store_true",
                    help="Disable preprocessing (preprocessing is ON by default)")
args = parser.parse_args()
use_preprocess = not args.no_preprocess

# --- Preprocessing functions ---
def preprocess_clahe(img):
    """CLAHE contrast enhancement."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def preprocess_sharpen(img):
    """Unsharp mask sharpening."""
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

def preprocess_binarize(img):
    """Adaptive binarization."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

def preprocess_pad(img, pad=10):
    """Add white border padding."""
    return cv2.copyMakeBorder(img, pad, pad, pad, pad,
                              cv2.BORDER_CONSTANT, value=(255, 255, 255))

PREPROCESS_PIPELINES = {
    "clahe+sharpen": lambda img: preprocess_pad(preprocess_sharpen(preprocess_clahe(img))),
    "clahe+binary":  lambda img: preprocess_pad(preprocess_binarize(preprocess_clahe(img))),
    "sharpen":       lambda img: preprocess_pad(preprocess_sharpen(img)),
}

# --- Load models ---
print(f"Loading YOLO model: {args.model}")
model = YOLO(args.model, task='detect')
labels = model.names

print("Loading PaddleOCR...")
reader = PaddleOCR(lang='en')

# Load SR models
sr_models = {}
model_types = ["fsrcnn", "edsr"] if args.sr == "both" else [args.sr]
scales = [2, 3, 4] if args.scale == "all" else [int(args.scale)]

for model_type in model_types:
    for scale in scales:
        pb_file = f"{model_type.upper()}_x{scale}.pb"
        pb_path = os.path.join(SR_MODEL_DIR, pb_file)
        if not os.path.exists(pb_path):
            print(f"  WARNING: {pb_path} not found, skipping")
            continue
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(pb_path)
        sr.setModel(model_type, scale)
        key = f"{model_type}_x{scale}"
        sr_models[key] = (sr, scale)
        print(f"  Loaded SR model: {key}")

print(f"  Preprocessing: {'ON' if use_preprocess else 'OFF'}")

# Set up output dir
if args.save:
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "results.txt")
    log_file = open(log_path, "w")
    print(f"\nSaving results to: {args.out}/")
else:
    log_file = None

def log(msg=""):
    """Print to terminal and optionally write to log file."""
    print(msg)
    if log_file:
        log_file.write(msg + "\n")

def run_ocr(img):
    """Run PaddleOCR on an image and return list of (text, conf) tuples."""
    result = reader.ocr(img)
    out = []
    if not result or result[0] is None:
        return out
    r0 = result[0]
    texts = getattr(r0, 'rec_texts', None) or (r0.get('rec_texts') if hasattr(r0, 'get') else None)
    scores = getattr(r0, 'rec_scores', None) or (r0.get('rec_scores') if hasattr(r0, 'get') else None)
    if texts:
        texts = list(texts)
        scores = list(scores) if scores is not None else [0.0] * len(texts)
        for t, s in zip(texts, scores):
            out.append((str(t).strip(), float(s)))
    else:
        for line in list(r0):
            if line and len(line) >= 2:
                second = line[1]
                if isinstance(second, (list, tuple)):
                    out.append((str(second[0]).strip(), float(second[1])))
                else:
                    out.append((str(second).strip(), 0.0))
    return out

def run_ocr_best(img):
    """Run OCR with all preprocessing pipelines and return the best result.
    Returns (results, pipeline_name) where results is list of (text, conf)."""
    # Always try the raw image first
    raw_results = run_ocr(img)
    raw_conf = _avg_conf(raw_results)
    best_results = raw_results
    best_name = "raw"
    best_conf = raw_conf

    if use_preprocess:
        for pipe_name, pipe_fn in PREPROCESS_PIPELINES.items():
            processed = pipe_fn(img)
            results = run_ocr(processed)
            conf = _avg_conf(results)
            if conf > best_conf:
                best_results = results
                best_name = pipe_name
                best_conf = conf

    return best_results, best_name

def _avg_conf(results):
    if not results:
        return 0.0
    return sum(c for _, c in results) / len(results)

def format_ocr_results(results, indent="  "):
    if not results:
        return [f"{indent}(no text detected)"]
    return [f"{indent}'{text}' (conf={conf:.2f})" for text, conf in results]

def build_comparison_image(images_dict):
    """Build a side-by-side comparison image from a dict of {name: img}."""
    max_h = max(img.shape[0] for img in images_dict.values())
    panels = []
    for name, img in images_dict.items():
        scale = max_h / img.shape[0]
        resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        labeled = cv2.copyMakeBorder(resized, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        cv2.putText(labeled, name.upper(), (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        panels.append(labeled)
    return np.hstack(panels)

# --- Process each image ---
if log_file:
    log(f"OCR Test Results - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"SR models: {', '.join(sr_models.keys()) or 'none'}")
    log(f"Scale(s): {', '.join(str(s) for s in scales)}")
    log(f"Preprocessing: {'ON' if use_preprocess else 'OFF'}")
    log()

for img_idx, image_path in enumerate(args.images):
    if not os.path.isfile(image_path):
        log(f"Skipping {image_path} (not found)")
        continue

    img_name = os.path.splitext(os.path.basename(image_path))[0]

    log(f"{'#'*60}")
    log(f"# Image: {image_path}")
    log(f"{'#'*60}")

    frame = cv2.imread(image_path)
    if frame is None:
        log(f"  Could not read image, skipping.\n")
        continue

    h_img, w_img = frame.shape[:2]
    log(f"  Size: {w_img}x{h_img}")

    results = model(frame, verbose=False, iou=0.45)[0]

    if len(results.boxes) == 0:
        log(f"  No detections.\n")
        continue

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = box.conf.item()
        cls_id = int(box.cls)
        label = labels[cls_id] if cls_id < len(labels) else "unknown"

        log(f"\n  {'='*56}")
        log(f"  Detection {i+1}: {label} (conf={conf:.2f}) at [{x1},{y1},{x2},{y2}]")
        log(f"  {'='*56}")

        # Crop with dynamic padding
        crop_h = y2 - y1
        if crop_h < 30:
            pad_x_pct, pad_y_pct = 0.40, 0.35
        elif crop_h < 60:
            pad_x_pct, pad_y_pct = 0.30, 0.25
        else:
            pad_x_pct, pad_y_pct = 0.20, 0.15

        px = int((x2 - x1) * pad_x_pct)
        py = int((y2 - y1) * pad_y_pct)
        crop = frame[max(0, y1-py):min(h_img, y2+py), max(0, x1-px):min(w_img, x2+px)]

        log(f"  Crop size: {crop.shape[1]}x{crop.shape[0]}")

        comparison_images = {"raw": crop}

        # --- Raw crop OCR ---
        log(f"\n  [RAW CROP]")
        raw_results, raw_pipe = run_ocr_best(crop)
        for line in format_ocr_results(raw_results, indent="    "):
            log(line)
        if use_preprocess and raw_pipe != "raw":
            log(f"    (best via: {raw_pipe})")

        # --- SR crop OCR ---
        for sr_name, (sr_model, sr_scale) in sr_models.items():
            sr_crop = sr_model.upsample(crop)
            log(f"\n  [{sr_name.upper()}] -> {sr_crop.shape[1]}x{sr_crop.shape[0]}")
            sr_results, sr_pipe = run_ocr_best(sr_crop)
            for line in format_ocr_results(sr_results, indent="    "):
                log(line)
            if use_preprocess and sr_pipe != "raw":
                log(f"    (best via: {sr_pipe})")
            comparison_images[sr_name] = sr_crop

        # --- Save comparison image ---
        if args.save:
            combined = build_comparison_image(comparison_images)
            save_name = f"{img_name}_det{i+1}_{label}.png"
            save_path = os.path.join(args.out, save_name)
            cv2.imwrite(save_path, combined)
            log(f"\n  Saved: {save_path}")

        # --- Show images ---
        if args.show:
            combined = build_comparison_image(comparison_images)
            win_name = f"{img_name} - Detection {i+1}: {label}"
            cv2.imshow(win_name, combined)
            log(f"\n  (press any key to continue)")
            cv2.waitKey(0)
            cv2.destroyWindow(win_name)

    log()

if log_file:
    log_file.close()
    print(f"\nLog saved to: {log_path}")

if args.show:
    cv2.destroyAllWindows()
