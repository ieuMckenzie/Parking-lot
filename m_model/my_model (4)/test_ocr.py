"""Quick OCR test - runs YOLO + PaddleOCR on a single image, prints results."""
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

image_path = sys.argv[1] if len(sys.argv) > 1 else "s3.png"
model_path = sys.argv[2] if len(sys.argv) > 2 else "my_model.pt"

print(f"Loading YOLO model: {model_path}")
model = YOLO(model_path, task='detect')
labels = model.names

print("Loading PaddleOCR...")
reader = PaddleOCR(lang='en')

print(f"Running on: {image_path}\n")
frame = cv2.imread(image_path)
if frame is None:
    sys.exit(f"Could not read image: {image_path}")

h_img, w_img = frame.shape[:2]
print(f"Image size: {w_img}x{h_img}\n")

results = model(frame, verbose=False, iou=0.45)[0]

for box in results.boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
    conf = box.conf.item()
    cls_id = int(box.cls)
    label = labels[cls_id] if cls_id < len(labels) else "unknown"

    print(f"--- YOLO Detection: {label} (conf={conf:.2f}) at [{x1},{y1},{x2},{y2}] ---")

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

    print(f"  Crop size: {crop.shape[1]}x{crop.shape[0]} (pad: {pad_x_pct:.0%}/{pad_y_pct:.0%})")

    # Run OCR on raw crop
    result = reader.ocr(crop)
    if result and result[0] is not None:
        r0 = result[0]
        texts = getattr(r0, 'rec_texts', None) or (r0.get('rec_texts') if hasattr(r0, 'get') else None)
        scores = getattr(r0, 'rec_scores', None) or (r0.get('rec_scores') if hasattr(r0, 'get') else None)

        if texts:
            texts = list(texts)
            scores = list(scores) if scores is not None else [0.0] * len(texts)
            for t, s in zip(texts, scores):
                print(f"  OCR result: '{t}' (conf={float(s):.2f})")
        else:
            # 2.x format
            for line in list(r0):
                if line and len(line) >= 2:
                    second = line[1]
                    if isinstance(second, (list, tuple)):
                        print(f"  OCR result: '{second[0]}' (conf={float(second[1]):.2f})")
                    else:
                        print(f"  OCR result: '{second}'")
    else:
        print("  OCR: no text detected")

    print()
