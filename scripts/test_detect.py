"""CLI: run YOLO detection on a video, draw bboxes, and save annotated output."""

import argparse
from pathlib import Path

import cv2

from backend.detection.detector import Detection, Detector

COLORS = {
    "LicensePlate": (0, 255, 0),
    "USDOT": (255, 0, 0),
    "TrailerNum": (0, 165, 255),
    "ContainerNum": (255, 255, 0),
    "ContainerPlate": (0, 255, 255),
}
DEFAULT_COLOR = (200, 200, 200)


def draw_detections(frame, detections: list[Detection]) -> None:
    for det in detections:
        color = COLORS.get(det.class_name, DEFAULT_COLOR)
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def process_video(video_path: str, model_path: str, output_path: str, confidence: float) -> None:
    detector = Detector(model_path=model_path, confidence=confidence)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FRAME_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    idx = 0
    det_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect(frame)
        det_count += len(detections)
        draw_detections(frame, detections)
        writer.write(frame)
        idx += 1
        if idx % 100 == 0:
            print(f"  {idx}/{total} frames processed...")

    cap.release()
    writer.release()
    print(f"Done: {idx} frames, {det_count} total detections → {output_path}")


def process_image(image_path: str, model_path: str, output_path: str, confidence: float) -> None:
    detector = Detector(model_path=model_path, confidence=confidence)
    frame = cv2.imread(image_path)
    if frame is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    detections = detector.detect(frame)
    draw_detections(frame, detections)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, frame)
    print(f"Done: {len(detections)} detections → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLO detection and save annotated output")
    parser.add_argument("input", help="Path to video or image file")
    parser.add_argument("-m", "--model", default="models/paddle/my_model.pt", help="YOLO model path")
    parser.add_argument("-o", "--output", default=None, help="Output path (default: input_detected.ext)")
    parser.add_argument("-c", "--confidence", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    inp = Path(args.input)
    if args.output:
        out = args.output
    else:
        out = str(inp.with_stem(f"{inp.stem}_detected"))

    if inp.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        process_image(args.input, args.model, out, args.confidence)
    else:
        process_video(args.input, args.model, out, args.confidence)
