"""Run the full GateVision pipeline on a video file or folder of images.

Runs: Detection → OCR → Fusion → Decision → DB + optional CSV log.

Usage:
    # Process a video
    uv run python -m scripts.run_pipeline video.mp4 -m models/yolo/my_model.pt

    # Process a folder of images (treated as sequential frames from one camera)
    uv run python -m scripts.run_pipeline data/frames/ -m models/yolo/my_model.pt

    # With CSV output and allowlist seeding
    uv run python -m scripts.run_pipeline video.mp4 -m models/yolo/my_model.pt --csv out.csv

    # Seed allowlist entries before processing
    uv run python -m scripts.run_pipeline video.mp4 -m models/yolo/my_model.pt --allow USDOT:1234567 --allow LicensePlate:ABC1234

    # Adjust fusion timeout (seconds of inactivity before closing a track)
    uv run python -m scripts.run_pipeline video.mp4 -m models/yolo/my_model.pt --timeout 5
"""

import argparse
import time
from pathlib import Path

import cv2
from sqlmodel import SQLModel, Session

from backend.config import settings
from backend.db.session import get_engine
from backend.decision.allowlist import add_entry
from backend.decision.handler import handle_track_closed
from backend.detection.detector import Detector
from backend.fusion.models import Read
from backend.fusion.pipeline import process_frame
from backend.fusion.tracker import TrackManager
from backend.recognition.ocr import OCREngine
from backend.utils.csv_logger import append_reads
from backend.utils.logging import setup_logging, get_logger

log = get_logger("pipeline")


def process_video(
    video_path: Path,
    detector: Detector,
    ocr: OCREngine,
    track_manager: TrackManager,
    session: Session,
    csv_path: str | None,
    camera_id: str = "cam1",
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FRAME_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {video_path.name}: {total} frames at {video_fps:.1f} fps")

    idx = 0
    start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = idx / video_fps
        reads = process_frame(frame, detector, ocr, camera_id, timestamp)

        if reads:
            print(f"  Frame {idx}: {len(reads)} reads → {', '.join(f'{r.class_name}={r.text} ({r.confidence:.2f})' for r in reads)}")
            if csv_path:
                append_reads(csv_path, reads)

        result = track_manager.update(reads, now=timestamp)
        if result is not None:
            track = track_manager.completed[-1][0]
            event = handle_track_closed(track_id=track.id, results=result, session=session)
            print(f"\n  === GATE EVENT ===")
            print(f"  Decision: {event.decision.value}")
            print(f"  Reason:   {event.decision_reason}")
            if event.usdot_number:
                print(f"  USDOT:    {event.usdot_number} (conf: {event.usdot_confidence:.2f})")
            if event.license_plate:
                print(f"  Plate:    {event.license_plate} (conf: {event.license_plate_confidence:.2f})")
            if event.trailer_number:
                print(f"  Trailer:  {event.trailer_number} (conf: {event.trailer_confidence:.2f})")
            print()

        idx += 1
        if idx % 200 == 0:
            elapsed = time.time() - start
            print(f"  ... {idx}/{total} frames ({elapsed:.1f}s, {idx/elapsed:.1f} fps)")

    cap.release()

    # Flush any remaining active track
    result = track_manager.flush()
    if result is not None:
        track = track_manager.completed[-1][0]
        event = handle_track_closed(track_id=track.id, results=result, session=session)
        print(f"\n  === GATE EVENT (final flush) ===")
        print(f"  Decision: {event.decision.value}")
        print(f"  Reason:   {event.decision_reason}")
        if event.usdot_number:
            print(f"  USDOT:    {event.usdot_number}")
        if event.license_plate:
            print(f"  Plate:    {event.license_plate}")
        if event.trailer_number:
            print(f"  Trailer:  {event.trailer_number}")
        print()

    elapsed = time.time() - start
    print(f"Done: {idx} frames in {elapsed:.1f}s ({idx/elapsed:.1f} fps)")
    print(f"Tracks completed: {len(track_manager.completed)}")


def process_images(
    image_dir: Path,
    detector: Detector,
    ocr: OCREngine,
    track_manager: TrackManager,
    session: Session,
    csv_path: str | None,
    camera_id: str = "cam1",
) -> None:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in extensions)

    if not images:
        print(f"No images found in {image_dir}")
        return

    print(f"Processing {len(images)} images from {image_dir}")

    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  Skipping unreadable: {img_path.name}")
            continue

        timestamp = float(idx)
        reads = process_frame(frame, detector, ocr, camera_id, timestamp)

        if reads:
            print(f"  {img_path.name}: {len(reads)} reads → {', '.join(f'{r.class_name}={r.text} ({r.confidence:.2f})' for r in reads)}")
            if csv_path:
                append_reads(csv_path, reads)

        result = track_manager.update(reads, now=timestamp)
        if result is not None:
            track = track_manager.completed[-1][0]
            event = handle_track_closed(track_id=track.id, results=result, session=session)
            print(f"\n  === GATE EVENT ===")
            print(f"  Decision: {event.decision.value} — {event.decision_reason}")
            print()

    # Flush remaining
    result = track_manager.flush()
    if result is not None:
        track = track_manager.completed[-1][0]
        event = handle_track_closed(track_id=track.id, results=result, session=session)
        print(f"\n  === GATE EVENT (final flush) ===")
        print(f"  Decision: {event.decision.value} — {event.decision_reason}")
        print()

    print(f"Done: {len(images)} images, {len(track_manager.completed)} tracks")


def main():
    parser = argparse.ArgumentParser(
        description="Run full GateVision pipeline on video or images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Video file or image directory")
    parser.add_argument("-m", "--model", default="models/yolo/my_model.pt", help="YOLO model path")
    parser.add_argument("-c", "--confidence", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--csv", default=None, help="CSV output path for per-frame reads")
    parser.add_argument("--db", default=None, help="SQLite database path (default: in-memory)")
    parser.add_argument("--timeout", type=float, default=5.0, help="Fusion track timeout in seconds (default: 5)")
    parser.add_argument("--camera-id", default="cam1", help="Camera ID label (default: cam1)")
    parser.add_argument(
        "--allow", action="append", default=[], metavar="TYPE:VALUE",
        help="Seed allowlist entry, e.g. --allow USDOT:1234567 --allow LicensePlate:ABC1234",
    )

    args = parser.parse_args()
    setup_logging(debug=True)

    # Database setup
    db_url = f"sqlite:///{args.db}" if args.db else "sqlite://"  # in-memory by default
    engine = get_engine(db_url)
    SQLModel.metadata.create_all(engine)
    session = Session(engine)

    # Seed allowlist
    for entry in args.allow:
        if ":" not in entry:
            parser.error(f"--allow must be TYPE:VALUE, got: {entry}")
        field_type, value = entry.split(":", 1)
        add_entry(session, field_type=field_type, value=value)
        print(f"Allowlist: {field_type} = {value}")

    # Initialize models
    print(f"Loading YOLO model: {args.model}")
    detector = Detector(model_path=args.model, confidence=args.confidence)
    print("Loading PaddleOCR engine...")
    ocr = OCREngine()

    track_manager = TrackManager(timeout=args.timeout)

    inp = Path(args.input)
    if inp.is_dir():
        process_images(inp, detector, ocr, track_manager, session, args.csv, args.camera_id)
    elif inp.is_file():
        process_video(inp, detector, ocr, track_manager, session, args.csv, args.camera_id)
    else:
        parser.error(f"Input not found: {inp}")

    session.close()


if __name__ == "__main__":
    main()
