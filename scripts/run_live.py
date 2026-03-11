"""Run GateVision in live/continuous mode.

Usage:
    # RTSP cameras
    uv run python -m scripts.run_live rtsp://host/cam1 rtsp://host/cam2

    # Video files as fake cameras
    uv run python -m scripts.run_live video1.mp4 video2.mp4 --realtime

    # USB webcam
    uv run python -m scripts.run_live webcam:0

    # Image folder
    uv run python -m scripts.run_live images:data/frames/

    # Mix sources
    uv run python -m scripts.run_live rtsp://host/cam1 video1.mp4 webcam:0
"""

import argparse

from sqlmodel import SQLModel, Session

from backend.config import settings
from backend.db.session import get_engine
from backend.decision.allowlist import add_entry
from backend.detection.detector import Detector
from backend.fusion.tracker import TrackManager
from backend.ingestion.camera import parse_source
from backend.ingestion.orchestrator import GateOrchestrator
from backend.recognition.ocr import OCREngine
from backend.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(
        description="Run GateVision in live/continuous mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("sources", nargs="+", help="Camera sources (rtsp://..., webcam:N, images:path, or video file)")
    parser.add_argument("-m", "--model", default="models/paddle/my_model.pt", help="YOLO model path")
    parser.add_argument("-c", "--confidence", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--csv", default=None, help="CSV output path for per-frame reads")
    parser.add_argument("--db", default=None, help="SQLite database path (default: in-memory)")
    parser.add_argument("--timeout", type=float, default=5.0, help="Fusion track timeout in seconds")
    parser.add_argument("--realtime", action="store_true", help="Play video files at real-time speed")
    parser.add_argument("--no-motion", action="store_true", help="Disable motion detection filter")
    parser.add_argument("--display", action="store_true", help="Show live OpenCV window with annotations")
    parser.add_argument("--output", default=None, metavar="PATH", help="Write annotated video to file (e.g. output.mp4)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress per-frame detection logs and gate event output")
    parser.add_argument("--no-banner", action="store_true", help="Hide decision banners (APPROVED/DENIED/FLAGGED) from display/video")
    parser.add_argument(
        "--allow", action="append", default=[], metavar="TYPE:VALUE",
        help="Seed allowlist entry, e.g. --allow USDOT:1234567",
    )

    args = parser.parse_args()
    setup_logging(debug=True)

    # Database
    db_url = f"sqlite:///{args.db}" if args.db else "sqlite://"
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

    # Parse camera sources
    cameras = [parse_source(spec, i, args.realtime) for i, spec in enumerate(args.sources)]
    print(f"Camera sources: {[f'{type(c).__name__}({c.camera_id})' for c in cameras]}")

    # Load models
    print(f"Loading YOLO model: {args.model}")
    detector = Detector(model_path=args.model, confidence=args.confidence)
    print("Loading PaddleOCR engine...")
    ocr = OCREngine()

    # Build orchestrator
    track_manager = TrackManager(timeout=args.timeout)
    orchestrator = GateOrchestrator(
        cameras=cameras,
        detector=detector,
        ocr=ocr,
        track_manager=track_manager,
        session=session,
        use_motion=not args.no_motion,
        motion_threshold=settings.camera.motion_threshold,
        motion_warmup=settings.camera.motion_warmup,
        csv_path=args.csv,
        display=args.display,
        output_path=args.output,
        quiet=args.quiet,
        show_banner=not args.no_banner,
    )

    quit_hint = " Press Q to quit." if args.display else " Press Ctrl+C to stop."
    print(f"Starting {len(cameras)} camera(s)...{quit_hint}")
    orchestrator.start()

    session.close()


if __name__ == "__main__":
    main()
