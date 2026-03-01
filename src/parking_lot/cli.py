"""CLI entry point for the parking lot scanner."""

import argparse
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parking Lot / Truck Logistics Scanner")
    parser.add_argument("--model", help="Path to YOLO model file", required=True)
    parser.add_argument("--source", help="Comma-separated sources (0, usb0, rtsp://..., file.mp4)", required=True)
    parser.add_argument("--thresh", help="Confidence threshold", default=0.5, type=float)
    parser.add_argument("--resolution", help="Resolution WxH (e.g. 1920x1080)", default=None)
    parser.add_argument("--record", help="Record video output", action="store_true")
    parser.add_argument("--grid-cols", help="Grid columns for multi-camera display", type=int, default=None)
    parser.add_argument("--ocr-workers", help="Number of OCR threads", type=int, default=2)
    parser.add_argument("--ocr-debug", help="Print OCR diagnostics", action="store_true")
    parser.add_argument("--use-gpu", help="Enable GPU for PaddleOCR", action="store_true")
    parser.add_argument("--no-enhance-feed", help="Disable frame enhancement", action="store_true")
    parser.add_argument("--no-sr", help="Disable super-resolution upscaling", action="store_true")
    parser.add_argument("--serve", help="Start FastAPI server instead of display", action="store_true")
    parser.add_argument("--port", help="API server port (with --serve)", type=int, default=8000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    from parking_lot.config import ScannerConfig, CameraConfig, SRConfig

    sources = [s.strip() for s in args.source.split(",") if s.strip()]

    if args.resolution:
        w, h = map(int, args.resolution.split("x"))
        resolution = (w, h)
    else:
        resolution = (1920, 1080)

    cfg = ScannerConfig(
        model_path=args.model,
        sources=sources,
        min_thresh=args.thresh,
        num_ocr_workers=max(1, args.ocr_workers),
        ocr_debug=args.ocr_debug,
        use_gpu=args.use_gpu,
        record=args.record,
        grid_cols=args.grid_cols,
        camera=CameraConfig(
            resolution=resolution,
            feed_enhance=not args.no_enhance_feed,
        ),
        sr=SRConfig(enabled=not args.no_sr),
    )

    from parking_lot.engine.scanner import ScannerEngine

    engine = ScannerEngine(cfg)
    engine.start()

    if args.serve:
        from parking_lot.api.app import create_app
        import uvicorn

        app = create_app(engine)
        print(f"\nStarting API server on port {args.port}...")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        engine.run_display_loop()


if __name__ == "__main__":
    main()
