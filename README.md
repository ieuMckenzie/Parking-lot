# Parking Lot Scanner

Truck logistics scanner using YOLO + PaddleOCR. Detects and reads license plates, container numbers, USDOT numbers, and trailer numbers from camera feeds.

## Quick Start

```bash
# Install
uv sync

# Run CLI scanner
parking-scanner --model models/paddle/my_model.pt --source usb0

# Run with API server
parking-scanner --model models/paddle/my_model.pt --source usb0 --serve
```

## Project Structure

- `src/parking_lot/` — Main Python package
  - `core/` — Detection + OCR pipeline (detector, OCR, preprocessing, super-resolution, validation)
  - `capture/` — Camera I/O (camera capture, grid display)
  - `engine/` — Threaded scanner orchestration (state, logger, scanner engine)
  - `api/` — FastAPI layer for integration
  - `cli.py` — CLI entry point
- `models/` — YOLO model weights + SR models (not tracked in git)
- `data/` — Runtime data (authorized plates list)
