# GateVision

Automated truck gate identification system using computer vision. Reads license plates, USDOT numbers, and trailer numbers from trucks arriving at a facility gate using overlapping cameras, then makes access decisions by cross-referencing against an allowlist.

## Setup

```bash
# Install uv package manager (If you don't have it already)
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
# On Windows.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install dependencies
uv sync
```
# Live / Multi-Camera Mode

The `scripts/run_live.py` script runs the full pipeline continuously on one or more camera sources (RTSP streams, video files, webcams, or image folders).

### Basic Usage

```bash
# Single video file
uv run python -m scripts.run_live data/DOT.mp4 --no-motion --csv detections.csv

# Multiple video files as simulated cameras
uv run python -m scripts.run_live video1.mp4 video2.mp4 --no-motion

# RTSP cameras
uv run python -m scripts.run_live rtsp://host/cam1 rtsp://host/cam2

# USB webcam
uv run python -m scripts.run_live webcam:0

# Image folder
uv run python -m scripts.run_live images:data/frames/
```

### All Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | `models/paddle/my_model.pt` | YOLO model weights path |
| `-c, --confidence` | `0.25` | Detection confidence threshold |
| `--csv` | _(disabled)_ | CSV output path for per-frame reads |
| `--db` | _(in-memory)_ | SQLite database file path |
| `--timeout` | `5.0` | Fusion track timeout in seconds |
| `--realtime` | _(off)_ | Play video files at real-time speed |
| `--no-motion` | _(off)_ | Disable motion detection filter |
| `--allow TYPE:VALUE` | _(none)_ | Seed an allowlist entry (repeatable) |

## Running the Pipeline

The `scripts/run_pipeline.py` script runs the full pipeline on recorded footage: YOLO detection, PaddleOCR recognition, temporal fusion, decision engine, and database persistence.

### Basic Usage

```bash
# Process a video file
uv run python -m scripts.run_pipeline path/to/video.mp4 -m models/yolo/my_model.pt

# Process a folder of images (treated as sequential frames)
uv run python -m scripts.run_pipeline path/to/frames/ -m models/yolo/my_model.pt
```

### CSV Output

Generate a CSV file compatible with the frontend dashboard:

```bash
uv run python -m scripts.run_pipeline video.mp4 -m models/yolo/my_model.pt --csv detections.csv
```

Output format: `Timestamp,Camera_ID,Value,Data_Type,Confidence`

### Allowlist

Seed allowlist entries so matching trucks are approved:

```bash
uv run python -m scripts.run_pipeline video.mp4 -m models/yolo/my_model.pt \
  --allow USDOT:1234567 \
  --allow LicensePlate:ABC1234
```

### Persistent Database

By default results are stored in-memory. To save to a SQLite file:

```bash
uv run python -m scripts.run_pipeline video.mp4 -m models/yolo/my_model.pt --db results.db
```

### All Options

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | `models/yolo/my_model.pt` | YOLO model weights path |
| `-c, --confidence` | `0.25` | Detection confidence threshold |
| `--csv` | _(disabled)_ | CSV output path for per-frame reads |
| `--db` | _(in-memory)_ | SQLite database file path |
| `--timeout` | `5.0` | Seconds of inactivity before closing a track |
| `--camera-id` | `cam1` | Camera ID label for the source |
| `--allow TYPE:VALUE` | _(none)_ | Seed an allowlist entry (repeatable) |

## Other Scripts

```bash
# Detection only — draw bounding boxes on a video/image (no OCR)
uv run python -m scripts.test_detect input.mp4 -m models/yolo/my_model.pt

# Extract frames from a video for annotation
uv run python -m scripts.annotate video.mp4 -o data/raw -f 2.0
```

## Running Tests

```bash
uv run pytest tests/ -v
```

## Project Structure

```
backend/
├── config.py              # Pydantic-settings config (.env driven)
├── main.py                # FastAPI app with health check
├── db/                    # SQLModel ORM (GateEvent, AllowlistEntry)
├── decision/              # Allowlist CRUD, decision engine, track close handler
├── detection/             # YOLO detector wrapper
├── fusion/                # Temporal fusion (tracker, voter, pipeline)
├── recognition/           # PaddleOCR 3.x full pipeline (det+rec) + regex postprocessing
└── utils/                 # Logging, CSV logger
scripts/                   # CLI tools for processing footage
tests/                     # Test suite
models/                    # YOLO and PaddleOCR weights
```
