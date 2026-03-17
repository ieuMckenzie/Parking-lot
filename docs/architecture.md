# System Architecture

## Overview

GateVision is a computer vision pipeline that reads identifying information from trucks arriving at a logistics gate — license plates, USDOT numbers, and trailer numbers — then makes an access decision by cross-referencing a carrier allowlist. It replaces the manual walk-up-and-read process used at most facilities today.

The system is designed to run standalone first (no YMS integration required), with a clean REST API layer for future webhook-based integration.

---

## Pipeline Architecture

```
Cameras (RTSP / file / webcam)
         │
         ▼
  ┌─────────────────┐
  │  Frame Ingestion│  ← CameraSource + MotionDetector
  │  (per-camera)   │
  └────────┬────────┘
           │ frames (with motion filter)
           ▼
  ┌─────────────────┐
  │    Detection    │  ← YOLO (ultralytics) multi-class
  │   (per frame)   │     classes: LicensePlate, USDOT,
  └────────┬────────┘     TrailerNum, ContainerNum, ContainerPlate
           │ bounding boxes + class names
           ▼
  ┌─────────────────┐
  │  Recognition    │  ← PaddleOCR 3.x (PP-OCRv5)
  │   (per crop)    │     padded crop → raw text + confidence
  └────────┬────────┘
           │ raw text
           ▼
  ┌─────────────────┐
  │ Post-processing │  ← regex validators per class
  │   (per read)    │     reject impossible outputs
  └────────┬────────┘
           │ validated reads
           ▼
  ┌─────────────────┐
  │ Temporal Fusion │  ← per-truck track window (10s)
  │  (track-level)  │     confidence-weighted majority vote
  └────────┬────────┘
           │ FusionResults (value + status + consensus)
           ▼
  ┌─────────────────┐
  │ Decision Engine │  ← allowlist lookup
  └────────┬────────┘
           │ APPROVED / DENIED / FLAGGED
           ▼
  ┌─────────────────┐
  │    Database     │  ← GateEvent persisted to CSV/SQLite/PostgreSQL
  │  (gate_events)  │
  └─────────────────┘
           │
           ▼
  REST API (FastAPI) + Dashboard (Streamlit)
```

---

## Multi-Camera Strategy

Three cameras cover overlapping fields of view at the gate. This redundancy is intentional:

- **License plate** cameras on the rear/front of the tractor
- **USDOT** camera on the cab door side (always high-contrast, federally required)
- **Container number** on the container body (Smaller, standardized size, and format)
- **Trailer number** on the trailer body (hardest: stenciled, faded fonts, inconsistent layout)

Reads from all cameras flow into the same track. The temporal fusion layer performs a confidence-weighted majority vote across all reads from all cameras, so agreement across cameras strongly boosts confidence.

---

## Data Flow: Single Truck Visit

1. Truck approaches gate. MotionDetectors (one per camera) begin firing.
2. Per frame, YOLO detects bounding boxes for all text classes present.
3. Each bbox is padded, cropped, and fed to PaddleOCR.
4. OCR output passes through a regex validator for the class (e.g. USDOT must be 5–8 digits). Invalid reads are discarded.
5. Valid reads accumulate in a `Track`. The track stays open as long as new reads arrive within the `window_seconds` timeout (default 10s).
6. When the track closes (timeout or end of source), the voter groups reads by class and runs weighted majority vote to pick the winner for each class.
7. The decision engine checks confirmed winners against the allowlist. First match → APPROVED. No match but high confidence → DENIED. Low confidence or ambiguous → FLAGGED.
8. A `GateEvent` is persisted to the database with the decision, reason, and matched values.

---

## Key Choices

### YOLO for Detection, PaddleOCR for Recognition

Running end-to-end OCR on full frames is slow and inaccurate on small text. Instead, YOLO first localizes regions containing text (fast, CPU-capable), then PaddleOCR operates on small, high-resolution crops. This split is both faster and more accurate.

### Bbox Padding Before OCR

Neural OCR fails on tight crops with no context. All crops are padded 20% beyond the bbox. Crops smaller than 150px (in either dimension) get 50% padding — these are typically distant or partially visible signs that need extra context for the recognition model.

### Temporal Fusion Window

Trucks move slowly through gates and are often stopped. The 10-second window (configurable) accumulates many reads from multiple frames and cameras. The confidence-weighted majority vote means a single bad frame doesn't affect the result — the correct text consistently outscores noise.

### Three Decision States

- **APPROVED** — At least one high-confidence read matches the allowlist. Gate opens.
- **DENIED** — High-confidence reads exist but none match. Gate stays closed, event is logged.
- **FLAGGED** — Insufficient confidence or ambiguous reads. Requires human review. Gate stays closed.

FLAGGED intentionally captures the "not sure" case rather than defaulting to DENIED. This is important because OCR on trailer numbers is genuinely hard (faded stencils, non-standard fonts), and false denies have real operational cost.

### SQLite-First, PostgreSQL-Ready

Development and single-server deployments can us CSV or SQLite (zero-config). The `database_url` config supports any SQLAlchemy-compatible URL, so PostgreSQL in production is a one-line change. Docker Compose includes a Postgres service behind a `--profile prod` flag.

---

## Module Map

| Package | Purpose |
|---------|---------|
| `backend/config.py` | Pydantic-settings config, `.env`-driven |
| `backend/db/` | SQLModel ORM (GateEvent, AllowlistEntry), engine/session setup |
| `backend/ingestion/` | Camera sources (RTSP, file, webcam, image folder), motion detection, orchestrator |
| `backend/detection/` | YOLO wrapper (Detection dataclass, Detector class) |
| `backend/recognition/` | PaddleOCR wrapper (OCREngine), regex post-processors |
| `backend/fusion/` | Read + FusionResult models, Track/TrackManager, voter, frame pipeline |
| `backend/decision/` | Decision engine, allowlist CRUD, track-close handler |
| `backend/api/` | FastAPI routes scaffold (in progress) |
| `backend/utils/` | Structlog setup, CSV logger, OpenCV visualization |
| `scripts/` | CLI entry points (run_live, run_pipeline, streamlit dashboard, annotate, test_detect) |
| `tests/` | Unit and integration tests |
