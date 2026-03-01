"""Detection and log endpoints."""

import os

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from parking_lot.api.deps import get_engine
from parking_lot.api.schemas import Detection, DetectionsResponse, LogEntry, LogResponse
from parking_lot.engine.scanner import ScannerEngine

router = APIRouter(prefix="/detections", tags=["detections"])


@router.get("", response_model=DetectionsResponse)
def current_detections(engine: ScannerEngine = Depends(get_engine)):
    raw = engine.state.get_all_detections()
    detections = {}
    for cam_id, dets in raw.items():
        detections[cam_id] = [Detection(**d) for d in dets]
    return DetectionsResponse(detections=detections)


@router.get("/log", response_model=LogResponse)
def detection_log(limit: int = Query(100, ge=1, le=10000), engine: ScannerEngine = Depends(get_engine)):
    entries = engine.logger.read_recent(limit)
    return LogResponse(
        entries=[LogEntry(**e) for e in entries],
        total=len(entries),
    )


@router.get("/log/export")
def export_log(engine: ScannerEngine = Depends(get_engine)):
    path = engine.logger.get_file_path()
    if not os.path.exists(path):
        return {"error": "No log file found"}
    return FileResponse(path, media_type="text/csv", filename="truck_logistics.csv")
