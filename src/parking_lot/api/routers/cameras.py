"""Camera status and snapshot endpoints."""

import cv2
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import io

from parking_lot.api.deps import get_engine
from parking_lot.api.schemas import CameraListResponse, CameraStatus
from parking_lot.engine.scanner import ScannerEngine

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("", response_model=CameraListResponse)
def list_cameras(engine: ScannerEngine = Depends(get_engine)):
    statuses = engine.get_camera_status()
    return CameraListResponse(
        cameras=[CameraStatus(**s) for s in statuses]
    )


@router.get("/{camera_id}/snapshot")
def snapshot(camera_id: str, engine: ScannerEngine = Depends(get_engine)):
    frame = engine.get_snapshot(camera_id)
    if frame is None:
        raise HTTPException(status_code=404, detail=f"Camera '{camera_id}' not found or no frame available")

    _, buf = cv2.imencode(".jpg", frame)
    return StreamingResponse(
        io.BytesIO(buf.tobytes()),
        media_type="image/jpeg",
        headers={"Content-Disposition": f"inline; filename={camera_id}.jpg"},
    )
