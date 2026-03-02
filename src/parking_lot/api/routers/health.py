"""Health check endpoints."""

from fastapi import APIRouter, Depends

from parking_lot.api.deps import get_engine
from parking_lot.api.schemas import HealthResponse
from parking_lot.engine.scanner import ScannerEngine

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(engine: ScannerEngine = Depends(get_engine)):
    q = engine.get_queue_sizes()
    return HealthResponse(
        status="running",
        uptime_seconds=round(engine.get_uptime(), 1),
        yolo_queue_size=q["yolo_queue"],
        ocr_queue_size=q["ocr_queue"],
        num_cameras=len(engine.cameras),
        sse_subscribers=engine.events.subscriber_count,
    )
