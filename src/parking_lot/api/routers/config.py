"""Configuration and authorized plates endpoints."""

import os

from fastapi import APIRouter, Depends, HTTPException

from parking_lot.api.deps import get_engine
from parking_lot.api.schemas import (
    AuthorizedListResponse,
    AuthorizedPlateRequest,
    ConfigResponse,
    MessageResponse,
    ThresholdsUpdate,
)
from parking_lot.engine.scanner import ScannerEngine

router = APIRouter(prefix="/config", tags=["config"])


@router.get("", response_model=ConfigResponse)
def get_config(engine: ScannerEngine = Depends(get_engine)):
    cfg = engine.cfg
    return ConfigResponse(
        min_thresh=cfg.min_thresh,
        ocr_thresh=cfg.ocr.thresh,
        similarity_thresh=cfg.detection.similarity_thresh,
        log_cooldown=cfg.detection.log_cooldown,
        detection_hold_time=cfg.detection.detection_hold_time,
        sr_enabled=cfg.sr.enabled,
        sr_target_width=cfg.sr.target_width,
        feed_enhance=cfg.camera.feed_enhance,
        num_ocr_workers=cfg.num_ocr_workers,
    )


@router.put("/thresholds", response_model=MessageResponse)
def update_thresholds(update: ThresholdsUpdate, engine: ScannerEngine = Depends(get_engine)):
    if update.min_thresh is not None:
        engine.cfg.min_thresh = update.min_thresh
    if update.ocr_thresh is not None:
        engine.cfg.ocr.thresh = update.ocr_thresh
    if update.similarity_thresh is not None:
        engine.cfg.detection.similarity_thresh = update.similarity_thresh
    if update.log_cooldown is not None:
        engine.cfg.detection.log_cooldown = update.log_cooldown
    return MessageResponse(message="Thresholds updated")


@router.get("/authorized", response_model=AuthorizedListResponse)
def list_authorized(engine: ScannerEngine = Depends(get_engine)):
    engine.validator.refresh_authorized()
    return AuthorizedListResponse(plates=sorted(engine.validator.authorized_plates))


@router.post("/authorized", response_model=MessageResponse)
def add_authorized(req: AuthorizedPlateRequest, engine: ScannerEngine = Depends(get_engine)):
    plate = req.plate.strip().upper()
    if not plate:
        raise HTTPException(status_code=400, detail="Plate cannot be empty")

    auth_file = engine.cfg.validation.authorized_file
    os.makedirs(os.path.dirname(auth_file), exist_ok=True)

    existing = set()
    if os.path.exists(auth_file):
        with open(auth_file, "r") as f:
            existing = {line.strip().upper() for line in f if line.strip()}

    if plate in existing:
        return MessageResponse(message=f"Plate '{plate}' already authorized")

    with open(auth_file, "a") as f:
        f.write(f"{plate}\n")

    engine.validator.authorized_plates.add(plate)
    return MessageResponse(message=f"Plate '{plate}' added")


@router.delete("/authorized/{plate}", response_model=MessageResponse)
def remove_authorized(plate: str, engine: ScannerEngine = Depends(get_engine)):
    plate = plate.strip().upper()
    auth_file = engine.cfg.validation.authorized_file

    if not os.path.exists(auth_file):
        raise HTTPException(status_code=404, detail="No authorized plates file")

    with open(auth_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if plate not in [l.upper() for l in lines]:
        raise HTTPException(status_code=404, detail=f"Plate '{plate}' not found")

    with open(auth_file, "w") as f:
        for line in lines:
            if line.upper() != plate:
                f.write(f"{line}\n")

    engine.validator.authorized_plates.discard(plate)
    return MessageResponse(message=f"Plate '{plate}' removed")
