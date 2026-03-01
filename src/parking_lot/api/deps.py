"""Dependency injection for FastAPI routes."""

from parking_lot.engine.scanner import ScannerEngine

_engine: ScannerEngine | None = None


def set_engine(engine: ScannerEngine):
    global _engine
    _engine = engine


def get_engine() -> ScannerEngine:
    if _engine is None:
        raise RuntimeError("ScannerEngine not initialized")
    return _engine
