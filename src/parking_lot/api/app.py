"""FastAPI application factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from parking_lot.api.deps import set_engine
from parking_lot.api.routers import cameras, config, detections, health
from parking_lot.engine.scanner import ScannerEngine


def create_app(engine: ScannerEngine) -> FastAPI:
    """Create a FastAPI app wired to the given ScannerEngine."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        set_engine(engine)
        yield
        engine.stop()

    app = FastAPI(
        title="Parking Lot Scanner API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = "/api/v1"
    app.include_router(health.router, prefix=prefix)
    app.include_router(cameras.router, prefix=prefix)
    app.include_router(detections.router, prefix=prefix)
    app.include_router(config.router, prefix=prefix)

    return app
