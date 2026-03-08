from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.config import settings
from backend.db import init_db
from backend.utils.logging import get_logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(debug=settings.debug)
    log = get_logger("main")
    log.info("starting", app=settings.app_name, debug=settings.debug)
    init_db(settings.database_url)
    log.info("database_ready", url=settings.database_url)
    yield
    log.info("shutting_down")


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok", "app": settings.app_name}
