from sqlalchemy import Engine
from sqlmodel import SQLModel

from backend.db.session import get_engine

engine: Engine | None = None


def init_db(database_url: str) -> None:
    global engine
    import backend.db.models  # noqa: F401 — register models with SQLModel metadata
    engine = get_engine(database_url)
    SQLModel.metadata.create_all(engine)
