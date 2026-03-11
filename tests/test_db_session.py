from sqlmodel import SQLModel, Session, select, Field

from backend.db.session import get_engine, get_session


class _Dummy(SQLModel, table=True):
    __tablename__ = "dummy"
    id: int | None = Field(default=None, primary_key=True)
    name: str


def test_engine_creates_tables_and_session_works(tmp_path):
    """Engine can create tables and session can read/write."""
    db_path = tmp_path / "test.db"
    engine = get_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)

    with get_session(engine) as session:
        session.add(_Dummy(name="hello"))
        session.commit()

    with get_session(engine) as session:
        result = session.exec(select(_Dummy)).first()
        assert result is not None
        assert result.name == "hello"
