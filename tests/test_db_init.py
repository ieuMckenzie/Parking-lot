from backend.db import init_db, engine


def test_init_db_creates_engine(tmp_path):
    """init_db creates the engine and tables."""
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    init_db(db_url)

    from backend.db import engine as eng
    assert eng is not None

    # Verify tables exist
    from sqlalchemy import inspect
    inspector = inspect(eng)
    tables = inspector.get_table_names()
    assert "gate_events" in tables
    assert "allowlist" in tables
