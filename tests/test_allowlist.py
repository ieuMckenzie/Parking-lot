from sqlmodel import SQLModel, Session

from backend.db.models import AllowlistEntry
from backend.db.session import get_engine
from backend.decision.allowlist import (
    add_entry,
    remove_entry,
    list_entries,
    lookup,
)


def _session(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path / 'test.db'}")
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def test_add_and_list(tmp_path):
    session = _session(tmp_path)
    add_entry(session, field_type="USDOT", value="1234567", carrier_name="Acme")
    add_entry(session, field_type="LicensePlate", value="ABC1234")
    entries = list_entries(session)
    assert len(entries) == 2
    session.close()


def test_remove_entry(tmp_path):
    session = _session(tmp_path)
    entry = add_entry(session, field_type="USDOT", value="9999999")
    assert remove_entry(session, entry.id) is True
    assert len(list_entries(session)) == 0
    session.close()


def test_remove_nonexistent(tmp_path):
    session = _session(tmp_path)
    assert remove_entry(session, 999) is False
    session.close()


def test_lookup_found(tmp_path):
    session = _session(tmp_path)
    add_entry(session, field_type="USDOT", value="1234567")
    assert lookup(session, field_type="USDOT", value="1234567") is True
    session.close()


def test_lookup_not_found(tmp_path):
    session = _session(tmp_path)
    add_entry(session, field_type="USDOT", value="1234567")
    assert lookup(session, field_type="USDOT", value="9999999") is False
    session.close()


def test_lookup_inactive_not_found(tmp_path):
    session = _session(tmp_path)
    entry = add_entry(session, field_type="USDOT", value="1234567")
    entry.active = False
    session.add(entry)
    session.commit()
    assert lookup(session, field_type="USDOT", value="1234567") is False
    session.close()


def test_list_filters_by_field_type(tmp_path):
    session = _session(tmp_path)
    add_entry(session, field_type="USDOT", value="1234567")
    add_entry(session, field_type="LicensePlate", value="ABC1234")
    assert len(list_entries(session, field_type="USDOT")) == 1
    assert len(list_entries(session, field_type="LicensePlate")) == 1
    session.close()
