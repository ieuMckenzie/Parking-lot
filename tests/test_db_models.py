from sqlmodel import SQLModel, Session, select

from backend.db.models import AllowlistEntry, GateDecision, GateEvent
from backend.db.session import get_engine


def _engine(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path / 'test.db'}")
    SQLModel.metadata.create_all(engine)
    return engine


def test_gate_event_roundtrip(tmp_path):
    engine = _engine(tmp_path)
    event = GateEvent(
        track_id="abc123",
        license_plate="ABC1234",
        license_plate_confidence=2.35,
        usdot_number="1234567",
        usdot_confidence=3.55,
        trailer_number=None,
        trailer_confidence=None,
        decision=GateDecision.APPROVED,
        decision_reason="USDOT 1234567 matches allowlist",
        review_status="resolved",
    )
    with Session(engine) as session:
        session.add(event)
        session.commit()
        session.refresh(event)
        assert event.id is not None
        assert event.created_at is not None

    with Session(engine) as session:
        loaded = session.exec(select(GateEvent)).first()
        assert loaded.track_id == "abc123"
        assert loaded.license_plate == "ABC1234"
        assert loaded.decision == GateDecision.APPROVED


def test_allowlist_entry_roundtrip(tmp_path):
    engine = _engine(tmp_path)
    entry = AllowlistEntry(
        field_type="USDOT",
        value="1234567",
        carrier_name="Test Carrier",
    )
    with Session(engine) as session:
        session.add(entry)
        session.commit()
        session.refresh(entry)
        assert entry.id is not None
        assert entry.active is True
        assert entry.created_at is not None

    with Session(engine) as session:
        loaded = session.exec(select(AllowlistEntry)).first()
        assert loaded.value == "1234567"
        assert loaded.field_type == "USDOT"


def test_gate_decision_enum_values():
    assert GateDecision.APPROVED == "APPROVED"
    assert GateDecision.DENIED == "DENIED"
    assert GateDecision.FLAGGED == "FLAGGED"
