from sqlmodel import SQLModel, Session, select

from backend.db.models import GateDecision, GateEvent
from backend.db.session import get_engine
from backend.decision.allowlist import add_entry
from backend.decision.handler import handle_track_closed
from backend.fusion.models import FusionResult, FusionStatus


def _session(tmp_path):
    engine = get_engine(f"sqlite:///{tmp_path / 'test.db'}")
    SQLModel.metadata.create_all(engine)
    return Session(engine)


def _result(cls, value, status=FusionStatus.CONFIRMED, confidence=3.0, reads=4):
    return FusionResult(
        class_name=cls,
        value=value,
        confidence=confidence,
        num_reads=reads,
        consensus_ratio=1.0,
        status=status,
    )


def test_handle_creates_approved_event(tmp_path):
    session = _session(tmp_path)
    add_entry(session, field_type="USDOT", value="1234567")
    results = [
        _result("USDOT", "1234567"),
        _result("LicensePlate", "ABC1234"),
    ]
    event = handle_track_closed(
        track_id="trk001",
        results=results,
        session=session,
    )
    assert event.id is not None
    assert event.decision == GateDecision.APPROVED
    assert event.usdot_number == "1234567"
    assert event.license_plate == "ABC1234"
    assert event.track_id == "trk001"


def test_handle_creates_denied_event(tmp_path):
    session = _session(tmp_path)
    results = [_result("USDOT", "9999999")]
    event = handle_track_closed(
        track_id="trk002",
        results=results,
        session=session,
    )
    assert event.decision == GateDecision.DENIED
    assert event.usdot_number == "9999999"


def test_handle_creates_flagged_event(tmp_path):
    session = _session(tmp_path)
    results = [_result("USDOT", "1234567", status=FusionStatus.NEEDS_REVIEW)]
    event = handle_track_closed(
        track_id="trk003",
        results=results,
        session=session,
    )
    assert event.decision == GateDecision.FLAGGED
    assert event.review_status == "pending"


def test_handle_persists_to_db(tmp_path):
    session = _session(tmp_path)
    results = [_result("USDOT", "1234567")]
    handle_track_closed(track_id="trk004", results=results, session=session)

    loaded = session.exec(select(GateEvent)).first()
    assert loaded is not None
    assert loaded.track_id == "trk004"


def test_handle_extracts_all_field_types(tmp_path):
    session = _session(tmp_path)
    results = [
        _result("USDOT", "1234567"),
        _result("LicensePlate", "ABC1234"),
        _result("TrailerNum", "TRL999"),
    ]
    event = handle_track_closed(track_id="trk005", results=results, session=session)
    assert event.usdot_number == "1234567"
    assert event.license_plate == "ABC1234"
    assert event.trailer_number == "TRL999"
