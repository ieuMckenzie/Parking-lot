from sqlmodel import SQLModel, Session, select

from backend.db.models import AllowlistEntry, GateDecision, GateEvent
from backend.db.session import get_engine
from backend.decision.allowlist import add_entry
from backend.decision.handler import handle_track_closed
from backend.fusion.models import FusionStatus, Read
from backend.fusion.tracker import TrackManager


def _read(text, conf, cls="USDOT", cam="cam1", ts=0.0):
    return Read(text=text, raw_text=text, confidence=conf,
                class_name=cls, camera_id=cam, timestamp=ts)


def test_truck_arrives_is_approved(tmp_path):
    """Full lifecycle: truck arrives -> fusion -> decision -> persisted event."""
    engine = get_engine(f"sqlite:///{tmp_path / 'test.db'}")
    SQLModel.metadata.create_all(engine)
    session = Session(engine)

    # Seed allowlist
    add_entry(session, field_type="USDOT", value="1234567", carrier_name="Acme Trucking")

    # Simulate truck arriving
    tm = TrackManager(timeout=5.0)
    tm.update([
        _read("1234567", 0.88, cls="USDOT", cam="cam1", ts=100.0),
        _read("ABC1234", 0.75, cls="LicensePlate", cam="cam1", ts=100.0),
    ], now=100.0)
    tm.update([
        _read("1234567", 0.92, cls="USDOT", cam="cam2", ts=101.0),
        _read("ABC1234", 0.80, cls="LicensePlate", cam="cam2", ts=101.0),
    ], now=101.0)
    tm.update([
        _read("1234567", 0.90, cls="USDOT", cam="cam1", ts=102.0),
        _read("ABC1234", 0.82, cls="LicensePlate", cam="cam3", ts=102.0),
    ], now=102.0)

    # Truck departs
    fusion_results = tm.update([], now=108.0)
    assert fusion_results is not None

    # Decision engine runs
    track = tm.completed[-1][0]
    event = handle_track_closed(
        track_id=track.id,
        results=fusion_results,
        session=session,
    )

    assert event.decision == GateDecision.APPROVED
    assert event.usdot_number == "1234567"
    assert event.license_plate == "ABC1234"
    assert "1234567" in event.decision_reason

    # Verify persisted
    loaded = session.exec(select(GateEvent).where(GateEvent.track_id == track.id)).first()
    assert loaded is not None
    assert loaded.decision == GateDecision.APPROVED
    session.close()


def test_unknown_truck_is_denied(tmp_path):
    """Truck with high-confidence reads but not on allowlist is denied."""
    engine = get_engine(f"sqlite:///{tmp_path / 'test.db'}")
    SQLModel.metadata.create_all(engine)
    session = Session(engine)

    tm = TrackManager(timeout=5.0)
    for i in range(4):
        tm.update([
            _read("9999999", 0.90, cls="USDOT", cam="cam1", ts=100.0 + i),
        ], now=100.0 + i)

    fusion_results = tm.update([], now=110.0)
    assert fusion_results is not None

    track = tm.completed[-1][0]
    event = handle_track_closed(
        track_id=track.id,
        results=fusion_results,
        session=session,
    )

    assert event.decision == GateDecision.DENIED
    session.close()


def test_low_confidence_truck_is_flagged(tmp_path):
    """Truck with too few reads gets flagged for review."""
    engine = get_engine(f"sqlite:///{tmp_path / 'test.db'}")
    SQLModel.metadata.create_all(engine)
    session = Session(engine)

    # Only 1 read — below min_reads threshold -> NEEDS_REVIEW
    tm = TrackManager(timeout=5.0)
    tm.update([
        _read("1234567", 0.5, cls="USDOT", cam="cam1", ts=100.0),
    ], now=100.0)

    fusion_results = tm.update([], now=106.0)
    assert fusion_results is not None
    assert fusion_results[0].status == FusionStatus.NEEDS_REVIEW

    track = tm.completed[-1][0]
    event = handle_track_closed(
        track_id=track.id,
        results=fusion_results,
        session=session,
    )

    assert event.decision == GateDecision.FLAGGED
    assert event.review_status == "pending"
    session.close()
