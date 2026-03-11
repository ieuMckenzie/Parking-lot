from sqlmodel import SQLModel, Session

from backend.db.models import AllowlistEntry, GateDecision
from backend.db.session import get_engine
from backend.decision.allowlist import add_entry
from backend.decision.engine import make_decision
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


def test_approved_when_usdot_on_allowlist(tmp_path):
    session = _session(tmp_path)
    add_entry(session, field_type="USDOT", value="1234567")
    results = [
        _result("USDOT", "1234567"),
        _result("LicensePlate", "ABC1234"),
    ]
    decision = make_decision(results, session)
    assert decision.decision == GateDecision.APPROVED
    assert "1234567" in decision.reason


def test_approved_when_plate_on_allowlist(tmp_path):
    session = _session(tmp_path)
    add_entry(session, field_type="LicensePlate", value="ABC1234")
    results = [_result("LicensePlate", "ABC1234")]
    decision = make_decision(results, session)
    assert decision.decision == GateDecision.APPROVED


def test_denied_when_no_match(tmp_path):
    session = _session(tmp_path)
    add_entry(session, field_type="USDOT", value="9999999")
    results = [_result("USDOT", "1234567")]
    decision = make_decision(results, session)
    assert decision.decision == GateDecision.DENIED


def test_flagged_when_needs_review(tmp_path):
    session = _session(tmp_path)
    add_entry(session, field_type="USDOT", value="1234567")
    results = [_result("USDOT", "1234567", status=FusionStatus.NEEDS_REVIEW)]
    decision = make_decision(results, session)
    assert decision.decision == GateDecision.FLAGGED


def test_flagged_when_all_need_review_and_no_match(tmp_path):
    session = _session(tmp_path)
    results = [_result("USDOT", "1234567", status=FusionStatus.NEEDS_REVIEW)]
    decision = make_decision(results, session)
    assert decision.decision == GateDecision.FLAGGED


def test_empty_results_flagged(tmp_path):
    session = _session(tmp_path)
    decision = make_decision([], session)
    assert decision.decision == GateDecision.FLAGGED


def test_approved_any_field_match(tmp_path):
    """If ANY identified number matches allowlist, approve."""
    session = _session(tmp_path)
    add_entry(session, field_type="TrailerNum", value="TRL999")
    results = [
        _result("USDOT", "1234567"),
        _result("LicensePlate", "ABC1234"),
        _result("TrailerNum", "TRL999"),
    ]
    decision = make_decision(results, session)
    assert decision.decision == GateDecision.APPROVED
    assert "TRL999" in decision.reason
