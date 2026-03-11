def test_decision_public_api():
    from backend.decision import make_decision, handle_track_closed, DecisionResult
    assert callable(make_decision)
    assert callable(handle_track_closed)
    assert DecisionResult is not None


def test_db_public_api():
    from backend.db import init_db, engine
    from backend.db.models import GateEvent, AllowlistEntry, GateDecision
    assert callable(init_db)
    assert GateEvent is not None
    assert AllowlistEntry is not None
    assert GateDecision is not None
