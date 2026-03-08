from sqlmodel import Session

from backend.db.models import GateDecision, GateEvent
from backend.decision.engine import make_decision
from backend.fusion.models import FusionResult
from backend.utils.logging import get_logger

log = get_logger("decision")

# Map fusion class names to GateEvent field names
_FIELD_MAP = {
    "USDOT": ("usdot_number", "usdot_confidence"),
    "LicensePlate": ("license_plate", "license_plate_confidence"),
    "TrailerNum": ("trailer_number", "trailer_confidence"),
}


def handle_track_closed(
    *,
    track_id: str,
    results: list[FusionResult],
    session: Session,
) -> GateEvent:
    decision_result = make_decision(results, session)

    event_data: dict = {
        "track_id": track_id,
        "decision": decision_result.decision,
        "decision_reason": decision_result.reason,
        "review_status": "pending" if decision_result.decision == GateDecision.FLAGGED else "resolved",
    }

    for r in results:
        if r.class_name in _FIELD_MAP:
            value_field, conf_field = _FIELD_MAP[r.class_name]
            event_data[value_field] = r.value
            event_data[conf_field] = r.confidence

    event = GateEvent(**event_data)
    session.add(event)
    session.commit()
    session.refresh(event)

    log.info(
        "gate_event_created",
        event_id=event.id,
        track_id=track_id,
        decision=decision_result.decision.value,
        reason=decision_result.reason,
    )
    return event
