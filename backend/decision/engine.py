from dataclasses import dataclass

from sqlmodel import Session

from backend.db.models import GateDecision
from backend.decision.allowlist import lookup
from backend.fusion.models import FusionResult, FusionStatus


@dataclass
class DecisionResult:
    decision: GateDecision
    reason: str
    matched_field: str | None = None
    matched_value: str | None = None


def make_decision(
    results: list[FusionResult],
    session: Session,
) -> DecisionResult:
    if not results:
        return DecisionResult(
            decision=GateDecision.FLAGGED,
            reason="No fusion results available",
        )

    # Check confirmed results against allowlist first
    confirmed = [r for r in results if r.status == FusionStatus.CONFIRMED]
    for r in confirmed:
        if lookup(session, field_type=r.class_name, value=r.value):
            return DecisionResult(
                decision=GateDecision.APPROVED,
                reason=f"{r.class_name} {r.value} matches allowlist",
                matched_field=r.class_name,
                matched_value=r.value,
            )

    # If all need review, flag regardless
    all_need_review = all(r.status == FusionStatus.NEEDS_REVIEW for r in results)
    if all_need_review:
        return DecisionResult(
            decision=GateDecision.FLAGGED,
            reason="All reads need review — low confidence or insufficient reads",
        )

    # High confidence but no match
    if confirmed:
        values = ", ".join(f"{r.class_name}={r.value}" for r in confirmed)
        return DecisionResult(
            decision=GateDecision.DENIED,
            reason=f"No allowlist match for confirmed reads: {values}",
        )

    # Remaining: some need review, none confirmed matched
    return DecisionResult(
        decision=GateDecision.FLAGGED,
        reason="Some reads need review, no confirmed allowlist match",
    )
