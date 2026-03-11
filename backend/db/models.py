import datetime
from enum import Enum

from sqlmodel import Field, SQLModel


class GateDecision(str, Enum):
    APPROVED = "APPROVED"
    DENIED = "DENIED"
    FLAGGED = "FLAGGED"


class GateEvent(SQLModel, table=True):
    __tablename__ = "gate_events"

    id: int | None = Field(default=None, primary_key=True)
    track_id: str = Field(index=True)
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC),
    )
    license_plate: str | None = None
    license_plate_confidence: float | None = None
    usdot_number: str | None = None
    usdot_confidence: float | None = None
    trailer_number: str | None = None
    trailer_confidence: float | None = None
    decision: GateDecision
    decision_reason: str = ""
    review_status: str = Field(default="pending")


class AllowlistEntry(SQLModel, table=True):
    __tablename__ = "allowlist"

    id: int | None = Field(default=None, primary_key=True)
    field_type: str = Field(index=True)
    value: str = Field(index=True)
    carrier_name: str | None = None
    active: bool = Field(default=True)
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC),
    )
