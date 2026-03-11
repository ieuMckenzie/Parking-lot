from sqlmodel import Session, select

from backend.db.models import AllowlistEntry


def add_entry(
    session: Session,
    *,
    field_type: str,
    value: str,
    carrier_name: str | None = None,
) -> AllowlistEntry:
    entry = AllowlistEntry(field_type=field_type, value=value, carrier_name=carrier_name)
    session.add(entry)
    session.commit()
    session.refresh(entry)
    return entry


def remove_entry(session: Session, entry_id: int) -> bool:
    entry = session.get(AllowlistEntry, entry_id)
    if entry is None:
        return False
    session.delete(entry)
    session.commit()
    return True


def list_entries(
    session: Session,
    *,
    field_type: str | None = None,
) -> list[AllowlistEntry]:
    stmt = select(AllowlistEntry).where(AllowlistEntry.active == True)  # noqa: E712
    if field_type is not None:
        stmt = stmt.where(AllowlistEntry.field_type == field_type)
    return list(session.exec(stmt).all())


def lookup(session: Session, *, field_type: str, value: str) -> bool:
    stmt = (
        select(AllowlistEntry)
        .where(AllowlistEntry.field_type == field_type)
        .where(AllowlistEntry.value == value)
        .where(AllowlistEntry.active == True)  # noqa: E712
    )
    return session.exec(stmt).first() is not None
