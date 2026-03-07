import uuid
from dataclasses import dataclass, field

from backend.fusion.models import FusionResult, Read
from backend.fusion.voter import vote
from backend.utils.logging import get_logger

log = get_logger("fusion")


@dataclass
class Track:
    """Accumulates OCR reads for a single truck gate event."""
    created_at: float
    last_seen: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    reads: list[Read] = field(default_factory=list)
    closed: bool = False

    def add_read(self, read: Read) -> None:
        self.reads.append(read)
        self.last_seen = max(self.last_seen, read.timestamp)

    def add_reads(self, reads: list[Read]) -> None:
        for r in reads:
            self.add_read(r)

    def is_expired(self, timeout: float, now: float) -> bool:
        return (now - self.last_seen) >= timeout

    def fuse(self, **kwargs) -> list[FusionResult]:
        """Run voter on accumulated reads."""
        return vote(self.reads, **kwargs)


class TrackManager:
    """Manages track lifecycle: create, accumulate, close on timeout."""

    def __init__(self, timeout: float | None = None):
        from backend.config import settings
        self.timeout = timeout if timeout is not None else settings.fusion.window_seconds
        self.active_track: Track | None = None
        self.completed: list[tuple[Track, list[FusionResult]]] = []

    def update(self, reads: list[Read], now: float) -> list[FusionResult] | None:
        """Process new reads. Returns FusionResult list if a track just closed."""
        closed_result = None

        # Check if active track expired
        if self.active_track and self.active_track.is_expired(self.timeout, now):
            closed_result = self._close_track()

        # If we have reads, add to track (or create new one)
        if reads:
            if self.active_track is None:
                self.active_track = Track(created_at=now, last_seen=now)
                log.info("track_created", track_id=self.active_track.id)
            self.active_track.add_reads(reads)

        return closed_result

    def flush(self) -> list[FusionResult] | None:
        """Force-close any active track."""
        if self.active_track:
            return self._close_track()
        return None

    def _close_track(self) -> list[FusionResult]:
        track = self.active_track
        results = track.fuse()
        track.closed = True
        self.completed.append((track, results))
        log.info("track_closed", track_id=track.id,
                 num_reads=len(track.reads), num_results=len(results))
        self.active_track = None
        return results
