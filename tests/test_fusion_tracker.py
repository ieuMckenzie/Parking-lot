from backend.fusion.models import Read, FusionStatus
from backend.fusion.tracker import Track, TrackManager


def _read(text: str, confidence: float, class_name: str = "USDOT",
          camera_id: str = "cam1", timestamp: float = 100.0) -> Read:
    return Read(text=text, raw_text=text, confidence=confidence,
                class_name=class_name, camera_id=camera_id, timestamp=timestamp)


class TestTrack:
    def test_new_track_empty(self):
        t = Track(created_at=100.0, last_seen=100.0)
        assert t.reads == []
        assert not t.closed
        assert t.id  # has an id

    def test_add_read(self):
        t = Track(created_at=100.0, last_seen=100.0)
        r = _read("1234567", 0.9, timestamp=101.0)
        t.add_read(r)
        assert len(t.reads) == 1
        assert t.last_seen == 101.0

    def test_add_reads_batch(self):
        t = Track(created_at=100.0, last_seen=100.0)
        reads = [
            _read("1234567", 0.9, timestamp=101.0),
            _read("ABC1234", 0.8, class_name="LicensePlate", timestamp=101.0),
        ]
        t.add_reads(reads)
        assert len(t.reads) == 2
        assert t.last_seen == 101.0

    def test_last_seen_updates_to_max(self):
        t = Track(created_at=100.0, last_seen=100.0)
        t.add_read(_read("A", 0.9, timestamp=105.0))
        t.add_read(_read("B", 0.9, timestamp=103.0))
        assert t.last_seen == 105.0

    def test_is_expired_false(self):
        t = Track(created_at=100.0, last_seen=105.0)
        assert not t.is_expired(timeout=10.0, now=110.0)

    def test_is_expired_true(self):
        t = Track(created_at=100.0, last_seen=105.0)
        assert t.is_expired(timeout=10.0, now=115.0)

    def test_is_expired_exact_boundary(self):
        t = Track(created_at=100.0, last_seen=105.0)
        assert t.is_expired(timeout=10.0, now=115.0)
        assert not t.is_expired(timeout=10.0, now=114.9)

    def test_fuse_delegates_to_voter(self):
        t = Track(created_at=100.0, last_seen=100.0)
        for i in range(4):
            t.add_read(_read("1234567", 0.9, timestamp=100.0 + i))
        results = t.fuse(min_reads=3, min_confidence=0.6)
        assert len(results) == 1
        assert results[0].value == "1234567"
        assert results[0].status == FusionStatus.CONFIRMED


class TestTrackManager:
    def test_no_reads_no_track(self):
        tm = TrackManager(timeout=10.0)
        result = tm.update([], now=100.0)
        assert result is None
        assert tm.active_track is None

    def test_first_reads_create_track(self):
        tm = TrackManager(timeout=10.0)
        reads = [_read("1234567", 0.9, timestamp=100.0)]
        result = tm.update(reads, now=100.0)
        assert result is None  # no track closed yet
        assert tm.active_track is not None
        assert len(tm.active_track.reads) == 1

    def test_subsequent_reads_accumulate(self):
        tm = TrackManager(timeout=10.0)
        tm.update([_read("1234567", 0.9, timestamp=100.0)], now=100.0)
        tm.update([_read("1234567", 0.85, timestamp=102.0)], now=102.0)
        assert len(tm.active_track.reads) == 2

    def test_expired_track_closes_on_update(self):
        tm = TrackManager(timeout=10.0)
        tm.update([_read("1234567", 0.9, timestamp=100.0)], now=100.0)
        tm.update([_read("1234567", 0.85, timestamp=101.0)], now=101.0)
        tm.update([_read("1234567", 0.80, timestamp=102.0)], now=102.0)
        # No reads for 10+ seconds
        result = tm.update([], now=112.0)
        assert result is not None
        assert len(result) == 1
        assert result[0].value == "1234567"
        assert tm.active_track is None
        assert len(tm.completed) == 1

    def test_new_reads_after_closure_create_new_track(self):
        tm = TrackManager(timeout=10.0)
        tm.update([_read("1234567", 0.9, timestamp=100.0)], now=100.0)
        tm.update([_read("1234567", 0.85, timestamp=101.0)], now=101.0)
        tm.update([_read("1234567", 0.80, timestamp=102.0)], now=102.0)
        # Close old track + start new one
        result = tm.update([_read("7654321", 0.9, timestamp=113.0)], now=113.0)
        assert result is not None  # old track closed
        assert tm.active_track is not None
        assert tm.active_track.reads[0].text == "7654321"
        assert len(tm.completed) == 1

    def test_flush_closes_active_track(self):
        tm = TrackManager(timeout=10.0)
        tm.update([_read("1234567", 0.9, timestamp=100.0)], now=100.0)
        tm.update([_read("1234567", 0.85, timestamp=101.0)], now=101.0)
        tm.update([_read("1234567", 0.80, timestamp=102.0)], now=102.0)
        result = tm.flush()
        assert result is not None
        assert result[0].value == "1234567"
        assert tm.active_track is None

    def test_flush_empty_returns_none(self):
        tm = TrackManager(timeout=10.0)
        assert tm.flush() is None

    def test_completed_tracks_accumulate(self):
        tm = TrackManager(timeout=10.0)
        # Track 1
        tm.update([_read("1234567", 0.9, timestamp=100.0)], now=100.0)
        tm.flush()
        # Track 2
        tm.update([_read("7654321", 0.9, timestamp=200.0)], now=200.0)
        tm.flush()
        assert len(tm.completed) == 2
