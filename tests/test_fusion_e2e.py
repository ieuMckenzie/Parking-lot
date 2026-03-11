from backend.fusion.models import FusionStatus, Read
from backend.fusion.tracker import TrackManager


def _read(text, conf, cls="USDOT", cam="cam1", ts=0.0):
    return Read(text=text, raw_text=text, confidence=conf,
                class_name=cls, camera_id=cam, timestamp=ts)


def test_full_truck_event_lifecycle():
    """Simulate a truck arriving, being read across frames/cameras, then departing."""
    tm = TrackManager(timeout=10.0)

    # Frame 1 (t=100): truck arrives, cam1 reads plate and USDOT
    tm.update([
        _read("1234567", 0.88, cls="USDOT", cam="cam1", ts=100.0),
        _read("ABC1234", 0.75, cls="LicensePlate", cam="cam1", ts=100.0),
    ], now=100.0)
    assert tm.active_track is not None

    # Frame 2 (t=101): cam2 also reads
    tm.update([
        _read("1234567", 0.92, cls="USDOT", cam="cam2", ts=101.0),
        _read("ABC1234", 0.80, cls="LicensePlate", cam="cam2", ts=101.0),
    ], now=101.0)

    # Frame 3 (t=102): cam1 reads again, slightly different plate OCR
    tm.update([
        _read("1234567", 0.90, cls="USDOT", cam="cam1", ts=102.0),
        _read("ABG1234", 0.60, cls="LicensePlate", cam="cam1", ts=102.0),
    ], now=102.0)

    # Frame 4 (t=103): cam3 confirms
    tm.update([
        _read("1234567", 0.85, cls="USDOT", cam="cam3", ts=103.0),
        _read("ABC1234", 0.82, cls="LicensePlate", cam="cam3", ts=103.0),
    ], now=103.0)

    # Truck departs, no reads for 10+ seconds
    result = tm.update([], now=113.0)
    assert result is not None
    assert len(result) == 2

    by_class = {r.class_name: r for r in result}

    usdot = by_class["USDOT"]
    assert usdot.value == "1234567"
    assert usdot.num_reads == 4
    assert usdot.status == FusionStatus.CONFIRMED

    plate = by_class["LicensePlate"]
    assert plate.value == "ABC1234"
    assert plate.num_reads == 3  # 3 reads of ABC1234 vs 1 of ABG1234
    assert plate.status == FusionStatus.CONFIRMED

    # Track completed
    assert tm.active_track is None
    assert len(tm.completed) == 1


def test_two_consecutive_trucks():
    """Two trucks arriving sequentially produce separate tracks."""
    tm = TrackManager(timeout=5.0)

    # Truck 1
    for i in range(3):
        tm.update([_read("1111111", 0.9, ts=100.0 + i)], now=100.0 + i)

    # Gap > 5s, then truck 2
    result1 = tm.update([_read("2222222", 0.9, ts=110.0)], now=110.0)
    assert result1 is not None
    assert result1[0].value == "1111111"

    for i in range(2):
        tm.update([_read("2222222", 0.9, ts=111.0 + i)], now=111.0 + i)

    result2 = tm.flush()
    assert result2 is not None
    assert result2[0].value == "2222222"
    assert len(tm.completed) == 2
