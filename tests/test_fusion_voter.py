from backend.fusion.models import Read, FusionStatus
from backend.fusion.voter import vote


def _read(text: str, confidence: float, class_name: str = "USDOT",
          camera_id: str = "cam1", timestamp: float = 0.0) -> Read:
    return Read(text=text, raw_text=text, confidence=confidence,
                class_name=class_name, camera_id=camera_id, timestamp=timestamp)


def test_vote_empty():
    assert vote([]) == []


def test_vote_single_class_clear_winner():
    reads = [
        _read("1234567", 0.95),
        _read("1234567", 0.90),
        _read("1234567", 0.88),
        _read("1234568", 0.40),
    ]
    results = vote(reads, min_reads=3, min_confidence=0.6)
    assert len(results) == 1
    r = results[0]
    assert r.class_name == "USDOT"
    assert r.value == "1234567"
    assert r.num_reads == 3
    assert r.confidence == 0.95 + 0.90 + 0.88
    assert r.consensus_ratio == 3 / 4
    assert r.status == FusionStatus.CONFIRMED


def test_vote_insufficient_reads():
    reads = [
        _read("1234567", 0.95),
        _read("1234568", 0.90),
    ]
    results = vote(reads, min_reads=3, min_confidence=0.6)
    assert len(results) == 1
    assert results[0].status == FusionStatus.NEEDS_REVIEW


def test_vote_low_total_confidence():
    reads = [
        _read("1234567", 0.15),
        _read("1234567", 0.15),
        _read("1234567", 0.15),
    ]
    results = vote(reads, min_reads=3, min_confidence=0.6)
    assert results[0].status == FusionStatus.NEEDS_REVIEW
    assert abs(results[0].confidence - 0.45) < 1e-9


def test_vote_multiple_classes():
    reads = [
        _read("1234567", 0.90, class_name="USDOT"),
        _read("1234567", 0.85, class_name="USDOT"),
        _read("1234567", 0.80, class_name="USDOT"),
        _read("ABC1234", 0.88, class_name="LicensePlate"),
        _read("ABC1234", 0.82, class_name="LicensePlate"),
        _read("ABC1234", 0.79, class_name="LicensePlate"),
    ]
    results = vote(reads, min_reads=3, min_confidence=0.6)
    assert len(results) == 2
    by_class = {r.class_name: r for r in results}
    assert by_class["USDOT"].value == "1234567"
    assert by_class["USDOT"].status == FusionStatus.CONFIRMED
    assert by_class["LicensePlate"].value == "ABC1234"
    assert by_class["LicensePlate"].status == FusionStatus.CONFIRMED


def test_vote_multi_camera():
    reads = [
        _read("1234567", 0.90, camera_id="cam1"),
        _read("1234567", 0.85, camera_id="cam2"),
        _read("1234567", 0.80, camera_id="cam3"),
    ]
    results = vote(reads, min_reads=3, min_confidence=0.6)
    assert results[0].status == FusionStatus.CONFIRMED


def test_vote_noisy_reads_correct_winner():
    reads = [
        _read("1234567", 0.92),
        _read("1234567", 0.88),
        _read("1234567", 0.85),
        _read("1234568", 0.70),
        _read("1234569", 0.30),
    ]
    results = vote(reads, min_reads=3, min_confidence=0.6)
    assert results[0].value == "1234567"
    assert results[0].status == FusionStatus.CONFIRMED
