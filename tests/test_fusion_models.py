from backend.fusion.models import Read, FusionResult, FusionStatus


def test_read_creation():
    r = Read(
        text="1234567",
        raw_text="l234567",
        confidence=0.92,
        class_name="USDOT",
        camera_id="cam1",
        timestamp=100.0,
    )
    assert r.text == "1234567"
    assert r.raw_text == "l234567"
    assert r.confidence == 0.92
    assert r.class_name == "USDOT"
    assert r.camera_id == "cam1"
    assert r.timestamp == 100.0


def test_fusion_result_confirmed():
    fr = FusionResult(
        class_name="USDOT",
        value="1234567",
        confidence=2.76,
        num_reads=3,
        consensus_ratio=1.0,
        status=FusionStatus.CONFIRMED,
    )
    assert fr.status == FusionStatus.CONFIRMED
    assert fr.value == "1234567"


def test_fusion_result_needs_review():
    fr = FusionResult(
        class_name="LicensePlate",
        value="ABC1234",
        confidence=0.45,
        num_reads=1,
        consensus_ratio=0.5,
        status=FusionStatus.NEEDS_REVIEW,
    )
    assert fr.status == FusionStatus.NEEDS_REVIEW


def test_fusion_status_values():
    assert FusionStatus.CONFIRMED == "CONFIRMED"
    assert FusionStatus.NEEDS_REVIEW == "NEEDS_REVIEW"
