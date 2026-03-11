# tests/test_fusion_init.py
def test_fusion_public_api():
    from backend.fusion import (
        Read, FusionResult, FusionStatus,
        Track, TrackManager,
        vote, process_frame,
    )
    assert Read is not None
    assert FusionResult is not None
    assert FusionStatus is not None
    assert Track is not None
    assert TrackManager is not None
    assert vote is not None
    assert process_frame is not None
