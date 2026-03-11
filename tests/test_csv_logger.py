import csv
from pathlib import Path

from backend.fusion.models import Read
from backend.utils.csv_logger import append_reads


def _read(text, conf, cls="LicensePlate", cam="cam1", ts=1708300776.0):
    return Read(text=text, raw_text=text, confidence=conf,
                class_name=cls, camera_id=cam, timestamp=ts)


def test_creates_file_with_header(tmp_path):
    csv_path = tmp_path / "detections.csv"
    append_reads(csv_path, [_read("ABC1234", 0.85)])

    lines = csv_path.read_text().strip().split("\n")
    assert lines[0] == "Timestamp,Camera_ID,Value,Data_Type,Confidence"
    assert len(lines) == 2


def test_appends_without_duplicating_header(tmp_path):
    csv_path = tmp_path / "detections.csv"
    append_reads(csv_path, [_read("ABC1234", 0.85)])
    append_reads(csv_path, [_read("XYZ9999", 0.90)])

    lines = csv_path.read_text().strip().split("\n")
    assert lines[0] == "Timestamp,Camera_ID,Value,Data_Type,Confidence"
    assert len(lines) == 3  # header + 2 data rows


def test_matches_legacy_format(tmp_path):
    csv_path = tmp_path / "detections.csv"
    reads = [
        _read("K8970006", 0.79, cls="LicensePlate", cam="s3.png", ts=1708300776.0),
        _read("1354U", 1.0, cls="TrailerNum", cam="s1.png", ts=1708300777.0),
        _read("1234567", 0.93, cls="USDOT", cam="s3.png", ts=1708300778.0),
    ]
    append_reads(csv_path, reads)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert len(rows) == 3
    assert rows[0]["Value"] == "K8970006"
    assert rows[0]["Data_Type"] == "licenseplate"
    assert rows[0]["Camera_ID"] == "s3.png"
    assert rows[1]["Data_Type"] == "trailernum"
    assert rows[2]["Data_Type"] == "usdot"


def test_multiple_reads_same_call(tmp_path):
    csv_path = tmp_path / "detections.csv"
    reads = [_read("A", 0.5), _read("B", 0.6), _read("C", 0.7)]
    append_reads(csv_path, reads)

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 3


def test_empty_reads_no_crash(tmp_path):
    csv_path = tmp_path / "detections.csv"
    append_reads(csv_path, [])
    assert not csv_path.exists()  # no file created for empty reads
