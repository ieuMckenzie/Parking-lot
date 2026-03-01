"""Tests for parking_lot.engine.logger.CSVLogger."""

import csv
import os


class TestCSVLoggerInit:
    def test_creates_file_with_header(self, csv_logger):
        assert os.path.exists(csv_logger.filename)
        with open(csv_logger.filename) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == ["Timestamp", "Camera_ID", "Value", "Data_Type", "Confidence"]

    def test_does_not_overwrite_existing(self, csv_logger):
        # Log an entry, then create a new logger on the same file
        csv_logger.log("cam0", "ABC123", "licenseplate", 0.9)

        from parking_lot.engine.logger import CSVLogger
        logger2 = CSVLogger(filename=csv_logger.filename)
        entries = logger2.read_recent()
        assert len(entries) == 1  # original entry still there


class TestLog:
    def test_log_appends_entry(self, csv_logger):
        csv_logger.log("cam0", "ABC1234", "licenseplate", 0.95)
        with open(csv_logger.filename) as f:
            lines = f.readlines()
        assert len(lines) == 2  # header + 1 entry

    def test_log_multiple_entries(self, csv_logger):
        csv_logger.log("cam0", "ABC1234", "licenseplate", 0.95)
        csv_logger.log("cam1", "XYZ9999", "usdot", 0.80)
        csv_logger.log("cam0", "MSCU123456", "containernum", 0.70)
        with open(csv_logger.filename) as f:
            lines = f.readlines()
        assert len(lines) == 4  # header + 3 entries

    def test_log_confidence_format(self, csv_logger):
        csv_logger.log("cam0", "ABC1234", "licenseplate", 0.9567)
        with open(csv_logger.filename) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            row = next(reader)
        assert row[4] == "0.96"  # formatted to 2 decimals


class TestReadRecent:
    def test_read_empty_log(self, csv_logger):
        entries = csv_logger.read_recent()
        assert entries == []

    def test_read_all_entries(self, csv_logger):
        csv_logger.log("cam0", "ABC1234", "licenseplate", 0.95)
        csv_logger.log("cam1", "XYZ9999", "usdot", 0.80)
        entries = csv_logger.read_recent()
        assert len(entries) == 2
        assert entries[0]["Value"] == "ABC1234"
        assert entries[1]["Camera_ID"] == "cam1"

    def test_read_with_limit(self, csv_logger):
        for i in range(10):
            csv_logger.log("cam0", f"PLATE{i:03d}", "licenseplate", 0.9)
        entries = csv_logger.read_recent(limit=3)
        assert len(entries) == 3
        # Should be the last 3
        assert entries[0]["Value"] == "PLATE007"
        assert entries[2]["Value"] == "PLATE009"

    def test_read_missing_file(self, tmp_path):
        from parking_lot.engine.logger import CSVLogger
        # Point at a file path but delete the file after init
        path = str(tmp_path / "missing.csv")
        logger = CSVLogger(filename=path)
        os.remove(path)
        entries = logger.read_recent()
        assert entries == []

    def test_dict_keys_match_header(self, csv_logger):
        csv_logger.log("cam0", "ABC1234", "licenseplate", 0.95)
        entries = csv_logger.read_recent()
        assert set(entries[0].keys()) == set(csv_logger.HEADER)


class TestGetFilePath:
    def test_returns_absolute_path(self, csv_logger):
        path = csv_logger.get_file_path()
        assert os.path.isabs(path)
