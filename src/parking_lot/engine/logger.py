"""Thread-safe CSV logging for detections."""

import csv
import datetime
import os
import threading


class CSVLogger:
    """Appends detection entries to a CSV file in a thread-safe manner."""

    HEADER = ["Timestamp", "Camera_ID", "Value", "Data_Type", "Confidence"]

    def __init__(self, filename: str = "truck_logistics.csv"):
        self.filename = filename
        self._lock = threading.Lock()
        self._init_file()

    def _init_file(self):
        if not os.path.exists(self.filename):
            with open(self.filename, "w", newline="") as f:
                csv.writer(f).writerow(self.HEADER)

    def log(self, camera_id: str, value: str, data_type: str, ocr_conf: float):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._lock:
            with open(self.filename, "a", newline="") as f:
                csv.writer(f).writerow([timestamp, camera_id, value, data_type, f"{ocr_conf:.2f}"])

    def read_recent(self, limit: int = 100) -> list[dict]:
        """Read the most recent log entries (for API)."""
        entries = []
        if not os.path.exists(self.filename):
            return entries
        with self._lock:
            with open(self.filename, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    entries.append(row)
        return entries[-limit:]

    def get_file_path(self) -> str:
        return os.path.abspath(self.filename)
