import csv
import datetime
from pathlib import Path

from backend.fusion.models import Read

_HEADER = ["Timestamp", "Camera_ID", "Value", "Data_Type", "Confidence"]

# Map internal class names to legacy CSV format
_CLASS_NAME_MAP = {
    "LicensePlate": "licenseplate",
    "TrailerNum": "trailernum",
    "USDOT": "usdot",
    "ContainerNum": "containernum",
    "ContainerPlate": "containerplate",
}


def append_reads(csv_path: str | Path, reads: list[Read]) -> None:
    if not reads:
        return

    csv_path = Path(csv_path)
    file_exists = csv_path.exists()

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(_HEADER)

        for read in reads:
            writer.writerow([
                datetime.datetime.fromtimestamp(read.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                read.camera_id,
                read.text,
                _CLASS_NAME_MAP.get(read.class_name, read.class_name.lower()),
                f"{read.confidence:.2f}",
            ])
