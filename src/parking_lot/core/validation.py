"""Text validation, smart correction, and authorized plate management."""

import itertools
import os
import re
import time

from parking_lot.config import ValidationConfig


class TextValidator:
    """Manages authorized plates and provides smart OCR correction."""

    def __init__(self, cfg: ValidationConfig):
        self.cfg = cfg
        self.authorized_plates: set[str] = set()
        self._last_auth_update: float = 0

    def refresh_authorized(self):
        """Reload authorized plates from file if stale."""
        now = time.time()
        if now - self._last_auth_update > self.cfg.auth_refresh_interval:
            if os.path.exists(self.cfg.authorized_file):
                try:
                    with open(self.cfg.authorized_file, "r") as f:
                        self.authorized_plates = {
                            line.strip().upper() for line in f if line.strip()
                        }
                except Exception:
                    pass
            self._last_auth_update = now

    def is_authorized(self, plate: str) -> bool:
        return plate in self.authorized_plates

    def smart_correction(self, text: str) -> str:
        """Try character substitutions to find an authorized plate or valid format."""
        if self.is_authorized(text):
            return text

        confusables = {
            "B": "8", "8": "B",
            "Q": "0", "0": "Q", "D": "0", "O": "0",
            "S": "5", "5": "S",
            "Z": "2", "2": "Z",
            "G": "6", "6": "G",
            "I": "1", "1": "I",
        }

        indices = [i for i, char in enumerate(text) if char in confusables]
        if not indices:
            return text

        chars = list(text)
        options = [[c, confusables[c]] for c in [text[i] for i in indices]]

        if len(options) > self.cfg.max_confusable_indices:
            options = options[: self.cfg.max_confusable_indices]
            indices = indices[: self.cfg.max_confusable_indices]

        for combo in itertools.product(*options):
            for i, idx in enumerate(indices):
                chars[idx] = combo[i]
            candidate = "".join(chars)

            if self.is_authorized(candidate):
                return candidate
            if re.match(self.cfg.container_regex, candidate.replace("-", "")) or re.match(
                self.cfg.plate_regex, candidate
            ):
                return candidate

        return text

    def is_valid_format(self, text: str, class_type: str) -> bool:
        """Check if text matches the expected regex for its class type."""
        if "containernum" in class_type or ("container" in class_type and "plate" not in class_type):
            return bool(re.match(self.cfg.container_regex, text.replace("-", "")))
        elif "trailernum" in class_type:
            return bool(re.match(self.cfg.trailer_regex, text))
        else:
            return bool(re.match(self.cfg.plate_regex, text))

    @staticmethod
    def clean_text(text: str) -> str:
        return re.sub(r"[^A-Z0-9\-]", "", text.upper())
