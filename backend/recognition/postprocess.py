import re

# Common OCR character confusions
CHAR_MAP = str.maketrans({
    "O": "0",
    "o": "0",
    "I": "1",
    "l": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
    "G": "6",
})

# Inverse: digits that should be letters in plate context
DIGIT_TO_ALPHA = str.maketrans({
    "0": "O",
    "1": "I",
    "8": "B",
    "5": "S",
    "2": "Z",
    "6": "G",
})


def normalize(text: str) -> str:
    """Strip whitespace, remove common OCR artifacts, uppercase."""
    text = text.strip().upper()
    text = re.sub(r"[^A-Z0-9\-\.]", "", text)
    return text


def validate_usdot(text: str) -> str | None:
    """Extract USDOT number: strip prefix, expect 5-8 digits."""
    text = normalize(text)
    # Remove USDOT/DOT prefix variants
    text = re.sub(r"^(US\s*D\.?O\.?T\.?|D\.?O\.?T\.?)\s*#?\s*", "", text)
    # Apply digit correction
    text = text.translate(CHAR_MAP)
    # Remove any remaining non-digits
    digits = re.sub(r"[^0-9]", "", text)
    if 5 <= len(digits) <= 8:
        return digits
    return None


def validate_plate(text: str) -> str | None:
    """Validate US license plate: 2-8 alphanumeric characters.

    If the full text is too long, tries to extract a substring matching
    common US plate patterns (e.g., ABC1234, 123ABC, AB1234).
    """
    text = normalize(text)
    text = re.sub(r"[^A-Z0-9]", "", text)
    if 2 <= len(text) <= 8:
        return text
    if len(text) > 8:
        return _extract_plate(text)
    return None


# Common US plate formats for fullmatch extraction
_PLATE_FORMATS = [
    re.compile(r"[A-Z]{3}\d{4}"),       # ABC1234
    re.compile(r"[A-Z]{2}\d{4}"),        # AB1234
    re.compile(r"\d{3}[A-Z]{3}"),        # 123ABC
    re.compile(r"\d{3}[A-Z]{4}"),        # 123ABCD
    re.compile(r"[A-Z]{3}\d{3}"),        # ABC123
    re.compile(r"[A-Z]\d{3}[A-Z]{3}"),   # A123BCD
    re.compile(r"\d[A-Z]{3}\d{3}"),       # 1ABC234
    re.compile(r"[A-Z]{2}\d{3}[A-Z]"),   # AB123C
    re.compile(r"[A-Z]{2}\d{2}[A-Z]{2}"),  # AB12CD
    re.compile(r"[A-Z]\d{2}[A-Z]{3}"),   # A12BCD
    re.compile(r"[A-Z]{2}\d{5}"),        # AB12345
    re.compile(r"\d{2}[A-Z]{3}"),        # 12ABC
]


def _extract_plate(text: str) -> str | None:
    """Try each plate pattern in priority order, return first match."""
    for fmt in _PLATE_FORMATS:
        m = fmt.search(text)
        if m:
            return m.group()
    return None


def validate_trailer(text: str) -> str | None:
    """Validate trailer number: company-specific, generally 3-12 alphanumeric."""
    text = normalize(text)
    text = re.sub(r"[^A-Z0-9\-]", "", text)
    if 3 <= len(text) <= 12:
        return text
    return None


def validate_container(text: str) -> str | None:
    """Validate container number: ISO 6346 format — 4 letters + 7 digits, or relaxed."""
    text = normalize(text)
    text = re.sub(r"[^A-Z0-9]", "", text)
    # Strict ISO 6346: XXXU1234567
    if re.match(r"^[A-Z]{4}\d{7}$", text):
        return text
    # Relaxed: 4+ alphanumeric
    if 4 <= len(text) <= 12:
        return text
    return None


# Map detection class names to validators
VALIDATORS: dict[str, callable] = {
    "USDOT": validate_usdot,
    "LicensePlate": validate_plate,
    "TrailerNum": validate_trailer,
    "ContainerNum": validate_container,
    "ContainerPlate": validate_container,
}


def postprocess(text: str, class_name: str) -> str | None:
    """Run the appropriate validator for a detection class. Returns cleaned text or None."""
    validator = VALIDATORS.get(class_name)
    if validator:
        return validator(text)
    return normalize(text) or None
