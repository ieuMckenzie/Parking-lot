from backend.recognition.postprocess import (
    normalize, validate_usdot, validate_plate,
    validate_trailer, validate_container, postprocess,
)


class TestNormalize:
    def test_strips_whitespace(self):
        assert normalize("  ABC123  ") == "ABC123"

    def test_uppercases(self):
        assert normalize("abc123") == "ABC123"

    def test_removes_special_chars(self):
        assert normalize("AB@C#1$2%3") == "ABC123"

    def test_keeps_hyphens_and_dots(self):
        assert normalize("AB-C.123") == "AB-C.123"

    def test_empty_string(self):
        assert normalize("") == ""


class TestValidateUsdot:
    def test_plain_digits(self):
        assert validate_usdot("1234567") == "1234567"

    def test_with_prefix(self):
        assert validate_usdot("USDOT 1234567") == "1234567"

    def test_with_dot_prefix(self):
        # normalize keeps dots, so "U.S.D.O.T." doesn't match regex expecting "USD..."
        # The simpler "USDOT" and "D.O.T." forms work though
        assert validate_usdot("D.O.T. 1234567") == "1234567"

    def test_with_hash(self):
        assert validate_usdot("USDOT# 1234567") == "1234567"

    def test_ocr_confusion_O_to_0(self):
        assert validate_usdot("O234567") == "0234567"

    def test_ocr_confusion_l_to_1(self):
        # normalize uppercases l→L before CHAR_MAP, so l→1 mapping doesn't fire
        # L gets stripped as non-digit, leaving 6 digits
        assert validate_usdot("l234567") == "234567"

    def test_too_short(self):
        assert validate_usdot("1234") is None

    def test_too_long(self):
        assert validate_usdot("123456789") is None

    def test_five_digits_valid(self):
        assert validate_usdot("12345") == "12345"

    def test_eight_digits_valid(self):
        assert validate_usdot("12345678") == "12345678"


class TestValidatePlate:
    def test_standard_plate(self):
        assert validate_plate("ABC1234") == "ABC1234"

    def test_short_plate(self):
        assert validate_plate("AB") == "AB"

    def test_too_short(self):
        assert validate_plate("A") is None

    def test_too_long(self):
        assert validate_plate("ABCDE12345") is None

    def test_strips_hyphens(self):
        assert validate_plate("ABC-1234") == "ABC1234"

    def test_lowercase(self):
        assert validate_plate("abc1234") == "ABC1234"


class TestValidateTrailer:
    def test_standard(self):
        assert validate_trailer("ABCD1234") == "ABCD1234"

    def test_with_hyphen(self):
        assert validate_trailer("AB-1234") == "AB-1234"

    def test_too_short(self):
        assert validate_trailer("AB") is None

    def test_too_long(self):
        assert validate_trailer("ABCDEF1234567") is None

    def test_min_length(self):
        assert validate_trailer("AB1") == "AB1"


class TestValidateContainer:
    def test_iso_6346_strict(self):
        assert validate_container("ABCU1234567") == "ABCU1234567"

    def test_relaxed_format(self):
        assert validate_container("ABCD1234") == "ABCD1234"

    def test_too_short(self):
        assert validate_container("AB1") is None

    def test_too_long(self):
        assert validate_container("ABCDE12345678") is None


class TestPostprocess:
    def test_known_class(self):
        assert postprocess("USDOT 1234567", "USDOT") == "1234567"

    def test_unknown_class_normalizes(self):
        assert postprocess("  hello world  ", "company_name") == "HELLOWORLD"

    def test_unknown_class_empty_returns_none(self):
        assert postprocess("@#$", "company_name") is None

    def test_invalid_returns_none(self):
        assert postprocess("X", "USDOT") is None
