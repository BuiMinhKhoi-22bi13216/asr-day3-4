from src.run_offline import normalize


def test_normalize_keeps_vietnamese_accents_and_digits():
    text = "  SÁNG   MAI, họp lúc 9 giờ! "
    assert normalize(text) == "sáng mai họp lúc 9 giờ"


def test_normalize_combines_unicode_marks():
    assert normalize("a\u0301") == "á"