"""Text extraction helpers: de-hyphenation and language detection."""

from src.ingestion.parse import dehyphenate
from src.language import detect_language


def test_dehyphenate_joins_wrapped_words():
    assert dehyphenate("przetwa-\nrzanie danych") == "przetwarzanie danych"
    assert dehyphenate("пер-\nсональних даних") == "персональних даних"
    assert dehyphenate("data pro-\ncessing") == "data processing"


def test_dehyphenate_keeps_real_hyphens():
    # hyphen before an uppercase letter or digit is not a line wrap
    assert dehyphenate("Стаття 8-\n1 щось") == "Стаття 8-\n1 щось"
    assert dehyphenate("e-mail") == "e-mail"


def test_dehyphenate_strips_soft_hyphens():
    assert dehyphenate("da\xadne") == "dane"


def test_detect_language():
    assert detect_language("Processing of personal data shall be lawful.") == "en"
    assert detect_language("Przetwarzanie danych osobowych jest zgodne z prawem.") == "pl"
    assert detect_language("Обробка персональних даних здійснюється відкрито.") == "uk"
