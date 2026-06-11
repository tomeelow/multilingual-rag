"""Language detection restricted to the three corpus languages.

lingua over langdetect: reliable on short snippets and Slavic scripts.
"""

from functools import lru_cache

from lingua import Language, LanguageDetectorBuilder

_TO_ISO = {Language.ENGLISH: "en", Language.POLISH: "pl", Language.UKRAINIAN: "uk"}


@lru_cache(maxsize=1)
def _detector():
    return LanguageDetectorBuilder.from_languages(*_TO_ISO).build()


def detect_language(text: str) -> str:
    return _TO_ISO.get(_detector().detect_language_of(text), "en")
