"""Lightweight query understanding for the retrieval layer.

Pure string heuristics — no models, no network. Legal queries carry strong
explicit signals (rights vs duties wording, article ranges, comparative
phrasing) that pure semantic similarity demonstrably gets wrong: a chunk
stuffed with "pracownik jest obowiązany" outscores the basic-principles
chapter on a *rights* question because the embedding only sees topical
closeness, not polarity.
"""

import re

from src.models import Chunk

# Stems, matched as lowercase substrings so one list covers inflected forms
# across en/pl/uk ("prawa", "praw", "prawach" → "praw" would also hit
# "prawodawstwo"; the longer stems below were chosen against the corpus
# section headings to avoid that).
_RIGHTS_TERMS = (
    "right",  # en: right / rights / rights of the data subject
    "prawa",  # pl: prawa pracownika, prawa osoby
    "prawo do",  # pl: prawo do wypoczynku
    "uprawnien",  # pl: uprawnienia / uprawnień
    "права",  # uk
    "право",  # uk
)
_DUTIES_TERMS = (
    "obowiązk",  # pl: obowiązki / obowiązkiem
    "obowiązan",  # pl: jest obowiązany / obowiązana
    "powinnoś",  # pl: powinności
    "dut",  # en: duty / duties
    "obligat",  # en: obligation / obligated
    "obliged",  # en
    "обов'язк",  # uk (U+0027 apostrophe)
    "обовʼязк",  # uk (U+02BC apostrophe)
    "зобов'язан",
    "зобовʼязан",
)

# Score multipliers applied on top of the sigmoid-normalized cross-encoder
# score. A section heading is the strongest signal we have — "Obowiązki
# pracownika" *is* the duties chapter no matter how the article text reads.
_OPPOSING_SECTION_WEIGHT = 0.35
_ALIGNED_SECTION_WEIGHT = 1.25
_OPPOSING_TEXT_WEIGHT = 0.7


# Comparative phrasing across en/pl/uk, matched as lowercase substrings
# ("differen" covers difference/different/differs, "porówn" covers
# porównaj/porównanie/w porównaniu, "порівня" covers порівняй/порівняння).
_COMPARATIVE_TERMS = (
    "differen",
    "compar",
    "contrast",
    "versus",
    " vs ",
    " vs.",
    "różni",
    "porówn",
    "w odróżnieniu",
    "різниц",
    "відрізня",
    "порівня",
    "на відміну",
)

# Aliases by which queries name corpus documents -> source_id groups.
# Keyed to the corpus in data/processed/; extend when documents are added.
# A "group" is every indexed language version of the same legal act.
_SOURCE_ALIASES: list[tuple[re.Pattern, tuple[str, ...]]] = [
    (
        re.compile(r"\bgdpr\b|\brodo\b|2016/679|general data protection", re.IGNORECASE),
        ("gdpr_en", "gdpr_pl"),
    ),
    (
        re.compile(r"ukrain\w*|україн\w*|ukraińsk\w*", re.IGNORECASE),
        ("ua_data_protection_law",),
    ),
    (
        re.compile(r"kodeks\w*\s+prac\w*|labou?r\s+code|трудов\w*", re.IGNORECASE),
        ("pl_labour_code",),
    ),
    (
        re.compile(
            r"\bai\s+act\b|akt\w*\s+o\s+sztucznej|artificial\s+intelligence\s+act"
            r"|штучн\w*\s+інтелект\w*",
            re.IGNORECASE,
        ),
        ("eu_ai_act_2024",),
    ),
    (
        re.compile(
            r"\bdsa\b|digital\s+services\s+act|akt\w*\s+o\s+usługach\s+cyfrowych", re.IGNORECASE
        ),
        ("dsa_en",),
    ),
    (
        re.compile(
            r"polish\s+(data\s+protection\s+)?(law|act)|ustaw\w*\s+o\s+ochronie\s+danych"
            r"|польськ\w*",
            re.IGNORECASE,
        ),
        ("pl_data_protection_act",),
    ),
]


def detect_comparative_intent(query: str) -> bool:
    q = query.lower()
    return any(t in q for t in _COMPARATIVE_TERMS)


def comparative_source_groups(query: str) -> list[list[str]]:
    """For a comparative query naming >= 2 corpus documents, the source_id
    groups to retrieve separately, in mention order. A useful comparison
    needs material from *every* side; one retrieval over the whole corpus
    demonstrably returns only the dominant one. Empty list otherwise."""
    if not detect_comparative_intent(query):
        return []
    mentioned: list[tuple[int, tuple[str, ...]]] = []
    for pattern, sources in _SOURCE_ALIASES:
        m = pattern.search(query)
        if m:
            mentioned.append((m.start(), sources))
    if len(mentioned) < 2:
        return []
    return [list(sources) for _, sources in sorted(mentioned)]


# "art. 10–18", "artykuły 10-18", "articles 10–18", "ст. 10–18",
# "статті 10–18". Superscript/letter suffixes on the bounds ("18³a") are
# consumed but ignored — ranges are resolved at base-article granularity.
_ARTICLE_RANGE = re.compile(
    r"(?:art(?:ykuł\w*|icles?)?|ст(?:атт\w*)?)\s*\.?\s*"
    r"(\d+)[\w¹²³⁴⁵⁶⁷⁸⁹⁰]*\s*[–—−-]\s*(\d+)[\w¹²³⁴⁵⁶⁷⁸⁹⁰]*",
    re.IGNORECASE,
)

_LEADING_DIGITS = re.compile(r"\d+")


def parse_article_range(query: str) -> tuple[int, int] | None:
    """Explicit article range in the query -> (lo, hi) base numbers."""
    m = _ARTICLE_RANGE.search(query)
    if not m:
        return None
    lo, hi = int(m.group(1)), int(m.group(2))
    return (lo, hi) if lo <= hi else (hi, lo)


def article_base_number(article_number: str | None) -> int | None:
    """Base integer of an article number: '11¹' -> 11, '18³a' -> 18,
    '8-1' -> 8 (UA inserted article). Roman annex numbers -> None."""
    if not article_number:
        return None
    m = _LEADING_DIGITS.match(article_number)
    return int(m.group()) if m else None


def in_article_range(chunk: Chunk, lo: int, hi: int) -> bool:
    if chunk.kind != "article":
        return False
    base = article_base_number(chunk.article_number)
    return base is not None and lo <= base <= hi


def _has_rights(text: str) -> bool:
    return any(t in text for t in _RIGHTS_TERMS)


def _has_duties(text: str) -> bool:
    return any(t in text for t in _DUTIES_TERMS)


def detect_rights_duties_intent(query: str) -> str | None:
    """-> "rights" | "duties" | None (absent or mixed — never guess)."""
    q = query.lower()
    rights, duties = _has_rights(q), _has_duties(q)
    if rights and not duties:
        return "rights"
    if duties and not rights:
        return "duties"
    return None


def intent_weight(intent: str | None, chunk: Chunk) -> float:
    """Multiplier for a chunk's rerank score under a rights/duties intent.

    Section heading first (chapter-level ground truth from ingestion), chunk
    text as fallback. Headings or texts mentioning both sides ("Prawa i
    obowiązki pracownika") stay neutral.
    """
    if intent is None:
        return 1.0
    aligned, opposing = (
        (_has_rights, _has_duties) if intent == "rights" else (_has_duties, _has_rights)
    )
    section = (chunk.section_title or "").lower()
    if aligned(section) and not opposing(section):
        return _ALIGNED_SECTION_WEIGHT
    if opposing(section) and not aligned(section):
        return _OPPOSING_SECTION_WEIGHT
    text = chunk.text.lower()
    if opposing(text) and not aligned(text):
        return _OPPOSING_TEXT_WEIGHT
    return 1.0
