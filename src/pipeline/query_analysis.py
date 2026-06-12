"""Lightweight query understanding for the retrieval layer.

Pure string heuristics — no models, no network. Legal queries carry strong
explicit signals (rights vs duties wording, article ranges, comparative
phrasing) that pure semantic similarity demonstrably gets wrong: a chunk
stuffed with "pracownik jest obowiązany" outscores the basic-principles
chapter on a *rights* question because the embedding only sees topical
closeness, not polarity.
"""

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
