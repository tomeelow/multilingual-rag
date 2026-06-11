"""PDF text extraction.

Strategy (verified against each source PDF, see docs/ingestion_notes.md):
all nine corpus PDFs are single-column with a good text layer. The noise to
remove is (a) footnotes — same or smaller font than body, always below a short
horizontal separator line near the page bottom, (b) running headers/footers —
stripped by per-publisher line patterns, and (c) superscript citation markers.
The Polish Labour Code additionally uses superscript digits *inside article
numbers* (Art. 18³ ≠ Art. 183), which must be preserved, not dropped.
"""

import re
import unicodedata

import pdfplumber
from pdfplumber.utils import extract_text as chars_to_text

from src.config import DATA_RAW
from src.ingestion.corpus import CorpusEntry

_SUPERSCRIPTS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")

# lowercase letters of the three corpus languages, for de-hyphenation
_LOWER = "a-ząćęłńóśźżа-яіїєґ"


def _footnote_cutoff(page) -> float:
    """Y coordinate of the footnote separator: the lowest short horizontal
    line starting at the left margin in the bottom half of the page.
    Everything below it is footnotes. The x0 condition matters: link
    underlines and table edges are also short horizontal lines, but they
    don't start at the margin."""
    cutoff = page.height
    for line in page.lines + page.rects:
        width = line["x1"] - line["x0"]
        if 30 < width < 250 and line["top"] > page.height / 2 and line["x0"] < 100:
            cutoff = min(cutoff, line["top"])
    return cutoff


def extract_page_text(page, entry: CorpusEntry) -> str:
    cutoff = _footnote_cutoff(page)
    chars = []
    for c in page.chars:
        if c["top"] >= cutoff or c["top"] < entry.crop_top:
            continue
        size = c["size"]
        if entry.superscript_size and abs(size - entry.superscript_size) < 0.3:
            # article numbers like Art. 18³a use superscript digits and letters;
            # anything else at superscript size is a citation marker — drop
            if c["text"].isdigit():
                chars.append({**c, "text": c["text"].translate(_SUPERSCRIPTS)})
            elif c["text"].isalpha():
                chars.append(c)
            continue
        if size >= entry.min_size:
            chars.append(c)
    if not chars:
        return ""
    # y_tolerance=6 keeps superscripts (raised baseline) on their parent line
    text = chars_to_text(chars, x_tolerance=1.5, y_tolerance=6)
    lines = [ln.strip() for ln in text.splitlines()]
    patterns = [re.compile(p) for p in entry.drop_line_patterns]
    lines = [ln for ln in lines if ln and not any(p.match(ln) for p in patterns)]
    return "\n".join(lines)


def dehyphenate(text: str) -> str:
    """Join words hyphenated across line breaks; keep genuine hyphens."""
    text = text.replace("\xad", "")  # soft hyphens
    return re.sub(rf"([{_LOWER}])-\n([{_LOWER}])", r"\1\2", text)


def parse_pdf(entry: CorpusEntry) -> list[str]:
    """Extract cleaned, NFC-normalized text per page."""
    pages = []
    with pdfplumber.open(DATA_RAW / entry.file) as pdf:
        entry.meta.page_count = len(pdf.pages)
        for page in pdf.pages:
            text = extract_page_text(page, entry)
            pages.append(unicodedata.normalize("NFC", text))
    return pages
