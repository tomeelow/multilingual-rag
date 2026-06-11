"""Split parsed page text into structural units (recitals, articles, annexes,
numbered paragraphs) with citation metadata.

Heading detection is regex-at-line-start plus a sequence check: a heading
match only counts if its number continues the running sequence. This rejects
look-alikes — e.g. GDPR Article 4 definitions are numbered "(1) ..." exactly
like recitals, and wrapped cross-references can leave "Article 99" alone on a
line.
"""

import re
from bisect import bisect_right
from dataclasses import dataclass

from src.ingestion.corpus import CorpusEntry
from src.models import Unit


def _main_number(num: str) -> int:
    """Leading integer of an article number: '18³a' -> 18, '8-1' -> 8.
    Superscript digits are not \\d, so they never merge into the main number."""
    return int(re.match(r"\d+", num).group(0))


def _ascii_number(num: str) -> str:
    """Unambiguous ASCII form for IDs: '18³a' -> '18^3a' (≠ article 183)."""
    return re.sub(
        r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+",
        lambda m: "^" + m.group(0).translate(str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")),
        num,
    )


@dataclass
class _Doc:
    """Joined page text with offset -> page-number lookup."""

    text: str
    page_starts: list[int]

    @classmethod
    def from_pages(cls, pages: list[str]) -> "_Doc":
        starts, pos = [], 0
        for p in pages:
            starts.append(pos)
            pos += len(p) + 1
        return cls(text="\n".join(pages), page_starts=starts)

    def page_of(self, offset: int) -> int:
        return bisect_right(self.page_starts, offset)  # 1-based


def _sequential(matches: list[re.Match], first: int = 1) -> list[re.Match]:
    """Keep only matches whose group(1) numbers form 1,2,3,... from `first`."""
    out, expected = [], first
    for m in matches:
        if int(m.group(1)) == expected:
            out.append(m)
            expected += 1
    return out


def _monotonic(matches: list[re.Match], max_gap: int = 100) -> list[re.Match]:
    """Keep matches whose main numbers never decrease (gaps from repealed
    articles are fine; an inline cross-reference jumping backwards is not)."""
    out, prev = [], 0
    for m in matches:
        n = _main_number(m.group(1))
        if prev <= n <= prev + max_gap:
            out.append(m)
            prev = n
    return out


def _slice_units(
    doc: _Doc,
    sid: str,
    matches: list[re.Match],
    end: int,
    kind: str,
    id_tag: str,
    title_from: str | None = "next_line",  # "next_line" | "inline" | None
    sections: list[tuple[int, str]] | None = None,
) -> list[Unit]:
    units = []
    for i, m in enumerate(matches):
        stop = matches[i + 1].start() if i + 1 < len(matches) else end
        text = doc.text[m.start() : stop].strip()
        number = m.group(1)
        title = None
        if title_from == "next_line":
            lines = text.split("\n", 2)
            if len(lines) > 1:
                title = lines[1].strip()
        elif title_from == "inline":
            title = m.group(2).strip() or None
        section = None
        if sections:
            for off, name in sections:
                if off < m.start():
                    section = name
        units.append(
            Unit(
                unit_id=f"{sid}:{id_tag}:{_ascii_number(number)}",
                kind=kind,
                number=number,
                title=title,
                section=section,
                text=text,
                pages=[doc.page_of(m.start()), doc.page_of(stop - 1)],
            )
        )
    return units


_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _roman_value(s: str) -> int:
    vals = [_ROMAN[c] for c in s]
    return sum(-v if i + 1 < len(vals) and v < vals[i + 1] else v for i, v in enumerate(vals))


def _parse_eurlex(doc: _Doc, sid: str, lang: str) -> list[Unit]:
    art_word = {"en": "Article", "pl": "Artykuł"}[lang]
    whereas = re.search(r"^(Whereas:?|a także mając na uwadze, co następuje:?)\s*$", doc.text, re.M)
    art_matches = list(re.finditer(rf"^{art_word} (\d+)\s*$", doc.text, re.M))
    art_matches = _sequential([m for m in art_matches if m.start() > whereas.end()])
    enacting = art_matches[0].start()

    rec_matches = [
        m
        for m in re.finditer(r"^\((\d+)\)\s", doc.text, re.M)
        if whereas.end() < m.start() < enacting
    ]
    rec_matches = _sequential(rec_matches)

    annex_matches = list(re.finditer(r"^(?:ANNEX|ZAŁĄCZNIK)\s+([IVXLC]+)\s*$", doc.text, re.M))
    annex_matches = [m for m in annex_matches if m.start() > enacting]
    annex_matches = [m for i, m in enumerate(annex_matches) if _roman_value(m.group(1)) == i + 1]
    body_end = annex_matches[0].start() if annex_matches else len(doc.text)

    units = _slice_units(doc, sid, rec_matches, enacting, "recital", "rec", title_from=None)
    units += _slice_units(doc, sid, art_matches, body_end, "article", "art")
    units += _slice_units(doc, sid, annex_matches, len(doc.text), "annex", "annex")

    # the signature block sits between the last article and the annexes
    last_art = units[len(rec_matches) + len(art_matches) - 1]
    cut = re.search(r"^(Done at|Sporządzono w)\b", last_art.text, re.M)
    if cut:
        last_art.text = last_art.text[: cut.start()].strip()
    return units


def _section_headings(doc: _Doc, pattern: str) -> list[tuple[int, str]]:
    """(offset, heading + next line) for chapter/section headings."""
    out = []
    for m in re.finditer(pattern, doc.text, re.M):
        nxt = doc.text.find("\n", m.end() + 1)
        title = doc.text[m.end() : nxt].strip() if nxt != -1 else ""
        name = m.group(0).strip()
        out.append((m.start(), f"{name} — {title}" if title else name))
    return out


def _parse_isap(doc: _Doc, sid: str) -> list[Unit]:
    matches = list(re.finditer(r"^Art\.\s+(\d+[⁰¹²³⁴⁵⁶⁷⁸⁹]*[a-z]{0,2})\.", doc.text, re.M))
    matches = _monotonic(matches)
    sections = _section_headings(doc, r"^(?:DZIAŁ\s+\S+|Rozdział\s+[IVXLC\d]+[a-z]?)\s*$")
    return _slice_units(
        doc, sid, matches, len(doc.text), "article", "art", title_from=None, sections=sections
    )


def _parse_ua_law(doc: _Doc, sid: str) -> list[Unit]:
    matches = list(re.finditer(r"^Стаття\s+(\d+(?:-\d+)?)\.\s*(.*)$", doc.text, re.M))
    matches = _monotonic(matches)
    sections = _section_headings(doc, r"^Розділ\s+[IVX]+\s*$")
    return _slice_units(
        doc, sid, matches, len(doc.text), "article", "art", title_from="inline", sections=sections
    )


def _parse_numbered_paragraphs(doc: _Doc, sid: str) -> list[Unit]:
    # strictly increasing with a small gap tolerance: a paragraph number lost
    # to extraction noise must not silently glue the rest of the document
    # into one unit; numbered section headings ("4. Uwagi końcowe") jump
    # backwards and are rejected
    matches, prev = [], 0
    for m in re.finditer(r"^(\d{1,3})\.\s+", doc.text, re.M):
        n = int(m.group(1))
        if prev < n <= prev + 5:
            matches.append(m)
            prev = n
    if len(matches) < 10:  # numbering not found — fall back to one unit
        return _parse_plain(doc, sid)
    return _slice_units(
        doc, sid, matches, len(doc.text), "paragraph", "para", title_from=None, sections=None
    )


def _parse_plain(doc: _Doc, sid: str) -> list[Unit]:
    return [
        Unit(
            unit_id=f"{sid}:doc",
            kind="document",
            number=None,
            title=None,
            section=None,
            text=doc.text.strip(),
            pages=[1, doc.page_of(len(doc.text) - 1)],
        )
    ]


def parse_units(pages: list[str], entry: CorpusEntry) -> list[Unit]:
    doc = _Doc.from_pages(pages)
    sid = entry.meta.source_id
    parser = {
        "eurlex": lambda: _parse_eurlex(doc, sid, entry.meta.language),
        "isap": lambda: _parse_isap(doc, sid),
        "ua_law": lambda: _parse_ua_law(doc, sid),
        "numbered_paragraphs": lambda: _parse_numbered_paragraphs(doc, sid),
        "plain": lambda: _parse_plain(doc, sid),
    }[entry.structure]
    return [u for u in parser() if u.text]
