"""Corpus registry: one entry per source document.

Each entry pairs citation metadata (DocumentMeta) with a parsing profile —
which file to read, how to filter the PDF text layer, and which structural
grammar to apply. Layout profiles were derived by inspecting the actual PDFs;
see docs/ingestion_notes.md for the findings.
"""

from dataclasses import dataclass, field
from datetime import date

from src.models import DocumentMeta


@dataclass
class CorpusEntry:
    meta: DocumentMeta
    file: str  # filename under data/raw/
    structure: str  # "eurlex" | "isap" | "ua_law" | "numbered_paragraphs" | "plain"
    min_size: float = 9.0  # drop chars below this font size (footnote/superscript text)
    superscript_size: float | None = None  # map digits at this size to unicode superscripts
    crop_top: float = 0.0  # drop objects above this y (positional headers)
    drop_line_patterns: list[str] = field(default_factory=list)  # regexes for header/footer lines


# Header/footer line patterns, per publisher
_EURLEX_STRIP = [
    r"^.{0,40}(Official Journal of the European Union|Dziennik Urzędowy Unii Europejskiej).{0,40}$",
    r"^(?:EN\s+)?OJ\s+[LC],\s+\d{1,2}\.\d{1,2}\.\d{4}(?:\s+EN)?$",
    r"^ELI:\s+\S+$",
    r"^\d{1,3}/\d{1,3}$",
    r"^(EN|PL)$",
]
_ISAP_STRIP = [
    r"^©\s*Kancelaria Sejmu.*$",
    r"^\d{4}-\d{2}-\d{2}$",
    r"^Dziennik Ustaw\s+.{0,8}\d+.{0,8}\s+Poz\.\s+\d+\s*$",
]
_PAGE_NUMBER_STRIP = [r"^\d{1,3}$"]

CORPUS: list[CorpusEntry] = [
    CorpusEntry(
        meta=DocumentMeta(
            source_id="eu_ai_act_2024",
            title="Regulation (EU) 2024/1689 — Artificial Intelligence Act",
            language="en",
            doc_type="regulation",
            jurisdiction="EU",
            effective_date=date(2024, 8, 1),
            url="https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689",
        ),
        file="OJ_L_202401689_EN_TXT.pdf",
        structure="eurlex",
        min_size=8.0,  # recital numbers are 8.5pt in the 2024 OJ format; footnotes go by zone
        drop_line_patterns=_EURLEX_STRIP,
    ),
    CorpusEntry(
        meta=DocumentMeta(
            source_id="gdpr_en",
            title="Regulation (EU) 2016/679 — General Data Protection Regulation",
            language="en",
            doc_type="regulation",
            jurisdiction="EU",
            effective_date=date(2018, 5, 25),  # date of application
            url="https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32016R0679",
        ),
        file="CELEX_32016R0679_EN_TXT.pdf",
        structure="eurlex",
        min_size=8.0,
        drop_line_patterns=_EURLEX_STRIP,
    ),
    CorpusEntry(
        meta=DocumentMeta(
            source_id="gdpr_pl",
            title="Rozporządzenie (UE) 2016/679 — RODO (ogólne rozporządzenie o ochronie danych)",
            language="pl",
            doc_type="regulation",
            jurisdiction="EU",
            effective_date=date(2018, 5, 25),
            url="https://eur-lex.europa.eu/legal-content/PL/TXT/?uri=CELEX:32016R0679",
        ),
        file="CELEX_32016R0679_PL_TXT.pdf",
        structure="eurlex",
        min_size=8.0,
        drop_line_patterns=_EURLEX_STRIP,
    ),
    CorpusEntry(
        meta=DocumentMeta(
            source_id="dsa_en",
            title="Regulation (EU) 2022/2065 — Digital Services Act",
            language="en",
            doc_type="regulation",
            jurisdiction="EU",
            effective_date=date(2024, 2, 17),
            url="https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32022R2065",
        ),
        file="CELEX_32022R2065_EN_TXT.pdf",
        structure="eurlex",
        min_size=8.0,
        drop_line_patterns=_EURLEX_STRIP,
    ),
    CorpusEntry(
        meta=DocumentMeta(
            source_id="pl_data_protection_act",
            title="Ustawa z dnia 10 maja 2018 r. o ochronie danych osobowych",
            language="pl",
            doc_type="act",
            jurisdiction="PL",
            effective_date=date(2018, 5, 25),
            url="https://isap.sejm.gov.pl/isap.nsf/DocDetails.xsp?id=WDU20180001000",
        ),
        file="D2018000100001.pdf",
        structure="isap",
        min_size=9.5,  # body is 10pt; header is stripped by pattern
        drop_line_patterns=_ISAP_STRIP,
    ),
    CorpusEntry(
        meta=DocumentMeta(
            source_id="pl_labour_code",
            title="Ustawa z dnia 26 czerwca 1974 r. — Kodeks pracy",
            language="pl",
            doc_type="act",
            jurisdiction="PL",
            effective_date=date(1975, 1, 1),
            url="https://isap.sejm.gov.pl/isap.nsf/DocDetails.xsp?id=WDU19740240141",
            consolidated_as_of=date(2026, 4, 16),  # validity date printed in the PDF footer
        ),
        file="D19740141Lj.pdf",
        structure="isap",
        min_size=11.0,  # body 12pt; 9pt header and 10pt footer drop out
        superscript_size=8.0,  # article numbers like Art. 18³ use 8pt superscript digits
        drop_line_patterns=_ISAP_STRIP,
    ),
    CorpusEntry(
        meta=DocumentMeta(
            source_id="ua_data_protection_law",
            title="Закон України «Про захист персональних даних» № 2297-VI",
            language="uk",
            doc_type="act",
            jurisdiction="UA",
            effective_date=date(2011, 1, 1),
            url="https://zakon.rada.gov.ua/laws/show/2297-17",
            consolidated_as_of=date(2025, 6, 14),  # from the source filename
        ),
        file=(
            "Про захист персональних даних - Закон № 2297-VI від 01.06.2010 - d325478-20250614.pdf"
        ),
        structure="ua_law",
        min_size=11.0,  # uniform 12pt body
        drop_line_patterns=_PAGE_NUMBER_STRIP,
    ),
    CorpusEntry(
        meta=DocumentMeta(
            source_id="edpb_opinion_28_2024_pl",
            title="Opinia EROD 28/2024 w sprawie modeli AI (tłumaczenie nieoficjalne)",
            language="pl",
            doc_type="guidance",
            jurisdiction="EU",
            effective_date=date(2024, 12, 17),
            url="https://www.edpb.europa.eu/our-work-tools/our-documents/opinion-board-art-64/opinion-282024-certain-data-protection-aspects_en",
            official=False,  # unofficial PL translation — not primary-authority guidance
        ),
        file="Opinia EROD w sp. sztucznej inteligencji - tłumaczenie nieoficjalne.pdf",
        structure="numbered_paragraphs",
        min_size=9.0,
        drop_line_patterns=_PAGE_NUMBER_STRIP,
    ),
    CorpusEntry(
        meta=DocumentMeta(
            source_id="uodo_ai_report_2024",
            title="UODO — Raport strategiczny: badanie potrzeb organizacji w zakresie"
            " wykorzystania sztucznej inteligencji",
            language="pl",
            doc_type="guidance",
            jurisdiction="PL",
            effective_date=date(2024, 12, 31),  # NOTE: publication date not printed in the PDF
            url="https://uodo.gov.pl/",
            official=True,
        ),
        file="Raport Strategiczny - Badanie potrzeb organizacji w zakresie wykorzystania sztuc.pdf",
        structure="plain",
        min_size=9.0,
        drop_line_patterns=_PAGE_NUMBER_STRIP,
    ),
]


def by_source_id(source_id: str) -> CorpusEntry:
    for entry in CORPUS:
        if entry.meta.source_id == source_id:
            return entry
    raise KeyError(source_id)
