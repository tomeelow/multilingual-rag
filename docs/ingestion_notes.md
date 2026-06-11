# Ingestion notes — PDF layout findings

Findings from inspecting each source PDF before writing the parsers
(per CLAUDE.md: verify before indexing). These drive the parsing profiles
in `src/ingestion/corpus.py`.

## Layout

All nine PDFs are **single-column** with a good text layer. The CELEX "TXT"
PDFs are not the two-column OJ reproduction; balanced left/right word counts
that suggest two columns are an artifact of justified full-width text.
Verified by measuring body line spans: every document has body lines spanning
>60% of page width.

## Noise sources and how each is removed

| Noise | Where | Removal |
|---|---|---|
| Footnotes | EUR-Lex (8.5pt), EDPB opinion (9pt) | Everything below the footnote separator: the lowest short (30–250pt) horizontal line starting at the left margin (x0 < 100) in the bottom half of the page. The x0 condition matters — link underlines and table edges are short horizontal lines too, but never start at the margin. |
| Superscript citation markers | EUR-Lex (4.8/5.4pt), ISAP (8pt) | Font-size floor (`min_size`). |
| Running headers/footers | All | Per-publisher line regexes (`drop_line_patterns`); positional cropping is unreliable because EUR-Lex headers overlap the body's vertical range and switch font size between pages. |

## Per-document traps found

- **AI Act (2024 OJ format):** recital numbers `(1)`…`(180)` are typeset at
  8.5pt — the same size as footnotes. A pure font-size filter erases them, so
  `min_size=8.0` plus zone-based footnote removal is required.
- **GDPR/DSA/AI Act:** GDPR Article 4 and AI Act Article 3 definitions are
  numbered `(1)`, `(2)`, … exactly like recitals; footnotes look like
  `(2) Directive 2000/31/EC …`. Recital detection therefore (a) only runs
  between the `Whereas:` line and the first `Article 1` heading and (b)
  enforces a strictly sequential 1,2,3,… numbering.
- **Polish Labour Code (ISAP):** article numbers contain meaningful 8pt
  superscripts: `Art. 18³` and `Art. 183` are different articles, and
  `Art. 18³a` adds superscript *letters*. Superscript digits are mapped to
  Unicode superscripts and letters kept; IDs use caret notation
  (`art:18^3a`) to stay unambiguous in ASCII.
- **Labour Code repealed ranges:** articles 19–21 (and others) are absent
  from the consolidated text — heading sequence validation must allow gaps
  (monotonic main number, gap ≤ 100) rather than require +1 steps.
- **EDPB Opinion 28/2024 (PL):** body paragraphs are numbered 1…135;
  numbered section headings (`4. Uwagi końcowe`) match the same pattern but
  jump backwards and are rejected by the increasing-sequence rule. Paragraph
  113 sits directly below a mid-page link underline — the original
  footnote-cutoff heuristic treated that underline as a footnote separator
  and silently dropped the paragraph (fixed by the x0 < 100 condition).
- **Enacting formula:** `HAVE ADOPTED THIS REGULATION` is typeset in a way
  that does not survive extraction in any of the four EUR-Lex files; the
  recitals/articles boundary is the first standalone `Article 1` /
  `Artykuł 1` line instead.
- **Signature blocks:** `Done at Brussels…` would otherwise glue to the last
  article; cut explicitly.

## Verified unit counts (against the official texts)

| Document | Expected | Parsed |
|---|---|---|
| GDPR EN / PL | 173 recitals, 99 articles | 173 + 99 (both) |
| DSA | 156 recitals, 93 articles | 156 + 93 |
| AI Act | 180 recitals, 113 articles, 13 annexes | 180 + 113 + 13 |
| PL data protection act | 176 articles | 176 |
| PL Labour Code | ~480 incl. superscript variants, ends at Art. 305 | 480, ends 305 |
| UA law 2297-VI | 30 articles | 30 |
| EDPB Opinion 28/2024 | 135 paragraphs | 132 (3 lost to extraction noise, glued to neighbours) |
