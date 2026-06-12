# Demo video script (90 seconds, no narration)

Per the plan: let the cross-linguality speak for itself. Record with the
full stack running (`docker compose up`) and an LLM key configured.

| t | action | what the viewer sees |
|---|---|---|
| 0:00–0:10 | Open http://localhost:8501 | UI title, sidebar with corpus (9 docs, 3 languages), language badges |
| 0:10–0:35 | Click 🇺🇦→🇬🇧 example: *Які системи штучного інтелекту заборонені в ЄС?* | Answer streams **in Ukrainian**; sources panel shows 🇬🇧-badged *AI Act, Article 5* — Ukrainian question, English statute |
| 0:35–0:55 | Type: *What does Polish law say about employee CCTV monitoring?* | Answer streams **in English**; sources show 🇵🇱 *Kodeks pracy, Art. 22²* and 🇵🇱 RODO — English question, Polish statutes |
| 0:55–1:15 | Click a source link; hover badges | Citation opens the official EUR-Lex/ISAP source; UNOFFICIAL badge visible on the EDPB translation |
| 1:15–1:30 | Toggle "Cross-lingual retrieval" off, re-ask the Ukrainian question | Sources now restricted to 🇺🇦 documents — the toggle visibly controls scope |

Keep the latency caption (retrieval/rerank/generation ms) in frame — it
reads as engineering, not magic.
