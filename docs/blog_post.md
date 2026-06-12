# Blog post draft (Medium / LinkedIn, ~550 words)

*Publish before applications go out; link from the README once live.*

---

## I built a legal assistant that answers in your language — and cites the law in its own

A Ukrainian founder running a business in Poland has a uniquely bad
information problem: her obligations live in the EU AI Act (English), RODO
(Polish), the Polish Labour Code (Polish), and Ukraine's data-protection law
(Ukrainian). Asking a generic chatbot gets fluent answers with invented
article numbers — useless, and dangerous, for compliance.

So I built a retrieval-augmented assistant over nine official legal sources
in three languages. You ask in English, Polish, or Ukrainian; it answers in
your language and pins every claim to a specific article — *Article 5 AI
Act*, *Art. 83 RODO*, *Стаття 8 Закону № 2297-VI* — keeping the citation in
the source document's language. No citation, no answer.

**The decision that mattered most: hybrid retrieval.** Legal text is
named-entity-heavy. Embeddings think "Article 6" and "Article 26" are nearly
identical; BM25 matches them exactly but can't connect a Ukrainian question
to an English regulation. So queries run through both legs — multilingual-e5
dense vectors in Qdrant and BM25 — fused by Reciprocal Rank Fusion, then
reranked by a cross-encoder. Chunking is parent-child: retrieval scores
256-token children for precision, but the LLM always reads the full parent
article, because a half-article is how you get a wrong legal answer.

**The part most tutorials skip: the PDFs.** EUR-Lex renders the AI Act's
recital numbers at 8.5pt — the same size as footnotes, so a naive font
filter deletes recital numbering entirely. The Polish Labour Code uses
superscript article numbers where *Art. 18³* and *Art. 183* are different
provisions. One paragraph of an EDPB opinion sat below a link underline that
my footnote detector mistook for a separator. Document parsing isn't
preprocessing; it is the product.

**The surprising finding from evaluation.** I hand-built a 57-question
golden set across the three languages, with per-article ground truth, and
ran a four-configuration ablation. For English queries, the standard recipe
works exactly as advertised: hybrid retrieval plus a cross-encoder reranker
hits the right article in the top five 83% of the time. But the stock
reranker (`ms-marco-MiniLM`, pinned in every RAG tutorial) is trained only
on English — and on cross-lingual questions it took accuracy to **zero**,
systematically demoting every correct foreign-language passage that dense
retrieval had surfaced. The component that adds the most precision
monolingually is the one that erases cross-linguality. If your RAG system is
multilingual, your reranker has to be too — and you will only ever learn
that from an evaluation set built to catch it.

Everything is reproducible: `docker compose up` bootstraps the whole stack —
parses the PDFs, builds both indexes, starts the API and UI — with no manual
steps. The eval harness, the golden set, and the honest numbers are in the
repo.

**Demo & code:** *(link to repo / deployed space)*

---

*Структура за planem: problem → approach → key decision → surprising
finding → demo link. ~550 words.*
