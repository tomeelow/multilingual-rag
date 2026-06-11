# CLAUDE.md

Guidance for Claude when working in this repository.

## Project context

RAG system for AI regulatory compliance and data protection law across **EU, Polish, and Ukrainian** jurisdictions. Sources include the EU AI Act, GDPR (EN/PL), the Polish Personal Data Protection Act, the Ukrainian Personal Data Protection Law, the DSA, the Polish Labour Code, and UODO guidance.

Document quality determines retrieval quality, which determines answer quality. Treat ingestion as the most critical phase, not a setup step to rush through.

## Core principles

- **Concise over clever.** Smallest amount of code that solves the problem clearly.
- **No premature abstraction.** Don't add layers, factories, or config switches until a second concrete use case exists.
- **Best practices, not ceremony.** Follow language idioms; don't import patterns from other ecosystems just because they're familiar.
- **Readable beats compact.** Concise is not the same as obfuscated.
- **Delete more than you add.** When refactoring, the diff should usually shrink.

## Git workflow

- **Branch per change.** Never commit features or fixes directly to `main`. Use `feat/<slug>`, `fix/<slug>`, `chore/<slug>`, `docs/<slug>`.
- **One concern per branch.** If scope grows, split.
- **Commit messages: short, imperative, descriptive. No filler.**
  - Good: `feat: add UODO opinion ingestion pipeline`
  - Good: `fix: handle two-column footnote interleaving in EUR-Lex PDFs`
  - Good: `chore: drop redundant HTML duplicates of GDPR PL`
  - Bad: `updates`, `WIP`, `various changes`, `fixed stuff`
- **Squash noisy histories before merging.** Keep `main` linear and readable.
- **Never commit secrets, API keys, or `.env`.** Provide `.env.example` instead.
- **Never commit large binaries.** Source PDFs go via Git LFS or a separate data store, not into the main tree.

## Code quality

- **Python:** type hints on public functions; `ruff` for lint, `black` (or `ruff format`) for formatting; `pytest` for tests.
- **Dependencies:** pin them. Pick one of `uv` / `poetry` / `pip-tools` and stick with it.
- **Configuration:** environment variables for secrets and deployment; YAML/TOML for pipeline parameters. No hardcoded paths or endpoints.
- **Logging over print.** Use `logging` with sensible levels. Never log raw document content at INFO in production.
- **Tests where it matters.** Required for text extraction, chunking, and retrieval scoring. Not required for throwaway scripts or exploratory notebooks.
- **Notebooks are scratch space.** If logic matures, move it into the package. Don't import from notebooks.

## RAG-specific practices

### Ingestion

- **Verify before indexing.** Two-column EUR-Lex PDFs (AI Act, GDPR, DSA) can interleave footnotes during extraction. Spot-check extracted text against the source PDF before it enters the index.
- **Preserve legal structure.** Article numbers, recital numbers, paragraph numbers, section IDs — these are how legal citations work. Keep them in metadata, not only inline.
- **Track provenance per chunk.** Every chunk must carry: source document, jurisdiction, language, article/recital reference, source URL, ingestion date. Without this, citations cannot be reconstructed and the system is unusable for compliance work.
- **One canonical version per document.** When multiple formats exist (HTML vs PDF), pick PDF and drop the rest unless there's a documented reason to keep both.
- **Distinguish official vs unofficial sources.** Unofficial translations (e.g. the Polish EDPB Opinion 28/2024) are not a substitute for primary-authority guidance and must be labeled as such in metadata.

### Chunking and embeddings

- **Chunk on semantic boundaries.** For legal texts, by article or recital — not by character count alone. Fall back to fixed-size chunking only for unstructured content.
- **Lock the embedding model per index.** Mixing embeddings from different models produces silent retrieval failures. If the model changes, re-embed everything and bump the index version.
- **Record index metadata.** Embedding model, chunking strategy, document-set version, and build date — kept in the repo next to the index config.

### Retrieval and generation

- **Citations are non-negotiable.** Every legal claim must point to a specific article or recital. No citation → no answer.
- **No hallucinated references.** If the retriever returns nothing relevant, the model says so. It does not invent a plausible-sounding article number.
- **Multilingual care.** EN/PL/UK queries retrieve from matching-language sources by default; cross-language fallback only when explicitly enabled. Legal terminology does not translate cleanly — preserve the original term alongside any rendering.
- **Golden set.** Maintain a small set of question → expected-citation pairs. Run it before merging any change that affects retrieval or generation.

### Reproducibility

- Set seeds for stochastic steps.
- Pin model versions explicitly: `text-embedding-3-large`, not `latest`.
- Record dataset versions. Laws get amended; the system needs to know which version it indexed.

## Privacy and compliance

This project is about data protection law. Apply the principles internally:

- No real personal data in fixtures, tests, or example queries.
- No persistent logging of full user queries in production.
- If processing documents that contain personal data (e.g. UODO case files), document the lawful basis and retention period in the repo.

## Ask before acting

When unsure, stop and ask. Specifically:

- Schema changes to the index or chunk metadata
- New external dependencies, especially any pulling from non-canonical legal sources
- Anything that re-embeds or re-indexes the full corpus
- Anything that rewrites git history on shared branches
