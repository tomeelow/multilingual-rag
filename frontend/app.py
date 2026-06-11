"""Streamlit chat frontend.

Run: uv run streamlit run frontend/app.py
Needs the API (src.api.main) running; set API_URL to point at it.
"""

import os
import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frontend.api_client import get_health, get_sources, stream_chat  # noqa: E402

API_URL = os.environ.get("API_URL", "http://localhost:8000")
MAX_HISTORY = 5  # exchanges kept on screen

# per-language colour + flag for source badges: a reviewer should see within
# seconds that an answer drew on sources in a different language
LANG_BADGE = {
    "en": ("🇬🇧 EN", "#1d6fb8"),
    "pl": ("🇵🇱 PL", "#c43c47"),
    "uk": ("🇺🇦 UK", "#b8980f"),
}

EXAMPLES = [
    ("🇬🇧", "Which AI practices are prohibited under the AI Act?"),
    ("🇵🇱", "Jakie kary pieniężne przewiduje RODO za naruszenia?"),
    ("🇺🇦", "Які права має суб'єкт персональних даних?"),
    ("🇺🇦→🇬🇧", "Які системи штучного інтелекту заборонені в ЄС?"),
]


def badge(language: str, official: bool) -> str:
    label, color = LANG_BADGE[language]
    extra = "" if official else " · UNOFFICIAL"
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:10px;font-size:0.75em;white-space:nowrap">{label}{extra}</span>'
    )


def render_sources(sources: list[dict]) -> None:
    for s in sources:
        st.markdown(
            f"{badge(s['language'], s['official'])} **{s['ref'] or '—'}** — "
            f"[{s['doc_title']}]({s['url']}), pp. {s['pages'][0]}–{s['pages'][1]}",
            unsafe_allow_html=True,
        )


st.set_page_config(page_title="Multilingual legal RAG", page_icon="⚖️", layout="wide")
st.title("⚖️ EU / PL / UA AI & data-protection law assistant")

if "history" not in st.session_state:
    st.session_state.history = []  # [{"query", "answer", "sources", "meta"}]
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

health = get_health(API_URL)
if not health:
    st.error(
        f"API is not reachable at {API_URL}. Start it with: "
        "`uv run uvicorn src.api.main:app --port 8000`"
    )
    st.stop()

with st.sidebar:
    st.caption(
        f"{health['indexed_chunks']} chunks · languages: {', '.join(health['languages'])} · "
        f"`{health['embedding_model']}`"
    )
    lang_choice = st.selectbox(
        "Query language",
        ["auto-detect", "en", "pl", "uk"],
        help="auto-detect uses lingua on the query text",
    )
    cross_lingual = st.toggle(
        "Cross-lingual retrieval",
        value=True,
        help="Retrieve from sources in all languages; answers cite the original. "
        "Off = same-language sources only.",
    )
    use_hyde = st.toggle(
        "HyDE retrieval",
        value=False,
        help="Hypothetical Document Embeddings: an LLM-written hypothetical excerpt "
        "drives dense retrieval. Costs one extra LLM call.",
    )

    st.divider()
    st.subheader("Try an example")
    for flag, example in EXAMPLES:
        if st.button(f"{flag} {example}", use_container_width=True):
            st.session_state.pending_query = example

    st.divider()
    with st.expander("Indexed corpus"):
        for doc in get_sources(API_URL):
            official = "" if doc["official"] else " *(unofficial translation)*"
            st.markdown(
                f"{badge(doc['language'], doc['official'])} [{doc['title']}]({doc['url']})"
                f"{official} — {doc['units']} units",
                unsafe_allow_html=True,
            )

# replay history (last MAX_HISTORY exchanges)
for exchange in st.session_state.history[-MAX_HISTORY:]:
    with st.chat_message("user"):
        st.write(exchange["query"])
    with st.chat_message("assistant"):
        st.write(exchange["answer"])
        if exchange["sources"]:
            with st.expander("Sources", expanded=False):
                render_sources(exchange["sources"])

query = st.chat_input("Ask in English, po polsku, або українською…")
if st.session_state.pending_query:
    query = st.session_state.pending_query
    st.session_state.pending_query = None

if query:
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        final: dict = {}

        def tokens():
            for event in stream_chat(
                API_URL,
                query,
                filter_language=None if lang_choice == "auto-detect" else lang_choice,
                cross_lingual=cross_lingual,
                use_hyde=use_hyde,
                session_id=st.session_state.session_id,
            ):
                if "token" in event:
                    yield event["token"]
                elif event.get("done"):
                    final.update(event)

        try:
            answer_text = st.write_stream(tokens)
        except Exception as exc:  # API/network failure mid-stream
            st.error(f"Request failed: {exc}")
            st.stop()

        if "error" in final:
            st.error(final["error"])
            st.stop()

        sources = final.get("sources", [])
        if sources:
            with st.expander("Sources", expanded=True):
                render_sources(sources)
        st.caption(
            f"lang: {final.get('language')} · retrieval {final.get('retrieval_ms')} ms · "
            f"rerank {final.get('rerank_ms')} ms · generation {final.get('generation_ms')} ms · "
            f"total {final.get('latency_ms')} ms"
        )

    st.session_state.history.append(
        {"query": query, "answer": answer_text, "sources": sources, "meta": final}
    )
    st.session_state.history = st.session_state.history[-MAX_HISTORY:]
