"""
app.py
------
Streamlit UI for the Gemini-powered RAG Document Q&A app.

Run:
    streamlit run app.py

Highlights
    * API key auto-detected from st.secrets / environment, with a status badge
      and an optional "Use Custom Gemini API Key" override expander.
    * Dark, gradient-styled UI with message bubbles and metadata badges.
    * Document stats card (filename, pages, characters, chunks) after indexing.
    * Quick-question buttons, citation cards per answer, and a markdown export
      of the whole discussion.
"""

from __future__ import annotations

import html
import os
import traceback
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from rag_engine import (
    API_KEY_ENV_VARS,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_OPTIONS,
    RETIRED_EMBEDDING_MODELS,
    RETIRED_LLM_MODELS,
    RAGEngine,
    auto_api_key,
    key_source,
    resolve_api_key,
)

load_dotenv()

st.set_page_config(
    page_title="Gemini RAG · Document Q&A",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Dark theme + custom CSS
# --------------------------------------------------------------------------- #

CUSTOM_CSS = """
<style>
:root {
    --gr-bg: #080c18;
    --gr-panel: rgba(255, 255, 255, 0.045);
    --gr-panel-2: rgba(255, 255, 255, 0.025);
    --gr-border: rgba(255, 255, 255, 0.10);
    --gr-text: #e9edf8;
    --gr-muted: #97a1bd;
    --gr-a: #7c9cff;
    --gr-b: #c08cff;
    --gr-c: #59e0c5;
}

html, body, [data-testid="stAppViewContainer"] { background: var(--gr-bg); }
.stApp {
    background:
        radial-gradient(1100px 620px at 6% -12%, rgba(124, 156, 255, 0.17), transparent 60%),
        radial-gradient(900px 520px at 100% -4%, rgba(192, 140, 255, 0.14), transparent 55%),
        radial-gradient(700px 400px at 60% 110%, rgba(89, 224, 197, 0.07), transparent 60%),
        var(--gr-bg);
    color: var(--gr-text);
}
.block-container { padding-top: 2rem; max-width: 1080px; }

/* ---------- headers ---------- */
.gr-hero { display: flex; align-items: center; gap: 0.9rem; margin-bottom: 0.3rem; }
.gr-hero-badge {
    width: 48px; height: 48px; border-radius: 15px; flex: 0 0 auto;
    display: grid; place-items: center; font-size: 1.4rem;
    background: linear-gradient(135deg, var(--gr-a), var(--gr-b));
    box-shadow: 0 10px 30px rgba(124, 156, 255, 0.38);
}
.gr-title {
    font-size: 1.8rem; font-weight: 750; letter-spacing: -0.025em; line-height: 1.12;
    background: linear-gradient(92deg, #ffffff 10%, #b9c6ff 55%, #d9bcff 95%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.gr-sub { color: var(--gr-muted); font-size: 0.92rem; margin: 0 0 1rem 0; }
.gr-h { font-size: 1.02rem; font-weight: 700; letter-spacing: -0.01em; margin: 0.2rem 0 0.15rem; }
.gr-muted { color: var(--gr-muted); }

/* ---------- cards ---------- */
.gr-card {
    background: var(--gr-panel);
    border: 1px solid var(--gr-border);
    border-radius: 16px;
    padding: 1rem 1.15rem;
    margin: 0.55rem 0;
    backdrop-filter: blur(7px);
    box-shadow: 0 6px 24px rgba(0, 0, 0, 0.22);
}
.gr-card.soft { background: var(--gr-panel-2); box-shadow: none; }

/* ---------- stats card ---------- */
.gr-stats {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.7rem; margin-top: 0.85rem;
}
.gr-stat {
    background: rgba(0, 0, 0, 0.28);
    border: 1px solid var(--gr-border);
    border-radius: 13px; padding: 0.65rem 0.8rem;
}
.gr-stat-k {
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.09em;
    color: var(--gr-muted); margin-bottom: 0.28rem;
}
.gr-stat-v { font-size: 1.24rem; font-weight: 750; letter-spacing: -0.02em; }
.gr-stat-v.accent {
    background: linear-gradient(135deg, var(--gr-a), var(--gr-b));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

/* ---------- badges ---------- */
.gr-badges { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.45rem 0 0.8rem; }
.gr-badge {
    display: inline-flex; align-items: center; gap: 0.35rem;
    padding: 0.24rem 0.62rem; border-radius: 999px;
    font-size: 0.73rem; font-weight: 650; letter-spacing: 0.01em;
    border: 1px solid var(--gr-border);
    background: rgba(124, 156, 255, 0.13); color: #cfd9ff;
}
.gr-badge.ok { background: rgba(89, 224, 197, 0.14); color: #a6f4e2; border-color: rgba(89,224,197,0.30); }
.gr-badge.warn { background: rgba(255, 186, 90, 0.14); color: #ffd79a; border-color: rgba(255,186,90,0.30); }
.gr-badge.dot::before {
    content: ""; width: 6px; height: 6px; border-radius: 50%;
    background: currentColor; box-shadow: 0 0 8px currentColor;
}

/* ---------- citation cards ---------- */
.gr-cite {
    border: 1px solid var(--gr-border);
    border-left: 3px solid var(--gr-a);
    border-radius: 4px 13px 13px 4px;
    background: rgba(0, 0, 0, 0.26);
    padding: 0.6rem 0.8rem; margin: 0 0 0.55rem 0;
}
.gr-cite-head {
    display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap;
    font-size: 0.78rem; color: var(--gr-muted); margin-bottom: 0.4rem;
}
.gr-cite-num {
    width: 20px; height: 20px; border-radius: 6px; display: grid; place-items: center;
    font-size: 0.68rem; font-weight: 800; color: #080c18;
    background: linear-gradient(135deg, var(--gr-a), var(--gr-b));
}
.gr-cite-body {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.79rem; line-height: 1.52; color: #ccd6f0;
    white-space: pre-wrap; word-break: break-word;
    max-height: 240px; overflow: auto;
}

/* ---------- chat bubbles ---------- */
[data-testid="stChatMessage"] {
    background: var(--gr-panel);
    border: 1px solid var(--gr-border);
    border-radius: 18px;
    padding: 0.9rem 1.05rem;
    margin-bottom: 0.65rem;
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.20);
}

/* ---------- controls ---------- */
.stButton > button {
    border-radius: 12px; font-weight: 650; border: 1px solid var(--gr-border);
    background: rgba(255, 255, 255, 0.05); color: var(--gr-text);
    transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease;
}
.stButton > button:hover {
    transform: translateY(-1px); border-color: rgba(124, 156, 255, 0.55);
    box-shadow: 0 9px 24px rgba(124, 156, 255, 0.22);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--gr-a), var(--gr-b));
    border: none; color: #080c18;
}
[data-testid="stSidebar"] {
    background: rgba(6, 9, 20, 0.80);
    border-right: 1px solid var(--gr-border);
}
[data-testid="stExpander"] {
    border: 1px solid var(--gr-border); border-radius: 14px;
    background: var(--gr-panel-2); overflow: hidden;
}

/* ---------- spinner ---------- */
@keyframes gr-orbit { to { transform: rotate(360deg); } }
@keyframes gr-pulse { 0%, 100% { opacity: 0.4; } 50% { opacity: 1; } }
.gr-spinner {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.7rem 0.95rem; border-radius: 14px;
    border: 1px solid var(--gr-border); background: var(--gr-panel);
    color: var(--gr-muted); font-size: 0.88rem;
}
.gr-spinner-ring {
    width: 18px; height: 18px; border-radius: 50%; flex: 0 0 auto;
    border: 2px solid rgba(124, 156, 255, 0.25);
    border-top-color: var(--gr-a);
    animation: gr-orbit 0.8s linear infinite;
}
.gr-spinner-dot {
    width: 6px; height: 6px; border-radius: 50%; background: var(--gr-b);
    animation: gr-pulse 1.1s ease-in-out infinite;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Small HTML helpers
# --------------------------------------------------------------------------- #

def esc(text) -> str:
    return html.escape(str(text if text is not None else ""))


def card(body: str, soft: bool = False) -> None:
    st.markdown(
        f"<div class='gr-card{' soft' if soft else ''}'>{body}</div>",
        unsafe_allow_html=True,
    )


def badges(items: list[tuple[str, str]], dot: bool = False) -> None:
    chips = "".join(
        f"<span class='gr-badge {tone}{' dot' if dot else ''}'>{esc(label)}</span>"
        for label, tone in items
    )
    st.markdown(f"<div class='gr-badges'>{chips}</div>", unsafe_allow_html=True)


def spinner(label: str) -> None:
    st.markdown(
        f"<div class='gr-spinner'><div class='gr-spinner-ring'></div>"
        f"<span>{esc(label)}</span><div class='gr-spinner-dot'></div></div>",
        unsafe_allow_html=True,
    )


def render_sources(sources: list[dict], expanded: bool = False) -> None:
    """Collapsible citation cards for the retrieved context."""
    if not sources:
        return
    pages = sorted({s["page"] for s in sources if s.get("page") is not None})
    label = f"📎 Sources · {len(sources)} snippet(s)"
    if pages:
        label += f" · page {', '.join(str(p) for p in pages)}"
    with st.expander(label, expanded=expanded):
        for i, src in enumerate(sources, start=1):
            page = src.get("page")
            page_txt = f"page {page}" if page is not None else "page —"
            st.markdown(
                "<div class='gr-cite'>"
                "<div class='gr-cite-head'>"
                f"<span class='gr-cite-num'>{i}</span>"
                f"<span><b>{esc(page_txt)}</b> · chunk {esc(src.get('chunk'))} · "
                f"{esc(src.get('characters', len(src.get('snippet', ''))))} chars</span>"
                "</div>"
                f"<div class='gr-cite-body'>{esc(src.get('snippet', ''))}</div>"
                "</div>",
                unsafe_allow_html=True,
            )


def stats_card(stats: dict) -> None:
    if not stats:
        return
    items = [
        ("Pages", f"{stats.get('pages', 0)}"),
        ("Characters", f"{stats.get('characters', 0):,}"),
        ("Chunks", f"{stats.get('chunks', 0)}"),
        ("Chunk size", f"{stats.get('chunk_size', '—')} / {stats.get('chunk_overlap', '—')}"),
    ]
    cells = "".join(
        f"<div class='gr-stat'><div class='gr-stat-k'>{esc(k)}</div>"
        f"<div class='gr-stat-v{' accent' if i == 2 else ''}'>{esc(v)}</div></div>"
        for i, (k, v) in enumerate(items)
    )
    card(
        "<div class='gr-h'>📄 " + esc(stats.get("filename", "document.pdf")) + "</div>"
        "<div class='gr-muted' style='font-size:0.82rem'>Indexed and ready for questions"
        "</div>"
        f"<div class='gr-stats'>{cells}</div>"
    )


def build_markdown() -> str:
    """Export the discussion, sources included, as markdown."""
    lines = [
        "# Document Q&A — Discussion",
        "",
        f"- **Document:** {st.session_state.indexed_name or '—'}",
        f"- **Model:** {st.session_state.indexed_with or LLM_MODEL}",
        f"- **Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
    ]
    for message in st.session_state.messages:
        if message["role"] == "user":
            lines += [f"## 🧑 Question", "", message["content"], ""]
        else:
            lines += ["## ✨ Answer", "", message["content"], ""]
            for i, src in enumerate(message.get("sources") or [], start=1):
                page = src.get("page")
                lines += [
                    f"<details><summary>Source {i} — page {page}, "
                    f"chunk {src.get('chunk')}</summary>",
                    "",
                    "```",
                    src.get("snippet", ""),
                    "```",
                    "",
                    "</details>",
                    "",
                ]
    if not st.session_state.messages:
        lines.append("_No questions asked yet._")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

st.session_state.setdefault("engine", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("indexed_name", None)
st.session_state.setdefault("chunk_count", 0)
st.session_state.setdefault("indexed_with", None)
st.session_state.setdefault("doc_stats", {})
st.session_state.setdefault("pending_question", None)

auto_key = auto_api_key()
auto_source = key_source()

QUICK_QUESTIONS = [
    ("📌 Summarize key takeaways", "Summarize the key takeaways of this document."),
    ("🧠 Explain main concepts", "Explain the main concepts in this document in simple terms."),
    ("🔑 List important terms", "List the important terms and definitions in this document."),
]


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("### ✨ Gemini RAG")
    st.caption("Upload a PDF, index it, ask questions.")

    # ---- API key -------------------------------------------------------- #
    if auto_key:
        st.success("API Key Loaded Automatically")
        st.caption(f"Source: {auto_source}")
    else:
        st.warning("No API key detected — add one below or in `.streamlit/secrets.toml`.")

    with st.expander("Use Custom Gemini API Key", expanded=not bool(auto_key)):
        custom_key = st.text_input(
            "Gemini API key",
            type="password",
            placeholder="AIza…",
            help="Overrides the key loaded from secrets / environment for this session.",
            label_visibility="collapsed",
        ).strip()
        if custom_key:
            st.markdown(
                "<span class='gr-badge ok'>using custom key</span>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Leave empty to use the automatically loaded key.")

    st.divider()

    # ---- models --------------------------------------------------------- #
    st.markdown("**Models**")
    embedding_model = st.selectbox(
        "Embedding model",
        [EMBEDDING_MODEL] + RETIRED_EMBEDDING_MODELS,
        help="gemini-embedding-001 is the live model. models/embedding-001 and "
             "text-embedding-004 are retired and return 404.",
    )
    llm_options = LLM_OPTIONS if LLM_MODEL in LLM_OPTIONS else [LLM_MODEL] + LLM_OPTIONS
    llm_model = st.selectbox(
        "LLM",
        llm_options,
        index=llm_options.index(LLM_MODEL),
        help="Gemini 3.x Flash is current. 1.5 / 2.0 / 2.5 are retired and return 404.",
    )
    custom_llm = st.text_input(
        "Or type a model id", placeholder="gemini-3.8-flash",
        help="Any live Gemini model id; overrides the dropdown.",
    ).strip()
    if custom_llm:
        llm_model = custom_llm
    if llm_model in RETIRED_LLM_MODELS:
        st.markdown(
            "<span class='gr-badge warn'>retired model — will 404</span>",
            unsafe_allow_html=True,
        )
    k = st.slider("Snippets to retrieve (k)", 1, 10, 4)

    st.divider()

    # ---- document ------------------------------------------------------- #
    st.markdown("**Document**")
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")

    already_indexed = uploaded is not None and st.session_state.indexed_name == uploaded.name
    if st.button("Index document", type="primary", use_container_width=True, disabled=already_indexed):
        if uploaded is None:
            st.warning("Upload a PDF first.")
        else:
            try:
                resolve_api_key(custom_key)  # fail fast, before any API call
            except RuntimeError as exc:
                st.error(str(exc))
            else:
                placeholder = st.empty()
                with placeholder.container():
                    spinner("Reading and chunking the PDF…")
                try:
                    engine = RAGEngine(
                        api_key=custom_key,
                        embedding_model=embedding_model,
                        llm_model=llm_model,
                        k=k,
                    )
                    engine._llm()  # fail fast on a retired model id, before embedding
                    count = engine.load_pdf(uploaded.getvalue(), filename=uploaded.name)
                except Exception as exc:
                    placeholder.empty()
                    st.error(f"Could not index the document: {exc}")
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc())
                else:
                    placeholder.empty()
                    st.session_state.engine = engine
                    st.session_state.indexed_name = uploaded.name
                    st.session_state.chunk_count = count
                    st.session_state.indexed_with = f"{llm_model}"
                    st.session_state.doc_stats = dict(engine.stats)
                    st.session_state.messages = []
                    st.success(f"Indexed {uploaded.name} — {count} chunks.")

    if st.session_state.indexed_name:
        badges([
            (f"{st.session_state.chunk_count} chunks", "ok"),
            (f"k={k}", ""),
        ])

    # ---- export --------------------------------------------------------- #
    st.divider()
    st.download_button(
        "⬇️ Download Discussion (.md)",
        data=build_markdown(),
        file_name=(
            f"discussion_{os.path.splitext(st.session_state.indexed_name)[0]}.md"
            if st.session_state.indexed_name else "discussion.md"
        ),
        mime="text/markdown",
        use_container_width=True,
        disabled=not st.session_state.messages,
    )
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(
        "PyPDFLoader → RecursiveCharacterTextSplitter (1000/200) → "
        "Gemini embeddings → Chroma → RetrievalQA."
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

st.markdown(
    "<div class='gr-hero'>"
    "<div class='gr-hero-badge'>✨</div>"
    "<div><div class='gr-title'>Document Q&amp;A with Gemini</div></div>"
    "</div>"
    "<p class='gr-sub'>Retrieval-augmented answers grounded in your PDF, with the exact "
    "snippets they came from.</p>",
    unsafe_allow_html=True,
)

badges([
    ("ChromaDB Active", "ok" if st.session_state.engine else ""),
    (f"Gemini {llm_model.replace('gemini-', '').replace('-', ' ').title()}", ""),
    ("gemini-embedding-001", ""),
    ("1000 / 200", ""),
], dot=True)

if st.session_state.indexed_name:
    stats_card(st.session_state.doc_stats)
else:
    card(
        "<b>No document indexed yet</b><br>"
        "<span class='gr-muted'>Upload a PDF in the sidebar and click "
        "<b>Index document</b> to start chatting with it.</span>"
    )

# ---- chat history -------------------------------------------------------- #
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "✨"):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources") or [])

# ---- quick questions ----------------------------------------------------- #
question = None
if st.session_state.indexed_name:
    st.markdown("<div class='gr-h'>Quick questions</div>", unsafe_allow_html=True)
    cols = st.columns(len(QUICK_QUESTIONS))
    for col, (label, prompt) in zip(cols, QUICK_QUESTIONS):
        if col.button(label, use_container_width=True):
            question = prompt

typed = st.chat_input(
    "Ask something about the document…",
    disabled=not st.session_state.indexed_name,
)
question = typed or question

# ---- answer -------------------------------------------------------------- #
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="✨"):
        slot = st.empty()
        with slot.container():
            spinner("Searching the document and writing the answer…")
        try:
            out = st.session_state.engine.answer(question)
            answer, sources = out["answer"], out["sources"]
        except Exception as exc:
            answer, sources = f"⚠️ {exc}", []
        slot.empty()

        st.markdown(answer)
        render_sources(sources, expanded=True)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    st.rerun()
