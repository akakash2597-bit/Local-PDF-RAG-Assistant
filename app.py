"""
app.py
------
Mobile-first & Desktop-Responsive Streamlit UI for Gemini RAG Document Q&A.

Key Features:
  - Responsive Layout: Primary document upload controls sit directly in the main page body.
  - Auto Collapsed Sidebar: Mobile visitors land on a clean view without sidebar overlays.
  - Fluid Styling: Uses CSS clamp() and flexwrap to ensure text and badges scale naturally.
  - Flexible API Key Fallback: Reads Streamlit secrets first, falling back gracefully to inline inputs.
"""

from __future__ import annotations

import html
import os
import traceback
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from rag_engine import (
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
    layout="centered",
    initial_sidebar_state="collapsed",  # collapsed by default for clean mobile startup
)

# --------------------------------------------------------------------------- #
# Theme & Mobile-Responsive CSS Overhaul
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
        radial-gradient(900px 520px at 8% -10%, rgba(124, 156, 255, 0.16), transparent 60%),
        radial-gradient(760px 440px at 100% -6%, rgba(192, 140, 255, 0.13), transparent 55%),
        var(--gr-bg);
    color: var(--gr-text);
}

/* Responsive shell container */
.block-container {
    padding: 3.5rem 0.9rem 7.5rem 0.9rem !important;   /* clears the mobile top bar */
    max-width: 900px;
}
@media (min-width: 768px) {
    .block-container { padding: 3.5rem 1.5rem 7rem 1.5rem !important; }
}

/* Top toolbar: transparent and collapsed so it never collides with the title */
header[data-testid="stHeader"] {
    background: transparent;
    height: 0;
    min-height: 0;
}
header[data-testid="stHeader"] > div:first-child { height: 0; }

/* keep the sidebar toggle reachable, but out of the title's way */
[data-testid="stSidebarCollapsedControl"] {
    top: 0.5rem;
    left: 0.3rem;
    z-index: 999;
}
@media (min-width: 768px) {
    [data-testid="stSidebarCollapsedControl"] { top: 0.75rem; left: 0.75rem; }
}

/* any deploy/menu toolbar sits top-right, below the title band */
[data-testid="stToolbar"], [data-testid="stStatusWidget"] {
    top: 0.5rem;
    right: 0.5rem;
    z-index: 999;
}

/* Fluid headers that never overflow on small screens */
.gr-hero { 
    display: flex; 
    align-items: center; 
    gap: 0.7rem; 
    margin-bottom: 0.2rem; 
    flex-wrap: wrap;
}
.gr-hero-badge {
    width: 38px; height: 38px; border-radius: 12px; flex: 0 0 auto;
    display: grid; place-items: center; font-size: 1.15rem;
    background: linear-gradient(135deg, var(--gr-a), var(--gr-b));
    box-shadow: 0 8px 24px rgba(124, 156, 255, 0.35);
}
.gr-title {
    font-size: clamp(1.2rem, 4vw, 1.8rem); 
    font-weight: 750; 
    letter-spacing: -0.02em; 
    line-height: 1.2;
    background: linear-gradient(92deg, #ffffff 10%, #b9c6ff 55%, #d9bcff 95%);
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent;
    word-break: break-word;
}
.gr-sub { color: var(--gr-muted); font-size: clamp(0.8rem, 2.6vw, 0.9rem); margin: 0 0 0.9rem 0; }
.gr-h { font-size: 0.95rem; font-weight: 700; margin: 0.2rem 0; }

/* ---- step markers ---- */
.gr-step {
    display: flex; align-items: center; gap: 0.5rem;
    font-size: clamp(0.85rem, 3vw, 0.95rem); font-weight: 700;
    letter-spacing: -0.01em; margin: 0.9rem 0 0.3rem 0;
}
.gr-step-num {
    width: 22px; height: 22px; border-radius: 7px; flex: 0 0 auto;
    display: grid; place-items: center; font-size: 0.72rem; font-weight: 800;
    color: #080c18;
    background: linear-gradient(135deg, var(--gr-a), var(--gr-b));
    box-shadow: 0 4px 14px rgba(124, 156, 255, 0.30);
}
.gr-muted { color: var(--gr-muted); }

/* Card components */
.gr-card {
    background: var(--gr-panel);
    border: 1px solid var(--gr-border);
    border-radius: 14px;
    padding: 0.85rem 0.95rem;
    margin: 0.5rem 0;
    backdrop-filter: blur(7px);
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.22);
}

/* Compact Key Warning */
.gr-warn {
    display: flex; gap: 0.6rem; align-items: flex-start;
    border: 1px solid rgba(255, 186, 90, 0.34);
    background: linear-gradient(135deg, rgba(255,186,90,0.13), rgba(255,186,90,0.05));
    border-radius: 12px; padding: 0.65rem 0.8rem; margin: 0.3rem 0;
}
.gr-warn-icon { font-size: 1rem; line-height: 1.25; }
.gr-warn-body { font-size: 0.82rem; color: #ffd79a; line-height: 1.4; }
.gr-warn-body b { color: #ffe6bd; }

/* ---------- system key status chip ---------- */
.gr-keychip {
    display: inline-flex; align-items: center; gap: 0.45rem; flex-wrap: wrap;
    padding: 0.42rem 0.7rem; border-radius: 12px; margin-bottom: 0.5rem;
    font-size: 0.83rem; font-weight: 650; color: #a6f4e2;
    border: 1px solid rgba(89, 224, 197, 0.32);
    background: linear-gradient(135deg, rgba(89,224,197,0.15), rgba(89,224,197,0.05));
}
.gr-keychip-dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--gr-c);
    box-shadow: 0 0 9px var(--gr-c);
}
.gr-keychip-src {
    font-size: 0.7rem; font-weight: 600; color: var(--gr-muted);
    border-left: 1px solid var(--gr-border); padding-left: 0.45rem;
}

/* Grid stats for document details */
.gr-stats {
    display: grid; 
    grid-template-columns: repeat(auto-fit, minmax(105px, 1fr));
    gap: 0.5rem; 
    margin-top: 0.65rem;
}
.gr-stat {
    background: rgba(0, 0, 0, 0.28); border: 1px solid var(--gr-border);
    border-radius: 11px; padding: 0.55rem 0.65rem;
}
.gr-stat-k {
    font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--gr-muted); margin-bottom: 0.2rem;
}
.gr-stat-v { font-size: 1.05rem; font-weight: 750; letter-spacing: -0.02em; }
.gr-stat-v.accent {
    background: linear-gradient(135deg, var(--gr-a), var(--gr-b));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}

/* Badges flexbox row */
.gr-badges { display: flex; flex-wrap: wrap; gap: 0.3rem; margin: 0.4rem 0 0.7rem; }
.gr-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.2rem 0.52rem; border-radius: 999px;
    font-size: 0.68rem; font-weight: 650; white-space: nowrap;
    border: 1px solid var(--gr-border);
    background: rgba(124, 156, 255, 0.13); color: #cfd9ff;
}
.gr-badge.ok { background: rgba(89, 224, 197, 0.14); color: #a6f4e2; border-color: rgba(89,224,197,0.30); }
.gr-badge.warn { background: rgba(255, 186, 90, 0.14); color: #ffd79a; border-color: rgba(255,186,90,0.30); }
.gr-badge.dot::before {
    content: ""; width: 5px; height: 5px; border-radius: 50%;
    background: currentColor; box-shadow: 0 0 6px currentColor;
}

/* Citation source blocks */
.gr-cite {
    border: 1px solid var(--gr-border);
    border-left: 3px solid var(--gr-a);
    border-radius: 4px 11px 11px 4px;
    background: rgba(0, 0, 0, 0.26);
    padding: 0.5rem 0.65rem; margin: 0 0 0.45rem 0;
}
.gr-cite-head {
    display: flex; align-items: center; gap: 0.4rem; flex-wrap: wrap;
    font-size: 0.74rem; color: var(--gr-muted); margin-bottom: 0.3rem;
}
.gr-cite-num {
    width: 18px; height: 18px; border-radius: 5px; display: grid; place-items: center;
    font-size: 0.64rem; font-weight: 800; color: #080c18;
    background: linear-gradient(135deg, var(--gr-a), var(--gr-b));
}
.gr-cite-body {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.75rem; line-height: 1.45; color: #ccd6f0;
    white-space: pre-wrap; word-break: break-word;
    max-height: 200px; overflow: auto;
}

/* Chat bubble optimization */
[data-testid="stChatMessage"] {
    background: var(--gr-panel);
    border: 1px solid var(--gr-border);
    border-radius: 14px;
    padding: 0.7rem 0.8rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.18);
}

/* Pinned chat input styling */
[data-testid="stChatInput"] {
    background: rgba(8, 12, 24, 0.94);
    backdrop-filter: blur(10px);
    border-top: 1px solid var(--gr-border);
    padding-top: 0.4rem;
}

/* Buttons and Mobile Touch Targets */
.stButton > button {
    border-radius: 11px; font-weight: 650; border: 1px solid var(--gr-border);
    background: rgba(255, 255, 255, 0.05); color: var(--gr-text);
    padding: 0.4rem 0.55rem; font-size: 0.83rem;
    transition: all 0.12s ease;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--gr-a), var(--gr-b));
    border: none; color: #080c18;
}
[data-testid="stSidebar"] {
    background: rgba(6, 9, 20, 0.95);
    border-right: 1px solid var(--gr-border);
}
[data-testid="stExpander"] {
    border: 1px solid var(--gr-border); border-radius: 12px;
    background: var(--gr-panel-2); overflow: hidden;
}

/* Animated Load Spinner */
@keyframes gr-orbit { to { transform: rotate(360deg); } }
@keyframes gr-pulse { 0%, 100% { opacity: 0.4; } 50% { opacity: 1; } }
.gr-spinner {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.6rem 0.8rem; border-radius: 12px;
    border: 1px solid var(--gr-border); background: var(--gr-panel);
    color: var(--gr-muted); font-size: 0.82rem;
}
.gr-spinner-ring {
    width: 16px; height: 16px; border-radius: 50%; flex: 0 0 auto;
    border: 2px solid rgba(124, 156, 255, 0.25);
    border-top-color: var(--gr-a);
    animation: gr-orbit 0.8s linear infinite;
}
.gr-spinner-dot {
    width: 5px; height: 5px; border-radius: 50%; background: var(--gr-b);
    animation: gr-pulse 1.1s ease-in-out infinite;
}

/* Media Query Touch Improvements */
@media (max-width: 640px) {
    .stButton > button { min-height: 42px; }
}

/* ---- touch targets: >= 42px everywhere ---- */
.stButton > button,
.stDownloadButton > button,
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"] {
    min-height: 42px;
    line-height: 1.25;
}

/* file uploader drop zone: tall enough to tap comfortably */
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzoneInput"],
section[data-testid="stFileUploader"] button {
    min-height: 42px;
}
[data-testid="stFileUploaderDropzone"] {
    padding: 1rem 0.75rem;
    border-radius: 14px;
    flex-direction: column;
    gap: 0.5rem;
    text-align: center;
}
@media (min-width: 768px) {
    [data-testid="stFileUploaderDropzone"] { flex-direction: row; padding: 0.9rem 1rem; }
}

/* sidebar toggle stays tappable */
[data-testid="stSidebarCollapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button {
    min-height: 42px; min-width: 42px;
}

/* viewport scaling: no horizontal overflow on phones */
html, body { -webkit-text-size-adjust: 100%; }
[data-testid="stAppViewContainer"], .stApp { max-width: 100vw; overflow-x: hidden; }
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { overflow-wrap: anywhere; }
img, pre, code, table { max-width: 100%; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Helper Render Functions
# --------------------------------------------------------------------------- #

def esc(text) -> str:
    return html.escape(str(text if text is not None else ""))


def card(body: str) -> None:
    st.markdown(f"<div class='gr-card'>{body}</div>", unsafe_allow_html=True)


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


def key_warning() -> None:
    st.markdown(
        "<div class='gr-warn'>"
        "<div class='gr-warn-icon'>🔑</div>"
        "<div class='gr-warn-body'><b>No Gemini API key found.</b> Save "
        "<code>GEMINI_API_KEY</code> under Streamlit Secrets, or paste a key below.</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_sources(sources: list[dict], expanded: bool = False) -> None:
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
        ("Size / overlap", f"{stats.get('chunk_size', '—')} / {stats.get('chunk_overlap', '—')}"),
    ]
    cells = "".join(
        f"<div class='gr-stat'><div class='gr-stat-k'>{esc(k)}</div>"
        f"<div class='gr-stat-v{' accent' if i == 2 else ''}'>{esc(v)}</div></div>"
        for i, (k, v) in enumerate(items)
    )
    card(
        "<div class='gr-h'>📄 " + esc(stats.get("filename", "document.pdf")) + "</div>"
        "<div class='gr-muted' style='font-size:0.78rem'>Indexed & Ready</div>"
        f"<div class='gr-stats'>{cells}</div>"
    )


def build_markdown() -> str:
    lines = [
        "# Document Q&A — Chat Log",
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
            lines += ["## 🧑 Question", "", message["content"], ""]
        else:
            lines += ["## ✨ Answer", "", message["content"], ""]
            for i, src in enumerate(message.get("sources") or [], start=1):
                lines += [
                    f"<details><summary>Source {i} — page {src.get('page')}, "
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
        lines.append("_No discussion history available._")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Application State Initialization
# --------------------------------------------------------------------------- #

st.session_state.setdefault("engine", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("indexed_name", None)
st.session_state.setdefault("chunk_count", 0)
st.session_state.setdefault("indexed_with", None)
st.session_state.setdefault("doc_stats", {})

auto_key = auto_api_key()
auto_source = key_source()

QUICK_QUESTIONS = [
    ("📌 Summarize key takeaways", "Summarize the key takeaways of this document."),
    ("🧠 Explain main concepts", "Explain the main concepts in this document in simple terms."),
    ("🔑 List important terms", "List the important terms and definitions in this document."),
]


# --------------------------------------------------------------------------- #
# Sidebar Drawer (Secondary Advanced Controls Only)
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.caption("Model configurations")

    if auto_key:
        st.markdown(
            "<span class='gr-badge ok'>Key Loaded · " + esc(auto_source) + "</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<span class='gr-badge warn'>No Key Detected</span>", unsafe_allow_html=True)

    embedding_model = st.selectbox(
        "Embedding model",
        [EMBEDDING_MODEL] + RETIRED_EMBEDDING_MODELS,
        help="gemini-embedding-001 is active.",
    )
    llm_options = LLM_OPTIONS if LLM_MODEL in LLM_OPTIONS else [LLM_MODEL] + LLM_OPTIONS
    llm_model = st.selectbox(
        "LLM",
        llm_options,
        index=llm_options.index(LLM_MODEL),
    )
    custom_llm = st.text_input(
        "Or type a model id", placeholder="gemini-3.8-flash",
    ).strip()
    if custom_llm:
        llm_model = custom_llm
    if llm_model in RETIRED_LLM_MODELS:
        st.markdown(
            "<span class='gr-badge warn'>Retired model</span>",
            unsafe_allow_html=True,
        )

    k = st.slider("Snippets to retrieve (k)", 1, 10, 4)

    st.divider()
    st.download_button(
        "⬇️ Download Chat History",
        data=build_markdown(),
        file_name=(
            f"discussion_{os.path.splitext(st.session_state.indexed_name)[0]}.md"
            if st.session_state.indexed_name else "discussion.md"
        ),
        mime="text/markdown",
        use_container_width=True,
        disabled=not st.session_state.messages,
    )
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------- #
# Main Body — Header, Key Gate, Document Uploader, Quick Prompts & Chat
# --------------------------------------------------------------------------- #

st.markdown(
    "<div class='gr-hero'>"
    "<div class='gr-hero-badge'>✨</div>"
    "<div><div class='gr-title'>Document Q&amp;A with Gemini</div></div>"
    "</div>"
    "<p class='gr-sub'>Answers grounded in your PDF, with exact source snippets.</p>",
    unsafe_allow_html=True,
)

# ---- API key configuration (right below the title) ---------------------- #
inline_key = ""
with st.expander("🔑 Gemini API Key Configuration", expanded=False):
    if auto_key:
        st.markdown(
            "<div class='gr-keychip'><span class='gr-keychip-dot'></span>"
            "✅ System API Key Active (Default)"
            f"<span class='gr-keychip-src'>{esc(auto_source)}</span></div>",
            unsafe_allow_html=True,
        )
        st.caption("Paste a key below only if you want to override it for this session.")
    else:
        key_warning()

    inline_key = st.text_input(
        "Custom Gemini API key",
        type="password",
        placeholder="Paste your Gemini API key (AIza…)",
        help="Overrides the system key for this session only.",
    ).strip()

    if inline_key:
        st.markdown(
            "<span class='gr-badge ok'>Custom key set for this session</span>",
            unsafe_allow_html=True,
        )
    elif auto_key:
        st.caption("Using the system key — nothing to do here.")
    else:
        st.caption("No system key found. A key is required to index and ask questions.")

# ---- Step 1 · Document Upload & Indexing -------------------------------- #
st.markdown(
    "<div class='gr-step'><span class='gr-step-num'>1</span>"
    "<span>Document Upload &amp; Indexing</span></div>",
    unsafe_allow_html=True,
)
with st.expander("📄 Upload & Index Document", expanded=not bool(st.session_state.indexed_name)):
    uploaded = st.file_uploader(
        "Choose a PDF", type=["pdf"], label_visibility="collapsed",
    )

    already_indexed = uploaded is not None and st.session_state.indexed_name == uploaded.name
    if st.button(
        "Index document", type="primary", use_container_width=True,
        disabled=(uploaded is None or already_indexed),
    ):
        try:
            resolve_api_key(inline_key)
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            placeholder = st.empty()
            with placeholder.container():
                spinner("Reading, chunking, and indexing the PDF...")
            try:
                engine = RAGEngine(
                    api_key=inline_key,
                    embedding_model=embedding_model,
                    llm_model=llm_model,
                    k=k,
                )
                engine._llm()  # Validate LLM model ID before processing embeddings
                count = engine.load_pdf(uploaded.getvalue(), filename=uploaded.name)
            except Exception as exc:
                placeholder.empty()
                st.error(f"Could not index document: {exc}")
                with st.expander("Traceback Details"):
                    st.code(traceback.format_exc())
            else:
                placeholder.empty()
                st.session_state.engine = engine
                st.session_state.indexed_name = uploaded.name
                st.session_state.chunk_count = count
                st.session_state.indexed_with = llm_model
                st.session_state.doc_stats = dict(engine.stats)
                st.session_state.messages = []
                st.success(f"Successfully indexed {uploaded.name} ({count} chunks).")

    if st.session_state.indexed_name and already_indexed:
        st.caption(f"✅ {st.session_state.indexed_name} is currently indexed.")

# Metadata status chips
badges([
    ("ChromaDB Active", "ok" if st.session_state.engine else ""),
    (llm_model, "warn" if llm_model in RETIRED_LLM_MODELS else ""),
    ("gemini-embedding-001", ""),
    (f"k={k}", ""),
    ("chunk 1000 / 200", ""),
], dot=True)

# Document Stats Container
if st.session_state.indexed_name:
    stats_card(st.session_state.doc_stats)

# ---- Step 2 · Chat Discussion ------------------------------------------- #
st.markdown(
    "<div class='gr-step'><span class='gr-step-num'>2</span>"
    "<span>Chat Discussion</span></div>",
    unsafe_allow_html=True,
)

# Active Discussion Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "✨"):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources") or [])

# Quick Starter Action Buttons
question = None
if st.session_state.indexed_name:
    st.markdown("<div class='gr-h'>Quick questions</div>", unsafe_allow_html=True)
    cols = st.columns([1, 1, 1], gap="small")
    for col, (label, prompt) in zip(cols, QUICK_QUESTIONS):
        if col.button(label, use_container_width=True):
            question = prompt
elif not st.session_state.messages:
    card(
        "<b>No document indexed yet</b><br>"
        "<span class='gr-muted'>Open <b>Upload &amp; Index Document</b> above, select a "
        "PDF file, and click <b>Index document</b>.</span>"
    )

# Fixed Bottom Input Bar
typed = st.chat_input(
    "Ask a question about your document...",
    disabled=not st.session_state.indexed_name,
)
question = typed or question

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="✨"):
        slot = st.empty()
        with slot.container():
            spinner("Searching document context & generating response...")
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