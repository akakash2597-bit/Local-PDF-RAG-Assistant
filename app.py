import os
import streamlit as st
from dotenv import load_dotenv

from rag_engine import RAGEngine, auto_api_key

load_dotenv()

# Page configuration - Clean title & auto-collapsed sidebar for mobile devices
st.set_page_config(
    page_title="PDF Document Assistant",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Subtle layout padding for a clean look across all devices
st.markdown("""
<style>
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 5.5rem !important;
        max-width: 800px;
    }
    .main-title {
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748b;
        font-size: 0.92rem;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Main Title Header
st.markdown('<div class="main-title">📄 PDF Document Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload a document to search its content and ask questions in real time.</div>', unsafe_allow_html=True)

# Application State Setup
st.session_state.setdefault("engine", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("indexed_file", None)

# API Key Validation
api_key = auto_api_key()
if not api_key:
    st.warning("⚠️ No Gemini API key detected in Streamlit Secrets.")
    custom_key = st.text_input("Enter your Gemini API Key:", type="password")
    if custom_key:
        os.environ["GEMINI_API_KEY"] = custom_key
        api_key = custom_key

# SECTION 1: Main Page Document Upload (No Sidebar Required)
st.subheader("1. Select Document")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], label_visibility="collapsed")

if uploaded_file:
    is_new_file = st.session_state.indexed_file != uploaded_file.name
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("⚡ Process & Index Document", type="primary", use_container_width=True, disabled=not is_new_file):
            if not api_key:
                st.error("Please provide a valid Gemini API Key first.")
            else:
                with st.spinner("Reading PDF and building vector search index..."):
                    try:
                        engine = RAGEngine(api_key=api_key)
                        chunk_count = engine.load_pdf(uploaded_file.getvalue(), filename=uploaded_file.name)
                        st.session_state.engine = engine
                        st.session_state.indexed_file = uploaded_file.name
                        st.session_state.messages = []  # Clear chat history on new document
                        st.success(f"Successfully processed **{uploaded_file.name}** ({chunk_count} text chunks).")
                    except Exception as e:
                        st.error(f"Failed to process document: {e}")
    
    with col2:
        if st.button("🗑️ Reset Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

st.divider()

# SECTION 2: Chat & Q&A Workflow
st.subheader("2. Ask Questions")

if not st.session_state.indexed_file:
    st.info("👆 Upload and index a PDF above to begin asking questions.")
else:
    # Display Active Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("🔍 View Retrieved Sources"):
                    for i, src in enumerate(msg["sources"], start=1):
                        st.markdown(f"**Source {i} (Page {src.get('page', 'N/A')}):**")
                        st.caption(src.get("snippet", ""))

    # Bottom Pinned Chat Bar
    if user_prompt := st.chat_input("Ask something about your document..."):
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching document context..."):
                try:
                    result = st.session_state.engine.answer(user_prompt)
                    answer = result.get("answer", "No answer generated.")
                    sources = result.get("sources", [])
                    
                    st.markdown(answer)
                    if sources:
                        with st.expander("🔍 View Retrieved Sources"):
                            for i, src in enumerate(sources, start=1):
                                st.markdown(f"**Source {i} (Page {src.get('page', 'N/A')}):**")
                                st.caption(src.get("snippet", ""))
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer, 
                        "sources": sources
                    })
                except Exception as e:
                    st.error(f"Error generating answer: {e}")

# Sidebar reserved purely for minimal metadata
with st.sidebar:
    st.title("About Project")
    st.write("This app uses Retrieval-Augmented Generation (RAG) to query document text locally using Google Gemini and ChromaDB.")
    if st.session_state.indexed_file:
        st.write(f"**Current File:** {st.session_state.indexed_file}")