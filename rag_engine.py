"""
rag_engine.py
-------------
Core engine for a Local RAG Document Q&A app powered by Google Gemini.

Pipeline:
    PDF file -> PyPDFLoader -> RecursiveCharacterTextSplitter
    (chunk_size=1000, chunk_overlap=200) -> GoogleGenerativeAIEmbeddings
    -> Chroma vector store -> RetrievalQA chain -> ChatGoogleGenerativeAI

Public API
----------
    RAGEngine(api_key=None, embedding_model=..., llm_model=..., k=4)
    engine.load_pdf(file_bytes, filename)   -> int   (# chunks indexed)
    engine.answer(question)                 -> {"answer": str, "sources": [dict]}

The API key is read from the argument first, then GOOGLE_API_KEY /
GEMINI_API_KEY in the environment (.env).
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DEFAULT_K = 4

# Live Gemini API model ids. Google has retired everything up to and including
# the 2.x line: models/embedding-001, gemini-1.5-flash and gemini-2.0-flash all
# return 404 now (Gemini 2.0 shut down 1 June 2026). See README.
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-3.8-flash"

# Other live LLMs you can pass to RAGEngine(llm_model=...) or pick in the sidebar.
# Fast/cheap first, strongest last.
LLM_OPTIONS = [
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

# Retired ids, kept only so the error is recognisable if someone selects them.
RETIRED_LLM_MODELS = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash"]
RETIRED_EMBEDDING_MODELS = ["models/embedding-001", "models/text-embedding-004"]

API_KEY_ENV_VARS = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
SECRET_KEY_NAME = "GEMINI_API_KEY"

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant answering questions about the user's "
        "document.\n"
        "Use ONLY the context below to answer. If the answer is not in the "
        "context, say plainly that the document does not contain it — do not "
        "invent anything.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)


# --------------------------------------------------------------------------- #
# API key
# --------------------------------------------------------------------------- #

def resolve_api_key(api_key: Optional[str] = None) -> str:
    """
    Return a Gemini API key, in priority order:

      1. the explicit argument (the sidebar's custom-key override)
      2. st.secrets['GEMINI_API_KEY'] (or GOOGLE_API_KEY)
      3. the environment / .env

    The Streamlit import is optional, so this module stays usable as a CLI.
    """
    if api_key and api_key.strip():
        return api_key.strip()

    try:  # Streamlit secrets, when running inside the app
        import streamlit as st

        for name in (SECRET_KEY_NAME, "GOOGLE_API_KEY"):
            try:
                value = st.secrets.get(name)
            except Exception:  # no secrets file configured
                value = None
            if isinstance(value, str) and value.strip():
                return value.strip()
    except ImportError:
        pass

    for var in API_KEY_ENV_VARS:
        value = os.getenv(var)
        if value and value.strip():
            return value.strip()

    raise RuntimeError(
        "No Gemini API key found. Add GEMINI_API_KEY to .streamlit/secrets.toml, "
        "put GOOGLE_API_KEY=... in a .env file, or paste a key under "
        "'Use Custom Gemini API Key' in the sidebar."
    )


def key_source() -> Optional[str]:
    """Where the key would come from automatically, or None. For UI badges."""
    try:
        import streamlit as st

        for name in (SECRET_KEY_NAME, "GOOGLE_API_KEY"):
            try:
                value = st.secrets.get(name)
            except Exception:
                value = None
            if isinstance(value, str) and value.strip():
                return f"st.secrets · {name}"
    except ImportError:
        pass

    for var in API_KEY_ENV_VARS:
        if (os.getenv(var) or "").strip():
            return f"environment · {var}"
    return None


def auto_api_key() -> Optional[str]:
    """The key available without user input, or None."""
    try:
        return resolve_api_key(None)
    except RuntimeError:
        return None


# --------------------------------------------------------------------------- #
# PDF loading and chunking
# --------------------------------------------------------------------------- #

def load_pdf_documents(file_bytes: bytes, filename: str = "document.pdf") -> List[Document]:
    """Parse a PDF with PyPDFLoader (writes bytes to a temp file first)."""
    suffix = Path(filename).suffix or ".pdf"
    tmp_dir = os.getenv("TMPDIR") or tempfile.gettempdir()
    tmp_path = Path(tempfile.mkstemp(suffix=suffix, dir=tmp_dir)[1])
    tmp_path.write_bytes(file_bytes)

    try:
        documents = PyPDFLoader(str(tmp_path)).load()
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass

    documents = [d for d in documents if d.page_content and d.page_content.strip()]
    if not documents:
        raise ValueError(
            "No extractable text found. The PDF may be scanned images — "
            "it would need OCR first."
        )
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """Chunk pages with RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #

@dataclass
class RAGEngine:
    """Holds the vector store + QA chain for one indexed document."""

    api_key: Optional[str] = None
    embedding_model: str = EMBEDDING_MODEL
    llm_model: str = LLM_MODEL
    k: int = DEFAULT_K
    temperature: float = 0.0

    vectorstore: Optional[Chroma] = field(default=None, init=False, repr=False)
    chain: Optional[Any] = field(default=None, init=False, repr=False)
    chunks: List[Document] = field(default_factory=list, init=False, repr=False)
    filename: Optional[str] = field(default=None, init=False)
    stats: Dict[str, Any] = field(default_factory=dict, init=False)

    # -- state ------------------------------------------------------------- #

    @property
    def is_ready(self) -> bool:
        return self.chain is not None

    def reset(self) -> None:
        """Drop the current index so a new PDF can be loaded."""
        self.vectorstore = None
        self.chain = None
        self.chunks = []
        self.filename = None
        self.stats = {}

    # -- build ------------------------------------------------------------- #

    def _embeddings(self):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=self.embedding_model,
            google_api_key=resolve_api_key(self.api_key),
        )

    def _llm(self):
        from langchain_google_genai import ChatGoogleGenerativeAI

        if self.llm_model in RETIRED_LLM_MODELS:
            raise RuntimeError(
                f"{self.llm_model} is retired by Google and returns 404. "
                f"Use one of: {', '.join(LLM_OPTIONS)}"
            )

        return ChatGoogleGenerativeAI(
            model=self.llm_model,
            google_api_key=resolve_api_key(self.api_key),
            temperature=self.temperature,
        )

    def load_pdf(self, file_bytes: bytes, filename: str = "document.pdf") -> int:
        """Chunk + embed a PDF and build the RetrievalQA chain. Returns chunk count."""
        self.reset()

        pages = load_pdf_documents(file_bytes, filename)
        chunks = split_documents(pages)

        for i, chunk in enumerate(chunks):
            chunk.metadata.update({"source": filename, "chunk": i})

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self._embeddings(),
            collection_name="gemini_rag_documents",
        )
        self.chunks = chunks
        self.filename = filename

        pages_with_text = len(pages)
        page_numbers = [p.metadata.get("page") for p in pages]
        max_page = max((p for p in page_numbers if isinstance(p, int)), default=None)

        self.stats = {
            "filename": filename,
            "pages": max_page + 1 if isinstance(max_page, int) else pages_with_text,
            "pages_with_text": pages_with_text,
            "characters": sum(len(d.page_content) for d in pages),
            "chunks": len(chunks),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }

        self.chain = RetrievalQA.from_chain_type(
            llm=self._llm(),
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.k},
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT},
        )
        return len(chunks)

    # -- query ------------------------------------------------------------- #

    def answer(self, question: str) -> Dict[str, Any]:
        """Run the chain and return the answer plus de-duplicated source snippets."""
        if not self.is_ready:
            raise RuntimeError("No document loaded. Call load_pdf() first.")
        if not question or not question.strip():
            raise ValueError("Question is empty.")

        result = self.chain.invoke({"query": question.strip()})
        answer = result.get("result", "")
        if not isinstance(answer, str):
            answer = getattr(answer, "content", str(answer))

        seen, sources = set(), []
        for doc in result.get("source_documents", []) or []:
            meta = doc.metadata or {}
            key = (meta.get("source"), meta.get("page"), meta.get("chunk"))
            if key in seen:
                continue
            seen.add(key)
            page = meta.get("page")
            sources.append({
                # PyPDFLoader pages are 0-based; show humans 1-based pages.
                "page": (page + 1) if isinstance(page, int) else page,
                "chunk": meta.get("chunk"),
                "source": meta.get("source"),
                "snippet": doc.page_content.strip(),
                "characters": len(doc.page_content),
            })

        return {"answer": answer.strip(), "sources": sources}


# --------------------------------------------------------------------------- #
# CLI:  python rag_engine.py file.pdf "your question"
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print('usage: python rag_engine.py <file.pdf> "<question>"')
        raise SystemExit(1)

    pdf_path, question = sys.argv[1], sys.argv[2]
    with open(pdf_path, "rb") as fh:
        engine = RAGEngine()
        count = engine.load_pdf(fh.read(), filename=os.path.basename(pdf_path))

    print(f"indexed {count} chunks")
    out = engine.answer(question)
    print("\nANSWER:\n", out["answer"])
    for s in out["sources"]:
        print(f"\n-- page {s['page']} (chunk {s['chunk']}) --\n{s['snippet'][:300]}...")
