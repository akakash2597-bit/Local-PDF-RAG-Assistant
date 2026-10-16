# Gemini RAG Document Q&A

Upload a PDF, index it, ask questions. Answers are grounded in the document and
ship with the exact retrieved snippets.

## Run

```bash
streamlit run app.py
```

Then paste your Gemini API key in the sidebar (or put it in `.env`), upload a PDF,
click **Index document**, and start asking.

## Files

- `rag_engine.py` — `PyPDFLoader` → `RecursiveCharacterTextSplitter` (chunk_size=1000, chunk_overlap=200) → `GoogleGenerativeAIEmbeddings` → Chroma → `RetrievalQA` with `ChatGoogleGenerativeAI`.
- `app.py` — Streamlit UI: custom CSS (cards, gradient header, badges, orbit spinner), API-key input, chat with collapsible snippet expanders.

## API key

Resolution order — first one found wins:

1. The sidebar's **Use Custom Gemini API Key** expander (session-only override)
2. `st.secrets['GEMINI_API_KEY']` (or `GOOGLE_API_KEY`)
3. Environment / `.env`: `GOOGLE_API_KEY` or `GEMINI_API_KEY`

With (2) or (3) in place the sidebar shows **API Key Loaded Automatically** and
names the source; the override expander stays collapsed.

`.streamlit/secrets.toml` (copy from `secrets.toml.example`):

```toml
GEMINI_API_KEY = "AIza..."
```

Or a `.env` file in this folder:

```
GOOGLE_API_KEY=AIza...
```

Get a key at https://aistudio.google.com/apikey.

## UI features

- Dark gradient theme, rounded message bubbles, metadata badges (`ChromaDB Active`, model name).
- **Document stats card** after indexing: filename, pages, characters, chunks, chunk size/overlap.
- **Quick questions** under the indexed document: summarize takeaways, explain main concepts, list important terms.
- **Citation cards**: collapsible `st.expander` per answer with page, chunk, character count and the snippet text.
- **Download Discussion (.md)** in the sidebar exports the whole chat, sources included.

## Models — read this before running

The model ids in the original brief are **retired by Google** and return
`404 models/... is not found`.

| Requested | Status | Used instead |
| --- | --- | --- |
| `models/embedding-001` | retired, 404 | `models/gemini-embedding-001` |
| `gemini-1.5-flash` | retired, 404 | `gemini-3.8-flash` |
| `gemini-2.0-flash` | shut down 1 Jun 2026, 404 | `gemini-3.8-flash` |
| `gemini-2.5-flash` | retired, 404 | `gemini-3.8-flash` |

`LLM_MODEL` in `rag_engine.py` is `gemini-3.8-flash`, and `LLM_OPTIONS` lists the
other live Flash models. The sidebar has a dropdown plus a free-text field, so any
live model id works without editing code. Picking a retired id raises a clear error
before the API call instead of a bare 404.

Live model ids as of now: `gemini-3.5-flash`, `gemini-3.6-flash`, `gemini-3.7-flash`,
`gemini-3.8-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`.

## CLI test

```bash
python rag_engine.py path/to/file.pdf "What is this document about?"
```

## Notes

- Chroma is **ephemeral** — re-index after restarting the app.
- Scanned/image-only PDFs have no extractable text; they need OCR first.
- Gemini embeddings are batched per chunk by the client; large PDFs take a while to index.
