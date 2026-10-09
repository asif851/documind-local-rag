# Mini RAG (small LLM + FAISS + Gradio)

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
- Web app:  `python app.py`  → open http://127.0.0.1:7860
- Notebook: `jupyter notebook rag_notebook.ipynb`

## Structure
| File | Purpose |
|---|---|
| `rag_core.py` | Loading, chunking, embeddings, FAISS index, LLM answer |
| `app.py` | Gradio UI (upload docs, ask questions, see sources) |
| `rag_notebook.ipynb` | Step-by-step walkthrough |
| `data/` | Your documents (.txt, .md, .pdf) |

## Customize
- Different LLM: change `LLM_MODEL` in `rag_core.py` (e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`).
- Better answers: raise `k`, adjust chunk `size`/`overlap`, or use a larger model.
- GPU: pass `torch_dtype=torch.float16` and `.to("cuda")` in `MiniRAG`.
