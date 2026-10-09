"""Gradio app for the mini RAG. Run: python app.py"""
from pathlib import Path

import gradio as gr

from rag_core import MiniRAG, load_chunks

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

print("Loading models (first run downloads ~1.2 GB)...")
rag = MiniRAG()
n = rag.build_index(load_chunks(DATA_DIR))
print(f"Indexed {n} chunks from {DATA_DIR}/")


def add_files(files):
    """Copy uploaded files into data/ and rebuild the index."""
    if not files:
        return "No files uploaded."
    for f in files:
        src = Path(f if isinstance(f, str) else f.name)
        (DATA_DIR / src.name).write_bytes(src.read_bytes())
    count = rag.build_index(load_chunks(DATA_DIR))
    return f"Index rebuilt: {count} chunks."


def ask(question, k):
    if not question.strip():
        return "Please type a question.", ""
    answer, hits = rag.answer(question, k=int(k))
    sources = "\n\n".join(f"**[{i+1}] {c.source}** (score {s:.2f})\n> {c.text[:300]}..."
                          for i, (c, s) in enumerate(hits))
    return answer, sources


with gr.Blocks(title="Mini RAG") as demo:
    gr.Markdown("# 🔎 Mini RAG\nSmall LLM (Qwen2.5-0.5B) + FAISS retrieval over your documents.")
    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(label="Add documents (.txt / .md / .pdf)", file_count="multiple")
            status = gr.Markdown()
            files.upload(add_files, files, status)
            k = gr.Slider(1, 6, value=3, step=1, label="Chunks to retrieve (k)")
        with gr.Column(scale=2):
            q = gr.Textbox(label="Question", placeholder="What is RAG?")
            btn = gr.Button("Ask", variant="primary")
            a = gr.Textbox(label="Answer", lines=6)
            s = gr.Markdown(label="Retrieved sources")
            btn.click(ask, [q, k], [a, s])
            q.submit(ask, [q, k], [a, s])

if __name__ == "__main__":
    demo.launch()
