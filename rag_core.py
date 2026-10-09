"""Minimal RAG pipeline: load -> chunk -> embed -> FAISS search -> small LLM answer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # ~0.5B params, runs on CPU


@dataclass
class Chunk:
    text: str
    source: str


# ---------- 1. Load & chunk ----------
def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, size: int = 500, overlap: int = 80) -> list[str]:
    words = text.split()
    step = max(size - overlap, 1)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step) if words[i:i + size]]


def load_chunks(folder: str | Path, size: int = 120, overlap: int = 30) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() in {".txt", ".md", ".pdf"}:
            for c in chunk_text(read_file(p), size, overlap):
                chunks.append(Chunk(c, p.name))
    return chunks


# ---------- 2. RAG engine ----------
class MiniRAG:
    def __init__(self, embed_model: str = EMBED_MODEL, llm_model: str = LLM_MODEL):
        self.embedder = SentenceTransformer(embed_model)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        self.llm.eval()
        self.chunks: list[Chunk] = []
        self.index: faiss.Index | None = None

    def build_index(self, chunks: list[Chunk]) -> int:
        if not chunks:
            raise ValueError("No chunks to index. Add .txt/.md/.pdf files first.")
        self.chunks = chunks
        vecs = self.embedder.encode(
            [c.text for c in chunks], normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        self.index = faiss.IndexFlatIP(vecs.shape[1])  # cosine sim (vectors normalized)
        self.index.add(vecs)
        return len(chunks)

    def retrieve(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        if self.index is None:
            raise RuntimeError("Index is empty. Call build_index() first.")
        q = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    @torch.inference_mode()
    def answer(self, question: str, k: int = 3, max_new_tokens: int = 200):
        hits = self.retrieve(question, k)
        context = "\n\n".join(f"[{i+1}] ({c.source}) {c.text}" for i, (c, _) in enumerate(hits))
        messages = [
            {"role": "system", "content": (
                "You are a helpful assistant. Answer ONLY using the context below. "
                "If the answer is not in the context, say you don't know. Be concise.")},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        out = self.llm.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip(), hits
