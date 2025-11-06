# vector_store.py
import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

EMB_MODEL = "all-MiniLM-L6-v2"

class VectorStore:
    def __init__(self, index_path="faiss.index", meta_path="faiss_meta.pkl"):
        self.index_path = index_path
        self.meta_path = meta_path
        self.embedder = SentenceTransformer(EMB_MODEL)
        self.index = None
        self.metadatas = []  # list of dicts corresponding to vectors

        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            self._load()

    def _load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.metadatas = pickle.load(f)

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadatas, f)

    def _ensure_index(self, dim):
        if self.index is None:
            # use IndexFlatIP (inner product) and normalize embeddings for cosine similarity
            self.index = faiss.IndexFlatIP(dim)

    def add_texts(self, texts: List[str], metadatas: List[dict]):
        embs = self.embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        # normalize for cosine
        faiss.normalize_L2(embs)
        dim = embs.shape[1]
        self._ensure_index(dim)
        self.index.add(embs)
        self.metadatas.extend(metadatas)
        self.save()

    def query(self, query_text: str, top_k=5) -> List[Tuple[dict, float, str]]:
        q_emb = self.embedder.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        if self.index is None or self.index.ntotal == 0:
            return []
        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < 0: 
                continue
            meta = self.metadatas[idx]
            results.append((meta, float(score), meta.get("text", "")))
        return results
