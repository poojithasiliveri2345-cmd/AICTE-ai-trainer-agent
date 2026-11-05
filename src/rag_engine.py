"""
RAG (Retrieval-Augmented Generation) engine using ChromaDB + sentence-transformers.
Indexes the local knowledge base JSON files and retrieves relevant context
to augment IBM Granite prompts.
"""

import json
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


KB_DIR = Path(__file__).parent.parent / "data" / "knowledge_base"
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "interview_knowledge_base")


class RAGEngine:
    """
    Manages the vector store and retrieval pipeline.
    Falls back to keyword-based search if ChromaDB/sentence-transformers are unavailable.
    """

    def __init__(self):
        self._docs: List[Dict[str, Any]] = []
        self._client: Optional[Any] = None
        self._collection: Optional[Any] = None
        self._embedder: Optional[Any] = None
        self._use_vector = CHROMA_AVAILABLE and ST_AVAILABLE
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize(self) -> str:
        """Load knowledge base and build/load vector index. Returns status message."""
        self._docs = self._load_knowledge_base()
        if self._use_vector:
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            self._client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIR,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            if self._collection.count() == 0:
                self._index_documents()
                status = f"Vector index built: {len(self._docs)} documents indexed."
            else:
                status = f"Vector index loaded: {self._collection.count()} documents."
        else:
            status = f"Keyword search mode: {len(self._docs)} documents loaded."
        self._initialized = True
        return status

    # ------------------------------------------------------------------
    # Document loading
    # ------------------------------------------------------------------
    def _load_knowledge_base(self) -> List[Dict[str, Any]]:
        docs = []
        for json_file in KB_DIR.glob("*.json"):
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
            docs.extend(self._flatten(data, source=json_file.stem))
        return docs

    def _flatten(self, data: Any, source: str, path: str = "") -> List[Dict[str, Any]]:
        """Recursively flatten nested JSON into flat document list."""
        documents = []
        if isinstance(data, list):
            for i, item in enumerate(data):
                documents.extend(self._flatten(item, source, f"{path}[{i}]"))
        elif isinstance(data, dict):
            # If the dict has a 'question' or 'scenario' key, treat as terminal document
            text_key = next((k for k in ("question", "scenario", "guideline") if k in data), None)
            if text_key:
                text = self._dict_to_text(data)
                doc_id = hashlib.md5(text.encode()).hexdigest()
                documents.append({
                    "id": doc_id,
                    "text": text,
                    "source": source,
                    "path": path,
                    "metadata": {k: str(v) for k, v in data.items() if isinstance(v, (str, int, float, bool))},
                })
            else:
                for k, v in data.items():
                    documents.extend(self._flatten(v, source, f"{path}/{k}"))
        return documents

    def _dict_to_text(self, d: Dict) -> str:
        """Serialize a knowledge base entry to a searchable text blob."""
        parts = []
        priority_keys = ["question", "scenario", "guideline", "category", "competency",
                         "model_answer", "model_answer_structure", "description", "tips"]
        for k in priority_keys:
            if k in d:
                v = d[k]
                if isinstance(v, list):
                    parts.append(f"{k}: {'; '.join(str(i) for i in v)}")
                else:
                    parts.append(f"{k}: {v}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------
    def _index_documents(self):
        ids, texts, metadatas = [], [], []
        for doc in self._docs:
            ids.append(doc["id"])
            texts.append(doc["text"])
            metadatas.append({**doc["metadata"], "source": doc["source"]})
        # Batch upsert in chunks of 100
        for i in range(0, len(ids), 100):
            embeddings = self._embedder.encode(texts[i : i + 100]).tolist()
            self._collection.upsert(
                ids=ids[i : i + 100],
                documents=texts[i : i + 100],
                embeddings=embeddings,
                metadatas=metadatas[i : i + 100],
            )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Retrieve top-k relevant documents for a query."""
        if not self._initialized:
            self.initialize()
        if self._use_vector and self._collection:
            return self._vector_retrieve(query, top_k, filters)
        return self._keyword_retrieve(query, top_k)

    def _vector_retrieve(self, query: str, top_k: int, filters: Optional[Dict]) -> List[Dict]:
        query_embedding = self._embedder.encode([query]).tolist()
        where = {k: v for k, v in (filters or {}).items() if v}
        kwargs = dict(query_embeddings=query_embedding, n_results=min(top_k, self._collection.count()))
        if where:
            kwargs["where"] = where
        results = self._collection.query(**kwargs)
        docs = []
        for i, doc_text in enumerate(results["documents"][0]):
            docs.append({
                "text": doc_text,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else 0.0,
            })
        return docs

    def _keyword_retrieve(self, query: str, top_k: int) -> List[Dict]:
        query_lower = query.lower()
        keywords = set(query_lower.split())
        scored = []
        for doc in self._docs:
            text_lower = doc["text"].lower()
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"text": d["text"], "metadata": d["metadata"], "distance": 1.0 / (s + 1)}
                for s, d in scored[:top_k]]

    def build_context(self, results: List[Dict]) -> str:
        """Format retrieved documents into a context block for LLM prompts."""
        if not results:
            return "No specific context retrieved."
        parts = []
        for i, r in enumerate(results, 1):
            source = r["metadata"].get("source", "knowledge_base")
            parts.append(f"[Context {i} — Source: {source}]\n{r['text']}")
        return "\n\n---\n\n".join(parts)
