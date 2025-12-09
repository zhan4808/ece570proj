"""Retriever model implementations."""
import time
import numpy as np
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import faiss

from .base import BaseRetriever


class EmbeddingRetriever(BaseRetriever):
    """Base class for embedding-based retrievers."""
    
    def __init__(self, model_name: str, display_name: str, corpus: Optional[List[str]] = None):
        super().__init__()
        self.model_name = model_name
        self.display_name = display_name
        self.model = SentenceTransformer(model_name)
        self.corpus = corpus or []
        self.index = None
        self._build_index()
    
    def _build_index(self):
        """Build FAISS index from corpus."""
        if len(self.corpus) == 0:
            # Create dummy index if corpus is empty
            self.corpus = ["Dummy document for retrieval."]
        
        # Encode all documents
        embeddings = self.model.encode(self.corpus, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
    
    def set_corpus(self, corpus: List[str]):
        """Set corpus and rebuild index."""
        self.corpus = corpus
        self._build_index()
    
    def retrieve(self, query: str, k: int = 8) -> List[str]:
        """Retrieve top-k documents."""
        if self.index is None or len(self.corpus) == 0:
            return []
        
        start_time = time.time()
        
        # Encode query
        query_embedding = self.model.encode([query], show_progress_bar=False)
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search
        k = min(k, len(self.corpus))
        distances, indices = self.index.search(query_embedding, k)
        
        # Get documents
        results = [self.corpus[idx] for idx in indices[0]]
        
        latency_ms = (time.time() - start_time) * 1000
        # Local computation cost is negligible
        self._record_query(0.0, latency_ms)
        
        return results
    
    @property
    def name(self) -> str:
        return self.display_name


class MiniLMRetriever(EmbeddingRetriever):
    """MiniLM-L6 retriever."""
    
    def __init__(self, corpus: Optional[List[str]] = None):
        super().__init__(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            display_name="MiniLM-L6",
            corpus=corpus
        )


class BGESmallRetriever(EmbeddingRetriever):
    """BGE-small retriever."""
    
    def __init__(self, corpus: Optional[List[str]] = None):
        super().__init__(
            model_name="BAAI/bge-small-en-v1.5",
            display_name="bge-small-en",
            corpus=corpus
        )


class BGEBaseRetriever(EmbeddingRetriever):
    """BGE-base retriever."""
    
    def __init__(self, corpus: Optional[List[str]] = None):
        super().__init__(
            model_name="BAAI/bge-base-en-v1.5",
            display_name="bge-base-en",
            corpus=corpus
        )

