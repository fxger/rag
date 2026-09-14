"""BM25 sparse retrieval implementation with Chinese tokenization."""
import json
import logging
import time
import pickle
from typing import Any
from urllib.parse import urlparse

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from src.config.settings import get_settings
from .models import SearchResult

logger = logging.getLogger(__name__)

# Redis keys
BM25_INDEX_KEY = "rag:bm25:index"
BM25_CORPUS_KEY = "rag:bm25:corpus"


class SparseRetriever:
    """Sparse retriever using BM25 algorithm with jieba tokenization.
    
    Provides BM25-based retrieval for keyword matching, particularly
    effective for Chinese text with proper tokenization.
    
    Attributes:
        bm25: BM25Okapi index instance.
        corpus: Original document corpus for retrieval.
        tokenized_corpus: Tokenized version of the corpus.
    """

    def __init__(self):
        """Initialize the sparse retriever."""
        self.settings = get_settings()
        self._init_redis()
        self.bm25: BM25Okapi | None = None
        self.corpus: list[dict[str, Any]] = []
        self.tokenized_corpus: list[list[str]] = []
        self._load_index()

    def _init_redis(self):
        """Initialize synchronous Redis client."""
        try:
            import redis
            
            # Parse Redis URL to get host and port
            redis_url = self.settings.REDIS_URL
            parsed = urlparse(redis_url)
            host = parsed.hostname or "localhost"
            port = parsed.port or 6379
            db = int(parsed.path.lstrip("/") or "0")
            
            self.redis = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=self.settings.REDIS_PASSWORD,
                decode_responses=True,
            )
            self.redis.ping()
            logger.info(f"Connected to Redis at {host}:{port}/{db} for BM25 index")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.redis = None

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text using jieba.
        
        Args:
            text: Input text to tokenize.
            
        Returns:
            List of tokens.
        """
        # Use jieba for Chinese tokenization, filter empty tokens
        tokens = list(jieba.cut(text))
        return [token.strip() for token in tokens if token.strip()]

    def _load_index(self) -> None:
        """Load BM25 index from Redis if available."""
        if not self.redis:
            return
            
        try:
            # Load corpus
            corpus_data = self.redis.get(BM25_CORPUS_KEY)
            if corpus_data:
                self.corpus = json.loads(corpus_data)
                logger.info(f"Loaded corpus with {len(self.corpus)} documents from Redis")
            
            # Load tokenized corpus and rebuild BM25 index
            index_data = self.redis.get(BM25_INDEX_KEY)
            if index_data:
                self.tokenized_corpus = pickle.loads(index_data)
                if self.tokenized_corpus:
                    self.bm25 = BM25Okapi(self.tokenized_corpus)
                    logger.info(f"Loaded BM25 index with {len(self.tokenized_corpus)} documents")
        except Exception as e:
            logger.warning(f"Failed to load BM25 index from Redis: {e}")
            self.bm25 = None
            self.corpus = []
            self.tokenized_corpus = []

    async def build_index(self, documents: list[dict[str, Any]]) -> None:
        """Build BM25 index from documents.
        
        Args:
            documents: List of document dictionaries with 'chunk_id', 'document_id',
                      'content', and optional 'metadata' fields.
        """
        start_time = time.time()
        
        try:
            self.corpus = documents
            
            # Tokenize all documents
            self.tokenized_corpus = [
                self._tokenize(doc.get("content", ""))
                for doc in documents
            ]
            
            # Build BM25 index
            if self.tokenized_corpus:
                self.bm25 = BM25Okapi(self.tokenized_corpus)
                logger.info(f"Built BM25 index with {len(documents)} documents")
            else:
                logger.warning("No documents to index")
                self.bm25 = None
            
            # Persist to Redis
            self._save_index()
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(f"BM25 index built in {elapsed_ms}ms")
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            raise

    def _save_index(self) -> None:
        """Save BM25 index and corpus to Redis."""
        if not self.redis:
            return
            
        try:
            # Save corpus
            self.redis.set(BM25_CORPUS_KEY, json.dumps(self.corpus))
            
            # Save tokenized corpus (for rebuilding index)
            self.redis.set(BM25_INDEX_KEY, pickle.dumps(self.tokenized_corpus))
            
            logger.info("Saved BM25 index to Redis")
        except Exception as e:
            logger.warning(f"Failed to save BM25 index to Redis: {e}")

    async def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search for documents using BM25 algorithm.
        
        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
            
        Returns:
            List of SearchResult objects sorted by BM25 score.
            
        Raises:
            RuntimeError: If BM25 index is not built.
        """
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_index() first.")
        
        start_time = time.time()
        
        try:
            # Tokenize query
            query_tokens = self._tokenize(query)
            
            if not query_tokens:
                logger.warning("Empty query after tokenization")
                return []
            
            # Get BM25 scores
            scores = self.bm25.get_scores(query_tokens)
            
            # Get top-k indices
            if len(scores) == 0:
                return []
            
            # Use argpartition for efficient top-k selection
            k = min(top_k, len(scores))
            top_indices = np.argpartition(scores, -k)[-k:]
            # Sort by score descending
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            
            # Build results
            search_results = []
            for rank, idx in enumerate(top_indices):
                if idx < len(self.corpus):
                    doc = self.corpus[idx]
                    metadata = doc.get("metadata", {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except json.JSONDecodeError:
                            metadata = {}
                    
                    result = SearchResult(
                        chunk_id=doc.get("chunk_id", ""),
                        document_id=doc.get("document_id", ""),
                        content=doc.get("content", ""),
                        score=float(scores[idx]),
                        rank=rank + 1,
                        metadata=metadata,
                        source_type="sparse",
                    )
                    search_results.append(result)
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"BM25 search completed: {len(search_results)} results in {elapsed_ms}ms"
            )
            
            return search_results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            raise

    def clear_index(self) -> None:
        """Clear the BM25 index from memory and Redis."""
        try:
            self.bm25 = None
            self.corpus = []
            self.tokenized_corpus = []
            if self.redis:
                self.redis.delete(BM25_INDEX_KEY, BM25_CORPUS_KEY)
            logger.info("Cleared BM25 index")
        except Exception as e:
            logger.warning(f"Failed to clear BM25 index: {e}")
