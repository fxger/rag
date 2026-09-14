"""Embedding generation client with intelligent caching support."""
import re
import hashlib
import logging
from typing import Optional
from difflib import SequenceMatcher

import httpx
from cachetools import TTLCache

logger = logging.getLogger(__name__)

# Qwen Embedding 配置
QWEN_EMBED_URL = "http://192.168.86.132:8000/v1/embeddings"
QWEN_EMBED_MODEL = "qwen-embedding"
EMBEDDING_DIMENSION = 1024


def normalize_query(text: str) -> str:
    """标准化查询文本 (去除标点、停用词)"""
    # 转小写
    text = text.lower().strip()
    # 去除标点符号
    text = re.sub(r'[？?！!。，,、\.\s\-\_\~\`\@\#\$\%\^\&\*\(\)\[\]\{\}\<\>]+', '', text)
    # 去除常见停用词
    stopwords = ['的', '了', '是', '在', '有', '吗', '呢', '吧', '啊', '么', '请', '告诉', '我']
    for word in stopwords:
        text = text.replace(word, '')
    return text


class IntelligentEmbeddingCache:
    """智能 Embedding 缓存 (标准化 + 模糊匹配)"""
    
    def __init__(self, l1_maxsize: int = 500, l1_ttl: int = 300, l2_ttl: int = 3600, fuzzy_threshold: float = 0.85):
        # L1: 进程内缓存 (标准化 key)
        self.l1 = TTLCache(maxsize=l1_maxsize, ttl=l1_ttl)
        self.l2_ttl = l2_ttl
        self.prefix = "rag:emb:"
        self.fuzzy_threshold = fuzzy_threshold
        
        # 模糊匹配索引: {normalized_key: (original_text, embedding)}
        self._fuzzy_index: dict[str, tuple[str, list[float]]] = {}
        
        self._hits = {"l1_exact": 0, "l1_fuzzy": 0, "l2": 0, "miss": 0}
    
    def _make_key(self, text: str) -> str:
        """生成缓存 key (基于标准化后的文本)"""
        normalized = normalize_query(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    
    def _fuzzy_search(self, text: str) -> Optional[list[float]]:
        """模糊搜索缓存"""
        normalized = normalize_query(text)
        
        for key, (original, embedding) in self._fuzzy_index.items():
            similarity = SequenceMatcher(None, normalized, key).ratio()
            if similarity >= self.fuzzy_threshold:
                logger.debug("Fuzzy cache hit: %.2f%% similar to '%s'", similarity * 100, original[:20])
                return embedding
        
        return None
    
    async def get(self, text: str) -> Optional[list[float]]:
        """从缓存获取 embedding"""
        # Step 1: L1 精确匹配
        key = self._make_key(text)
        if key in self.l1:
            self._hits["l1_exact"] += 1
            return self.l1[key]
        
        # Step 2: L1 模糊匹配
        fuzzy_result = self._fuzzy_search(text)
        if fuzzy_result:
            self._hits["l1_fuzzy"] += 1
            return fuzzy_result
        
        # Step 3: L2 Redis 缓存
        try:
            from src.config.redis import get_cache_redis
            import json
            redis = await get_cache_redis()
            raw = await redis.get(f"{self.prefix}{key}")
            if raw is not None:
                value = json.loads(raw)
                self.l1[key] = value
                self._hits["l2"] += 1
                return value
        except Exception as e:
            logger.warning("L2 cache read error: %s", e)
        
        self._hits["miss"] += 1
        return None
    
    async def set(self, text: str, embedding: list[float], ttl: int | None = None):
        """写入缓存"""
        effective_ttl = ttl or self.l2_ttl
        key = self._make_key(text)
        normalized = normalize_query(text)
        
        # 写入 L1
        self.l1[key] = embedding
        
        # 更新模糊索引
        self._fuzzy_index[normalized] = (text, embedding)
        
        # 写入 L2 Redis
        try:
            from src.config.redis import get_cache_redis
            import json
            redis = await get_cache_redis()
            await redis.set(
                f"{self.prefix}{key}",
                json.dumps(embedding, separators=(",", ":")),
                ex=effective_ttl,
            )
        except Exception as e:
            logger.warning("L2 cache write error: %s", e)
    
    @property
    def stats(self) -> dict:
        total = sum(self._hits.values())
        return {
            "l1_exact_hits": self._hits["l1_exact"],
            "l1_fuzzy_hits": self._hits["l1_fuzzy"],
            "l2_hits": self._hits["l2"],
            "misses": self._hits["miss"],
            "hit_rate": (self._hits["l1_exact"] + self._hits["l1_fuzzy"] + self._hits["l2"]) / max(total, 1),
            "fuzzy_index_size": len(self._fuzzy_index),
        }


class EmbeddingGenerator:
    """Generates vector embeddings using Qwen embedding API."""

    def __init__(self):
        self.model = QWEN_EMBED_MODEL
        self.dimension = EMBEDDING_DIMENSION
        self._cache = IntelligentEmbeddingCache()

    def _get_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=30.0)

    async def generate(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        async with self._get_client() as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    response = await client.post(
                        QWEN_EMBED_URL,
                        json={"input": batch, "model": self.model},
                    )
                    result = response.json()
                    batch_embeddings = [item["embedding"] for item in result["data"]]
                    all_embeddings.extend(batch_embeddings)
                except Exception as e:
                    logger.error("Embedding batch failed: %s", e)
                    raise

        return all_embeddings

    async def generate_single(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        async with self._get_client() as client:
            response = await client.post(
                QWEN_EMBED_URL,
                json={"input": text, "model": self.model},
            )
            result = response.json()
            return result["data"][0]["embedding"]

    async def generate_with_cache(self, text: str) -> list[float]:
        """带智能缓存的 embedding 生成"""
        # 尝试缓存 (精确 + 模糊)
        cached = await self._cache.get(text)
        if cached:
            logger.debug("Embedding cache hit for: %s...", text[:20])
            return cached
        
        # 缓存未命中, 调用 API
        embedding = await self.generate_single(text)
        
        # 写入缓存
        await self._cache.set(text, embedding)
        
        return embedding
    
    @property
    def cache_stats(self) -> dict:
        return self._cache.stats

    @staticmethod
    def text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


# 全局单例
_embedding_generator: Optional[EmbeddingGenerator] = None


def get_embedding_generator() -> EmbeddingGenerator:
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator
