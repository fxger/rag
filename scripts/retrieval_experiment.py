"""Four-way retrieval comparison: dense / sparse / hybrid / hybrid+rerank.

Why a separate script from ``test_retrieval_accuracy.py``: that one carries
its own self-contained implementation (its own Milvus client, its own BM25,
its own fixed 0.7/0.3 weighting) and imports nothing from ``src/`` — so its
numbers describe a baseline, not this project's pipeline.  This script calls
the real components (``src.config.milvus``, ``SparseRetriever``,
``ResultFusion``, ``Reranker``), so the comparison reflects what actually
runs.

Usage:
    .venv\\Scripts\\python.exe scripts/retrieval_experiment.py

Outputs:
    console table + ``retrieval_experiment.json``

Metric definitions (stated explicitly because the corpus has no graded
ground truth — relevance is keyword-based binary):

    hit_rate@k : fraction of queries with >= 1 relevant item in the top k
    MRR@k      : mean of 1/rank of the first relevant item (0 if none)
    nDCG@k     : binary gain, IDCG assumed 1.0 (one relevant item per query),
                 i.e. it equals 1/log2(rank+1) of the first relevant hit
"""
import asyncio
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.milvus import search_vectors
from src.config.settings import get_settings
from src.core.embedding.embedder import EmbeddingGenerator
from src.core.retrieval.fusion import ResultFusion
from src.core.retrieval.models import SearchResult
from src.core.retrieval.reranker import Reranker
from src.core.retrieval.sparse_retriever import SparseRetriever
from test_retrieval_accuracy import TEST_CASES

TOP_K = 5
CANDIDATES = 20          # per-path candidate pool before fusion / rerank
OUTPUT = ROOT / "retrieval_experiment.json"


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def _relevant(content: str, expected: list[str]) -> bool:
    return any(kw and kw in content for kw in expected)


def _score(ranked_contents: list[list[str]], expected: list[list[str]], k: int) -> dict:
    hits = 0
    rr_sum = 0.0
    ndcg_sum = 0.0

    for contents, expect in zip(ranked_contents, expected):
        flags = [_relevant(c, expect) for c in contents[:k]]
        if any(flags):
            hits += 1
            rank = flags.index(True) + 1
            rr_sum += 1.0 / rank
            ndcg_sum += 1.0 / math.log2(rank + 1)

    n = max(len(ranked_contents), 1)
    return {
        "hit_rate": round(hits / n, 4),
        "mrr": round(rr_sum / n, 4),
        "ndcg": round(ndcg_sum / n, 4),
        "hits": hits,
        "total": len(ranked_contents),
    }


def _to_search_result(hit: dict, rank: int, source: str) -> SearchResult:
    return SearchResult(
        chunk_id=str(hit.get("chunk_id", "")),
        document_id=str(hit.get("document_id", "")),
        content=hit.get("content", "") or "",
        score=float(hit.get("score", 0.0) or 0.0),
        rank=rank,
        metadata=hit.get("metadata") or {},
        source_type=source,
    )


# ----------------------------------------------------------------------
# Retrieval configurations
# ----------------------------------------------------------------------
async def main() -> int:
    settings = get_settings()

    # ── corpus (also feeds the BM25 index) ──────────────────────────
    from pymilvus import Collection, connections

    connections.connect(
        "default",
        host=settings.MILVUS_HOST,
        port=str(settings.MILVUS_PORT),
    )
    collection = Collection(settings.MILVUS_COLLECTION)
    collection.load()
    corpus = collection.query(
        expr="",
        output_fields=["chunk_id", "document_id", "content", "metadata"],
        limit=16384,
    )
    print(f"语料库：{len(corpus)} 个分块\n")

    if not corpus:
        print("语料库为空，实验无法进行。")
        return 1

    sparse = SparseRetriever()
    await sparse.build_index(corpus)

    fusion = ResultFusion()
    reranker = Reranker()
    embedder = EmbeddingGenerator()

    async def dense_only(query: str, k: int) -> list[str]:
        emb = await embedder.generate_single(query)
        hits = await asyncio.to_thread(search_vectors, emb, k)
        return [h.get("content", "") or "" for h in hits]

    async def sparse_only(query: str, k: int) -> list[str]:
        results = await sparse.search(query, top_k=k)
        return [r.content for r in results]

    async def _hybrid_fused(query: str) -> list[SearchResult]:
        emb = await embedder.generate_single(query)
        dense_raw = await asyncio.to_thread(search_vectors, emb, CANDIDATES)
        sparse_res = await sparse.search(query, top_k=CANDIDATES)

        dense = [_to_search_result(h, i + 1, "dense") for i, h in enumerate(dense_raw)]
        return fusion.rrf_fusion([dense, sparse_res], k=60)

    async def hybrid(query: str, k: int) -> list[str]:
        return [r.content for r in (await _hybrid_fused(query))[:k]]

    async def hybrid_rerank(query: str, k: int) -> list[str]:
        fused = await _hybrid_fused(query)
        reranked = await reranker.rerank(query, fused[:CANDIDATES], top_n=k)
        return [r.content for r in reranked]

    methods = {
        "dense_only": ("纯稠密向量", dense_only),
        "sparse_only": ("纯稀疏 BM25", sparse_only),
        "hybrid": ("混合检索 (RRF)", hybrid),
        "hybrid_rerank": ("混合 + Cross-Encoder 重排", hybrid_rerank),
    }

    expected = [case["expected"] for case in TEST_CASES]
    report: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus_size": len(corpus),
        "query_count": len(TEST_CASES),
        "top_k": TOP_K,
        "config": {
            "milvus_collection": settings.MILVUS_COLLECTION,
            "embedding_model": settings.QWEN_EMBED_MODEL,
            "fusion": "rrf(k=60)",
            "reranker_model": reranker.model_name,
            "candidates_per_path": CANDIDATES,
        },
        "results": {},
    }

    print(f"{'配置':<28}{'HitRate@5':>12}{'MRR@5':>10}{'nDCG@5':>10}{'耗时':>10}")
    print("-" * 72)

    for key, (label, fn) in methods.items():
        started = time.time()
        ranked: list[list[str]] = []
        for case in TEST_CASES:
            try:
                ranked.append(await fn(case["query"], TOP_K))
            except Exception as exc:  # noqa: BLE001 — one bad query must not abort
                print(f"  ! {key} 查询失败 '{case['query'][:20]}': {type(exc).__name__}: {exc}")
                ranked.append([])

        elapsed = time.time() - started
        stats = _score(ranked, expected, TOP_K)
        stats["elapsed_s"] = round(elapsed, 2)
        stats["label"] = label
        report["results"][key] = stats

        print(
            f"{label:<28}{stats['hit_rate']:>12.4f}{stats['mrr']:>10.4f}"
            f"{stats['ndcg']:>10.4f}{elapsed:>9.1f}s"
        )

    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {OUTPUT.name}")

    # 与基线的差值（相对纯稠密）
    base = report["results"]["dense_only"]
    print("\n相对「纯稠密向量」的增量：")
    for key, stats in report["results"].items():
        if key == "dense_only":
            continue
        print(
            f"  {stats['label']:<28}"
            f"HitRate {stats['hit_rate'] - base['hit_rate']:+.4f}  "
            f"MRR {stats['mrr'] - base['mrr']:+.4f}  "
            f"nDCG {stats['ndcg'] - base['ndcg']:+.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
