"""Retrieval comparison: dense / sparse / hybrid / hybrid + rerank (N models).

Why a separate script from ``test_retrieval_accuracy.py``: that one carries
its own self-contained implementation (its own Milvus client, its own BM25,
its own fixed 0.7/0.3 weighting) and imports nothing from ``src/`` — so its
numbers describe a baseline, not this project's pipeline.  This script calls
the real components (``src.config.milvus``, ``SparseRetriever``,
``ResultFusion``, ``Reranker``), so the comparison reflects what actually
runs.

Design note — the rerank stage is a *controlled* comparison.  Fusion is
computed once per query and the resulting candidate list is cached; every
reranker then re-orders the exact same inputs.  Without that, differences
between rerankers would be polluted by retrieval noise and by re-embedding
the query each time (the embedding service is slow and non-deterministic at
the millisecond level).

Usage:
    .venv\\Scripts\\python.exe scripts/retrieval_experiment.py
    .venv\\Scripts\\python.exe scripts/retrieval_experiment.py \\
        --rerankers "cross-encoder/ms-marco-MiniLM-L-6-v2,BAAI/bge-reranker-base" \\
        --output retrieval_experiment_multilingual.json

Outputs:
    console table + JSON report (default ``retrieval_experiment.json``)

Metric definitions (stated explicitly because the corpus has no graded
ground truth — relevance is keyword-based binary):

    hit_rate@k : fraction of queries with >= 1 relevant item in the top k
    MRR@k      : mean of 1/rank of the first relevant item (0 if none)
    nDCG@k     : binary gain, IDCG assumed 1.0 (one relevant item per query),
                 i.e. it equals 1/log2(rank+1) of the first relevant hit
"""
import argparse
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

DEFAULT_RERANKERS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "BAAI/bge-reranker-base",
]


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def _relevant(content: str, expected: list[str]) -> bool:
    return any(kw and kw in content for kw in expected)


def _first_relevant_rank(contents: list[str], expect: list[str], k: int) -> int | None:
    """1-based rank of the first relevant item, or None if none in top-k."""
    for i, c in enumerate(contents[:k]):
        if _relevant(c, expect):
            return i + 1
    return None


def _score(ranked_contents: list[list[str]], expected: list[list[str]], k: int) -> dict:
    ranks = [_first_relevant_rank(c, e, k) for c, e in zip(ranked_contents, expected)]
    found = [r for r in ranks if r is not None]

    n = max(len(ranked_contents), 1)
    return {
        "hit_rate": round(len(found) / n, 4),
        "mrr": round(sum(1.0 / r for r in found) / n, 4),
        "ndcg": round(sum(1.0 / math.log2(r + 1) for r in found) / n, 4),
        "hits": len(found),
        "total": len(ranked_contents),
        "ranks": ranks,
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


def _short(model_name: str) -> str:
    return model_name.split("/")[-1]


# ----------------------------------------------------------------------
# Retrieval configurations
# ----------------------------------------------------------------------
async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rerankers",
        default=",".join(DEFAULT_RERANKERS),
        help="逗号分隔的 Cross-Encoder 模型名，逐个与同一候选集对照",
    )
    ap.add_argument(
        "--output",
        default="retrieval_experiment.json",
        help="结果 JSON 的文件名（写入项目根目录）",
    )
    args = ap.parse_args()
    reranker_names = [m.strip() for m in args.rerankers.split(",") if m.strip()]
    output = ROOT / args.output

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
    print(f"语料库：{len(corpus)} 个分块")
    print(f"查询数：{len(TEST_CASES)}   候选池：{CANDIDATES}   截断：top-{TOP_K}")
    print(f"重排模型：{', '.join(_short(n) for n in reranker_names)}\n")

    if not corpus:
        print("语料库为空，实验无法进行。")
        return 1

    sparse = SparseRetriever()
    await sparse.build_index(corpus)

    fusion = ResultFusion()
    embedder = EmbeddingGenerator()

    async def dense_only(query: str, k: int) -> list[str]:
        emb = await embedder.generate_single(query)
        hits = await asyncio.to_thread(search_vectors, emb, k)
        return [h.get("content", "") or "" for h in hits]

    async def sparse_only(query: str, k: int) -> list[str]:
        results = await sparse.search(query, top_k=k)
        return [r.content for r in results]

    # ── fusion is computed once per query and cached ────────────────
    fused_cache: dict[str, list[SearchResult]] = {}

    async def _fused(query: str) -> list[SearchResult]:
        if query not in fused_cache:
            emb = await embedder.generate_single(query)
            dense_raw = await asyncio.to_thread(search_vectors, emb, CANDIDATES)
            sparse_res = await sparse.search(query, top_k=CANDIDATES)
            dense = [
                _to_search_result(h, i + 1, "dense") for i, h in enumerate(dense_raw)
            ]
            fused_cache[query] = fusion.rrf_fusion([dense, sparse_res], k=60)
        return fused_cache[query]

    async def hybrid(query: str, k: int) -> list[str]:
        return [r.content for r in (await _fused(query))[:k]]

    def make_rerank(reranker: Reranker):
        async def _run(query: str, k: int) -> list[str]:
            candidates = (await _fused(query))[:CANDIDATES]
            reranked = await reranker.rerank(query, candidates, top_n=k)
            return [r.content for r in reranked]

        return _run

    methods: dict[str, tuple[str, object]] = {
        "dense_only": ("纯稠密向量", dense_only),
        "sparse_only": ("纯稀疏 BM25", sparse_only),
        "hybrid": ("混合检索 (RRF)", hybrid),
    }
    reranker_meta: dict[str, dict] = {}
    loaded_rerankers: list[Reranker] = []

    for name in reranker_names:
        key = f"hybrid_rerank__{_short(name)}"
        methods[key] = (f"混合 + 重排 ({_short(name)})", make_rerank(Reranker(model_name=name)))
        # Load explicitly so a download/offline failure is reported up-front
        # rather than surfacing as 15 identical per-query warnings.
        probe = Reranker(model_name=name)
        started = time.time()
        available = await asyncio.to_thread(probe._ensure_model_loaded)
        load_s = round(time.time() - started, 2)
        reranker_meta[name] = {"available": bool(available), "load_s": load_s}
        if not available:
            print(f"  ! {_short(name)} 未能加载（离线且未缓存？），其对照行将等同混合检索")
        else:
            print(f"  · {_short(name)} 加载完成于 {load_s}s")
        loaded_rerankers.append(probe)
    print()

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
            "reranker_models": reranker_names,
            "reranker_availability": reranker_meta,
            "candidates_per_path": CANDIDATES,
        },
        "results": {},
    }

    print(f"{'配置':<38}{'HitRate@5':>12}{'MRR@5':>10}{'nDCG@5':>10}{'耗时':>10}")
    print("-" * 82)

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
            f"{label:<38}{stats['hit_rate']:>12.4f}{stats['mrr']:>10.4f}"
            f"{stats['ndcg']:>10.4f}{elapsed:>9.1f}s"
        )

    report["per_query"] = {
        case["query"]: {
            "expected": case["expected"],
            **{
                key: report["results"][key]["ranks"][i]
                for key in methods
            },
        }
        for i, case in enumerate(TEST_CASES)
    }

    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已写入 {output.name}")

    # 与基线的差值（相对纯稠密）
    base = report["results"]["dense_only"]
    print("\n相对「纯稠密向量」的增量：")
    for key, stats in report["results"].items():
        if key == "dense_only":
            continue
        print(
            f"  {stats['label']:<38}"
            f"HitRate {stats['hit_rate'] - base['hit_rate']:+.4f}  "
            f"MRR {stats['mrr'] - base['mrr']:+.4f}  "
            f"nDCG {stats['ndcg'] - base['ndcg']:+.4f}"
        )

    hybrid_stats = report["results"]["hybrid"]
    print("\n相对「混合检索 (RRF)」的增量（重排的真实效果）：")
    for key, stats in report["results"].items():
        if not key.startswith("hybrid_rerank"):
            continue
        print(
            f"  {stats['label']:<38}"
            f"HitRate {stats['hit_rate'] - hybrid_stats['hit_rate']:+.4f}  "
            f"MRR {stats['mrr'] - hybrid_stats['mrr']:+.4f}  "
            f"nDCG {stats['ndcg'] - hybrid_stats['ndcg']:+.4f}"
        )

    # 逐查询胜负：判断整体提升是否由个别查询驱动
    print("\n逐查询胜负（相对「混合检索 (RRF)」，比较首个相关项的名次）：")
    base_ranks = hybrid_stats["ranks"]
    for key, stats in report["results"].items():
        if not key.startswith("hybrid_rerank"):
            continue
        win = loss = tie = 0
        for a, b in zip(base_ranks, stats["ranks"]):
            av = a if a is not None else 10**6   # 未命中排到最后
            bv = b if b is not None else 10**6
            if bv < av:
                win += 1
            elif bv > av:
                loss += 1
            else:
                tie += 1
        print(f"  {stats['label']:<38} 改善 {win:>2} · 变差 {loss:>2} · 持平 {tie:>2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
