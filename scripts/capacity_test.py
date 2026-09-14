"""Capacity / latency curve test — how retrieval latency scales with corpus size.

Answers a question the resume asserts but nothing measured: does retrieval
still hold up at 10k chunks?  The script builds a **separate** collection
(``enterprise_rag_capacity``) with the same schema and index parameters as
production, fills it in stages, and measures search latency at each stage so
the growth curve is visible rather than assumed.  The production collection
is never touched.

Usage:
    .venv\\Scripts\\python.exe scripts/capacity_test.py                 # full run
    .venv\\Scripts\\python.exe scripts/capacity_test.py --sizes 500,2000
    .venv\\Scripts\\python.exe scripts/capacity_test.py --cleanup       # drop test collection

Output: console table + ``capacity_report.json``
"""
import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.milvus import get_milvus_client, search_vectors
from src.config.settings import get_settings
from src.core.embedding.embedder import EmbeddingGenerator

TEST_COLLECTION = "enterprise_rag_capacity"
DEFAULT_SIZES = [500, 2000, 5000, 10000]
QUERIES_PER_STAGE = 30
# Qwen embedding 服务对单次请求体量敏感：100 条 × 200 字会触发 30s 读超时，
# 20 条是实测稳定的批量。
BATCH = 20

_TOPICS = [
    "混合检索", "向量索引", "多级缓存", "并发控制", "指标采集",
    "文档分块", "结果重排", "连接池", "限流熔断", "上下文组装",
]
_TEMPLATES = [
    "{t}的核心目标是在可控延迟下提高召回质量，通常需要同时权衡容量、成本与稳定性。",
    "在高并发场景中，{t}往往成为瓶颈，需要通过批处理、缓存与背压机制来缓解。",
    "{t}的常见失效模式包括下游超时、连接耗尽与缓存击穿，需要配套的降级策略。",
    "评测{t}时应当区分检索阶段与生成阶段的耗时，二者的优化手段完全不同。",
    "为{t}建立可观测性时，除了耗时直方图，还应关注排队长度与错误率的变化趋势。",
]


def _make_corpus(n: int, seed: int = 20260914) -> list[str]:
    """Deterministic synthetic Chinese corpus (no MinIO needed)."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        parts = [rng.choice(_TEMPLATES).format(t=rng.choice(_TOPICS)) for _ in range(rng.randint(2, 5))]
        out.append(f"[{i}] " + "".join(parts))
    return out


def _random_vector(dim: int, rng: random.Random) -> list[float]:
    """Unit-norm synthetic vector.

    Used when ``--synthetic`` is passed (e.g. the embedding service is
    unavailable).  Search *latency* depends on the index, the dimension and
    the corpus size — not on where the vectors came from — so the latency
    curve stays meaningful, even though the retrieved content is not.
    """
    raw = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct), len(ordered) - 1)
    return ordered[idx]


async def _build_collection(client, dim: int) -> None:
    if TEST_COLLECTION in client.list_collections():
        return

    from pymilvus import CollectionSchema, DataType, FieldSchema

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=8192),
    ]
    client.create_collection(
        collection_name=TEST_COLLECTION,
        schema=CollectionSchema(fields, description="Capacity test corpus"),
    )
    params = client.prepare_index_params()
    params.add_index(
        field_name="embedding",
        index_type="IVF_FLAT",
        metric_type="COSINE",
        params={"nlist": 1024},
    )
    client.create_index(TEST_COLLECTION, params)
    print(f"  已创建 {TEST_COLLECTION}（schema / 索引与生产一致）")


async def _measure(
    client,
    embedder: EmbeddingGenerator,
    queries: list[str],
    top_k: int = 5,
    synthetic: bool = False,
    rng: random.Random | None = None,
) -> dict:
    """Search latency for a set of queries (embedding time excluded)."""
    if synthetic:
        dim = get_settings().MILVUS_DIMENSION
        gen = rng or random.Random(0)
        embeddings = [_random_vector(dim, gen) for _ in queries]
    else:
        embeddings = await embedder.generate(queries)
    latencies: list[float] = []

    for emb in embeddings:
        t0 = time.perf_counter()
        # Reuse the project's own search wrapper (it already builds the
        # request correctly for MilvusClient, which rejects `param=`).
        await asyncio.to_thread(search_vectors, emb, top_k, TEST_COLLECTION)
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "p50_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(_percentile(latencies, 0.95), 2),
        "p99_ms": round(_percentile(latencies, 0.99), 2),
        "mean_ms": round(statistics.fmean(latencies), 2),
        "samples": len(latencies),
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES))
    parser.add_argument("--cleanup", action="store_true", help="drop the test collection and exit")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="use synthetic unit vectors instead of calling the embedding service",
    )
    args = parser.parse_args()

    settings = get_settings()
    client = get_milvus_client()

    if args.cleanup:
        if TEST_COLLECTION in client.list_collections():
            client.drop_collection(TEST_COLLECTION)
            print(f"已删除 {TEST_COLLECTION}")
        else:
            print(f"{TEST_COLLECTION} 不存在")
        return 0

    sizes = sorted(int(s) for s in args.sizes.split(",") if s.strip())
    dim = settings.MILVUS_DIMENSION
    embedder = EmbeddingGenerator()

    await _build_collection(client, dim)

    queries = [
        "混合检索如何提高召回",
        "缓存击穿怎么处理",
        "检索耗时如何观测",
        "连接池耗尽的表现",
        "重排阶段为什么慢",
    ]

    report: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "collection": TEST_COLLECTION,
        "synthetic_vectors": args.synthetic,
        "index": {"type": "IVF_FLAT", "metric": "COSINE", "nlist": 1024, "nprobe": 16},
        "embedding_dim": dim,
        "stages": [],
    }

    inserted = 0
    corpus = _make_corpus(max(sizes))

    print(f"\n{'语料规模':>10}{'P50(ms)':>10}{'P95(ms)':>10}{'P99(ms)':>10}{'均值(ms)':>11}{'灌库耗时':>11}")
    print("-" * 66)

    for size in sizes:
        if size <= inserted:
            continue

        stage_started = time.perf_counter()
        batch = corpus[inserted:size]
        rng = random.Random(20260914)
        for i in range(0, len(batch), BATCH):
            piece = batch[i:i + BATCH]
            if args.synthetic:
                vectors = [_random_vector(dim, rng) for _ in piece]
            else:
                try:
                    vectors = await embedder.generate(piece)
                except Exception as exc:  # noqa: BLE001 — a slow batch must not abort the run
                    print(
                        f"  ! 嵌入服务不可用（{type(exc).__name__}），本批次改用合成向量"
                        " —— 检索结果无语义，但索引/延迟仍可比"
                    )
                    vectors = [_random_vector(dim, rng) for _ in piece]
            client.insert(
                collection_name=TEST_COLLECTION,
                data=[
                    {
                        "document_id": f"cap-{inserted + i + j}",
                        "chunk_id": f"cap-{inserted + i + j}",
                        "content": text,
                        "embedding": vec,
                        "metadata": '{"capacity_test": true}',
                    }
                    for j, (text, vec) in enumerate(zip(piece, vectors))
                ],
            )
        # 新插入的数据需要重新加载才可检索
        client.load_collection(TEST_COLLECTION)
        ingest_s = time.perf_counter() - stage_started
        inserted = size

        stats = await _measure(client, embedder, queries, synthetic=args.synthetic)
        stats.update({"corpus_size": size, "ingest_s": round(ingest_s, 1)})
        report["stages"].append(stats)

        print(
            f"{size:>10}{stats['p50_ms']:>10.2f}{stats['p95_ms']:>10.2f}"
            f"{stats['p99_ms']:>10.2f}{stats['mean_ms']:>11.2f}{ingest_s:>10.1f}s"
        )

    out = ROOT / "capacity_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if len(report["stages"]) >= 2:
        first, last = report["stages"][0], report["stages"][-1]
        growth = last["corpus_size"] / max(first["corpus_size"], 1)
        latency_ratio = last["p95_ms"] / max(first["p95_ms"], 0.01)
        print(
            f"\n语料从 {first['corpus_size']} 增至 {last['corpus_size']}（{growth:.0f}×），"
            f"P95 从 {first['p95_ms']}ms 变为 {last['p95_ms']}ms（{latency_ratio:.2f}×）"
        )

    print(f"报告已写入 {out.name}")
    print(f"清理测试数据：python scripts/capacity_test.py --cleanup")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
