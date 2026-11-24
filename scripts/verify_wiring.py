"""验收脚本：确认 6 个「实现完整但从未被调用」的模块真的接进了链路。

背景
----
这些模块此前在仓库里是**死的**：代码完整、测试可过，但请求路径上没有任何一处
调用它们。只做静态检查（import 成功）无法区分「接进来了」和「只是能被导入」，
所以这里用假组件驱动 RAGService 跑一遍，断言每个模块**确实被调用**。

不涉及真实网络：检索器、生成器、缓存、重写器全部注入假实现；
Redis / PostgreSQL / Milvus 不可达时相关分支应降级而非抛错 —— 这也一并验证。

用法::

    python scripts/verify_wiring.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.cache.warmer import CacheWarmer
from src.core.generation.context_builder import ContextBuilder
from src.core.retrieval.models import QueryAnalysis, RetrievalResult, SearchResult
from src.infrastructure.pool.connection_pool import pool_manager
from src.services.rag_service import RAGService, SearchCache

PASS, FAIL = "PASS", "FAIL"
_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, ok, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))


# ── 假组件 ──────────────────────────────────────────────────────
class FakeEmbeddingCache:
    def __init__(self):
        self.l1: dict = {}

    async def get(self, text):
        return self.l1.get(text)

    async def set(self, text, embedding):
        self.l1[text] = embedding


class FakeEmbedder:
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.calls = 0

    async def generate_single(self, text: str) -> list[float]:
        self.calls += 1
        return [0.1] * self.dim


class FakeRetriever:
    def __init__(self):
        self.calls = 0

    async def retrieve(self, query, top_k=5, embedding=None, fusion_method=None, rerank=False):
        self.calls += 1
        results = [
            SearchResult(
                chunk_id=f"c{i}",
                document_id="d1",
                content=f"关于「{query}」的第 {i} 段内容",
                score=1.0 - i * 0.1,
                rank=i,
                metadata={},
                source_type="dense",
            )
            for i in range(1, 3)
        ]
        return RetrievalResult(results=results, query_analysis=QueryAnalysis(
            original_query=query,
            query_type="factual",
            keywords=[],
            entities=[],
        ), retrieval_time_ms=12)


class FakeGenerator:
    def __init__(self):
        self.model = "fake-model"
        self.last_usage: dict = {}
        self.stream_calls = 0

    async def generate_with_sources(self, query, context, sources, system_prompt=None):
        from src.core.generation.generator import GenerationResult

        assert system_prompt is not None, "system prompt 必须被注入，否则会退回英文默认提示词"
        return GenerationResult(
            answer=f"ANS:{query}",
            sources=sources,
            model=self.model,
            tokens_used=42,
        )

    async def generate_stream(self, query, context, system_prompt=None):
        self.stream_calls += 1
        assert system_prompt is not None
        for tok in ["你", "好", "，", "世界"]:
            yield tok


class FakeAntiAvalanche:
    """记录自己是否被调用——这是「防雪崩是否接进链路」的直接证据。"""

    def __init__(self):
        self.calls = 0

    async def get_with_lock(self, key, fetch_fn, ttl=300, *args, cache=None, **kwargs):
        self.calls += 1
        assert cache is not None, "必须注入调用方的 cache，否则读写的是另一个 keyspace"
        return await fetch_fn(*args, **kwargs)


class FakeRewriter:
    def __init__(self):
        self.rewrite_calls = 0
        self.hyde_calls = 0

    async def rewrite(self, query, num_variants=3):
        self.rewrite_calls += 1
        return [query, f"{query} 的流程", f"{query} 怎么办理"]

    async def hyde(self, query):
        self.hyde_calls += 1
        return f"假设文档：{query} 需要提交申请。"


def build_service() -> tuple[RAGService, dict]:
    fakes = {
        "retriever": FakeRetriever(),
        "generator": FakeGenerator(),
        "embedder": FakeEmbedder(),
        "embedding_cache": FakeEmbeddingCache(),
        "anti_avalanche": FakeAntiAvalanche(),
        "rewriter": FakeRewriter(),
    }
    svc = RAGService(
        retriever=fakes["retriever"],
        generator=fakes["generator"],
        context_builder=ContextBuilder(),
        embedder=fakes["embedder"],
        embedding_cache=fakes["embedding_cache"],
        search_cache=SearchCache(),
        anti_avalanche=fakes["anti_avalanche"],
        rewriter=fakes["rewriter"],
    )
    return svc, fakes


# ── 用例 ────────────────────────────────────────────────────────
async def test_full_pipeline():
    print("\n1) RAGService 编排层：embed → retrieve → context → generate → log")
    svc, fakes = build_service()

    result = await svc.query("如何报销差旅费", user_id="u1", top_k=5)

    check("返回答案", result["answer"] == "ANS:如何报销差旅费", result["answer"])
    check("返回 sources", len(result["sources"]) == 2, f"{len(result['sources'])} 条")
    check("检索器被调用", fakes["retriever"].calls == 1, f"{fakes['retriever'].calls} 次")
    check("token 数被带回", result["tokens_used"] == 42)
    check("查询已记入内存日志", len(svc.history("u1")) == 1)

    # 直接验证 ContextBuilder：它必须给出带编号的上下文块，而不是原始的拼接
    search_results = await svc.retrieve("上下文验证", top_k=5)
    context, sources = svc.build_context(search_results)
    check(
        "ContextBuilder 参与构建（带 [Source N] 编号）",
        "关于「上下文验证」" in context and "Source" in context,
        context[:60].replace("\n", " "),
    )
    check("sources 结构未变（SSE 契约）", len(sources) == 2 and hasattr(sources[0], "chunk_id"))


async def test_search_cache_and_anti_avalanche():
    print("\n2) 检索缓存复用 MultiLevelCache + 防雪崩 single-flight")
    svc, fakes = build_service()

    await svc.retrieve("缓存键查询", top_k=5, rerank=False)
    first_calls = fakes["retriever"].calls
    check("首次未命中 → 走检索器", first_calls == 1, f"{first_calls} 次")
    check("防雪崩被调用", fakes["anti_avalanche"].calls == 1, f"{fakes['anti_avalanche'].calls} 次")

    await svc.retrieve("缓存键查询", top_k=5, rerank=False)
    check(
        "二次命中缓存 → 不再查检索器",
        fakes["retriever"].calls == first_calls,
        f"累计 {fakes['retriever'].calls} 次",
    )

    stats = svc.search_cache.stats
    check("L1 命中被统计", stats["l1_hits"] >= 1, str(stats))

    # rerank 必须进入 cache key，否则流式/非流式会互相污染
    await svc.retrieve("缓存键查询", top_k=5, rerank=True)
    check(
        "rerank 参与缓存键（不同端点不互相污染）",
        fakes["retriever"].calls == first_calls + 1,
        f"累计 {fakes['retriever'].calls} 次",
    )


async def test_query_rewrite_switch():
    print("\n3) 查询重写 / HyDE：默认关闭，开启后生效")
    svc, fakes = build_service()

    check("默认开关关闭", svc.settings.ENABLE_QUERY_REWRITE is False)
    variants = await svc.expand_query("报销")
    check("关闭时不做任何 LLM 调用", variants == ["报销"] and fakes["rewriter"].rewrite_calls == 0)

    svc.settings.ENABLE_QUERY_REWRITE = True
    svc.settings.QUERY_REWRITE_MAX_CHARS = 10
    try:
        await svc.expand_query("报销")
        check("开启后短查询触发重写", fakes["rewriter"].rewrite_calls == 1)

        await svc.expand_query("差旅费报销的具体流程和所需材料有哪些注意事项")
        check(
            "长查询不触发（阈值生效）",
            fakes["rewriter"].rewrite_calls == 1,
            "仍为 1 次",
        )

        svc.settings.QUERY_REWRITE_MODE = "hyde"
        await svc.expand_query("报销")
        check("HyDE 模式可达", fakes["rewriter"].hyde_calls == 1)
        svc.settings.QUERY_REWRITE_MODE = "rewrite"
    finally:
        svc.settings.ENABLE_QUERY_REWRITE = False

    # 多变体检索的召回必须去重
    svc2, _ = build_service()
    svc2.settings.ENABLE_QUERY_REWRITE = True
    try:
        deduped = svc2._dedupe([
            {"chunk_id": "c1", "score": 0.5, "content": "a", "rank": 1},
            {"chunk_id": "c1", "score": 0.9, "content": "a", "rank": 2},
            {"chunk_id": "c2", "score": 0.7, "content": "b", "rank": 3},
        ])
        check("多变体召回按 chunk_id 去重且保留最高分", len(deduped) == 2 and deduped[0]["score"] == 0.9)
    finally:
        svc2.settings.ENABLE_QUERY_REWRITE = False


async def test_warmer():
    print("\n4) 缓存预热：CacheWarmer 注入调用方的 cache")
    svc, _ = build_service()

    payload = await svc.build_warm_payloads(limit=2)
    check("预热载荷可构造", isinstance(payload, dict))

    # 用注入了同一 cache 的 warmer 写一条，再确认能被 SearchCache 读到
    warmer = CacheWarmer(default_ttl=60, cache=svc.search_cache.cache)
    key = svc.search_cache.key_for("预热键", 5, False)
    warmed = await warmer.warm_up([(key, lambda: _async_list())])
    check("warmer 写入成功", warmed == 1, f"{warmed} 条")

    cached = await svc.search_cache.get("预热键", 5, False)
    check("写入的键能被 SearchCache 读到（说明是同一个 keyspace）", cached == [{"ok": True}])

    # 无 Redis 时预热应降级为 0，而不是把启动拖垮
    count = await svc.warm_hot_queries(limit=3)
    check("无查询日志时预热安全降级", count == 0, f"warmed={count}")


async def _async_list():
    return [{"ok": True}]


async def test_connection_pool():
    print("\n5) 连接池管理器：注册 + 探活（外部服务不可达时应返回 False 而非抛错）")
    pool_manager.register_pool("postgresql", None)
    pool_manager.register_pool("redis", None)
    pool_manager.register_pool("milvus", None)
    check("三个池已注册", len(pool_manager._metrics) >= 3)

    results = await pool_manager.check_all()
    check("check_all 返回全部池的状态", set(results) >= {"postgresql", "redis", "milvus"}, str(results))
    check("不可达时返回 False 而不是抛异常", all(v is False for v in results.values()), str(results))

    metrics = pool_manager.all_metrics
    check("指标可读（healthy 字段存在）", all("healthy" in m for m in metrics.values()))


async def test_streaming_and_logging():
    print("\n6) 流式路径：prepare_stream → stream_generate → log_stream_query")
    svc, fakes = build_service()

    sources, context = await svc.prepare_stream("流式查询", top_k=5)
    check("先拿到 sources（前端可在首 token 前渲染引用）", len(sources) == 2)

    tokens = [t async for t in svc.stream_generate("流式查询", context)]
    check("生成器被驱动", "".join(tokens) == "你好，世界", "".join(tokens))
    check("流式不精排（rerank=False 进缓存键）", fakes["retriever"].calls == 1)

    await svc.log_stream_query("u2", "流式查询", "你好，世界", 2, 123)
    check("流式查询被记入日志", len(svc.history("u2")) == 1)


async def test_pipeline_features():
    print("\n7) /search-mode 反映真实开关状态")
    svc, _ = build_service()
    feats = svc.pipeline_features()
    joined = " | ".join(feats)
    check("防雪崩状态在接口里可见", "Anti-avalanche: on" in joined)
    check("预热状态在接口里可见", "Cache warmer: on" in joined)
    check("重写状态在接口里可见", "Query rewrite: off" in joined)


async def main() -> int:
    await test_full_pipeline()
    await test_search_cache_and_anti_avalanche()
    await test_query_rewrite_switch()
    await test_warmer()
    await test_connection_pool()
    await test_streaming_and_logging()
    await test_pipeline_features()

    failed = [n for n, ok, _ in _results if not ok]
    print(f"\n{'=' * 60}")
    print(f"总计 {len(_results)} 项，通过 {len(_results) - len(failed)}，失败 {len(failed)}")
    if failed:
        print("失败项：")
        for n in failed:
            print("  -", n)
        return 1
    print("6 个模块全部确认接入链路。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
