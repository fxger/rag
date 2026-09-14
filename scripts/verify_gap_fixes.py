"""Regression checks for the resume-gap fixes (B1 / B3 / B4 / B5 / B9 / B10).

Run from the project root:

    .venv\\Scripts\\python.exe scripts/verify_gap_fixes.py

Each check asserts the *observable* behaviour that was broken, so a
re-introduced regression fails loudly instead of silently degrading.
"""
import asyncio
import sys
import threading
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASSED: list[str] = []


def _ok(label: str, detail: str = "") -> None:
    PASSED.append(label)
    print(f"  [PASS] {label}" + (f" — {detail}" if detail else ""))


# ----------------------------------------------------------------------
# B1 · Chinese chunking
# ----------------------------------------------------------------------
def check_b1() -> None:
    from src.core.document.chunker import SmartChunker

    chunker = SmartChunker(chunk_size=512, chunk_overlap=50)
    text = "人工智能正在改变世界。" * 90          # 990 chars, no spaces
    chunks = chunker.chunk(text, file_type="txt")

    assert len(chunks) > 1, f"中文仍只切出 {len(chunks)} 块（应 > 1）"
    over = [c.token_count for c in chunks if c.token_count > 512]
    assert not over, f"存在超限块: {over}"
    for c in chunks:
        assert text[c.start_offset:c.end_offset] == c.content, "偏移无法还原原文"
    assert " " not in chunks[0].content, "中文内容被插入空格"
    _ok("B1 中文分块", f"990 字 -> {len(chunks)} 块, tokens={[c.token_count for c in chunks]}")


def check_b1_english_limit() -> None:
    """The 512-token ceiling must hold for English too, not just Chinese."""
    from src.core.document.chunker import SmartChunker

    chunker = SmartChunker(chunk_size=512, chunk_overlap=50)
    text = "Artificial intelligence is transforming the world. " * 60
    chunks = chunker.chunk(text, file_type="txt")
    over = [c.token_count for c in chunks if c.token_count > 512]
    assert not over, f"英文块超限: {over}（修复前实测报出 764）"
    _ok("B1 英文 512 上限", f"{len(text)} 字符 -> {len(chunks)} 块, tokens={[c.token_count for c in chunks]}")


# ----------------------------------------------------------------------
# B4 · Token budget & truncation
# ----------------------------------------------------------------------
def check_b4() -> None:
    from src.core.generation.context_builder import ContextBuilder
    from src.core.retrieval.models import SearchResult
    from src.core.tokenization import estimate_tokens

    builder = ContextBuilder()

    # Truncation must respect the budget for CJK (was 4x over).
    cjk = "人工智能技术正在快速发展。" * 300
    out = builder._truncate_to_tokens(cjk, 100)
    got = estimate_tokens(out)
    assert got <= 130, f"中文截断超预算: {got} > 130"

    # And the context budget must actually bound what gets assembled.
    docs = [
        SearchResult(
            chunk_id=f"c{i}", document_id=f"d{i}",
            content="人工智能技术正在快速发展。" * 300,
            score=0.9 - i * 0.01, rank=i + 1, metadata={}, source_type="test",
        )
        for i in range(10)
    ]
    ctx, sources = builder.build_with_sources("q", docs, max_tokens=4000)
    actual = estimate_tokens(ctx)
    assert actual <= 4600, f"上下文预算被突破: {actual} token"
    _ok("B4 上下文预算", f"4000 预算(中文) -> 收录 {len(sources)} 篇 / 实际 {actual} token")


# ----------------------------------------------------------------------
# B5 · RetrievalResult unwrapping
# ----------------------------------------------------------------------
def check_b5() -> None:
    from src.core.retrieval.models import RetrievalResult

    # Bypass __init__ (query_analysis is required) — we only care that the
    # wrapper exposes no length.
    result = RetrievalResult.__new__(RetrievalResult)
    result.results = []
    try:
        len(result)  # noqa: B018 — asserting the absence of __len__
        raise AssertionError("RetrievalResult 竟带有 __len__，本检查已失效")
    except TypeError:
        pass  # expected: the wrapper has no length

    assert isinstance(result.results, list), ".results 应为 list"

    src = (Path(__file__).resolve().parent.parent / "src/services/rag_service.py").read_text(
        encoding="utf-8"
    )
    assert src.count("results = retrieval.results") >= 2, (
        "非流式与流式两条路径都应从 RetrievalResult 解包出 .results"
    )
    assert src.count("retrieval = await self.retriever.retrieve(") >= 2, (
        "两处 retrieve() 都应先赋给 retrieval 变量，避免把包装对象传给下游"
    )
    _ok("B5 RetrievalResult 解包", "非流式与流式两条路径均已修复")


# ----------------------------------------------------------------------
# B9 · Connection pool accounting
# ----------------------------------------------------------------------
def check_b9() -> None:
    from src.core.concurrent_singleton import ConnectionPool

    pool = ConnectionPool(factory=lambda: "conn", min_size=1, max_size=3)
    assert pool.stats["active"] == 0, pool.stats

    c1 = pool.acquire()
    assert pool.stats["active"] == 1, pool.stats
    c2 = pool.acquire()
    assert pool.stats["active"] == 2, pool.stats
    pool.release(c1)
    assert pool.stats["active"] == 1, pool.stats
    pool.release(c2)
    assert pool.stats["active"] == 0, pool.stats

    errors: list[Exception] = []

    def worker() -> None:
        for _ in range(100):
            try:
                conn = pool.acquire(timeout=5)
                pool.release(conn)
            except Exception as exc:  # noqa: BLE001 — recorded for the assertion
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors[:3]
    assert pool.stats["active"] == 0, f"并发后 active 未归零: {pool.stats}"
    assert pool.stats["created"] <= 3, f"连接数超出 max_size: {pool.stats}"
    _ok("B9 连接池计数", "顺序 0→1→2→1→0；并发 2000 次零错误")


# ----------------------------------------------------------------------
# B3 / B10 · Reranker concurrency
# ----------------------------------------------------------------------
def _install_fake_cross_encoder(load_s: float = 0.3, infer_s: float = 0.3) -> None:
    """Patch sentence_transformers with a slow fake to expose blocking."""
    fake = types.ModuleType("sentence_transformers")

    class FakeCrossEncoder:
        def __init__(self, name, max_length=512):
            time.sleep(load_s)
            self.name = name

        def predict(self, pairs, batch_size=32, show_progress_bar=False):
            time.sleep(infer_s)
            return [0.5] * len(pairs)

    fake.CrossEncoder = FakeCrossEncoder
    sys.modules["sentence_transformers"] = fake


def check_b3() -> None:
    _install_fake_cross_encoder()
    from src.core.retrieval.reranker import Reranker

    reranker = Reranker(model_name="verify-b3")
    outcomes: list[bool] = []

    def worker() -> None:
        outcomes.append(reranker._ensure_model_loaded())

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(outcomes), f"并发加载出现静默失败: {outcomes}"
    _ok("B3 模型加载竞态", "10 线程并发全部成功（修复前部分返回 False）")


def check_b10() -> None:
    from src.core.retrieval.models import SearchResult
    from src.core.retrieval.reranker import Reranker

    async def main() -> float:
        ticks: list[float] = []

        async def heartbeat() -> None:
            for _ in range(30):
                ticks.append(time.monotonic())
                await asyncio.sleep(0.02)

        beat = asyncio.create_task(heartbeat())
        reranker = Reranker(model_name="verify-b10")   # fresh name -> first load
        docs = [
            SearchResult(chunk_id=f"c{i}", document_id="d", content="x" * 100,
                         score=0.5, rank=i, metadata={}, source_type="test")
            for i in range(5)
        ]
        await asyncio.gather(*[reranker.rerank("q", docs) for _ in range(8)])
        await beat
        gaps = [b - a for a, b in zip(ticks, ticks[1:])]
        return max(gaps) * 1000

    worst = asyncio.run(main())
    assert worst < 150, f"事件循环被阻塞 {worst:.0f}ms（修复前为 302ms）"
    _ok("B10 事件循环阻塞", f"8 路并发下心跳最大间隔 {worst:.0f}ms（修复前 302ms）")


# ----------------------------------------------------------------------
# 第二批 · 接线（A1 / A2 / A3 / A5 / A6 / A7 / A8）
# ----------------------------------------------------------------------
def check_a1() -> None:
    # Deliberately avoids instantiating the retriever: building one pulls in
    # the analyzers and Milvus clients, which is heavy (and crashes outright
    # in this environment).  The wiring is asserted from source + the cache.
    from src.api.v1.query import SearchCache

    cache = SearchCache()
    assert cache._make_key("q", 5, True) != cache._make_key("q", 5, False), (
        "rerank 未参与缓存 key —— 非流式与流式会互相污染"
    )

    src = (ROOT / "src/api/v1/query.py").read_text(encoding="utf-8")
    assert "enable_reranker=True" in src, "检索器仍禁止加载重排器"
    assert "enable_reranker=False" not in src, "仍有位置关闭重排器加载"
    assert "ENABLE_RERANK" in src, "非流式端点未接入精排配置开关"
    assert "rerank=True" not in src, "仍有硬编码开启精排的位置"
    assert "rerank=False" in src, "流式端点未显式关闭重排"

    from src.config.settings import get_settings

    assert get_settings().ENABLE_RERANK is False, (
        "ENABLE_RERANK 默认应为关闭：实测英文 Cross-Encoder 在中文语料上降低命中率"
    )
    _ok(
        "A1 重排开关",
        "开关默认关闭（实测无效，见 retrieval_experiment.json）；流式显式关闭；缓存 key 隔离",
    )


def check_a2() -> None:
    from src.api.middleware.circuit_breaker import CircuitBreakerError, CircuitState
    from src.api.v1.query import _llm_breaker

    _llm_breaker.reset()

    async def boom() -> None:
        raise RuntimeError("upstream unavailable")

    async def run() -> bool:
        for _ in range(_llm_breaker.failure_threshold):
            try:
                await _llm_breaker.call(boom)
            except RuntimeError:
                pass
        if _llm_breaker.state != CircuitState.OPEN:
            return False
        try:
            await _llm_breaker.call(boom)      # must be rejected without calling
        except CircuitBreakerError:
            return True
        return False

    threshold = _llm_breaker.failure_threshold
    opened = asyncio.run(run())
    _llm_breaker.reset()
    assert opened, "熔断器未在连续失败后打开"
    _ok("A2 熔断器", f"连续 {threshold} 次失败后打开并拒绝调用，reset 后恢复")


def check_a3() -> None:
    from src.main import create_app

    app = create_app()
    # Starlette keeps user_middleware outermost-first: index 0 runs first.
    names = [m.cls.__name__ for m in app.user_middleware]
    assert "MetricsMiddleware" in names, names
    assert "RateLimitMiddleware" in names, names
    assert names[0] == "MetricsMiddleware", f"Metrics 应在最外层: {names}"
    assert names.index("RateLimitMiddleware") > names.index("CORSMiddleware"), (
        f"限流应在 CORS 内层: {names}"
    )
    _ok("A3 限流注册", f"执行顺序（外→内）: {' -> '.join(names)}")


def check_a5() -> None:
    src = (ROOT / "src/api/v1/query.py").read_text(encoding="utf-8")
    assert "IntelligentEmbeddingCache" in src, "未接入多级嵌入缓存"
    assert "get_qwen_embedding_sync" not in src, "旧的同步缓存实现仍然存在"
    assert "await asyncio.to_thread(_request_embedding_sync" in src, (
        "嵌入 HTTP 请求未移出事件循环"
    )
    _ok("A5 嵌入缓存", "统一用 IntelligentEmbeddingCache；请求移出事件循环")


def check_a6() -> None:
    from src.api.v1.query import _MilvusSession

    assert hasattr(_MilvusSession, "_get_pool"), "_MilvusSession 缺少连接池访问器"
    pool_src = (ROOT / "src/core/concurrent_singleton.py").read_text(encoding="utf-8")
    assert 'alias="default"' in pool_src, "连接池 factory 未使用 default alias"
    assert "current_thread()" not in pool_src.split("def factory()")[1][:200], (
        "仍按线程 id 建 alias，池中连接与 Collection 使用的不是同一个"
    )
    _ok("A6 连接池", "factory 使用 default alias；query 经 _MilvusSession 借还")


def check_a7() -> None:
    src = (ROOT / "src/api/v1/query.py").read_text(encoding="utf-8")
    required = (
        "RAG_QUERY_TOTAL", "RAG_QUERY_DURATION", "RAG_RETRIEVAL_DURATION",
        "RAG_RETRIEVAL_RESULTS", "RAG_GENERATION_DURATION",
        "AI_GENERATION_TOTAL", "AI_GENERATION_TOKENS",
        "CACHE_HITS_TOTAL", "CACHE_MISSES_TOTAL", "CACHE_SIZE",
    )
    missing = [n for n in required if n not in src]
    assert not missing, f"未埋点的指标: {missing}"

    rerank_src = (ROOT / "src/core/retrieval/hybrid_retriever.py").read_text(encoding="utf-8")
    assert "RAG_RERANK_DURATION" in rerank_src, "重排耗时未埋点"
    _ok("A7 业务指标埋点", f"{len(required)} 个指标已埋点 + 重排耗时")


def check_a8() -> None:
    from src.core.retrieval.reranker import Reranker

    stats = Reranker(model_name="verify-a8").stats
    required = ("active", "waiting", "max_concurrent", "total_requests",
                "inference_count", "cache_hits", "cache_size")
    missing = [k for k in required if k not in stats]
    assert not missing, f"Reranker.stats 缺少字段: {missing}"

    body = (ROOT / "src/api/metrics.py").read_text(encoding="utf-8").split(
        "def update_concurrent_metrics"
    )[1]
    assert "from src.core.singleton import get_reranker" in body, (
        "并发指标采集源未切到生产单例"
    )
    _ok("A8 指标采集源", "Reranker.stats 字段齐备；采集源指向生产单例")


def check_a7_metrics_exposed() -> None:
    """Every metric referenced by the collector must be importable."""
    from src.api.metrics import update_concurrent_metrics

    update_concurrent_metrics()      # must not raise with no traffic
    _ok("A8 采集函数可执行", "无流量时调用不抛异常")


def main() -> int:
    print("验证简历能力补齐 —— 第一批 + 第二批\n")
    checks = (
        # 第一批 · 修地基
        check_b1, check_b1_english_limit, check_b4, check_b5,
        check_b9, check_b3, check_b10,
        # 第二批 · 接线
        check_a1, check_a2, check_a3, check_a5, check_a6, check_a7, check_a8,
        check_a7_metrics_exposed,
    )
    for check in checks:
        try:
            check()
        except AssertionError as exc:
            print(f"  [FAIL] {check.__name__}: {exc}")
            return 1
        except Exception as exc:  # noqa: BLE001 — surface unexpected errors
            print(f"  [ERROR] {check.__name__}: {type(exc).__name__}: {exc}")
            return 1

    print(f"\n全部通过（{len(PASSED)} 项）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
