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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def main() -> int:
    print("验证简历能力补齐 —— 第一批修复\n")
    checks = (
        check_b1, check_b1_english_limit, check_b4, check_b5,
        check_b9, check_b3, check_b10,
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
