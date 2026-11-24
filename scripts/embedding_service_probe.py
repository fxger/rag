#!/usr/bin/env python
"""Probe the Qwen embedding service to characterise its real capacity.

The service (VM1, container ``qwen-embeddings-2.0-small``) runs an
OpenAI-compatible ``/v1/embeddings`` endpoint backed by
``Qwen/Qwen3-Embedding-0.6B`` on **CPU only** (2 vCPU, no GPU).

Its endpoint is declared ``async def`` but calls ``model.encode()``
synchronously, so an in-flight embedding blocks the whole event loop.  This
script demonstrates that and measures practical throughput so that bulk
indexing jobs can size their batches sensibly.

Usage:
    python scripts/embedding_service_probe.py                 # full report
    python scripts/embedding_service_probe.py --health-only   # event-loop test
"""
from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import get_settings  # noqa: E402

# A typical document chunk produced by the project's SmartChunker.
CHUNK = (
    "企业级 RAG 系统的检索层采用混合检索策略，稠密向量由 Qwen 嵌入服务生成，"
    "稀疏侧使用 BM25 配合 jieba 分词；两路结果经 RRF 融合后再送入生成模型。"
) * 3  # ~ 360 chars, ~ 330 tokens


def _post(client: httpx.Client, texts: list[str], timeout: float) -> tuple[int, float]:
    started = time.perf_counter()
    try:
        resp = client.post(
            get_settings().QWEN_EMBED_URL,
            json={"input": texts, "model": get_settings().QWEN_EMBED_MODEL},
            timeout=timeout,
        )
        return resp.status_code, time.perf_counter() - started
    except Exception as exc:  # noqa: BLE001
        print(f"      ! {type(exc).__name__}: {exc}")
        return 0, time.perf_counter() - started


def probe_health(url: str, timeout: float = 5.0) -> float | None:
    started = time.perf_counter()
    try:
        r = httpx.get(url, timeout=timeout)
        return (time.perf_counter() - started) * 1000 if r.status_code == 200 else None
    except Exception:  # noqa: BLE001
        return None


def test_event_loop_blocking() -> None:
    """Fire a heavy embedding request and watch /health latency meanwhile."""
    base = get_settings().QWEN_EMBED_URL.split("/v1/")[0]
    health = f"{base}/health"

    print("── 1. 事件循环阻塞验证 ─────────────────────────────────")
    base_latency = probe_health(health)
    print(f"  空闲时 /health 延迟: {base_latency:.0f} ms" if base_latency else "  空闲时 /health 无响应")

    done = threading.Event()
    heavy_result: list[tuple[int, float]] = []

    def heavy() -> None:
        with httpx.Client() as c:
            heavy_result.append(_post(c, [CHUNK] * 64, timeout=900))
        done.set()

    threading.Thread(target=heavy, daemon=True).start()
    time.sleep(1.0)  # let the server pick up the request

    samples: list[float] = []
    deadline = time.time() + 40
    while time.time() < deadline:
        ms = probe_health(health, timeout=6.0)
        samples.append(-1.0 if ms is None else ms)
        if done.is_set():
            break
        time.sleep(0.5)

    timeouts = sum(1 for s in samples if s < 0)
    ok = [s for s in samples if s >= 0]
    print(f"  重负载期间采样 {len(samples)} 次: 超时/失败 {timeouts} 次")
    if ok:
        print(f"  /health 延迟 中位数 {statistics.median(ok):.0f} ms, 最大 {max(ok):.0f} ms")
    if timeouts:
        print("  ⇒ 结论：嵌入请求阻塞了事件循环，连 /health 都拿不到响应")
    elif ok and max(ok) > (base_latency or 0) * 5:
        print("  ⇒ 结论：/health 延迟显著上升，事件循环被挤压")
    else:
        print("  ⇒ 结论：未观察到阻塞，端点可能已改为线程池执行")

    done.wait(timeout=900)
    if heavy_result:
        code, elapsed = heavy_result[0]
        print(f"  重负载请求: 64 条 x ~360 字 -> HTTP {code}, 耗时 {elapsed:.1f}s "
              f"({64 / elapsed:.2f} 条/秒)")


def test_batch_scaling() -> None:
    """Throughput at several batch sizes, using realistic chunk lengths."""
    print()
    print("── 2. 批量吞吐标定 ─────────────────────────────────────")
    print(f"  {'批量':>6} {'耗时(s)':>10} {'条/秒':>10} {'单条(ms)':>10}")
    with httpx.Client() as c:
        for n in (1, 4, 8, 16, 32):
            code, elapsed = _post(c, [CHUNK] * n, timeout=900)
            if code != 200:
                print(f"  {n:>6}  HTTP {code} —— 跳过")
                continue
            print(f"  {n:>6} {elapsed:>10.2f} {n / elapsed:>10.2f} {elapsed / n * 1000:>10.0f}")


def test_concurrency() -> None:
    """Two simultaneous requests: serialised (total ~2x) or parallel (~1x)?"""
    print()
    print("── 3. 并发行为（2 个并发请求 x 8 条）────────────────────")
    results: list[tuple[int, float]] = []
    lock = threading.Lock()

    def worker() -> None:
        with httpx.Client() as c:
            r = _post(c, [CHUNK] * 8, timeout=900)
        with lock:
            results.append(r)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    started = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - started

    for i, (code, elapsed) in enumerate(results, 1):
        print(f"  请求{i}: HTTP {code}, 自身耗时 {elapsed:.2f}s")
    print(f"  墙钟总耗时: {wall:.2f}s")
    slowest = max(e for _, e in results) if results else 0
    if wall > slowest * 1.5:
        print("  ⇒ 串行处理：并发不会提升吞吐，只会拉长排队")
    else:
        print("  ⇒ 存在并行度")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--health-only", action="store_true", help="只跑事件循环阻塞验证")
    args = ap.parse_args()

    settings = get_settings()
    print(f"目标服务: {settings.QWEN_EMBED_URL}")
    print(f"样本长度: {len(CHUNK)} 字（模拟 SmartChunker 产出的真实分块）")
    print()
    test_event_loop_blocking()
    if not args.health_only:
        test_batch_scaling()
        test_concurrency()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
