"""Demonstrate why a synchronous CPU call inside `async def` freezes the loop.

Companion to docs/EMBEDDING_SERVICE_INCIDENT.md.  It runs the same CPU-bound
workload three ways and measures, for each, how long the event loop went
without being able to run anything else (the "heartbeat gap").

  A. async def + plain call            -> loop frozen for the whole duration
  B. async def + await to_thread(...)  -> loop stays responsive
  C. async def + await sleep(0) yields -> responsive, but NOT concurrent

Numbers are machine-dependent; the *relative* ordering is the point.

Usage:
    python scripts/async_blocking_demo.py
"""
from __future__ import annotations

import asyncio
import sys
import time

HEARTBEAT_S = 0.0           # yield immediately: record whenever the loop runs
TARGET_TOTAL_S = 1.0        # total CPU work per request
CHUNKS = 20                 # split into chunks so variant C has yield points
CONCURRENCY = 2             # simultaneous requests


def _benchmark_once(iters: int) -> float:
    started = time.perf_counter()
    x = 0
    for i in range(iters):
        x += i * i
    return time.perf_counter() - started


def _calibrate() -> int:
    """Chunk iterations such that CHUNKS of them take about TARGET_TOTAL_S."""
    probe = 200_000
    per_iter = _benchmark_once(probe) / probe
    return max(1_000, int((TARGET_TOTAL_S / CHUNKS) / per_iter))


def _spin(iters: int) -> int:
    x = 0
    for i in range(iters):
        x += i * i
    return x


async def cpu_plain(iters: int) -> int:
    """A. Blocking compute called directly from a coroutine."""
    return _spin(iters)


async def cpu_threadpool(iters: int) -> int:
    """B. Same compute, handed off to a worker thread."""
    return await asyncio.to_thread(_spin, iters)


async def cpu_yielding(iters: int) -> int:
    """C. Same compute, but with `await` points spliced in.

    This models "what if model.encode() had an await inside it" — the coroutine
    yields regularly, so the loop stays responsive, but the work still runs on
    the event loop thread.  Responsiveness is restored; concurrency is not.
    """
    total = 0
    for _ in range(CHUNKS):
        total += _spin(iters // CHUNKS)
        await asyncio.sleep(0)
    return total


async def measure(label: str, handler) -> dict:
    # Track only the maximum gap (O(1) memory).  Appending every tick to a list
    # grows into millions of entries, and the resulting allocation/GC pauses
    # show up as fake "freezes".
    state = {"last": time.perf_counter(), "max_gap": 0.0}
    stop = False

    async def heartbeat() -> None:
        # `sleep(0)` rather than a timer: this records "the loop got control"
        # without depending on OS timer granularity (~15.6ms on Windows), so
        # the measured gap is exactly the period the loop could not run.
        while not stop:
            now = time.perf_counter()
            gap = now - state["last"]
            if gap > state["max_gap"]:
                state["max_gap"] = gap
            state["last"] = now
            await asyncio.sleep(HEARTBEAT_S)

    hb = asyncio.create_task(heartbeat())
    await asyncio.sleep(0.05)       # let the heartbeat settle
    state["last"] = time.perf_counter()
    state["max_gap"] = 0.0
    started = time.perf_counter()
    await asyncio.gather(*[handler() for _ in range(CONCURRENCY)])
    wall = time.perf_counter() - started
    # measure the final gap up to the moment the loop came back
    state["max_gap"] = max(state["max_gap"], time.perf_counter() - state["last"])
    stop = True
    await hb

    frozen = state["max_gap"] * 1000
    print(f"  {label:<34}{wall:>9.2f}s{frozen:>12.0f}ms"
          f"{'   <-- 事件循环被冻住' if frozen > wall / CONCURRENCY * 500 else ''}")
    return {"wall": wall, "frozen_ms": frozen}


def main() -> int:
    iters = _calibrate() * CHUNKS          # iterations for ONE full request
    single = _benchmark_once(iters)
    print(f"标定：每次请求约 {single:.2f}s 纯 CPU 计算，"
          f"并发 {CONCURRENCY} 个，心跳期望间隔 {HEARTBEAT_S * 1000:.0f}ms\n")
    print(f"  {'写法':<34}{'墙钟耗时':>10}{'最长冻结':>12}")
    print("  " + "-" * 56)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    a = loop.run_until_complete(measure("A. async def + 直接调用", lambda: cpu_plain(iters)))
    c = loop.run_until_complete(measure("C. 函数里插入 await 让出", lambda: cpu_yielding(iters)))
    b = loop.run_until_complete(measure("B. await asyncio.to_thread(...)", lambda: cpu_threadpool(iters)))
    loop.close()

    print()
    print("  读法：")
    print(f"    A 墙钟 {a['wall']:.2f}s ≈ 单次 {single:.2f}s × {CONCURRENCY} ⇒ 完全串行（并发度为 1）")
    print(f"    A 最长冻结 {a['frozen_ms']:.0f}ms ≈ 整段墙钟 ⇒ 全程无法服务任何其他任务")
    print(f"    C 最长冻结 {c['frozen_ms']:.0f}ms ⇒ 让出控制权**确实**恢复了响应性")
    print(f"    但 C 墙钟 {c['wall']:.2f}s 并不比 A 低（本例反而更慢，让出本身有开销）")
    print(f"      ⇒ 让出 ≠ 并发：吞吐一点没变，只是把「卡住」切碎了")
    print(f"    B 最长冻结 {b['frozen_ms']:.0f}ms ⇒ 换线程执行，事件循环全程可用")
    print()
    print("  注：纯 Python 循环持 GIL，B 的墙钟不会比 A 快（甚至略慢）。")
    print("      真正的并行收益来自底层库释放 GIL —— torch 做张量运算时就会释放，")
    print("      这正是本项目的修复能同时改善响应性与吞吐的原因。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
