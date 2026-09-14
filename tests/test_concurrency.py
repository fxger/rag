"""Concurrency test for singleton manager."""

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys
sys.path.insert(0, '.')

from src.core.concurrent_singleton import (
    ThreadSafeReranker,
    ConnectionPool,
    SingletonManager,
)


def test_thread_safe_reranker():
    """Test thread-safe reranker with concurrent requests."""
    print("=== Testing Thread-Safe Reranker ===")
    
    reranker = SingletonManager.get_reranker(max_concurrent=4)
    
    def predict_task(task_id: int) -> dict:
        """Single prediction task."""
        start = time.time()
        pairs = [
            ("公司营收是多少", f"文档{task_id}：公司2025年营收5亿元"),
            ("技术架构是怎样的", f"文档{task_id}：采用微服务架构"),
        ]
        scores = reranker.predict(pairs, timeout=5.0)
        elapsed = time.time() - start
        return {
            "task_id": task_id,
            "scores": scores,
            "elapsed": elapsed,
        }
    
    # Run 10 concurrent tasks
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(predict_task, i) for i in range(10)]
        for future in as_completed(futures):
            results.append(future.result())
    
    # Print results
    for r in sorted(results, key=lambda x: x["task_id"]):
        print(f"  Task {r['task_id']}: {r['elapsed']:.3f}s, scores={[f'{s:.3f}' for s in r['scores']]}")
    
    print(f"\n  Stats: {reranker.stats}")
    print()


def test_connection_pool():
    """Test connection pool with concurrent access."""
    print("=== Testing Connection Pool ===")
    
    # Create a mock connection pool
    mock_connections = []
    
    def mock_factory():
        conn_id = len(mock_connections)
        conn = {"id": conn_id, "created": time.time()}
        mock_connections.append(conn)
        return conn
    
    def mock_health_check(conn):
        return True
    
    pool = ConnectionPool(
        factory=mock_factory,
        min_size=2,
        max_size=5,
        health_check=mock_health_check,
    )
    
    def access_pool(task_id: int) -> dict:
        """Access pool connection."""
        start = time.time()
        conn = pool.acquire(timeout=2.0)
        time.sleep(0.1)  # Simulate work
        pool.release(conn)
        elapsed = time.time() - start
        return {"task_id": task_id, "conn_id": conn["id"], "elapsed": elapsed}
    
    # Run concurrent pool access
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(access_pool, i) for i in range(15)]
        for future in as_completed(futures):
            results.append(future.result())
    
    # Print results
    for r in sorted(results, key=lambda x: x["task_id"])[:5]:
        print(f"  Task {r['task_id']}: conn_{r['conn_id']}, {r['elapsed']:.3f}s")
    print(f"  ... ({len(results)} total)")
    print(f"\n  Pool stats: {pool.stats}")
    print()


def test_concurrent_initialization():
    """Test that singletons are created only once under concurrency."""
    print("=== Testing Concurrent Initialization ===")
    
    instances = []
    lock = threading.Lock()
    
    def get_instance(task_id: int):
        reranker = SingletonManager.get_reranker()
        with lock:
            instances.append(id(reranker))
    
    # Run concurrent initialization
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(get_instance, i) for i in range(20)]
        for future in as_completed(futures):
            future.result()
    
    unique_instances = len(set(instances))
    print(f"  Total requests: {len(instances)}")
    print(f"  Unique instances: {unique_instances}")
    print(f"  Singleton working: {'✓' if unique_instances == 1 else '✗'}")
    print()


def main():
    """Run all concurrency tests."""
    print("Concurrency Tests for Singleton Manager")
    print("=" * 50)
    print()
    
    test_thread_safe_reranker()
    test_connection_pool()
    test_concurrent_initialization()
    
    print("=" * 50)
    print("All tests completed!")


if __name__ == "__main__":
    main()
