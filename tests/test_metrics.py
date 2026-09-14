"""
Prometheus Metrics Integration Test Script.

Run: D:\Desktop\RAG\.venv\Scripts\python.exe tests/test_metrics.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_metrics_import():
    """Test that all metrics can be imported."""
    from src.api.metrics import (
        # HTTP Metrics
        HTTP_REQUESTS_TOTAL,
        HTTP_REQUEST_DURATION,
        HTTP_REQUESTS_IN_PROGRESS,
        # RAG Query Metrics
        RAG_QUERY_TOTAL,
        RAG_QUERY_DURATION,
        RAG_RETRIEVAL_DURATION,
        RAG_RERANK_DURATION,
        RAG_GENERATION_DURATION,
        RAG_RETRIEVAL_RESULTS,
        # Concurrency Metrics
        SEMAPHORE_ACTIVE,
        SEMAPHORE_WAITING,
        SEMAPHORE_MAX_CONCURRENT,
        # Reranker Metrics
        RERANKER_REQUESTS_TOTAL,
        RERANKER_INFERENCE_TOTAL,
        RERANKER_CACHE_HITS_TOTAL,
        RERANKER_CACHE_SIZE,
        # Connection Pool Metrics
        MILVUS_POOL_SIZE,
        MILVUS_POOL_ACTIVE,
        REDIS_POOL_SIZE,
        REDIS_POOL_ACTIVE,
        # Document Metrics
        DOCUMENTS_UPLOADED_TOTAL,
        DOCUMENTS_PROCESSED_TOTAL,
        DOCUMENTS_FAILED_TOTAL,
        DOCUMENT_PROCESSING_DURATION,
        # AI Metrics
        AI_GENERATION_TOTAL,
        AI_GENERATION_TOKENS,
        # Cache Metrics
        CACHE_HITS_TOTAL,
        CACHE_MISSES_TOTAL,
        CACHE_SIZE,
        # App Info
        APP_INFO,
        # Utility
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    print("[PASS] All metrics imported successfully")
    return True


def test_middleware_import():
    """Test that middleware can be imported."""
    from src.api.middleware.metrics import MetricsMiddleware, normalize_path
    print("[PASS] MetricsMiddleware imported successfully")
    return True


def test_path_normalization():
    """Test path normalization function."""
    from src.api.middleware.metrics import normalize_path
    
    test_cases = [
        ("/api/v1/query/abc123-def456", "/api/v1/query/abc123-def456"),
        ("/api/v1/documents/123/chunks", "/api/v1/documents/{id}/chunks"),
        ("/api/v1/auth/login", "/api/v1/auth/login"),
        ("/metrics", "/metrics"),
    ]
    
    for input_path, expected in test_cases:
        result = normalize_path(input_path)
        assert result == expected, f"normalize_path('{input_path}') = '{result}', expected '{expected}'"
    
    print("[PASS] Path normalization works correctly")
    return True


def test_metrics_generation():
    """Test that metrics can be generated."""
    from src.api.metrics import (
        HTTP_REQUESTS_TOTAL,
        generate_latest,
    )
    
    # Add some test data
    HTTP_REQUESTS_TOTAL.labels(method="GET", endpoint="/test", status_code="200").inc()
    
    # Generate output
    output = generate_latest()
    assert len(output) > 0, "Metrics output is empty"
    assert b"rag_http_requests_total" in output, "rag_http_requests_total not found in output"
    
    print("[PASS] Metrics generation works correctly")
    return True


def test_fastapi_integration():
    """Test FastAPI app integration."""
    from fastapi.testclient import TestClient
    from src.main import app
    
    client = TestClient(app)
    
    # Test /metrics endpoint
    response = client.get("/metrics")
    assert response.status_code == 200, f"/metrics returned status {response.status_code}"
    assert "text/plain" in response.headers.get("content-type", ""), "Invalid content-type"
    assert len(response.text) > 0, "/metrics response is empty"
    
    # Verify our custom metrics are present
    assert "rag_http_requests_total" in response.text, "rag_http_requests_total not in /metrics output"
    
    print("[PASS] FastAPI /metrics endpoint works correctly")
    return True


def test_all_metrics_present():
    """Test that all defined metrics are present in output."""
    from src.api.metrics import generate_latest
    
    output = generate_latest().decode("utf-8")
    
    expected_metrics = [
        "rag_http_requests_total",
        "rag_http_request_duration_seconds",
        "rag_http_requests_in_progress",
        "rag_query_total",
        "rag_query_duration_seconds",
        "rag_retrieval_duration_seconds",
        "rag_rerank_duration_seconds",
        "rag_generation_duration_seconds",
        "rag_retrieval_results_count",
        "rag_semaphore_active",
        "rag_semaphore_waiting",
        "rag_semaphore_max_concurrent",
        "rag_reranker_requests_total",
        "rag_reranker_inference_total",
        "rag_reranker_cache_hits_total",
        "rag_reranker_cache_size",
        "rag_milvus_pool_size",
        "rag_milvus_pool_active",
        "rag_redis_pool_size",
        "rag_redis_pool_active",
        "rag_documents_uploaded_total",
        "rag_documents_processed_total",
        "rag_documents_failed_total",
        "rag_document_processing_duration_seconds",
        "rag_ai_generation_total",
        "rag_ai_generation_tokens_total",
        "rag_cache_hits_total",
        "rag_cache_misses_total",
        "rag_cache_size",
        "rag_app_info",
    ]
    
    missing = []
    for metric in expected_metrics:
        if metric not in output:
            missing.append(metric)
    
    if missing:
        print(f"[FAIL] Missing metrics: {missing}")
        return False
    
    print(f"[PASS] All {len(expected_metrics)} custom metrics are present")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Prometheus Metrics Integration Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_metrics_import,
        test_middleware_import,
        test_path_normalization,
        test_metrics_generation,
        test_fastapi_integration,
        test_all_metrics_present,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
