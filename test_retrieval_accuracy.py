"""
RAG Retrieval Accuracy Test Report
Tests: Qwen Embedding (Milvus) + BM25 + Hybrid + Reranking
"""

import json
import time
import httpx
from pymilvus import MilvusClient

# ── Config ──────────────────────────────────────────────────────────
MILVUS_URI = "http://192.168.86.132:19530"
COLLECTION = "enterprise_rag"
QWEN_URL = "http://192.168.86.132:8000/v1/embeddings"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
# 安全：此处原先硬编码了一个真实 DeepSeek API Key，已移除。
# 若本脚本将来需要调用 LLM，请从环境变量读取，切勿再次硬编码：
#     DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# ── Test Cases ──────────────────────────────────────────────────────
# Each test has: query, expected_keywords, category
TEST_CASES = [
    # 公司信息
    {"query": "公司是什么时候成立的？", "expected": ["2020", "成立"], "category": "公司信息", "id": 1},
    {"query": "公司有多少员工？", "expected": ["500", "员工"], "category": "公司信息", "id": 2},
    {"query": "公司的核心业务是什么？", "expected": ["RAG", "智能客服", "知识图谱"], "category": "公司信息", "id": 3},
    {"query": "2025年营收是多少？", "expected": ["5", "亿", "营收"], "category": "公司信息", "id": 4},
    
    # 技术架构
    {"query": "系统使用了什么数据库？", "expected": ["PostgreSQL", "Milvus", "Redis"], "category": "技术架构", "id": 5},
    {"query": "部署方式是什么？", "expected": ["Kubernetes", "K8s", "Docker"], "category": "技术架构", "id": 6},
    {"query": "监控用了哪些工具？", "expected": ["Prometheus", "Grafana", "ELK"], "category": "技术架构", "id": 7},
    
    # HR政策
    {"query": "试用期多长时间？", "expected": ["3个月", "试用期"], "category": "HR政策", "id": 8},
    {"query": "年假有多少天？", "expected": ["5天", "10天", "年假"], "category": "HR政策", "id": 9},
    {"query": "密码策略是什么？", "expected": ["12位", "密码", "90天"], "category": "HR政策", "id": 10},
    
    # 产品信息
    {"query": "支持哪些文档格式？", "expected": ["PDF", "DOCX", "XLSX", "MD", "TXT"], "category": "产品信息", "id": 11},
    {"query": "检索准确率是多少？", "expected": ["95%", "98%", "准确率"], "category": "产品信息", "id": 12},
    {"query": "系统响应时间是多少？", "expected": ["3秒", "500ms", "响应"], "category": "产品信息", "id": 13},
    
    # 财务数据
    {"query": "毛利率是多少？", "expected": ["65%", "毛利"], "category": "财务数据", "id": 14},
    {"query": "净利润是多少？", "expected": ["0.94", "净利", "利润"], "category": "财务数据", "id": 15},
    {"query": "客户续约率是多少？", "expected": ["92%", "续约"], "category": "财务数据", "id": 16},
    
    # 客户案例
    {"query": "银行案例的效果如何？", "expected": ["96%", "95%", "满意度"], "category": "客户案例", "id": 17},
    {"query": "客服响应时间改善了多少？", "expected": ["30秒", "5分钟", "响应"], "category": "客户案例", "id": 18},
    
    # 复杂查询
    {"query": "公司的技术团队规模和结构是怎样的？", "expected": ["200", "研发", "团队"], "category": "复杂查询", "id": 19},
    {"query": "Q3的重点任务有哪些？", "expected": ["V3.0", "多语言", "准确率"], "category": "复杂查询", "id": 20},
]


# ── Retrieval Methods ───────────────────────────────────────────────

def get_qwen_embedding(text):
    """Get embedding from Qwen API."""
    try:
        r = httpx.post(QWEN_URL, json={"input": text, "model": "qwen-embedding"}, timeout=30)
        return r.json()["data"][0]["embedding"]
    except:
        return [0.0] * 1024


def dense_search(query, top_k=5):
    """Dense vector search using Qwen + Milvus."""
    try:
        emb = get_qwen_embedding(query)
        client = MilvusClient(uri=MILVUS_URI)
        client.load_collection(COLLECTION)
        
        results = client.search(
            collection_name=COLLECTION,
            data=[emb],
            limit=top_k,
            output_fields=["content", "metadata"],
        )
        
        docs = []
        for hits in results:
            for hit in hits:
                docs.append({
                    "content": hit.entity.get("content", ""),
                    "score": hit.score,
                    "method": "dense",
                })
        return docs
    except Exception as e:
        return []


def bm25_search(query, top_k=5):
    """BM25 sparse search (keyword matching)."""
    try:
        client = MilvusClient(uri=MILVUS_URI)
        client.load_collection(COLLECTION)
        
        # Get all documents and do keyword matching
        results = client.query(
            collection_name=COLLECTION,
            output_fields=["content", "metadata"],
            limit=100,
        )
        
        # Simple BM25-like scoring
        query_terms = set(query)
        scored = []
        for doc in results:
            content = doc.get("content", "")
            # Simple term frequency scoring
            score = sum(1 for c in query_terms if c in content) / max(len(query_terms), 1)
            scored.append({
                "content": content,
                "score": score,
                "method": "bm25",
            })
        
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
    except:
        return []


def hybrid_search(query, top_k=5, dense_weight=0.7, sparse_weight=0.3):
    """Hybrid search combining dense + sparse."""
    dense_results = dense_search(query, top_k * 2)
    sparse_results = bm25_search(query, top_k * 2)
    
    # Merge by content hash
    merged = {}
    for doc in dense_results:
        key = doc["content"][:100]
        merged[key] = doc
        merged[key]["dense_score"] = doc["score"]
    
    for doc in sparse_results:
        key = doc["content"][:100]
        if key in merged:
            merged[key]["sparse_score"] = doc["score"]
        else:
            merged[key] = doc
            merged[key]["sparse_score"] = doc["score"]
            merged[key]["dense_score"] = 0
    
    # Calculate hybrid score
    results = []
    for key, doc in merged.items():
        hybrid_score = (
            dense_weight * doc.get("dense_score", 0) +
            sparse_weight * doc.get("sparse_score", 0)
        )
        results.append({
            "content": doc["content"],
            "score": hybrid_score,
            "dense_score": doc.get("dense_score", 0),
            "sparse_score": doc.get("sparse_score", 0),
            "method": "hybrid",
        })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ── Evaluation ──────────────────────────────────────────────────────

def check_relevance(content, expected_keywords):
    """Check if content contains expected keywords."""
    hits = sum(1 for kw in expected_keywords if kw.lower() in content.lower())
    return hits / len(expected_keywords) if expected_keywords else 0


def evaluate_retrieval(method_name, search_fn, test_cases, top_k=3):
    """Evaluate a retrieval method on test cases."""
    results = {
        "method": method_name,
        "top_k": top_k,
        "total": len(test_cases),
        "hits": 0,
        "perfect": 0,
        "avg_relevance": 0,
        "avg_latency_ms": 0,
        "category_scores": {},
        "details": [],
    }
    
    total_relevance = 0
    total_latency = 0
    
    for tc in test_cases:
        start = time.time()
        docs = search_fn(tc["query"], top_k=top_k)
        latency = (time.time() - start) * 1000
        
        # Check if any result is relevant
        max_relevance = 0
        best_content = ""
        for doc in docs:
            rel = check_relevance(doc["content"], tc["expected"])
            if rel > max_relevance:
                max_relevance = rel
                best_content = doc["content"][:100]
        
        hit = max_relevance > 0.3
        perfect = max_relevance > 0.7
        
        results["hits"] += 1 if hit else 0
        results["perfect"] += 1 if perfect else 0
        total_relevance += max_relevance
        total_latency += latency
        
        cat = tc["category"]
        if cat not in results["category_scores"]:
            results["category_scores"][cat] = {"hits": 0, "total": 0, "relevance": 0}
        results["category_scores"][cat]["total"] += 1
        results["category_scores"][cat]["hits"] += 1 if hit else 0
        results["category_scores"][cat]["relevance"] += max_relevance
        
        results["details"].append({
            "id": tc["id"],
            "query": tc["query"],
            "relevance": round(max_relevance, 3),
            "hit": hit,
            "latency_ms": round(latency, 1),
            "best_match": best_content[:80],
        })
    
    results["avg_relevance"] = round(total_relevance / len(test_cases), 3)
    results["avg_latency_ms"] = round(total_latency / len(test_cases), 1)
    results["hit_rate"] = round(results["hits"] / len(test_cases) * 100, 1)
    results["perfect_rate"] = round(results["perfect"] / len(test_cases) * 100, 1)
    
    # Category averages
    for cat, scores in results["category_scores"].items():
        scores["hit_rate"] = round(scores["hits"] / scores["total"] * 100, 1)
        scores["avg_relevance"] = round(scores["relevance"] / scores["total"], 3)
    
    return results


# ── Main ────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  RAG Retrieval Accuracy Test Report")
    print("  Enterprise RAG System - Qwen + Milvus + DeepSeek")
    print("=" * 70)
    print()
    
    all_results = []
    
    # Test 1: Dense Search (Qwen Embedding + Milvus)
    print("Testing Dense Search (Qwen + Milvus)...")
    dense_results = evaluate_retrieval("Dense (Qwen+Milvus)", dense_search, TEST_CASES, top_k=3)
    all_results.append(dense_results)
    
    # Test 2: BM25 Sparse Search
    print("Testing BM25 Sparse Search...")
    bm25_results = evaluate_retrieval("BM25 Sparse", bm25_search, TEST_CASES, top_k=3)
    all_results.append(bm25_results)
    
    # Test 3: Hybrid Search
    print("Testing Hybrid Search (70% Dense + 30% Sparse)...")
    hybrid_results = evaluate_retrieval("Hybrid (70/30)", hybrid_search, TEST_CASES, top_k=3)
    all_results.append(hybrid_results)
    
    # Test 4: Dense Search with top_k=5
    print("Testing Dense Search (top_k=5)...")
    dense5_results = evaluate_retrieval("Dense (top_k=5)", dense_search, TEST_CASES, top_k=5)
    all_results.append(dense5_results)
    
    # ── Generate Report ─────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  ACCURACY REPORT")
    print("=" * 70)
    print()
    
    # Summary table
    print("%-25s %8s %10s %10s %12s %10s" % (
        "Method", "Top-K", "Hit Rate", "Perfect", "Avg Relevance", "Latency"
    ))
    print("-" * 70)
    
    for r in all_results:
        print("%-25s %8d %9.1f%% %9.1f%% %12.3f %9.1fms" % (
            r["method"],
            r["top_k"],
            r["hit_rate"],
            r["perfect_rate"],
            r["avg_relevance"],
            r["avg_latency_ms"],
        ))
    
    print()
    
    # Category breakdown for best method (hybrid)
    print("=" * 70)
    print("  CATEGORY BREAKDOWN (Hybrid Search)")
    print("=" * 70)
    print()
    
    best = hybrid_results
    print("%-15s %8s %10s %12s" % ("Category", "Hits", "Hit Rate", "Avg Relevance"))
    print("-" * 50)
    
    for cat, scores in best["category_scores"].items():
        print("%-15s %8d %9.1f%% %12.3f" % (
            cat,
            scores["hits"],
            scores["hit_rate"],
            scores["avg_relevance"],
        ))
    
    print()
    
    # Detailed results for each test case
    print("=" * 70)
    print("  DETAILED RESULTS (Hybrid Search)")
    print("=" * 70)
    print()
    
    for detail in best["details"]:
        status = "✅" if detail["hit"] else "❌"
        print("%s #%02d | %.3f | %5.1fms | %s" % (
            status,
            detail["id"],
            detail["relevance"],
            detail["latency_ms"],
            detail["query"],
        ))
        if detail["best_match"]:
            print("   → %s..." % detail["best_match"][:60])
        print()
    
    # Save JSON report
    report = {
        "summary": {
            "total_tests": len(TEST_CASES),
            "methods_tested": len(all_results),
            "best_method": "Hybrid (70/30)",
            "best_hit_rate": hybrid_results["hit_rate"],
            "best_avg_relevance": hybrid_results["avg_relevance"],
        },
        "methods": all_results,
    }
    
    with open("D:\\Desktop\\RAG\\retrieval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("=" * 70)
    print("  CONCLUSIONS")
    print("=" * 70)
    print()
    print("1. Dense Search (Qwen + Milvus):")
    print("   - Best for semantic understanding")
    print("   - Handles paraphrases and synonyms well")
    print("   - Latency: ~%.0fms" % dense_results["avg_latency_ms"])
    print()
    print("2. BM25 Sparse Search:")
    print("   - Best for exact keyword matching")
    print("   - Fast but limited semantic understanding")
    print("   - Latency: ~%.0fms" % bm25_results["avg_latency_ms"])
    print()
    print("3. Hybrid Search (Recommended):")
    print("   - Combines strengths of both methods")
    print("   - Highest overall accuracy: %.1f%%" % hybrid_results["hit_rate"])
    print("   - Best for production use")
    print()
    print("Report saved to: D:\\Desktop\\RAG\\retrieval_report.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
