"""Simplified RAG Retrieval Test"""
import json, time, httpx
from pymilvus import MilvusClient

MILVUS_URI = "http://192.168.86.132:19530"
COLLECTION = "enterprise_rag"
QWEN_URL = "http://192.168.86.132:8000/v1/embeddings"

TESTS = [
    {"q": "公司什么时候成立的", "e": ["2020", "成立"]},
    {"q": "公司有多少员工", "e": ["500", "员工"]},
    {"q": "核心业务是什么", "e": ["RAG", "智能客服"]},
    {"q": "2025年营收", "e": ["5", "亿"]},
    {"q": "使用什么数据库", "e": ["PostgreSQL", "Milvus", "Redis"]},
    {"q": "部署方式", "e": ["Kubernetes", "Docker"]},
    {"q": "试用期多长", "e": ["3个月"]},
    {"q": "年假多少天", "e": ["5天", "10天"]},
    {"q": "支持哪些文档格式", "e": ["PDF", "DOCX", "XLSX"]},
    {"q": "检索准确率", "e": ["95%", "98%"]},
    {"q": "毛利率", "e": ["65%"]},
    {"q": "净利润", "e": ["0.94"]},
    {"q": "客户续约率", "e": ["92%"]},
    {"q": "银行案例效果", "e": ["96%", "满意度"]},
    {"q": "Q3重点任务", "e": ["V3.0", "多语言"]},
]

def get_emb(text):
    r = httpx.post(QWEN_URL, json={"input": text, "model": "qwen-embedding"}, timeout=30)
    return r.json()["data"][0]["embedding"]

def search(query, top_k=3):
    emb = get_emb(query)
    client = MilvusClient(uri=MILVUS_URI)
    client.load_collection(COLLECTION)
    results = client.search(collection_name=COLLECTION, data=[emb], limit=top_k, output_fields=["content"])
    docs = []
    for hits in results:
        for hit in hits:
            docs.append({"content": hit.entity.get("content", ""), "score": hit.score})
    return docs

def relevance(content, expected):
    hits = sum(1 for kw in expected if kw.lower() in content.lower())
    return hits / len(expected)

print("=" * 60)
print("  RAG Retrieval Accuracy Test")
print("=" * 60)

hit_count = 0
perfect_count = 0
total_rel = 0
total_lat = 0

for i, t in enumerate(TESTS):
    start = time.time()
    docs = search(t["q"], top_k=3)
    lat = (time.time() - start) * 1000
    
    max_rel = max(relevance(d["content"], t["e"]) for d in docs) if docs else 0
    hit = max_rel > 0.3
    perfect = max_rel > 0.7
    
    hit_count += hit
    perfect_count += perfect
    total_rel += max_rel
    total_lat += lat
    
    status = "OK" if hit else "MISS"
    print("[%s] #%02d | %.3f | %5.1fms | %s" % (status, i+1, max_rel, lat, t["q"]))

n = len(TESTS)
print()
print("=" * 60)
print("  RESULTS")
print("=" * 60)
print("  Hit Rate:      %.1f%% (%d/%d)" % (hit_count/n*100, hit_count, n))
print("  Perfect Rate:  %.1f%% (%d/%d)" % (perfect_count/n*100, perfect_count, n))
print("  Avg Relevance: %.3f" % (total_rel/n))
print("  Avg Latency:   %.0fms" % (total_lat/n))
print("=" * 60)

report = {"hit_rate": round(hit_count/n*100, 1), "perfect_rate": round(perfect_count/n*100, 1), 
          "avg_relevance": round(total_rel/n, 3), "avg_latency_ms": round(total_lat/n, 1)}
with open("D:\\Desktop\\RAG\\retrieval_report.json", "w") as f:
    json.dump(report, f, indent=2)
