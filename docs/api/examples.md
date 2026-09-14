# RAG Query Examples

## Basic Query

```bash
curl -X POST http://localhost:8000/api/v1/query/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the company revenue growth strategy?",
    "top_k": 10,
    "use_reranking": true
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "answer": "Based on the Q3 financial report, the company's revenue growth strategy focuses on...",
    "sources": [
      {
        "chunk_id": "...",
        "document_id": "...",
        "content": "relevant text snippet...",
        "score": 0.95,
        "document_title": "Q3 Financial Report"
      }
    ],
    "latency_ms": 1250
  }
}
```

## Streaming Query

```bash
curl -X POST http://localhost:8000/api/v1/query/stream \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the market analysis", "top_k": 5}' \
  --no-buffer
```

## Upload Document

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@quarterly_report.pdf"
```