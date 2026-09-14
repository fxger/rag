# Architecture Overview

## System Architecture

The Enterprise RAG System follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Layer                                │
│  FastAPI + Uvicorn │ JWT Auth │ Rate Limiting │ CORS           │
├─────────────────────────────────────────────────────────────────┤
│                     Service Layer                               │
│  RAGService │ DocumentService │ AuthService │ AdminService     │
├─────────────────────────────────────────────────────────────────┤
│                      Core Layer                                 │
│  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌────────────────┐  │
│  │ Document  │ │ Retrieval │ │Embedding │ │   Generation   │  │
│  │Processing │ │  Engine   │ │Generator │ │    (LLM)       │  │
│  └──────────┘ └───────────┘ └──────────┘ └────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                   Infrastructure Layer                          │
│  PostgreSQL │ Redis │ Milvus │ MinIO │ Kafka (optional)       │
└─────────────────────────────────────────────────────────────────┘
```

## Key Components

### Document Processing Pipeline
1. **Upload** → MinIO storage
2. **Parse** → Format-specific parsers (PDF, DOCX, XLSX, MD, TXT)
3. **Clean** → HTML strip, encoding normalization, whitespace cleanup
4. **Chunk** → SmartChunker with type-aware strategies
5. **Embed** → OpenAI text-embedding-3-large (3072-dim)
6. **Store** → Milvus vectors + PostgreSQL metadata

### Hybrid Retrieval Pipeline
1. **Query Analysis** → Type classification, keyword extraction
2. **Dense Search** → Milvus cosine similarity
3. **Sparse Search** → BM25 with jieba tokenization
4. **Fusion** → Weighted + Reciprocal Rank Fusion
5. **Reranking** → Cross-encoder (ms-marco-MiniLM-L-6-v2)
6. **Generation** → GPT-4-turbo with source citations

### Multi-Level Cache
- **L1**: In-process TTLCache (sub-ms, hot data)
- **L2**: Redis distributed cache (1-5ms, shared)
- **Anti-avalanche**: Random TTL + mutex locks

## Infrastructure

| Service    | Location          | Port  | Purpose                |
|------------|-------------------|-------|------------------------|
| PostgreSQL | 192.168.86.134    | 5432  | Relational data        |
| Redis      | 192.168.86.132    | 6379  | Cache, rate limiting   |
| Milvus     | 192.168.86.132    | 19530 | Vector storage         |
| MinIO      | 192.168.86.132    | 9000  | Object storage         |
| Kafka      | 192.168.86.134    | 9092  | Message queue (v1 opt) |