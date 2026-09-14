# System Design

## Data Flow

### Document Ingestion Flow
```
User Upload → API → DocumentService → MinIO (store file)
                                     → PostgreSQL (create record)
                                     → DocumentProcessor
                                        → Parse → Clean → Chunk
                                        → Embed → Milvus (vectors)
                                        → PostgreSQL (update chunks/status)
```

### Query Flow
```
User Query → API → RAGService
                   → EmbeddingCache.get(query)
                   → [Cache miss] → Embedder.generate(query)
                   → HybridRetriever.retrieve(query, embedding)
                      → QueryAnalyzer.analyze(query)
                      → VectorRetriever.search(embedding) [parallel]
                      → SparseRetriever.search(query)     [parallel]
                      → ResultFusion.fuse(dense, sparse)
                      → Reranker.rerank(query, fused, top_n)
                   → ContextBuilder.build(query, results)
                   → AnswerGenerator.generate_with_sources(query, context)
                   → QueryLog (store to Redis)
                   → Return answer + sources
```

## Database Schema

### ER Diagram (Simplified)
```
users ─────┬── documents ────┬── document_chunks
           │                 └── document_metadata
           ├── query_logs
           └── audit_logs
```

### Key Tables
- **users**: Authentication, roles, department
- **documents**: File metadata, processing status, content hash (dedup)
- **document_chunks**: Text chunks with Milvus embedding_id reference
- **document_metadata**: Typed key-value pairs with confidence scores
- **query_logs**: Query/response pairs with latency and feedback
- **audit_logs**: Compliance trail for all system actions

## Milvus Collection Schema
| Field        | Type          | Description                    |
|-------------|---------------|--------------------------------|
| id          | INT64 (PK)    | Auto-generated                 |
| document_id | VARCHAR(64)   | FK to PostgreSQL documents     |
| chunk_id    | VARCHAR(64)   | FK to PostgreSQL chunks        |
| content     | VARCHAR(65535)| Original text content          |
| embedding   | FLOAT_VECTOR  | 3072-dim embedding             |
| metadata    | VARCHAR(8192) | JSON metadata                  |

**Index**: IVF_FLAT with COSINE metric, nlist=1024