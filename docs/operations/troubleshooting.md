# Troubleshooting Guide

## Common Issues

### 1. Database Connection Failed
**Symptoms**: `ConnectionRefusedError` on startup
**Fix**: 
- Verify PostgreSQL is running on 192.168.86.134:5432
- Check firewall rules
- Verify credentials in `.env`

### 2. Redis Connection Failed
**Symptoms**: Cache errors in logs
**Fix**:
- Verify Redis is running on 192.168.86.132:6379
- Check `REDIS_URL` in `.env`

### 3. Milvus Connection Failed
**Symptoms**: Vector search errors
**Fix**:
- Verify Milvus is running on 192.168.86.132:19530
- Check `MILVUS_HOST` and `MILVUS_PORT`

### 4. OpenAI API Errors
**Symptoms**: Embedding/generation failures
**Fix**:
- Verify `OPENAI_API_KEY` is valid
- Check API quota and rate limits
- Set `OPENAI_BASE_URL` if using a proxy

### 5. Document Processing Fails
**Symptoms**: Status stuck at "processing"
**Fix**:
- Check file format is supported (PDF, DOCX, XLSX, MD, TXT)
- Verify file is not corrupted
- Check disk space for temp files

### 6. Slow Queries
**Symptoms**: High latency on `/api/v1/query/`
**Fix**:
- Enable caching (`MultiLevelCache`)
- Reduce `DEFAULT_TOP_K`
- Check Milvus index is loaded
- Monitor with Grafana dashboard

## Logs
```bash
# View logs
docker compose logs -f rag-api

# Filter by level
docker compose logs rag-api | grep ERROR
```

## Performance Tuning
- **Pool size**: Adjust `DATABASE_POOL_SIZE` based on concurrent users
- **Cache TTL**: Tune L1/L2 TTLs based on data freshness requirements
- **Chunk size**: Adjust `CHUNK_SIZE` for accuracy vs. speed tradeoff
- **Rate limits**: Adjust `RATE_LIMIT_*` based on expected traffic