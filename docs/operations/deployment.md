# Deployment Guide

## Docker Deployment

### Development
```bash
docker compose up -d
```

### Production
```bash
# Build
docker build -t enterprise-rag:latest .

# Run with production compose
docker compose -f docker-compose.prod.yml up -d
```

## Manual Deployment

### 1. Install Dependencies
```bash
pip install -e .
```

### 2. Configure Environment
```bash
copy .env.example .env
# Edit .env for production
```

### 3. Initialize Database
```bash
python scripts/init_db.py
```

### 4. Run with Uvicorn
```bash
uvicorn src.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level warning
```

## Health Checks
```bash
# Quick check
python scripts/health_check.py

# HTTP check
curl http://localhost:8000/health/ready
```

## Monitoring
- **Prometheus**: Scrapes metrics from `/metrics` endpoint
- **Grafana**: Import dashboard from `monitoring/grafana/dashboards/rag.json`
- **Logs**: Structured JSON logging to stdout