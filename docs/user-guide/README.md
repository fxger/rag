# Quick Start Guide

## Prerequisites
- Python 3.11+
- Access to VM services (PostgreSQL, Redis, Milvus, MinIO)

## 1. Installation

```bash
# Clone and enter project
cd enterprise-rag-system

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e ".[dev]"

# Copy and edit environment variables
copy .env.example .env
# Edit .env with your OPENAI_API_KEY
```

## 2. Initialize Database

```bash
python scripts/init_db.py
```

## 3. Start the Server

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## 4. Access the API

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## 5. First Query

```bash
# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# Use the access_token from response:
curl -X POST http://localhost:8000/api/v1/query/ \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello, what can you do?"}'
```