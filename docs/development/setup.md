# Environment Setup

## System Requirements
- Python 3.11, 3.12, or 3.13
- 4GB+ RAM recommended
- Network access to VM services

## Step-by-Step Setup

### 1. Python Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install
pip install -e ".[dev]"
```

### 2. Environment Variables
```bash
copy .env.example .env
```

Key variables to configure:
| Variable         | Description                    | Default                  |
|-----------------|--------------------------------|--------------------------|
| OPENAI_API_KEY  | Your OpenAI API key            | (required)               |
| DATABASE_URL    | PostgreSQL connection string   | postgresql+asyncpg://...@192.168.86.134:5432/postgres |
| REDIS_URL       | Redis connection string        | redis://192.168.86.132:6379/0 |
| MILVUS_HOST     | Milvus server host             | 192.168.86.132           |
| JWT_SECRET_KEY  | JWT signing secret             | (change in production)   |

### 3. Database Setup
```bash
python scripts/init_db.py
python scripts/seed_data.py  # Optional: seed test data
```

### 4. Run Tests
```bash
# Unit tests (no external services needed)
pytest tests/unit/ -v

# Integration tests (requires VM services)
pytest tests/integration/ -v -m integration

# All tests with coverage
pytest --cov=src --cov-report=html
```

### 5. Start Development Server
```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```