# Enterprise RAG System

A production-grade Retrieval-Augmented Generation system supporting multi-format document processing, hybrid retrieval, metadata governance, and high-concurrency handling.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │   Auth   │  │  Query   │  │Document  │  │  Admin   │          │
│  │   API    │  │   API    │  │   API    │  │   API    │          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│       │              │              │              │                │
│  ┌────┴──────────────┴──────────────┴──────────────┴─────┐         │
│  │                   Core Services                       │         │
│  │  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌────────┐ │         │
│  │  │Document  │ │ Retrieval │ │Embedding │ │  RAG   │ │         │
│  │  │Processing│ │  Engine   │ │Generator │ │Service │ │         │
│  │  └──────────┘ └───────────┘ └──────────┘ └────────┘ │         │
│  └───────────────────────────────────────────────────────┘         │
│       │              │              │              │                │
│  ┌────┴──────────────┴──────────────┴──────────────┴─────┐         │
│  │              Infrastructure Layer                     │         │
│  │  ┌──────────┐ ┌───────────┐ ┌──────────┐ ┌────────┐ │         │
│  │  │PostgreSQL│ │   Redis   │ │  Milvus  │ │ MinIO  │ │         │
│  │  │ VM2:5432 │ │ VM1:6379  │ │VM1:19530 │ │VM1:9000│ │         │
│  │  └──────────┘ └───────────┘ └──────────┘ └────────┘ │         │
│  └───────────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component       | Technology                     |
|-----------------|--------------------------------|
| Web Framework   | FastAPI + Uvicorn              |
| Database        | PostgreSQL (asyncpg)           |
| Cache/Queue     | Redis                          |
| Vector DB       | Milvus 2.x                    |
| Object Storage  | MinIO (S3-compatible)          |
| Message Queue   | Kafka (optional)               |
| AI Models       | OpenAI (text-embedding-3-large, gpt-4-turbo-preview) |
| ORM             | SQLAlchemy 2.0 (async)         |

## Quick Start

### Prerequisites

- Python 3.11+
- Access to VM services (PostgreSQL, Redis, Milvus, MinIO)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd enterprise-rag-system

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -e ".[dev]"

# Copy environment variables
copy .env.example .env
# Edit .env with your actual configuration

# Run the application
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
enterprise-rag-system/
├── src/
│   ├── api/                    # API endpoints and middleware
│   │   ├── v1/                 # Versioned API routes
│   │   ├── middleware/          # Auth, rate limiting, CORS
│   │   ├── schemas/            # Pydantic request/response schemas
│   │   └── deps.py             # FastAPI dependency injection
│   ├── core/                   # Core business logic
│   │   ├── document/           # Document parsing, chunking, processing
│   │   ├── retrieval/          # Vector, sparse, hybrid retrieval
│   │   ├── generation/         # Query rewriting, context building, LLM
│   │   ├── embedding/          # Embedding generation and caching
│   │   ├── cache/              # Multi-level caching
│   │   └── metadata/           # Metadata extraction and management
│   ├── models/                 # SQLAlchemy ORM models
│   ├── services/               # Business service layer
│   ├── config/                 # Configuration management
│   ├── infrastructure/         # Connection pools, external clients
│   └── utils/                  # Shared utilities
├── tests/                      # Test suite (unit, integration, API, perf)
├── config/                     # Environment-specific configs
├── scripts/                    # Utility scripts
├── docker/                     # Docker configurations
├── docs/                       # Technical documentation
└── monitoring/                 # Prometheus + Grafana configs
```

## Development

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Lint
ruff check src/ tests/
ruff format src/ tests/

# Type check
mypy src/
```

## License

MIT