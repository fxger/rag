# API Overview

## Base URL
```
http://localhost:8000
```

## Authentication
All protected endpoints require a JWT Bearer token:
```
Authorization: Bearer <access_token>
```

Obtain tokens via `/api/v1/auth/login`.

## Endpoints Summary

| Method | Endpoint                        | Description         | Auth     |
|--------|---------------------------------|---------------------|----------|
| GET    | /health                         | Liveness check      | No       |
| GET    | /health/ready                   | Readiness check     | No       |
| POST   | /api/v1/auth/login              | User login          | No       |
| POST   | /api/v1/auth/register           | User registration   | No       |
| POST   | /api/v1/auth/refresh            | Refresh token       | Yes      |
| GET    | /api/v1/auth/me                 | Current user        | Yes      |
| POST   | /api/v1/query/                  | RAG query           | Yes      |
| POST   | /api/v1/query/stream            | Streaming query     | Yes      |
| GET    | /api/v1/query/history           | Query history       | Yes      |
| POST   | /api/v1/query/{id}/feedback     | Submit feedback     | Yes      |
| POST   | /api/v1/documents/upload        | Upload document     | Yes      |
| GET    | /api/v1/documents/              | List documents      | Yes      |
| GET    | /api/v1/documents/{id}          | Document details    | Yes      |
| DELETE | /api/v1/documents/{id}          | Delete document     | Yes      |
| GET    | /api/v1/admin/stats             | System statistics   | Admin    |
| GET    | /api/v1/admin/users             | List all users      | Admin    |
| POST   | /api/v1/admin/reindex           | Rebuild index       | Admin    |
| GET    | /api/v1/admin/audit-logs        | Audit logs          | Admin    |

## Rate Limits
- **Users**: 100 requests/minute
- **Global**: 10,000 requests/minute