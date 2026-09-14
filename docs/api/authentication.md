# Authentication Guide

## Login

```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "admin123"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "access_token": "eyJhbGciOiJIUzI1NiIs...",
    "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
    "token_type": "bearer",
    "expires_in": 1800
  }
}
```

## Using the Token

Include the access token in the `Authorization` header:
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

## Register

```http
POST /api/v1/auth/register
Content-Type: application/json

{
  "username": "newuser",
  "email": "user@company.com",
  "password": "SecurePass123!",
  "department": "Engineering"
}
```

## Refresh Token

```http
POST /api/v1/auth/refresh
Content-Type: application/json

{
  "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
}
```

## Roles

| Role     | Permissions                                      |
|----------|--------------------------------------------------|
| admin    | Full access: users, docs, admin, reindex, audit  |
| analyst  | Query, document management, own history          |
| user     | Basic query, own document upload, own history    |