"""CORS middleware configuration helper."""
from typing import Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def configure_cors(
    app: FastAPI,
    allowed_origins: Optional[list[str]] = None,
    allow_credentials: bool = True,
    allowed_methods: Optional[list[str]] = None,
    allowed_headers: Optional[list[str]] = None,
    expose_headers: Optional[list[str]] = None,
    max_age: int = 600,
) -> None:
    """Configure CORS middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
        allowed_origins: List of allowed origins (default: ["*"])
        allow_credentials: Whether to allow credentials
        allowed_methods: List of allowed HTTP methods (default: ["*"])
        allowed_headers: List of allowed headers (default: ["*"])
        expose_headers: List of headers to expose to the browser
        max_age: Max age for preflight cache in seconds
    """
    if allowed_origins is None:
        allowed_origins = ["*"]
    if allowed_methods is None:
        allowed_methods = ["*"]
    if allowed_headers is None:
        allowed_headers = ["*"]
    
    cors_params = {
        "allow_origins": allowed_origins,
        "allow_credentials": allow_credentials,
        "allow_methods": allowed_methods,
        "allow_headers": allowed_headers,
        "max_age": max_age,
    }
    
    if expose_headers:
        cors_params["expose_headers"] = expose_headers
    
    app.add_middleware(CORSMiddleware, **cors_params)


def get_test_cors_config() -> dict:
    """Get a CORS configuration suitable for testing.
    
    Returns:
        Dictionary with test CORS configuration
    """
    return {
        "allowed_origins": ["http://localhost:3000", "http://localhost:8000"],
        "allow_credentials": True,
        "allowed_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allowed_headers": ["Authorization", "Content-Type"],
        "expose_headers": ["X-Request-ID", "X-Process-Time"],
        "max_age": 3600,
    }
