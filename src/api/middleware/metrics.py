"""
Prometheus HTTP Metrics 自动采集中间件.

自动记录:
- 请求总数 (按 method, endpoint, status_code)
- 请求延迟 (Histogram)
- 当前进行中的请求数 (Gauge)
"""

import time
import re
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.api.metrics import (
    HTTP_REQUESTS_TOTAL,
    HTTP_REQUEST_DURATION,
    HTTP_REQUESTS_IN_PROGRESS,
)


def normalize_path(path: str) -> str:
    """标准化路径，合并动态路径段.
    
    Examples:
        /api/v1/query/abc123 → /api/v1/query/{id}
        /api/v1/documents/123/chunks → /api/v1/documents/{id}/chunks
    """
    # 替换 UUID
    path = re.sub(
        r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
        '{id}',
        path,
        flags=re.IGNORECASE,
    )
    
    # 替换纯数字 ID
    path = re.sub(r'/\d+', '/{id}', path)
    
    # 替换文件名
    path = re.sub(r'/[^/]+\.[^/]+$', '/{filename}', path)
    
    return path


class MetricsMiddleware(BaseHTTPMiddleware):
    """HTTP 请求指标自动采集中间件."""
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # 跳过 /metrics 端点自身
        if request.url.path == "/metrics":
            return await call_next(request)
        
        # 标准化路径
        path = normalize_path(request.url.path)
        method = request.method
        
        # 标记请求开始
        HTTP_REQUESTS_IN_PROGRESS.labels(method=method, endpoint=path).inc()
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # 记录成功请求
            duration = time.time() - start_time
            status_code = response.status_code
            
            HTTP_REQUESTS_TOTAL.labels(
                method=method,
                endpoint=path,
                status_code=str(status_code),
            ).inc()
            
            HTTP_REQUEST_DURATION.labels(
                method=method,
                endpoint=path,
            ).observe(duration)
            
            return response
            
        except Exception as e:
            # 记录失败请求
            duration = time.time() - start_time
            
            HTTP_REQUESTS_TOTAL.labels(
                method=method,
                endpoint=path,
                status_code="500",
            ).inc()
            
            HTTP_REQUEST_DURATION.labels(
                method=method,
                endpoint=path,
            ).observe(duration)
            
            raise
            
        finally:
            # 减少进行中的请求数
            HTTP_REQUESTS_IN_PROGRESS.labels(method=method, endpoint=path).dec()
