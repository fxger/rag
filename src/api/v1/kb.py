"""Knowledge-base retrieval API — retrieval only, tenant scoped.

Why this endpoint exists instead of reusing ``/api/v1/query/``:

* ``/query/`` always runs a second LLM call to synthesise an answer.  The
  customer-service agent already does its own generation; the answer would be
  paid for and discarded.
* ``/query/`` truncates every ``sources[].content`` to
  ``content[:200] + "..."`` — useless as evidence for the agent.
* ``/query/`` has no tenant dimension at all: one collection, no filter, and
  its result cache key is ``sha256(query:top_k:rerank)``, so two stores asking
  the same question would collide.

The contract is written down in
``D:\\Desktop\\agent\\customer_service\\docs\\RAG多租户知识库接入方案.md`` §5.1.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends

from src.api.middleware.auth import get_current_user
from src.api.schemas.kb import (
    PLATFORM_TENANT,
    KbChunk,
    KbRetrieveRequest,
    KbRetrieveResponse,
)
from src.config.milvus import get_collection_fields, search_vectors
from src.config.settings import get_settings
from src.core.embedding.embedder import get_embedding_generator
from src.core.retrieval.models import SearchResult

logger = logging.getLogger(__name__)

# No prefix here on purpose: ``main.py`` already prefixes every router with
# ``/api/v1/<name>``, and the four pre-existing routers *also* declare their
# own prefix, which is why the live paths are ``/api/v1/query/query/``.
# Declaring none makes this router's real path ``/api/v1/kb/retrieve``.
router = APIRouter(tags=["Knowledge Base"])

_settings = get_settings()

# A chunk longer than this is almost always a whole document mis-chunked.
# Truncating keeps one bad chunk from eating the agent's context window.
_MAX_CONTENT_CHARS = 4000


def _elapsed_ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _degraded(
    tenant_id: str,
    started: float,
    reason: str,
    mode: str = "vector",
) -> KbRetrieveResponse:
    """Uniform "no knowledge" answer — never raises, never blocks the caller."""
    logger.warning("kb.retrieve degraded: %s (tenant=%s)", reason, tenant_id)
    return KbRetrieveResponse(
        tenant_id=tenant_id,
        chunks=[],
        degraded=True,
        degraded_reason=reason,
        latency_ms=_elapsed_ms(started),
        mode=mode,
    )


def _row_tenant(row: dict[str, Any]) -> str:
    """Tenant of a Milvus row.

    Legacy collections have no ``tenant_id`` field; the value then only exists
    inside the ``metadata`` JSON string.  Rows written before tenancy existed
    count as platform content (they are the old demo corpus) — that is a
    deliberate, documented leak: legacy rows are visible to every tenant until
    the collection is rebuilt with the tenant field.
    """
    metadata = row.get("metadata") or {}
    if isinstance(metadata, dict):
        return str(metadata.get("tenant_id") or PLATFORM_TENANT)
    return PLATFORM_TENANT


def _build_search_expr(tenant_id: str) -> str:
    """Milvus expression: this tenant's chunks plus the shared platform pool."""
    return f'tenant_id in ["{tenant_id}", "{PLATFORM_TENANT}"]'


async def _embed_with_budget(query: str, budget_s: float) -> list[float]:
    """Embed the query, honouring the caller's deadline.

    The embedding service measured 2–8 s per call on the VM, so a deadline is
    not paranoia: without it one slow embedding turns into a 30 s agent turn.
    """
    return await asyncio.wait_for(
        get_embedding_generator().generate_with_cache(query),
        timeout=max(0.5, budget_s),
    )


@router.post("/retrieve", response_model=KbRetrieveResponse)
async def retrieve(
    payload: KbRetrieveRequest,
    user: dict = Depends(get_current_user),  # noqa: B008 — FastAPI 依赖注入的标准写法
) -> KbRetrieveResponse:
    """Retrieve raw chunks for one query under one tenant.

    Never raises for an infrastructure problem: it reports ``degraded`` so the
    agent can fall back to its own FAQ table instead of failing the turn.
    """
    started = time.perf_counter()
    tenant_id = (payload.tenant_id or PLATFORM_TENANT).strip() or PLATFORM_TENANT
    deadline_s = payload.deadline_ms / 1000.0
    collection = _settings.MILVUS_COLLECTION

    # ── 1. Which filtering strategy does this collection support? ────────
    fields = await asyncio.to_thread(get_collection_fields, collection)
    server_side_filter = "tenant_id" in fields
    mode = "vector+tenant-filter" if server_side_filter else "vector+post-filter"

    # ── 2. Embedding (largest single cost) ──────────────────────────────
    try:
        embedding = await _embed_with_budget(payload.query, deadline_s * 0.55)
    except TimeoutError:
        return _degraded(tenant_id, started, "embedding_timeout", mode)
    except Exception as exc:
        return _degraded(
            tenant_id, started, f"embedding_error:{type(exc).__name__}", mode,
        )

    # ── 3. Vector search ────────────────────────────────────────────────
    # Legacy collections cannot filter server-side, so over-fetch and drop
    # other tenants' rows afterwards.  Over-fetching is capped because the
    # result set crosses the wire in full.
    fetch_k = payload.top_k if server_side_filter else min(payload.top_k * 5, 100)
    expr = _build_search_expr(tenant_id) if server_side_filter else None

    try:
        rows = await asyncio.wait_for(
            asyncio.to_thread(search_vectors, embedding, fetch_k, collection, expr),
            timeout=max(0.5, deadline_s * 0.35),
        )
    except TimeoutError:
        return _degraded(tenant_id, started, "search_timeout", mode)
    except Exception as exc:
        return _degraded(tenant_id, started, f"search_error:{type(exc).__name__}", mode)

    if not server_side_filter:
        rows = [r for r in rows if _row_tenant(r) in (tenant_id, PLATFORM_TENANT)]

    if not rows:
        return KbRetrieveResponse(
            tenant_id=tenant_id,
            chunks=[],
            latency_ms=_elapsed_ms(started),
            mode=mode,
        )

    # ── 4. Optional rerank (off by default: CPU cross-encoder is slow) ──
    results = [
        SearchResult(
            chunk_id=row.get("chunk_id", ""),
            document_id=row.get("document_id", ""),
            content=row.get("content", ""),
            score=float(row.get("score", 0.0)),
            rank=index + 1,
            metadata=row.get("metadata") or {},
            source_type="vector",
        )
        for index, row in enumerate(rows)
    ]

    if payload.rerank and _settings.ENABLE_RERANK:
        remaining = deadline_s - (time.perf_counter() - started)
        try:
            from src.core.singleton import get_reranker

            reranker = get_reranker(model_name=_settings.RERANKER_MODEL)
            results = await asyncio.wait_for(
                reranker.rerank(payload.query, results, top_n=payload.top_k),
                timeout=max(0.3, remaining * 0.8),
            )
        except Exception as exc:
            logger.warning("kb.retrieve rerank skipped: %s", exc)

    # ── 5. Trim to top_k and serialise ──────────────────────────────────
    chunks = [
        KbChunk(
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            content=item.content[:_MAX_CONTENT_CHARS],
            score=round(float(item.score), 4),
            tenant_id=_row_tenant(
                {"metadata": item.metadata if isinstance(item.metadata, dict) else {}},
            ),
            metadata=item.metadata if isinstance(item.metadata, dict) else {},
        )
        for item in results[: payload.top_k]
    ]

    logger.info(
        "kb.retrieve tenant=%s q=%r -> %d chunks in %dms (%s)",
        tenant_id,
        payload.query[:40],
        len(chunks),
        _elapsed_ms(started),
        mode,
    )

    return KbRetrieveResponse(
        tenant_id=tenant_id,
        chunks=chunks,
        degraded=False,
        latency_ms=_elapsed_ms(started),
        mode=mode,
    )
