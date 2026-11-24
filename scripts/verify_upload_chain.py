#!/usr/bin/env python
"""End-to-end check of the document ingestion chain.

Covers the path that C1 wired up:

    local file -> MinIO -> PostgreSQL(Document) -> parse/chunk -> PostgreSQL(chunks)
               -> embed (Qwen on VM1) -> Milvus -> retrievable

Unlike ``scripts/verify_gap_fixes.py`` (which asserts the *wiring* from source),
this script exercises the live services, so it needs MinIO, PostgreSQL, Milvus
and the embedding service to be reachable.  It cleans up after itself.

Why this exists: the earlier C1 verification had to bypass MinIO because the
container was down, and that hid a real defect — ``src/config/minio.py``
declared its three helpers ``async def`` while every call site went through
``asyncio.to_thread``, so the coroutine was never awaited and the upload
silently did nothing.

Usage:
    .venv\\Scripts\\python.exe -u scripts/verify_upload_chain.py
    .venv\\Scripts\\python.exe -u scripts/verify_upload_chain.py --file test_data/company_overview.txt
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import text  # noqa: E402

from src.config.database import async_session_factory  # noqa: E402
from src.config.milvus import get_milvus_client, search_vectors
from src.config.minio import get_minio_client  # noqa: E402
from src.config.settings import get_settings  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def _ok(name: str, detail: str = "") -> None:
    PASSED.append(name)
    print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))


def _fail(name: str, detail: str) -> None:
    FAILED.append(name)
    print(f"  [FAIL] {name} — {detail}")


async def _ensure_owner() -> str:
    """Return an existing user id, creating a throwaway one if the table is empty."""
    async with async_session_factory() as s:
        row = (await s.execute(text("SELECT id FROM users LIMIT 1"))).fetchone()
        if row:
            return str(row[0])

    from src.models import User, UserRole

    uid = uuid.uuid4()
    async with async_session_factory() as s:
        async with s.begin():
            s.add(
                User(
                    id=uid,
                    username=f"chainprobe_{uid.hex[:8]}",
                    email=f"{uid.hex[:8]}@probe.local",
                    hashed_password="x",
                    role=UserRole.USER,
                    is_active=True,
                )
            )
    print(f"  (users 表为空，已创建临时用户 {uid})")
    return str(uid)


def _milvus_count() -> int:
    """Number of *queryable* rows in the production collection.

    Deliberately not ``Collection.num_entities``: that counter includes
    logically-deleted rows until a compaction runs, so it drifts upward every
    time a document is created and deleted.  Counting query results reflects
    what retrieval can actually see.

    A flush is issued first so freshly inserted rows are included.
    """
    client = get_milvus_client()
    s = get_settings()
    client.flush(s.MILVUS_COLLECTION)
    rows = client.query(
        collection_name=s.MILVUS_COLLECTION,
        filter="",
        output_fields=["chunk_id"],
        limit=16384,        # Milvus caps (offset + limit) at 16384
    )
    return len(rows)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="test_data/company_overview.txt")
    # Pick a query that is distinctive to the test file.  "2025年营收是多少"
    # looks like the obvious choice, but the corpus already contains a very
    # similar 《2025财务报告》 chunk, so the test file ranks outside the top 30
    # and the assertion fails for reasons unrelated to the ingestion chain.
    ap.add_argument("--query", default="总部设在哪里")
    args = ap.parse_args()

    local_file = ROOT / args.file
    if not local_file.exists():
        print(f"测试文件不存在: {local_file}")
        return 2

    settings = get_settings()
    print("文档入库链路端到端验证")
    print(f"  文件   : {args.file} ({local_file.stat().st_size} bytes)")
    print(f"  查询   : {args.query}")
    print(f"  MinIO  : {settings.MINIO_ENDPOINT}/{settings.MINIO_BUCKET}")
    print(f"  Milvus : {settings.MILVUS_HOST}:{settings.MILVUS_PORT}/{settings.MILVUS_COLLECTION}")
    print()

    from src.services.document_service import DocumentService

    service = DocumentService()
    client = get_minio_client()

    owner_id = await _ensure_owner()
    doc_id: str | None = None
    object_path: str | None = None
    created_user = None
    baseline = _milvus_count()

    try:
        print("── 1. 落 MinIO + 建库记录 ──")
        created = await service.upload_document(
            file_path=str(local_file),
            owner_id=owner_id,
            content_type="text/plain",
        )
        doc_id = created["document_id"]
        record = await service.get_document(doc_id)
        assert record is not None, "get_document 返回 None"
        object_path = record.get("file_path")
        print(f"  document_id = {doc_id}")
        print(f"  file_path   = {object_path}  status={record.get('status')}")

        try:
            stat = client.stat_object(settings.MINIO_BUCKET, object_path)
            assert stat.size == local_file.stat().st_size, (
                f"MinIO 对象大小 {stat.size} != 本地 {local_file.stat().st_size}"
            )
            _ok("1a 对象已落入 MinIO", f"{stat.size} bytes")
        except Exception as exc:  # noqa: BLE001
            _fail("1a 对象已落入 MinIO", f"{type(exc).__name__}: {exc}")
            return 1

        assert record["status"] == "pending", f"初始状态应为 pending，实际 {record['status']}"
        _ok("1b 库记录已建", f"file_name={record.get('file_name')} status=pending")

        print("── 2. 下载 → 解析 → 分块 → 写 PostgreSQL ──")
        result = await service.process_document(doc_id)
        print(f"  process_document -> {result}")
        assert result.get("status") == "completed", f"处理未完成: {result}"

        async with async_session_factory() as s:
            chunk_rows = (
                await s.execute(
                    text(
                        # id first: Milvus stores DocumentChunk.id as chunk_id
                        # (see DocumentService._index_chunks), so that is what
                        # the retrieval results must be matched against.
                        "SELECT id, chunk_index, content, token_count FROM document_chunks "
                        "WHERE document_id = :d ORDER BY chunk_index"
                    ),
                    {"d": doc_id},
                )
            ).fetchall()
        if chunk_rows:
            _ok(
                "2a 分块已入库",
                f"{len(chunk_rows)} 块, tokens={[r[3] for r in chunk_rows]}",
            )
        else:
            _fail("2a 分块已入库", "document_chunks 无记录")
            return 1

        print("── 3. 向量化 → 写入 Milvus ──")
        indexed = result.get("indexed_chunks", 0)
        after = _milvus_count()
        if indexed and after - baseline == indexed:
            _ok("3a 向量已入 Milvus", f"新增 {after - baseline} 条（{baseline} -> {after}）")
        else:
            _fail("3a 向量已入 Milvus", f"indexed={indexed}, 实体数 {baseline} -> {after}")

        print("── 4. 可被检索命中 ──")
        from src.core.embedding.embedder import EmbeddingGenerator

        embedding = await EmbeddingGenerator().generate_single(args.query)
        hits = await asyncio.to_thread(search_vectors, embedding, 20)
        expected_ids = {str(r[0]) for r in chunk_rows}
        ranks = {i + 1 for i, h in enumerate(hits) if str(h.get("chunk_id")) in expected_ids}
        print(f"  检索「{args.query}」命中 {len(hits)} 条（top-20）")
        for i, h in enumerate(hits[:5], 1):
            mark = "★本次入库" if str(h.get("chunk_id")) in expected_ids else " "
            print(f"    #{i} {h.get('score', 0):.4f} {mark} {str(h.get('content',''))[:40]}…")
        if ranks:
            rank = min(ranks)
            _ok("4 本次入库内容可被检索", f"在 top-20 中排 #{rank}")
        else:
            _fail("4 本次入库内容可被检索", "top-20 未命中本次写入的分块")

        print("── 5. 列表与计数 ──")
        docs = await service.list_documents(limit=200)
        total = await service.count_documents()
        if any(d["id"] == uuid.UUID(doc_id) for d in docs):
            _ok("5 列表/计数", f"list={len(docs)} 条, count={total}")
        else:
            _fail("5 列表/计数", "新建的文档未出现在 list_documents 中")

    finally:
        print("── 6. 删除（含对象清理）──")
        if doc_id:
            try:
                async with async_session_factory() as s:
                    row = (
                        await s.execute(
                            text("SELECT file_path FROM documents WHERE id = :d"),
                            {"d": doc_id},
                        )
                    ).fetchone()
                obj = row[0] if row else object_path

                removed = await service.delete_document(doc_id)
                assert removed, "delete_document 返回 False"

                gone_pg = False
                async with async_session_factory() as s:
                    left = (
                        await s.execute(
                            text("SELECT count(*) FROM documents WHERE id = :d"),
                            {"d": doc_id},
                        )
                    ).scalar_one()
                gone_pg = left == 0

                gone_minio = False
                if obj:
                    try:
                        client.stat_object(settings.MINIO_BUCKET, obj)
                    except Exception:  # noqa: BLE001 — NoSuchKey is the success case
                        gone_minio = True

                if gone_pg and gone_minio:
                    _ok("6 删除已清理库记录与对象", "PostgreSQL + MinIO 均已清理")
                else:
                    _fail("6 删除已清理库记录与对象", f"pg_gone={gone_pg} minio_gone={gone_minio}")
            except Exception as exc:  # noqa: BLE001
                _fail("6 删除已清理库记录与对象", f"{type(exc).__name__}: {exc}")

        # 清理临时用户（仅当本次新建）
        async with async_session_factory() as s:
            row = (
                await s.execute(
                    text("SELECT username FROM users WHERE id = :u"), {"u": owner_id}
                )
            ).fetchone()
        if row and str(row[0]).startswith("chainprobe_"):
            async with async_session_factory() as s:
                async with s.begin():
                    await s.execute(text("DELETE FROM users WHERE id = :u"), {"u": owner_id})
            print("  (已清理本次创建的临时用户)")

    print()
    print(f"结果：通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        for name in FAILED:
            print(f"  - {name}")
        return 1
    print("文档入库链路端到端通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
