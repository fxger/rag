"""
MinIO object storage client configuration.

Manages document file storage on MinIO at VM1 (192.168.86.132:9000).
"""

import logging
from typing import Optional

from minio import Minio
from minio.error import S3Error

from src.config.settings import get_settings

logger = logging.getLogger(__name__)

_client: Optional[Minio] = None


def get_minio_client() -> Minio:
    """Get or create MinIO client singleton."""
    global _client
    if _client is not None:
        return _client

    settings = get_settings()
    _client = Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )

    # Ensure bucket exists
    if not _client.bucket_exists(settings.MINIO_BUCKET):
        _client.make_bucket(settings.MINIO_BUCKET)
        logger.info("Created MinIO bucket: %s", settings.MINIO_BUCKET)
    else:
        logger.info("MinIO bucket exists: %s", settings.MINIO_BUCKET)

    return _client


def upload_file(
    local_path: str,
    object_name: str,
    content_type: Optional[str] = None,
) -> str:
    """Upload a local file to MinIO.

    **Deliberately synchronous.**  The minio SDK is blocking, and every caller
    wraps this in ``asyncio.to_thread``.  These three functions used to be
    declared ``async def`` while the call sites called them through
    ``to_thread`` — which meant ``to_thread`` invoked the coroutine function,
    received an un-awaited coroutine object back, and silently discarded it.
    The upload never happened and no error was raised.

    Args:
        local_path: Path of the file to upload.
        object_name: Object key in the bucket.
        content_type: MIME type; guessed from the filename when omitted.

    Returns:
        The object name used.
    """
    client = get_minio_client()
    settings = get_settings()
    kwargs = {"content_type": content_type} if content_type else {}
    client.fput_object(
        settings.MINIO_BUCKET,
        object_name,
        local_path,
        **kwargs,
    )
    logger.info(
        "Uploaded %s -> %s/%s", local_path, settings.MINIO_BUCKET, object_name
    )
    return object_name


def download_file(object_name: str, dest_path: str) -> str:
    """Download an object from MinIO to a local path (synchronous, see above).

    Args:
        object_name: Object key in the bucket.
        dest_path: Local path to write to; its parent directory must exist.

    Returns:
        The destination path.
    """
    client = get_minio_client()
    settings = get_settings()
    client.fget_object(settings.MINIO_BUCKET, object_name, dest_path)
    logger.info(
        "Downloaded %s/%s -> %s", settings.MINIO_BUCKET, object_name, dest_path
    )
    return dest_path


def delete_file(object_name: str) -> bool:
    """Delete a file from MinIO. Returns True if successful."""
    client = get_minio_client()
    settings = get_settings()
    try:
        client.remove_object(settings.MINIO_BUCKET, object_name)
        logger.info("Deleted file from MinIO: %s/%s", settings.MINIO_BUCKET, object_name)
        return True
    except S3Error as e:
        logger.error("Failed to delete file from MinIO: %s", e)
        return False