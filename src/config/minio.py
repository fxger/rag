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


async def upload_file(object_name: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """
    Upload a file to MinIO.

    Args:
        object_name: Object key in the bucket.
        data: File content bytes.
        content_type: MIME type of the file.

    Returns:
        The object name used.
    """
    import io
    client = get_minio_client()
    settings = get_settings()
    data_stream = io.BytesIO(data)
    client.put_object(
        bucket_name=settings.MINIO_BUCKET,
        object_name=object_name,
        data=data_stream,
        length=len(data),
        content_type=content_type,
    )
    logger.info("Uploaded file to MinIO: %s/%s", settings.MINIO_BUCKET, object_name)
    return object_name


async def download_file(object_name: str) -> bytes:
    """
    Download a file from MinIO.

    Args:
        object_name: Object key in the bucket.

    Returns:
        File content bytes.
    """
    client = get_minio_client()
    settings = get_settings()
    response = client.get_object(settings.MINIO_BUCKET, object_name)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


async def delete_file(object_name: str) -> bool:
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