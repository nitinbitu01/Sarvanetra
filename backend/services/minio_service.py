"""
backend/services/minio_service.py — MinIO S3 Object Storage Service
with presigned URL generation and robust local disk fallback.
"""
import io
import os
import logging
from typing import Optional

logger = logging.getLogger("sentinel.minio")


class MinioService:
    def __init__(self, endpoint: Optional[str] = None, access_key: Optional[str] = None, secret_key: Optional[str] = None, secure: bool = False):
        self.endpoint = endpoint or os.getenv("MINIO_ENDPOINT", "minio:9000")
        self.access_key = (
            access_key
            or os.getenv("MINIO_ACCESS_KEY")
            or os.getenv("MINIO_ROOT_USER")
            or os.getenv("AWS_ACCESS_KEY_ID")
            or "sentineladmin"
        )
        self.secret_key = (
            secret_key
            or os.getenv("MINIO_SECRET_KEY")
            or os.getenv("MINIO_ROOT_PASSWORD")
            or os.getenv("AWS_SECRET_ACCESS_KEY")
            or "sentinel_dev_minio_admin_2026"
        )
        self.secure = secure or (os.getenv("MINIO_SECURE", "false").lower() == "true")
        self.client = None
        self._local_fallback_dir = "output/evidence_vault"
        os.makedirs(self._local_fallback_dir, exist_ok=True)
        self._init_client()

    def _init_client(self):
        try:
            from minio import Minio
            self.client = Minio(
                endpoint=self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure,
            )
            # Check connection
            self.client.list_buckets()
            logger.info(f"Connected to MinIO at {self.endpoint}")
        except Exception as e:
            logger.warning(f"MinIO initialization deferred/failed ({e}); using local storage fallback.")
            self.client = None

    async def upload_object(self, bucket: str, path: str, data: bytes, content_type: str = "image/jpeg") -> str:
        if self.client:
            try:
                if not self.client.bucket_exists(bucket):
                    self.client.make_bucket(bucket)
                self.client.put_object(
                    bucket_name=bucket,
                    object_name=path,
                    data=io.BytesIO(data),
                    length=len(data),
                    content_type=content_type,
                )
                return path
            except Exception as e:
                logger.warning(f"MinIO put_object failed ({e}), writing to local disk")

        # Local fallback
        local_path = os.path.join(self._local_fallback_dir, path.replace("/", "_"))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(data)
        return path

    async def delete_object(self, bucket: str, path: str):
        if self.client:
            try:
                self.client.remove_object(bucket_name=bucket, object_name=path)
            except Exception as e:
                logger.warning(f"MinIO delete_object error: {e}")
        local_path = os.path.join(self._local_fallback_dir, path.replace("/", "_"))
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except Exception:
                pass

    async def get_presigned_url(self, bucket: str, path: str, expiry_seconds: int = 3600) -> str:
        if self.client:
            try:
                from datetime import timedelta
                return self.client.presigned_get_object(bucket, path, expires=timedelta(seconds=expiry_seconds))
            except Exception as e:
                logger.warning(f"MinIO presigned_get_object error ({e})")
        return f"/api/v1/evidence/view/{path}"
