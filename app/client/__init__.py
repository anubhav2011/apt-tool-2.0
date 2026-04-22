"""External service clients (S3, etc.)."""

from app.client.s3_client import S3ClientFactory

__all__ = ["S3ClientFactory"]
