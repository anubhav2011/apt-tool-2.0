"""External service clients (S3, detection, etc.)."""

from app.client.s3_client import S3ClientFactory
from app.client.detection import get_detection_client, BaseDetectionClient

__all__ = [
    "S3ClientFactory",
    "BaseDetectionClient",
    "get_detection_client",
]
