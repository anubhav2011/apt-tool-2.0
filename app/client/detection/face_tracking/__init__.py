"""Face tracking detection clients module."""

from app.client.detection.face_tracking.base import BaseDetectionClient
from app.client.detection.face_tracking.mediapipe_client import (
    MediaPipeDetectionClient,
)
from app.client.detection.face_tracking.intel_client import IntelDetectionClient
from app.client.detection.factory import get_detection_client

__all__ = [
    "BaseDetectionClient",
    "MediaPipeDetectionClient",
    "IntelDetectionClient",
    "get_detection_client",
]
