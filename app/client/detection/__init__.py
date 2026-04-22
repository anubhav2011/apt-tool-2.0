"""Detection clients package.

Switch between backends by setting the DETECTION_CLIENT environment variable:

    DETECTION_CLIENT=mediapipe   (default)
    DETECTION_CLIENT=intel

The factory function ``get_detection_client`` reads this variable at runtime and
returns the matching client instance, so no application code needs to change when
you rotate backends.

Client implementations live in the ``face_tracking`` subfolder:
  - face_tracking/base.py              — Abstract base class
  - face_tracking/mediapipe_client.py  — MediaPipe backend
  - face_tracking/intel_client.py      — Intel OpenVINO backend
  - face_tracking/model_downloader.py  — Auto-download Intel models at startup
"""

from app.client.detection.face_tracking.base import BaseDetectionClient
from app.client.detection.factory import get_detection_client
from app.client.detection.face_tracking.model_downloader import (
    ensure_intel_models,
)

__all__ = [
    "BaseDetectionClient",
    "get_detection_client",
    "ensure_intel_models",
]

