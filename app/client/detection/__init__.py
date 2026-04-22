"""Detection clients package.

Switch between backends by setting the DETECTION_CLIENT environment variable:

    DETECTION_CLIENT=mediapipe   (default)
    DETECTION_CLIENT=intel

The factory function ``get_detection_client`` reads this variable at runtime and
returns the matching client instance, so no application code needs to change when
you rotate backends.
"""

from app.client.detection.base import BaseDetectionClient
from app.client.detection.factory import get_detection_client

__all__ = [
    "BaseDetectionClient",
    "get_detection_client",
]
