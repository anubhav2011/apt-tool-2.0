"""
Detection Client Factory
========================
Reads the ``DETECTION_CLIENT`` environment variable and returns the
corresponding concrete ``BaseDetectionClient`` instance.

Supported values
----------------
``mediapipe``  (default, case-insensitive)
    Use ``MediaPipeDetectionClient``.

``intel``
    Use ``IntelDetectionClient`` (requires openvino + model files).

Switching at runtime
--------------------
Change ``DETECTION_CLIENT`` in your environment **before** starting the
application (or worker).  No code changes are needed — the factory will
pick up the new value on next process start.

Example — switch to Intel::

    export DETECTION_CLIENT=intel
    uvicorn app.main:app

Example — switch back to MediaPipe::

    export DETECTION_CLIENT=mediapipe
    uvicorn app.main:app

Adding new backends
-------------------
1. Create ``app/client/detection/<name>_client.py`` that subclasses
   ``BaseDetectionClient``.
2. Add a branch to the ``_REGISTRY`` dict below.
3. Set ``DETECTION_CLIENT=<name>`` in your environment.
"""

from __future__ import annotations

import os
from typing import Dict, Type

from app.client.detection.face_tracking.base import BaseDetectionClient
from app.utils.logger import debug_logger


# ---------------------------------------------------------------------------
# Registry — maps env-var value → concrete class (lazy import to avoid
# importing mediapipe when only intel is needed, and vice versa).
# ---------------------------------------------------------------------------

def _load_mediapipe_class() -> Type[BaseDetectionClient]:
    from app.client.detection.face_tracking.mediapipe_client import (
        MediaPipeDetectionClient,
    )
    return MediaPipeDetectionClient


def _load_intel_class() -> Type[BaseDetectionClient]:
    from app.client.detection.face_tracking.intel_client import IntelDetectionClient
    return IntelDetectionClient


_REGISTRY: Dict[str, callable] = {
    "mediapipe": _load_mediapipe_class,
    "intel":     _load_intel_class,
}

_DEFAULT_CLIENT = "mediapipe"


# ---------------------------------------------------------------------------
# Public factory function
# ---------------------------------------------------------------------------

def get_detection_client(config) -> BaseDetectionClient:
    """
    Instantiate and return the detection client specified by the
    ``DETECTION_CLIENT`` environment variable.

    Parameters
    ----------
    config:
        The ``ProctoringConfig`` instance to pass to the client constructor.

    Returns
    -------
    BaseDetectionClient
        A fully initialised detection client ready to call ``detect_gaze``
        and ``detect_head_pose`` on.

    Raises
    ------
    ValueError
        If ``DETECTION_CLIENT`` is set to an unrecognised value.
    """
    raw     = (os.getenv("DETECTION_CLIENT") or _DEFAULT_CLIENT).strip().lower()
    loader  = _REGISTRY.get(raw)

    if loader is None:
        valid = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unknown DETECTION_CLIENT='{raw}'. "
            f"Valid options: {valid}. "
            f"Defaulting is not done when an explicit unknown value is set."
        )

    client_class = loader()
    client       = client_class(config)

    debug_logger.info(
        f"Detection client initialised: {client.client_name} "
        f"(DETECTION_CLIENT={raw!r})"
    )
    return client
