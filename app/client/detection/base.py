"""
Base Detection Client
=====================
Abstract contract that every detection backend must satisfy.

All concrete clients (MediaPipe, Intel, …) must subclass
``BaseDetectionClient`` and implement every abstract method.
The public method signatures are intentionally identical to those
previously on ``DetectionService`` so swapping backends requires
zero changes in calling code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


class BaseDetectionClient(ABC):
    """
    Contract for face-landmark, gaze, and head-pose detection.

    Lifecycle
    ---------
    1. ``__init__(config)`` — store config, call ``_setup()``.
    2. ``detect_gaze(frame)`` / ``detect_head_pose(frame)``  — per-frame calls.
    3. ``cleanup()`` — release native resources (called once, on shutdown).

    Thread safety
    -------------
    Clients are NOT required to be thread-safe internally. The caller
    (``VideoProcessingService``) serialises calls per client instance.
    """

    def __init__(self, config) -> None:
        self.config = config
        self._setup()

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _setup(self) -> None:
        """Initialise native resources (models, pipelines, etc.)."""

    @abstractmethod
    def cleanup(self) -> None:
        """Release native resources. Must be idempotent."""

    # ------------------------------------------------------------------
    # Core detection API
    # ------------------------------------------------------------------

    @abstractmethod
    def detect_gaze(
        self, frame: NDArray[np.uint8]
    ) -> Tuple[
        Optional[float],   # horizontal gaze angle (degrees), None if unavailable
        Optional[float],   # vertical gaze angle   (degrees), None if unavailable
        int,               # number of detected faces
        Optional[Tuple[float, float]],  # bbox centre (px), None if unavailable
        float,             # gaze confidence  [0, 1]
        float,             # face occlusion ratio [0, 1]
    ]:
        """
        Detect gaze direction from a single BGR frame.

        Returns
        -------
        (gaze_h, gaze_v, num_faces, bbox_center, gaze_confidence, occlusion_ratio)
        """

    @abstractmethod
    def detect_head_pose(
        self, frame: NDArray[np.uint8]
    ) -> Tuple[
        Optional[float],  # yaw   (degrees)
        Optional[float],  # pitch (degrees)
        Optional[float],  # roll  (degrees)
        float,            # head-pose confidence [0, 1]
    ]:
        """
        Estimate head orientation from a single BGR frame.

        Returns
        -------
        (yaw, pitch, roll, head_confidence)
        """

    @abstractmethod
    def get_landmark_vector(
        self, frame: NDArray[np.uint8]
    ) -> Optional[NDArray[np.float32]]:
        """
        Return a flat (936,) landmark vector for the primary face,
        or None if no face is detected.

        Used by the TVT temporal model. Clients that do not support
        468-landmark face meshes should return None.
        """

    # ------------------------------------------------------------------
    # Client identity
    # ------------------------------------------------------------------

    @property
    def client_name(self) -> str:
        """Human-readable name of the backend (e.g. 'mediapipe', 'intel')."""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"<{self.client_name} detection client>"
