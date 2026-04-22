"""
Intel OpenVINO Detection Client
================================
Concrete ``BaseDetectionClient`` backed by Intel's OpenVINO runtime.

Models used
-----------
* **Face detection** — ``face-detection-retail-0004`` (or any SSD-MobileNet
  compatible model; path controlled by ``INTEL_FACE_MODEL_XML``).
* **Facial landmarks / head-pose** — ``head-pose-estimation-adas-0001``
  (path controlled by ``INTEL_HEAD_POSE_MODEL_XML``).
* **Gaze estimation** — ``gaze-estimation-adas-0002``
  (path controlled by ``INTEL_GAZE_MODEL_XML``).

All model paths default to the standard Open Model Zoo layout:

    /opt/intel/openvino/models/<model-name>/FP32/<model-name>.xml

Override with environment variables if your layout differs.

Switching to this client
------------------------
Set the environment variable::

    DETECTION_CLIENT=intel

Environment variables
---------------------
INTEL_FACE_MODEL_XML      Path to face-detection IR .xml
INTEL_HEAD_POSE_MODEL_XML Path to head-pose IR .xml
INTEL_GAZE_MODEL_XML      Path to gaze-estimation IR .xml
INTEL_DEVICE              Inference device, e.g. CPU (default), GPU, AUTO

Graceful degradation
--------------------
If OpenVINO is not installed or any model file is missing the client falls
back to returning safe ``None`` values on every call rather than crashing the
pipeline.  A warning is logged once during ``_setup``.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from app.client.detection.base import BaseDetectionClient
from app.utils.logger import debug_logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_model_path(name: str) -> str:
    """Return the conventional Open Model Zoo path for *name*."""
    return os.path.join(
        "/opt/intel/openvino/models", name, "FP32", f"{name}.xml"
    )


# ---------------------------------------------------------------------------
# Intel detection client
# ---------------------------------------------------------------------------

class IntelDetectionClient(BaseDetectionClient):
    """
    Gaze + head-pose detection backed by Intel OpenVINO models.

    The public API is identical to ``MediaPipeDetectionClient`` so it can be
    swapped in transparently via the factory.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        self._available = False  # flipped to True only when all models load OK

        # ── Config shortcuts ──────────────────────────────────────────
        c = self.config
        self.landmark_indices = list(c.HEAD_POSE_LANDMARK_INDICES)
        self.model_points     = np.array(c.HEAD_POSE_MODEL_POINTS_MM, dtype=np.float64)

        # ── Kalman + gaze history (same as MediaPipe client) ──────────
        self._init_kalman_filters()
        self.gaze_history: List[Tuple[float, float]] = []
        self.max_history_size = 5

        # ── OpenVINO model loading ────────────────────────────────────
        try:
            from openvino.runtime import Core  # type: ignore[import]
        except ImportError:
            debug_logger.warning(
                "IntelDetectionClient: openvino package not installed. "
                "All detection calls will return None values. "
                "Install it with: pip install openvino"
            )
            return

        device = os.getenv("INTEL_DEVICE", "CPU").strip().upper()

        face_xml      = os.getenv(
            "INTEL_FACE_MODEL_XML",
            _default_model_path("face-detection-retail-0004"),
        )
        head_pose_xml = os.getenv(
            "INTEL_HEAD_POSE_MODEL_XML",
            _default_model_path("head-pose-estimation-adas-0001"),
        )
        gaze_xml      = os.getenv(
            "INTEL_GAZE_MODEL_XML",
            _default_model_path("gaze-estimation-adas-0002"),
        )

        try:
            ie = Core()

            self._face_model      = ie.compile_model(
                ie.read_model(face_xml), device
            )
            self._head_pose_model = ie.compile_model(
                ie.read_model(head_pose_xml), device
            )
            self._gaze_model      = ie.compile_model(
                ie.read_model(gaze_xml), device
            )

            # Cache output layer names so we don't call them per-frame
            self._face_out      = self._face_model.output(0)
            self._hp_yaw_out    = self._head_pose_model.output("angle_y_fc")
            self._hp_pitch_out  = self._head_pose_model.output("angle_p_fc")
            self._hp_roll_out   = self._head_pose_model.output("angle_r_fc")
            self._gaze_out      = self._gaze_model.output("gaze_vector")

            self._available = True
            debug_logger.info(
                f"IntelDetectionClient: models loaded on device={device}"
            )
        except Exception as exc:
            debug_logger.warning(
                f"IntelDetectionClient: model loading failed ({exc}). "
                f"Falling back to pass-through (None) responses."
            )

    def cleanup(self) -> None:
        # OpenVINO compiled models are garbage-collected; nothing explicit needed.
        self.gaze_history.clear()
        self._available = False

    # ------------------------------------------------------------------
    # Kalman smoothing (shared implementation)
    # ------------------------------------------------------------------

    def _init_kalman_filters(self) -> None:
        def _make() -> cv2.KalmanFilter:
            kf                     = cv2.KalmanFilter(2, 1)
            kf.measurementMatrix   = np.array([[1, 0]], np.float32)
            kf.transitionMatrix    = np.array([[1, 1], [0, 1]], np.float32)
            kf.processNoiseCov     = np.eye(2, dtype=np.float32) * 0.03
            kf.measurementNoiseCov = np.array([[1]], np.float32) * 0.1
            return kf

        self.kf_horizontal      = _make()
        self.kf_vertical        = _make()
        self.kalman_initialized = False

    def _smooth_gaze_with_kalman(self, h: float, v: float) -> Tuple[float, float]:
        try:
            if not self.kalman_initialized:
                self.kf_horizontal.statePre = np.array([[h], [0]], np.float32)
                self.kf_vertical.statePre   = np.array([[v], [0]], np.float32)
                self.kalman_initialized     = True
                return h, v
            self.kf_horizontal.predict()
            self.kf_vertical.predict()
            h_c = self.kf_horizontal.correct(np.array([[h]], np.float32))
            v_c = self.kf_vertical.correct(np.array([[v]], np.float32))
            return float(h_c[0][0]), float(v_c[0][0])
        except Exception:
            return h, v

    # ------------------------------------------------------------------
    # Face detection (internal)
    # ------------------------------------------------------------------

    def _detect_faces(
        self, frame: NDArray[np.uint8]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Run face-detection model and return list of bounding boxes in
        (x1, y1, x2, y2) pixel coordinates, sorted by confidence desc.
        """
        h, w   = frame.shape[:2]
        inp_h, inp_w = 300, 300
        blob   = cv2.resize(frame, (inp_w, inp_h))
        blob   = blob.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        detections = self._face_model({self._face_model.input(0): blob})[self._face_out]
        # shape: (1, 1, N, 7)  — [image_id, label, conf, x1, y1, x2, y2]
        detections = detections[0][0]

        boxes: List[Tuple[int, int, int, int]] = []
        for det in detections:
            conf = float(det[2])
            if conf < 0.5:
                continue
            x1 = int(np.clip(det[3] * w, 0, w))
            y1 = int(np.clip(det[4] * h, 0, h))
            x2 = int(np.clip(det[5] * w, 0, w))
            y2 = int(np.clip(det[6] * h, 0, h))
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        return boxes

    # ------------------------------------------------------------------
    # Head-pose from face crop (internal)
    # ------------------------------------------------------------------

    def _head_pose_from_crop(
        self, crop: NDArray[np.uint8]
    ) -> Tuple[float, float, float, float]:
        """Return (yaw_deg, pitch_deg, roll_deg, confidence)."""
        blob = cv2.resize(crop, (60, 60)).transpose(2, 0, 1)[np.newaxis].astype(np.float32)
        infer_req = self._head_pose_model
        result    = infer_req({infer_req.input(0): blob})

        yaw_deg   = float(result[self._hp_yaw_out][0][0])
        pitch_deg = float(result[self._hp_pitch_out][0][0])
        roll_deg  = float(result[self._hp_roll_out][0][0])

        # Clamp pathological values (mirror MediaPipe client behaviour)
        if pitch_deg > 90.0:
            pitch_deg = 180.0 - pitch_deg
        elif pitch_deg < -90.0:
            pitch_deg = -180.0 - pitch_deg
        if abs(pitch_deg) > 70.0:
            pitch_deg = 0.0
        if abs(yaw_deg) > 85.0:
            yaw_deg = 0.0

        # Simple confidence from head-pose magnitude proximity to centre
        yaw_conf   = float(np.clip(1.0 - abs(yaw_deg) / 90.0, 0.0, 1.0))
        pitch_conf = float(np.clip(1.0 - abs(pitch_deg) / 70.0, 0.0, 1.0))
        head_conf  = float(0.6 * yaw_conf + 0.4 * pitch_conf)

        return round(yaw_deg, 2), round(pitch_deg, 2), round(roll_deg, 2), round(head_conf, 2)

    # ------------------------------------------------------------------
    # Gaze from face crop + head pose (internal)
    # ------------------------------------------------------------------

    def _gaze_from_crop(
        self,
        left_eye_crop:  NDArray[np.uint8],
        right_eye_crop: NDArray[np.uint8],
        head_pose_angles: Tuple[float, float, float],
    ) -> Tuple[float, float, float]:
        """
        Run gaze-estimation model.
        Returns (gaze_h_deg, gaze_v_deg, confidence).
        """
        def _eye_blob(crop: NDArray[np.uint8]) -> NDArray[np.float32]:
            return cv2.resize(crop, (60, 60)).transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        yaw_rad, pitch_rad, _ = head_pose_angles
        hp_blob = np.array([[yaw_rad, pitch_rad, _]], dtype=np.float32)

        gaze_model  = self._gaze_model
        gaze_vector = gaze_model({
            gaze_model.input("left_eye_image"):  _eye_blob(left_eye_crop),
            gaze_model.input("right_eye_image"): _eye_blob(right_eye_crop),
            gaze_model.input("head_pose_angles"): hp_blob,
        })[self._gaze_out][0]

        # gaze_vector: (x, y, z) unit vector
        gx, gy = float(gaze_vector[0]), float(gaze_vector[1])
        h_deg  = float(np.degrees(np.arctan2(gx, gaze_vector[2])))
        v_deg  = float(np.degrees(np.arctan2(gy, gaze_vector[2])))
        conf   = float(np.clip(np.linalg.norm(gaze_vector[:2]), 0.0, 1.0))
        return round(h_deg, 2), round(v_deg, 2), round(conf, 2)

    # ------------------------------------------------------------------
    # Occlusion (simple proxy — no landmark mesh available)
    # ------------------------------------------------------------------

    def _occlusion_from_box(
        self, box: Tuple[int, int, int, int], frame_w: int, frame_h: int
    ) -> float:
        x1, y1, x2, y2 = box
        margin = 0.05
        left_ok   = x1 / frame_w > margin
        top_ok    = y1 / frame_h > margin
        right_ok  = x2 / frame_w < (1.0 - margin)
        bottom_ok = y2 / frame_h < (1.0 - margin)
        clipped   = sum([not left_ok, not top_ok, not right_ok, not bottom_ok])
        return round(min(clipped * 0.25, 1.0), 2)

    # ------------------------------------------------------------------
    # Eye-crop extraction helper
    # ------------------------------------------------------------------

    @staticmethod
    def _eye_crop(
        frame: NDArray[np.uint8],
        x1: int, y1: int, x2: int, y2: int,
        side: str,
    ) -> Optional[NDArray[np.uint8]]:
        """Return left or right half of a face bounding box as a crop."""
        fh, fw = frame.shape[:2]
        bw = x2 - x1
        bh = y2 - y1
        if bw < 4 or bh < 4:
            return None
        if side == "left":
            ex1, ex2 = x1, x1 + bw // 2
        else:
            ex1, ex2 = x1 + bw // 2, x2
        ey1 = y1 + int(bh * 0.15)
        ey2 = y1 + int(bh * 0.55)
        ex1, ex2 = max(ex1, 0), min(ex2, fw)
        ey1, ey2 = max(ey1, 0), min(ey2, fh)
        if ex2 <= ex1 or ey2 <= ey1:
            return None
        crop = frame[ey1:ey2, ex1:ex2]
        return crop if crop.size > 0 else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_gaze(
        self, frame: NDArray[np.uint8]
    ) -> Tuple[
        Optional[float], Optional[float], int,
        Optional[Tuple[float, float]], float, float,
    ]:
        if not self._available:
            return None, None, 0, None, 0.0, 0.0

        h, w = frame.shape[:2]

        try:
            boxes = self._detect_faces(frame)
        except Exception as exc:
            debug_logger.debug(f"IntelDetectionClient.detect_gaze face-det error: {exc}")
            return None, None, 0, None, 0.0, 0.0

        num_faces = len(boxes)
        if num_faces == 0:
            return None, None, 0, None, 0.0, 0.0

        x1, y1, x2, y2 = boxes[0]
        bbox_center = (float((x1 + x2) / 2), float((y1 + y2) / 2))
        occlusion   = self._occlusion_from_box((x1, y1, x2, y2), w, h)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None, None, num_faces, bbox_center, 0.0, occlusion

        try:
            yaw_d, pitch_d, roll_d, _ = self._head_pose_from_crop(face_crop)
        except Exception as exc:
            debug_logger.debug(f"IntelDetectionClient head-pose error: {exc}")
            return None, None, num_faces, bbox_center, 0.0, occlusion

        left_crop  = self._eye_crop(frame, x1, y1, x2, y2, "left")
        right_crop = self._eye_crop(frame, x1, y1, x2, y2, "right")
        if left_crop is None or right_crop is None:
            return None, None, num_faces, bbox_center, 0.0, occlusion

        try:
            h_deg, v_deg, conf = self._gaze_from_crop(
                left_crop, right_crop,
                (float(np.radians(yaw_d)), float(np.radians(pitch_d)), float(np.radians(roll_d))),
            )
        except Exception as exc:
            debug_logger.debug(f"IntelDetectionClient gaze-est error: {exc}")
            return None, None, num_faces, bbox_center, 0.0, occlusion

        h_deg, v_deg = self._smooth_gaze_with_kalman(h_deg, v_deg)

        self.gaze_history.append((h_deg, v_deg))
        if len(self.gaze_history) > self.max_history_size:
            self.gaze_history.pop(0)

        # Weighted temporal smoothing (same weights as MediaPipe client)
        if len(self.gaze_history) >= 5:
            wts    = np.array([0.05, 0.10, 0.15, 0.25, 0.45])
            recent = self.gaze_history[-5:]
            h_deg  = float(np.average([x[0] for x in recent], weights=wts))
            v_deg  = float(np.average([x[1] for x in recent], weights=wts))
        elif len(self.gaze_history) >= 3:
            wts    = np.array([0.2, 0.3, 0.5])
            recent = self.gaze_history[-3:]
            h_deg  = float(np.average([x[0] for x in recent], weights=wts))
            v_deg  = float(np.average([x[1] for x in recent], weights=wts))

        return (
            round(h_deg, 2), round(v_deg, 2),
            num_faces, bbox_center,
            conf, occlusion,
        )

    def detect_head_pose(
        self, frame: NDArray[np.uint8]
    ) -> Tuple[Optional[float], Optional[float], Optional[float], float]:
        if not self._available:
            return None, None, None, 0.0

        try:
            boxes = self._detect_faces(frame)
        except Exception as exc:
            debug_logger.debug(f"IntelDetectionClient.detect_head_pose face-det error: {exc}")
            return None, None, None, 0.0

        if not boxes:
            return None, None, None, 0.0

        x1, y1, x2, y2 = boxes[0]
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None, None, None, 0.0

        try:
            return self._head_pose_from_crop(face_crop)
        except Exception as exc:
            debug_logger.debug(f"IntelDetectionClient head-pose error: {exc}")
            return None, None, None, 0.0

    def get_landmark_vector(
        self, frame: NDArray[np.uint8]
    ) -> Optional[NDArray[np.float32]]:
        """
        The Intel pipeline does not produce a 468-landmark mesh compatible
        with the TVT model, so this always returns None.

        TVT inference is therefore automatically skipped when using the
        Intel client — this is the correct safe default.
        """
        return None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def client_name(self) -> str:
        return "intel"
