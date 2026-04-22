"""
Intel OpenVINO Detection Client
================================
Concrete ``BaseDetectionClient`` backed by the four Intel Open Model Zoo
ADAS / gaze models.  The model wrapper classes below are taken verbatim
from the reference implementation (testing.py) so inference behaviour is
identical.

Models used (downloaded automatically by ``model_downloader.ensure_intel_models``)
---------
1. face-detection-adas-0001          — SSD face detector
2. facial-landmarks-35-adas-0002     — 35-point landmark detector
3. head-pose-estimation-adas-0001    — yaw / pitch / roll regressor
4. gaze-estimation-adas-0002         — gaze vector estimator

Path resolution
---------------
Model paths are resolved in this order (first match wins):

1. Per-model environment variable override
   (INTEL_FACE_MODEL_XML, INTEL_LANDMARKS_MODEL_XML,
    INTEL_HEAD_POSE_MODEL_XML, INTEL_GAZE_MODEL_XML)
2. Auto-downloaded location via ``model_downloader.get_model_paths()``
   (``<this_file_dir>/intel/<model>/FP32/<model>.xml``)
3. Graceful no-op — all detection calls return ``None`` values with a
   one-time warning logged.

Switching to this client
------------------------
Set the environment variable before starting the server::

    DETECTION_CLIENT=intel
    uvicorn app.main:app

Switching back to MediaPipe::

    DETECTION_CLIENT=mediapipe   # or unset entirely
    uvicorn app.main:app
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from app.client.detection.face_tracking.base import BaseDetectionClient
from app.client.detection.face_tracking.model_downloader import get_model_paths
from app.utils.logger import debug_logger


# ============================================================================
#  Model wrapper classes  (ported verbatim from reference testing.py)
# ============================================================================

class _FaceDetector:
    """
    Wrapper for face-detection-adas-0001.
    Input : [1, 3, 384, 672]  BGR image
    Output: [1, 1, N, 7]  each row = [img_id, label, conf, x1, y1, x2, y2]
    """

    def __init__(self, core, model_path: str, device: str = "CPU", threshold: float = 0.5):
        self.threshold = threshold
        model = core.read_model(model=model_path)
        self.compiled = core.compile_model(model=model, device_name=device)
        self.input_layer  = self.compiled.input(0)
        self.output_layer = self.compiled.output(0)
        self.n, self.c, self.h, self.w = self.input_layer.shape
        debug_logger.info(
            f"[IntelClient] FaceDetector loaded  "
            f"input={self.input_layer.shape}  output={self.output_layer.shape}"
        )

    def detect(self, frame: NDArray[np.uint8]) -> List[Tuple[int, int, int, int]]:
        """Return list of (x1, y1, x2, y2) bounding boxes in pixel coords."""
        frame_h, frame_w = frame.shape[:2]

        blob = cv2.resize(frame, (self.w, self.h))
        blob = blob.transpose((2, 0, 1))
        blob = blob.reshape((self.n, self.c, self.h, self.w)).astype(np.float32)

        detections = self.compiled([blob])[self.output_layer][0][0]

        faces: List[Tuple[int, int, int, int]] = []
        for det in detections:
            conf = float(det[2])
            if conf > self.threshold:
                x1 = int(np.clip(det[3] * frame_w, 0, frame_w - 1))
                y1 = int(np.clip(det[4] * frame_h, 0, frame_h - 1))
                x2 = int(np.clip(det[5] * frame_w, 0, frame_w - 1))
                y2 = int(np.clip(det[6] * frame_h, 0, frame_h - 1))
                if x2 > x1 and y2 > y1:
                    faces.append((x1, y1, x2, y2))
        return faces


class _FacialLandmarksDetector:
    """
    Wrapper for facial-landmarks-35-adas-0002.
    Input : [1, 3, 60, 60]  BGR face crop
    Output: [1, 70]  → 35 landmarks × (x, y), normalised 0..1
    Landmark 0,1 = left-eye corners; 2,3 = right-eye corners.
    """

    def __init__(self, core, model_path: str, device: str = "CPU"):
        model = core.read_model(model=model_path)
        self.compiled = core.compile_model(model=model, device_name=device)
        self.input_layer  = self.compiled.input(0)
        self.output_layer = self.compiled.output(0)
        self.n, self.c, self.h, self.w = self.input_layer.shape
        debug_logger.info(
            f"[IntelClient] LandmarksDetector loaded  "
            f"input={self.input_layer.shape}  output={self.output_layer.shape}"
        )

    def detect(
        self, face_image: NDArray[np.uint8]
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Returns
        -------
        left_eye_center  : ndarray [x, y] in face-crop pixel coords
        right_eye_center : ndarray [x, y] in face-crop pixel coords
        all_landmarks    : ndarray (35, 2) normalised
        """
        face_h, face_w = face_image.shape[:2]

        blob = cv2.resize(face_image, (self.w, self.h))
        blob = blob.transpose((2, 0, 1))
        blob = blob.reshape((self.n, self.c, self.h, self.w)).astype(np.float32)

        raw = self.compiled([blob])[self.output_layer]
        pts = raw.reshape(-1, 2)          # (35, 2) normalised

        # Eye corners → centres (same arithmetic as reference)
        le_left   = pts[0] * [face_w, face_h]
        le_right  = pts[1] * [face_w, face_h]
        re_left   = pts[2] * [face_w, face_h]
        re_right  = pts[3] * [face_w, face_h]

        left_eye_center  = ((le_left + le_right) / 2.0).astype(int)
        right_eye_center = ((re_left + re_right) / 2.0).astype(int)

        return left_eye_center, right_eye_center, pts


class _HeadPoseEstimator:
    """
    Wrapper for head-pose-estimation-adas-0001.
    Input : [1, 3, 60, 60]  BGR face crop
    Outputs: angle_y_fc (yaw), angle_p_fc (pitch), angle_r_fc (roll) in degrees.
    """

    def __init__(self, core, model_path: str, device: str = "CPU"):
        model = core.read_model(model=model_path)
        self.compiled = core.compile_model(model=model, device_name=device)
        self.input_layer = self.compiled.input(0)
        self.n, self.c, self.h, self.w = self.input_layer.shape

        self.out_y = self.compiled.output("angle_y_fc")
        self.out_p = self.compiled.output("angle_p_fc")
        self.out_r = self.compiled.output("angle_r_fc")
        debug_logger.info(
            f"[IntelClient] HeadPoseEstimator loaded  input={self.input_layer.shape}"
        )

    def estimate(self, face_image: NDArray[np.uint8]) -> Tuple[float, float, float]:
        """Return (yaw_deg, pitch_deg, roll_deg)."""
        blob = cv2.resize(face_image, (self.w, self.h))
        blob = blob.transpose((2, 0, 1))
        blob = blob.reshape((self.n, self.c, self.h, self.w)).astype(np.float32)

        res   = self.compiled([blob])
        yaw   = float(res[self.out_y][0][0])
        pitch = float(res[self.out_p][0][0])
        roll  = float(res[self.out_r][0][0])
        return yaw, pitch, roll


class _GazeEstimator:
    """
    Wrapper for gaze-estimation-adas-0002.
    Inputs:
        left_eye_image   : [1, 3, 60, 60]
        right_eye_image  : [1, 3, 60, 60]
        head_pose_angles : [1, 3]  (yaw, pitch, roll) in degrees
    Output: [1, 3]  gaze vector (x, y, z)
    """

    def __init__(self, core, model_path: str, device: str = "CPU"):
        model = core.read_model(model=model_path)
        self.compiled = core.compile_model(model=model, device_name=device)

        self.in_le = self.compiled.input("left_eye_image")
        self.in_re = self.compiled.input("right_eye_image")
        self.in_hp = self.compiled.input("head_pose_angles")
        self.output_layer = self.compiled.output(0)
        debug_logger.info(
            f"[IntelClient] GazeEstimator loaded  "
            f"left_eye={self.in_le.shape}  right_eye={self.in_re.shape}  "
            f"head_pose={self.in_hp.shape}  output={self.output_layer.shape}"
        )

    @staticmethod
    def _prep_eye(eye_bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Resize to 60×60, HWC→CHW, add batch dim, float32."""
        img = cv2.resize(eye_bgr, (60, 60))
        img = img.transpose((2, 0, 1))
        return img.reshape(1, 3, 60, 60).astype(np.float32)

    def estimate(
        self,
        left_eye_img:  NDArray[np.uint8],
        right_eye_img: NDArray[np.uint8],
        head_pose_angles: Tuple[float, float, float],
    ) -> NDArray[np.float32]:
        """
        Parameters
        ----------
        left_eye_img, right_eye_img : BGR eye crops (any size; resized to 60×60 internally)
        head_pose_angles            : (yaw, pitch, roll) in **degrees**

        Returns
        -------
        gaze_vector : ndarray (3,)  — (x, y, z) direction vector
        """
        result = self.compiled(
            {
                self.in_le: self._prep_eye(left_eye_img),
                self.in_re: self._prep_eye(right_eye_img),
                self.in_hp: np.array([head_pose_angles], dtype=np.float32),
            }
        )[self.output_layer]
        return result[0]


# ============================================================================
#  Eye-crop helper  (ported verbatim from reference testing.py)
# ============================================================================

def _crop_eye(
    face_image: NDArray[np.uint8], center: NDArray, size: int = 60
) -> Optional[NDArray[np.uint8]]:
    """Crop a square patch around *center* from *face_image*."""
    h, w   = face_image.shape[:2]
    half   = size // 2
    x, y   = int(center[0]), int(center[1])
    x1     = max(0, x - half)
    y1     = max(0, y - half)
    x2     = min(w, x + half)
    y2     = min(h, y + half)
    patch  = face_image[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    return cv2.resize(patch, (size, size))


# ============================================================================
#  IntelDetectionClient
# ============================================================================

class IntelDetectionClient(BaseDetectionClient):
    """
    Gaze + head-pose detection backed by four Intel OpenVINO ADAS models.

    The public API is identical to ``MediaPipeDetectionClient`` so it can be
    swapped in transparently via the factory (``DETECTION_CLIENT=intel``).

    Models are resolved from the auto-downloaded local ``intel/`` directory
    (populated by ``model_downloader.ensure_intel_models`` at startup) and can
    be overridden per-model via environment variables.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        self._available = False  # set True only when all models load OK

        # Kalman smoothing + gaze history (mirrors MediaPipe client behaviour)
        self._init_kalman_filters()
        self.gaze_history: List[Tuple[float, float]] = []
        self.max_history_size = 5

        # ── Resolve OpenVINO ─────────────────────────────────────────
        try:
            from openvino.runtime import Core  # type: ignore[import]
        except ImportError:
            debug_logger.warning(
                "[IntelClient] openvino package not installed. "
                "All detection calls will return None values. "
                "Install with:  pip install openvino openvino-dev"
            )
            return

        device = (os.getenv("INTEL_DEVICE") or "CPU").strip().upper()

        # ── Resolve model paths ──────────────────────────────────────
        # Per-model env-var overrides take precedence; fall back to the
        # auto-downloaded paths from model_downloader.
        downloaded = get_model_paths() or {}

        face_xml      = (os.getenv("INTEL_FACE_MODEL_XML") or "").strip() \
                        or downloaded.get("face_det", "")
        landmarks_xml = (os.getenv("INTEL_LANDMARKS_MODEL_XML") or "").strip() \
                        or downloaded.get("landmarks", "")
        head_pose_xml = (os.getenv("INTEL_HEAD_POSE_MODEL_XML") or "").strip() \
                        or downloaded.get("head_pose", "")
        gaze_xml      = (os.getenv("INTEL_GAZE_MODEL_XML") or "").strip() \
                        or downloaded.get("gaze", "")

        missing = [
            name for name, path in [
                ("face-detection-adas-0001",        face_xml),
                ("facial-landmarks-35-adas-0002",   landmarks_xml),
                ("head-pose-estimation-adas-0001",  head_pose_xml),
                ("gaze-estimation-adas-0002",       gaze_xml),
            ] if not path
        ]
        if missing:
            debug_logger.warning(
                f"[IntelClient] Missing model paths for: {missing}. "
                "Ensure DETECTION_CLIENT=intel was set before startup so "
                "model_downloader ran, or set INTEL_*_MODEL_XML env vars manually."
            )
            return

        # ── Load all four models ─────────────────────────────────────
        try:
            core = Core()
            debug_logger.info(
                f"[IntelClient] OpenVINO available devices: {core.available_devices}"
            )

            self._face_detector       = _FaceDetector(core, face_xml, device, threshold=0.5)
            self._landmarks_detector  = _FacialLandmarksDetector(core, landmarks_xml, device)
            self._head_pose_estimator = _HeadPoseEstimator(core, head_pose_xml, device)
            self._gaze_estimator      = _GazeEstimator(core, gaze_xml, device)

            self._available = True
            debug_logger.info(
                f"[IntelClient] All four models loaded on device={device}. "
                "Intel detection client ready."
            )
        except Exception as exc:
            debug_logger.warning(
                f"[IntelClient] Model loading failed ({exc}). "
                "Detection calls will return None values."
            )

    def cleanup(self) -> None:
        # OpenVINO compiled models are garbage-collected; no explicit close needed.
        self.gaze_history.clear()
        self._available = False

    # ------------------------------------------------------------------
    # Kalman smoothing
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
    # Occlusion proxy
    # ------------------------------------------------------------------

    @staticmethod
    def _occlusion_from_box(
        box: Tuple[int, int, int, int], frame_w: int, frame_h: int
    ) -> float:
        x1, y1, x2, y2 = box
        margin    = 0.05
        clipped   = sum([
            x1 / frame_w <= margin,
            y1 / frame_h <= margin,
            x2 / frame_w >= (1.0 - margin),
            y2 / frame_h >= (1.0 - margin),
        ])
        return round(min(clipped * 0.25, 1.0), 2)

    # ------------------------------------------------------------------
    # Gaze vector → angle degrees  (matches reference testing.py formulas)
    # ------------------------------------------------------------------

    @staticmethod
    def _gaze_vec_to_degrees(
        gaze_vec: NDArray[np.float32],
    ) -> Tuple[float, float, float]:
        """
        Convert raw (x, y, z) gaze unit vector to (h_deg, v_deg, confidence).
        Formula is identical to the reference testing.py on-screen display.
        """
        gx, gy, gz = float(gaze_vec[0]), float(gaze_vec[1]), float(gaze_vec[2])
        h_deg = math.degrees(math.atan2(gx, -gz))      # yaw component
        v_deg = math.degrees(math.atan2(
            -gy, math.sqrt(gx ** 2 + gz ** 2)           # pitch component
        ))
        conf  = float(np.clip(np.linalg.norm(gaze_vec[:2]), 0.0, 1.0))
        return round(h_deg, 2), round(v_deg, 2), round(conf, 2)

    # ------------------------------------------------------------------
    # Public API — detect_gaze
    # ------------------------------------------------------------------

    def detect_gaze(
        self, frame: NDArray[np.uint8]
    ) -> Tuple[
        Optional[float], Optional[float], int,
        Optional[Tuple[float, float]], float, float,
    ]:
        if not self._available:
            return None, None, 0, None, 0.0, 0.0

        fh, fw = frame.shape[:2]

        # ── Face detection ───────────────────────────────────────────
        try:
            boxes = self._face_detector.detect(frame)
        except Exception as exc:
            debug_logger.debug(f"[IntelClient] face-det error: {exc}")
            return None, None, 0, None, 0.0, 0.0

        num_faces = len(boxes)
        if num_faces == 0:
            return None, None, 0, None, 0.0, 0.0

        x1, y1, x2, y2 = boxes[0]
        bbox_center = (float((x1 + x2) / 2), float((y1 + y2) / 2))
        occlusion   = self._occlusion_from_box((x1, y1, x2, y2), fw, fh)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None, None, num_faces, bbox_center, 0.0, occlusion

        # ── Head pose (needed as input to gaze model) ─────────────────
        try:
            yaw_d, pitch_d, roll_d = self._head_pose_estimator.estimate(face_crop)
        except Exception as exc:
            debug_logger.debug(f"[IntelClient] head-pose error: {exc}")
            return None, None, num_faces, bbox_center, 0.0, occlusion

        # Clamp pathological values (matches MediaPipe client behaviour)
        if pitch_d > 90.0:
            pitch_d = 180.0 - pitch_d
        elif pitch_d < -90.0:
            pitch_d = -180.0 - pitch_d
        if abs(pitch_d) > 70.0:
            pitch_d = 0.0
        if abs(yaw_d) > 85.0:
            yaw_d = 0.0

        # ── Landmarks → eye centres ───────────────────────────────────
        try:
            le_center, re_center, _ = self._landmarks_detector.detect(face_crop)
        except Exception as exc:
            debug_logger.debug(f"[IntelClient] landmarks error: {exc}")
            return None, None, num_faces, bbox_center, 0.0, occlusion

        # ── Eye crops ─────────────────────────────────────────────────
        le_img = _crop_eye(face_crop, le_center, size=60)
        re_img = _crop_eye(face_crop, re_center, size=60)
        if le_img is None or re_img is None:
            return None, None, num_faces, bbox_center, 0.0, occlusion

        # ── Gaze estimation ───────────────────────────────────────────
        try:
            # gaze model expects head_pose_angles in degrees (yaw, pitch, roll)
            gaze_vec = self._gaze_estimator.estimate(
                le_img, re_img, (yaw_d, pitch_d, roll_d)
            )
        except Exception as exc:
            debug_logger.debug(f"[IntelClient] gaze-est error: {exc}")
            return None, None, num_faces, bbox_center, 0.0, occlusion

        h_deg, v_deg, conf = self._gaze_vec_to_degrees(gaze_vec)

        # ── Kalman + temporal smoothing ───────────────────────────────
        h_deg, v_deg = self._smooth_gaze_with_kalman(h_deg, v_deg)

        self.gaze_history.append((h_deg, v_deg))
        if len(self.gaze_history) > self.max_history_size:
            self.gaze_history.pop(0)

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

    # ------------------------------------------------------------------
    # Public API — detect_head_pose
    # ------------------------------------------------------------------

    def detect_head_pose(
        self, frame: NDArray[np.uint8]
    ) -> Tuple[Optional[float], Optional[float], Optional[float], float]:
        if not self._available:
            return None, None, None, 0.0

        try:
            boxes = self._face_detector.detect(frame)
        except Exception as exc:
            debug_logger.debug(f"[IntelClient] detect_head_pose face-det error: {exc}")
            return None, None, None, 0.0

        if not boxes:
            return None, None, None, 0.0

        x1, y1, x2, y2 = boxes[0]
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None, None, None, 0.0

        try:
            yaw_d, pitch_d, roll_d = self._head_pose_estimator.estimate(face_crop)
        except Exception as exc:
            debug_logger.debug(f"[IntelClient] head-pose error: {exc}")
            return None, None, None, 0.0

        # Clamp pathological values
        if pitch_d > 90.0:
            pitch_d = 180.0 - pitch_d
        elif pitch_d < -90.0:
            pitch_d = -180.0 - pitch_d
        if abs(pitch_d) > 70.0:
            pitch_d = 0.0
        if abs(yaw_d) > 85.0:
            yaw_d = 0.0

        # Confidence from pose proximity to centre
        yaw_conf   = float(np.clip(1.0 - abs(yaw_d) / 90.0, 0.0, 1.0))
        pitch_conf = float(np.clip(1.0 - abs(pitch_d) / 70.0, 0.0, 1.0))
        head_conf  = round(float(0.6 * yaw_conf + 0.4 * pitch_conf), 2)

        return round(yaw_d, 2), round(pitch_d, 2), round(roll_d, 2), head_conf

    # ------------------------------------------------------------------
    # Public API — get_landmark_vector
    # ------------------------------------------------------------------

    def get_landmark_vector(
        self, frame: NDArray[np.uint8]
    ) -> Optional[NDArray[np.float32]]:
        """
        The 35-point landmark model does not produce a 468-point mesh
        compatible with the TVT model, so this always returns None.
        TVT inference is automatically skipped when using the Intel client.
        """
        return None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def client_name(self) -> str:
        return "intel"
