# detection_service.py
"""
Detection Service — Gaze detection, head pose estimation, violation tracking.

Production-safe build:
  - Uses MediaPipe FaceMesh ONLY (no Tasks / FaceLandmarker API).
  - Eliminates libGLESv2 / native-lib dependency issues on headless servers.
  - Retains full gaze + head-pose + ViolationTracker pipeline from v2.4.
  - Retains optional TVT (Temporal Violation Tracker) path.
  - Retains Kalman smoothing, hysteresis, gap-bridging, safe-look suppression.
"""

import os

# Must run before `import mediapipe` to suppress TFLite / absl logging.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from typing import Dict, List, Optional, Tuple

from numpy.typing import NDArray

from .base_proctoring_processing_service import BaseService

# Letterbox to square before FaceMesh.process() so MediaPipe's
# landmark_projection_calculator does not warn on non-square ROI
# (NORM_RECT without IMAGE_DIMENSIONS). Landmarks are mapped back to
# original-frame normalized coordinates.

SquarePackMeta = Tuple[int, int, int, int, int]  # w, h, S, ox, oy


# ======================================================================
# DetectionService
# ======================================================================

class DetectionService(BaseService):
    """
    Gaze + head-pose detection backed exclusively by MediaPipe FaceMesh.

    Drop-in replacement for the original service; public API is identical.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        self._last_tvt_prediction: Optional[Dict] = None
        self._last_tvt_time: float = 0.0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        c = self.config

        # ── Landmark index lists ──────────────────────────────────────
        self.LEFT_EYE         = list(c.FACE_MESH_LEFT_EYE_INDICES)
        self.RIGHT_EYE        = list(c.FACE_MESH_RIGHT_EYE_INDICES)
        self.LEFT_IRIS        = list(c.FACE_MESH_LEFT_IRIS_INDICES)
        self.RIGHT_IRIS       = list(c.FACE_MESH_RIGHT_IRIS_INDICES)
        self.LEFT_EYE_CORNERS = list(c.FACE_MESH_LEFT_EYE_CORNER_INDICES)
        self.RIGHT_EYE_CORNERS= list(c.FACE_MESH_RIGHT_EYE_CORNER_INDICES)
        self.landmark_indices = list(c.HEAD_POSE_LANDMARK_INDICES)
        self.model_points     = np.array(c.HEAD_POSE_MODEL_POINTS_MM, dtype=np.float64)

        # ── MediaPipe FaceMesh (server-safe, no Tasks API) ────────────
        _fm = mp.solutions.face_mesh  # type: ignore[attr-defined]

        self.gaze_face_mesh = _fm.FaceMesh(
            max_num_faces=2,
            refine_landmarks=True,          # iris landmarks required for gaze
            min_detection_confidence=0.60,
            min_tracking_confidence=0.60,
        )
        self.pose_face_mesh = _fm.FaceMesh(
            max_num_faces=2,
            refine_landmarks=False,
            min_detection_confidence=0.60,
            min_tracking_confidence=0.60,
        )

        # ── Kalman filters ────────────────────────────────────────────
        self._init_kalman_filters()

        # ── Gaze + motion history ─────────────────────────────────────
        self.gaze_history: List[Tuple[float, float]] = []
        self.max_history_size  = 5
        self.yaw_history       = deque(maxlen=c.VELOCITY_HISTORY_SIZE)
        self.pitch_history     = deque(maxlen=c.VELOCITY_HISTORY_SIZE)
        self.eye_angle_history = deque(maxlen=c.VELOCITY_HISTORY_SIZE)
        self.timestamp_history = deque(maxlen=c.VELOCITY_HISTORY_SIZE)

        # ── Violation tracker ─────────────────────────────────────────
        self.violation_tracker = ViolationTracker(self.config)

        # ── Optional TVT ──────────────────────────────────────────────
        self._tvt_buffer: Optional[object] = None
        self._tvt_model:  Optional[object] = None
        self._last_tvt_prediction = None
        self._last_tvt_time       = 0.0

        if getattr(c, "ENABLE_TVT", False):
            try:
                from .temporal_buffer import TemporalBuffer
                from .tvt_lite_model  import create_tvt_model
                self._tvt_buffer = TemporalBuffer(
                    window_size=getattr(c, "TVT_TEMPORAL_WINDOW", 24),
                    landmark_dim=936,
                )
                self._tvt_model = create_tvt_model(c)
            except Exception:
                self._tvt_buffer = None
                self._tvt_model  = None

    @staticmethod
    def _letterbox_square_rgb(rgb: NDArray[np.uint8]) -> Tuple[NDArray[np.uint8], SquarePackMeta]:
        h, w = rgb.shape[:2]
        S    = max(w, h)
        if w == h == S:
            return rgb, (w, h, S, 0, 0)
        square = np.zeros((S, S, 3), dtype=rgb.dtype)
        ox, oy = (S - w) // 2, (S - h) // 2
        square[oy : oy + h, ox : ox + w] = rgb
        return square, (w, h, S, ox, oy)

    @staticmethod
    def _square_norm_to_orig_norm(
        x: float, y: float, w: int, h: int, S: int, ox: int, oy: int
    ) -> Tuple[float, float]:
        xo = (x * float(S) - float(ox)) / float(w)
        yo = (y * float(S) - float(oy)) / float(h)
        return xo, yo

    def _landmarks_orig_norm(
        self, face_landmarks, meta: SquarePackMeta
    ) -> List[Tuple[float, float]]:
        w, h, S, ox, oy = meta
        return [
            self._square_norm_to_orig_norm(lm.x, lm.y, w, h, S, ox, oy)
            for lm in face_landmarks
        ]

    # ------------------------------------------------------------------
    # Kalman helpers
    # ------------------------------------------------------------------

    def _init_kalman_filters(self) -> None:
        def _make() -> cv2.KalmanFilter:
            kf = cv2.KalmanFilter(2, 1)
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
    # Eye / occlusion helpers
    # ------------------------------------------------------------------

    def _calculate_eye_aspect_ratio(self, coords: List[List[float]]) -> float:
        try:
            ea = np.array(coords, dtype=np.float64)
            v1 = np.linalg.norm(ea[1] - ea[5])
            v2 = np.linalg.norm(ea[2] - ea[4])
            h  = np.linalg.norm(ea[0] - ea[3])
            return float((v1 + v2) / (2.0 * h)) if h > 0 else 0.0
        except Exception:
            return 0.0

    def _calculate_gaze_confidence(
        self, avg_ear: float, iris_valid: bool, eye_width: float
    ) -> float:
        conf = 0.0
        if avg_ear > 0.08:
            conf += min((avg_ear - 0.08) / 0.12, 1.0) * 0.35
        if iris_valid:
            conf += 0.45
        if eye_width > 3.0:
            conf += min((eye_width - 3.0) / 12.0, 1.0) * 0.20
        return min(conf, 1.0)

    def _calculate_face_occlusion(
            self, orig_xy: List[Tuple[float, float]]
        ) -> Tuple[float, float]:        
        try:
            key_indices   = [1, 234, 454, 10, 152, 33, 263]
            visible_count = 0
            x_coords: List[float] = []
            y_coords: List[float] = []

            for idx in key_indices:
                if idx < len(orig_xy):
                    nx, ny = orig_xy[idx]
                    if 0.05 <= nx <= 0.95 and 0.05 <= ny <= 0.95:
                        visible_count += 1
                        x_coords.append(nx)
                        y_coords.append(ny)

            visibility_ratio = visible_count / len(key_indices)
            spread_penalty   = 0.0

            if len(x_coords) >= 3:
                if (max(x_coords) - min(x_coords)) < 0.10 and \
                   (max(y_coords) - min(y_coords)) < 0.10:
                    spread_penalty = 0.15
            else:
                spread_penalty = 0.40

            return (
                min((1.0 - visibility_ratio) + spread_penalty, 1.0),
                min(visibility_ratio * 1.2, 1.0),
            )
        except Exception:
            return 0.0, 0.0

    # ------------------------------------------------------------------
    # TVT helpers
    # ------------------------------------------------------------------

    def get_landmark_vector(
        self, frame: NDArray[np.uint8]
    ) -> Optional[NDArray[np.float32]]:
        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sq, meta     = self._letterbox_square_rgb(rgb)
        results      = self.gaze_face_mesh.process(sq)
        mlm          = getattr(results, "multi_face_landmarks", None)
        if not mlm:
            return None
        on = self._landmarks_orig_norm(mlm[0].landmark, meta)
        out = np.zeros(936, dtype=np.float32)
        n   = min(468, len(on))
        for i in range(n):
            out[i * 2]     = on[i][0]
            out[i * 2 + 1] = on[i][1]
        return out

    def push_landmark_and_maybe_run_tvt(
        self, lv: Optional[NDArray[np.float32]], ts: float
    ) -> None:
        if self._tvt_buffer is None or self._tvt_model is None or lv is None:
            return
        self._tvt_buffer.push(lv, ts)
        interval = float(getattr(self.config, "TVT_INFERENCE_INTERVAL_SEC", 1.0))
        if not self._tvt_buffer.is_ready() or ts - self._last_tvt_time < interval:
            return
        window = self._tvt_buffer.get_window()
        if window is not None:
            try:
                self._last_tvt_prediction = self._tvt_model.predict(window)
                self._last_tvt_time       = ts
            except Exception:
                pass

    def get_tvt_prediction(self) -> Optional[Dict]:
        return self._last_tvt_prediction

    # ------------------------------------------------------------------
    # Gaze detection
    # ------------------------------------------------------------------

    def detect_gaze(self, frame: NDArray[np.uint8]) -> Tuple[
        Optional[float], Optional[float], int,
        Optional[Tuple[float, float]], float, float,
    ]:
        rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sq, meta     = self._letterbox_square_rgb(rgb)
        results      = self.gaze_face_mesh.process(sq)
        mlm          = getattr(results, "multi_face_landmarks", None)

        if not mlm:
            return None, None, 0, None, 0.0, 0.0

        num_faces      = len(mlm)
        face_landmarks = mlm[0].landmark
        h, w           = frame.shape[:2]
        on             = self._landmarks_orig_norm(face_landmarks, meta)

        bbox_center = (
            float(np.mean([on[i][0] * w for i in range(len(on))])),
            float(np.mean([on[i][1] * h for i in range(len(on))])),
        )
        occlusion_ratio, _ = self._calculate_face_occlusion(on)

        try:
            le = [[on[i][0] * w, on[i][1] * h] for i in self.LEFT_EYE]
            re = [[on[i][0] * w, on[i][1] * h] for i in self.RIGHT_EYE]
            li = [[on[i][0] * w, on[i][1] * h] for i in self.LEFT_IRIS]
            ri = [[on[i][0] * w, on[i][1] * h] for i in self.RIGHT_IRIS]

            avg_ear = (
                self._calculate_eye_aspect_ratio(le)
                + self._calculate_eye_aspect_ratio(re)
            ) / 2.0

            if avg_ear < 0.05:  # 0.10
                return None, None, num_faces, bbox_center, 0.0, occlusion_ratio

            iris_valid = (
                all(0 <= c[0] <= w and 0 <= c[1] <= h for c in li)
                and all(0 <= c[0] <= w and 0 <= c[1] <= h for c in ri)
            )
           
            if not iris_valid:
                return None, None, num_faces, bbox_center, 0.0, occlusion_ratio

            le_c = np.mean(np.array(le, dtype=np.float64), axis=0)
            re_c = np.mean(np.array(re, dtype=np.float64), axis=0)
            li_c = np.mean(np.array(li, dtype=np.float64), axis=0)
            ri_c = np.mean(np.array(ri, dtype=np.float64), axis=0)

            def _ew(corners: List[int]) -> float:
                p1 = np.array([on[corners[0]][0] * w, on[corners[0]][1] * h])
                p2 = np.array([on[corners[1]][0] * w, on[corners[1]][1] * h])
                return float(np.linalg.norm(p2 - p1))

            avg_eye_width = (_ew(self.LEFT_EYE_CORNERS) + _ew(self.RIGHT_EYE_CORNERS)) / 2.0
            if avg_eye_width < 0.8:  # 1.8
                return None, None, num_faces, bbox_center, 0.0, occlusion_ratio
         
            avg_disp = ((li_c - le_c) + (ri_c - re_c)) / 2.0
            h_ratio  = float(np.clip(avg_disp[0] / avg_eye_width, -1.0, 1.0))
            v_ratio  = float(np.clip(avg_disp[1] / avg_eye_width, -1.0, 1.0))
       
            h_angle = float(np.arctan(h_ratio) * (180.0 / np.pi))
            v_angle = float(np.arctan(v_ratio) * (180.0 / np.pi))
            h_angle, v_angle = self._smooth_gaze_with_kalman(h_angle, v_angle)

            self.gaze_history.append((h_angle, v_angle))
            if len(self.gaze_history) > self.max_history_size:
                self.gaze_history.pop(0)

            # Weighted temporal smoothing
            if len(self.gaze_history) >= 5:
                wts    = np.array([0.05, 0.10, 0.15, 0.25, 0.45])
                recent = self.gaze_history[-5:]
                h_angle = float(np.average([x[0] for x in recent], weights=wts))
                v_angle = float(np.average([x[1] for x in recent], weights=wts))
            elif len(self.gaze_history) >= 3:
                wts    = np.array([0.2, 0.3, 0.5])
                recent = self.gaze_history[-3:]
                h_angle = float(np.average([x[0] for x in recent], weights=wts))
                v_angle = float(np.average([x[1] for x in recent], weights=wts))

            confidence = self._calculate_gaze_confidence(avg_ear, iris_valid, avg_eye_width)
            return (
                round(h_angle, 2), round(v_angle, 2),
                num_faces, bbox_center,
                confidence, occlusion_ratio,
            )
        except (IndexError, ValueError, TypeError):
            return None, None, num_faces, bbox_center, 0.0, occlusion_ratio

    # ------------------------------------------------------------------
    # Head pose detection
    # ------------------------------------------------------------------

    def _estimate_head_pose(
        self,
        frame: np.ndarray,
        results,
        meta: SquarePackMeta,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], float]:
        mlm = getattr(results, "multi_face_landmarks", None)
        if not mlm:
            return None, None, None, 0.0

        face_landmarks = mlm[0].landmark
        h, w           = frame.shape[:2]
        on             = self._landmarks_orig_norm(face_landmarks, meta)

        image_points = np.array(
            [[on[i][0] * w, on[i][1] * h] for i in self.landmark_indices],
            dtype=np.float64,
        )
        focal_length  = float(w)
        camera_matrix = np.array(
            [[focal_length, 0, w / 2.0],
             [0, focal_length, h / 2.0],
             [0, 0, 1]],
            dtype=np.float64,
        )

        try:
            success, rvec, _ = cv2.solvePnP(
                self.model_points, image_points,
                camera_matrix, np.zeros((4, 1), dtype=np.float64),
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not success or rvec is None:
                return None, None, None, 0.0

            r, _ = cv2.Rodrigues(rvec)
            if r is None:
                return None, None, None, 0.0

            sy       = float(np.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2))
            singular = sy < 1e-6

            if not singular:
                pitch_rad = np.arctan2(r[2, 1], r[2, 2])
                yaw_rad   = np.arctan2(-r[2, 0], sy)
                roll_rad  = np.arctan2(r[1, 0], r[0, 0])
            else:
                pitch_rad = np.arctan2(-r[1, 2], r[1, 1])
                yaw_rad   = np.arctan2(-r[2, 0], sy)
                roll_rad  = 0.0

            yaw_deg   = float(np.degrees(yaw_rad))
            pitch_deg = float(np.degrees(pitch_rad))
            roll_deg  = float(np.degrees(roll_rad))

            # Clamp pathological flips
            if pitch_deg > 90.0:
                pitch_deg = 180.0 - pitch_deg
            elif pitch_deg < -90.0:
                pitch_deg = -180.0 - pitch_deg
            if abs(pitch_deg) > 70.0:
                pitch_deg = 0.0
            if abs(yaw_deg) > 85.0:
                yaw_deg = 0.0

            # Confidence from landmark spread and rotation singularity margin
            x_coords = [on[i][0] for i in self.landmark_indices]
            y_coords = [on[i][1] for i in self.landmark_indices]
            spread_x = float(np.clip((max(x_coords) - min(x_coords) - 0.08) / 0.14, 0.0, 1.0))
            spread_y = float(np.clip((max(y_coords) - min(y_coords) - 0.08) / 0.14, 0.0, 1.0))
            lm_conf  = 0.5 * spread_x + 0.5 * spread_y
            sy_conf  = float(np.clip((sy - 0.08) / 0.18, 0.0, 1.0))
            head_conf = float(np.clip(0.15 + 0.85 * (0.6 * lm_conf + 0.4 * sy_conf), 0.0, 1.0))

            return round(yaw_deg, 2), round(pitch_deg, 2), round(roll_deg, 2), round(head_conf, 2)

        except (cv2.error, ValueError, TypeError):
            return None, None, None, 0.0

    def detect_head_pose(
        self, frame: NDArray[np.uint8]
    ) -> Tuple[Optional[float], Optional[float], Optional[float], float]:
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sq, meta = self._letterbox_square_rgb(rgb)
        results  = self.pose_face_mesh.process(sq)
        if results is None:
            return None, None, None, 0.0
        return self._estimate_head_pose(frame, results, meta)

    # ------------------------------------------------------------------
    # Violation update
    # ------------------------------------------------------------------

    def update_violations(
        self,
        timestamp:        float,
        gaze_h:           Optional[float],
        gaze_v:           Optional[float],
        yaw:              Optional[float],
        pitch:            Optional[float],
        roll:             Optional[float],
        num_faces:        int,
        thresholds:       Dict[str, float],
        gaze_confidence:  float = 0.0,
        head_confidence:  float = 0.0,
        occlusion_ratio:  float = 0.0,
        landmark_vector:  Optional[NDArray[np.float32]] = None,
    ) -> None:
        _ = roll  # not used downstream; kept for API compatibility

        if getattr(self.config, "ENABLE_TVT", False) and landmark_vector is not None:
            self.push_landmark_and_maybe_run_tvt(landmark_vector, timestamp)

        tvt_prediction = (
            self.get_tvt_prediction()
            if getattr(self.config, "ENABLE_TVT", False)
            else None
        )

        self.timestamp_history.append(timestamp)
        if yaw   is not None: self.yaw_history.append(yaw)
        if pitch  is not None: self.pitch_history.append(pitch)
        if gaze_h is not None and gaze_v is not None:
            self.eye_angle_history.append(np.sqrt(gaze_h ** 2 + gaze_v ** 2))

        self.violation_tracker.update(
            timestamp, gaze_h, gaze_v, yaw, pitch, num_faces, thresholds,
            gaze_confidence, head_confidence, occlusion_ratio,
            self.yaw_history, self.pitch_history,
            self.eye_angle_history, self.timestamp_history,
            tvt_prediction=tvt_prediction,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_counts(self) -> Dict[str, int]:
        return self.violation_tracker.get_counts()

    def get_all_timestamps(self) -> Dict[str, List[float]]:
        return {k: self.violation_tracker.get_timestamps(k)
                for k in self.violation_tracker.counts}

    def get_all_max_intensities(self) -> Dict[str, float]:
        return self.violation_tracker.max_intensities.copy()

    def get_violation_events(self) -> List[Dict]:
        return self.violation_tracker.get_all_events()

    def cleanup(self) -> None:
        for attr in ("gaze_face_mesh", "pose_face_mesh"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        self.gaze_history.clear()
        if getattr(self, "_tvt_buffer", None) is not None:
            self._tvt_buffer.clear()
        self.violation_tracker.finalize()


# ======================================================================
# ViolationTracker  (v2.4 — unchanged from original)
# ======================================================================

class ViolationTracker:
    """
    Duration-based violation tracker with hysteresis, gap bridging,
    and context-aware safe-look suppression.

    v2.4 fixes:
      - _HYSTERESIS_FRAMES: 2 → 3  (167 ms @ 18 fps; avoids micro-movement noise)
      - _calculate_velocity: fixed silent negative-index overflow on short histories.
      - _is_safe_down_look: max_dur now reads SAFE_DOWN_MAX_DURATION_SEC from config.
    """

    _HYSTERESIS_FRAMES: int = 3

    def __init__(self, config) -> None:
        self.config      = config
        self.counts: Dict[str, int] = {
            "gaze_left": 0, "gaze_right": 0, "gaze_up": 0, "gaze_down": 0,
            "head_left": 0, "head_right": 0, "head_up": 0, "head_down": 0,
            "face_missing": 0, "multiple_faces": 0, "face_occluded": 0,
        }
        self.timestamps:      Dict[str, List[float]] = {k: [] for k in self.counts}
        self.max_intensities: Dict[str, float]       = {k: 0.0 for k in self.counts}
        self.violation_events: List[Dict]            = []

        self.gap_tolerance          = float(getattr(config, "EVENT_GAP_TOLERANCE", 0.55))
        self._target_fps            = float(getattr(config, "TARGET_FPS", 18))
        self._last_active_yaw_time: float = -999.0

        self._hysteresis:     Dict[str, int] = {"yaw": 0, "pitch": 0, "eye_h": 0, "eye_v": 0}
        self._hysteresis_dir: Dict[str, str] = {"yaw": "", "pitch": "", "eye_h": "", "eye_v": ""}

        _blank: Dict = {
            "active": False, "start_time": 0.0, "last_update_time": 0.0,
            "direction": "", "max_intensity": 0.0, "confidence": 0.0,
            "velocity": 0.0, "confidence_samples": [], "velocity_samples": [],
            "last_intensity": 0.0,
        }
        self.current_yaw_state   = dict(_blank)
        self.current_pitch_state = dict(_blank)
        self.current_eye_h_state = dict(_blank)
        self.current_eye_v_state = dict(_blank)

        self.current_face_missing_state: Dict = {
            "active": False, "start_time": 0.0,
            "last_update_time": 0.0, "first_no_face_time": None,
        }
        self.current_face_occluded_state: Dict = {
            "active": False, "start_time": 0.0,
            "last_update_time": 0.0, "max_occlusion": 0.0,
        }
        self.active_violations: Dict[str, Dict] = {
            "multiple_faces": {
                "active": False, "start_time": 0.0, "last_update_time": 0.0,
                "duration": 0.0, "max_intensity": 0.0, "confidence": 0.9, "velocity": 0.0,
            }
        }

    # ------------------------------------------------------------------
    # Gap tolerance
    # ------------------------------------------------------------------

    def _effective_gap_tolerance(self) -> float:
        frame_gap = 1.0 / max(self._target_fps, 1.0)
        return max(self.gap_tolerance, frame_gap * 4)

    # ------------------------------------------------------------------
    # Velocity  (v2.4 fix: no negative-index overflow)
    # ------------------------------------------------------------------

    def _calculate_velocity(self, angle_history: deque, ts_history: deque) -> float:
        if len(angle_history) < 2 or len(ts_history) < 2:
            return 0.0
        try:
            angles = list(angle_history)
            times  = list(ts_history)
            n      = min(len(angles), len(times))
            velocities: List[float] = []
            # Forward absolute indexing — iterate from most-recent pair backwards
            for i in range(n - 1, max(n - 10, 0), -1):
                da = abs(angles[i] - angles[i - 1])
                if da > 180:
                    da = 360 - da
                dt = times[i] - times[i - 1]
                if dt > 0:
                    velocities.append(da / dt)
            return sum(velocities) / len(velocities) if velocities else 0.0
        except (IndexError, ZeroDivisionError):
            return 0.0

    # ------------------------------------------------------------------
    # Safe down-look suppression  (v2.4 fix: uses config value 2.5 s)
    # ------------------------------------------------------------------

    def _is_safe_down_look(
        self,
        direction:          str,
        duration_sec:       float,
        velocity_deg_per_s: float,
        max_pitch_deg:      float,
    ) -> bool:
        if direction != "down":
            return False
        max_vel = float(getattr(self.config, "SAFE_DOWN_MAX_VELOCITY_DEG_PER_S", 4.0))
        max_dur = float(getattr(self.config, "SAFE_DOWN_MAX_DURATION_SEC", 2.5))
        return velocity_deg_per_s < max_vel and duration_sec < max_dur

    # ------------------------------------------------------------------
    # Direction helpers
    # ------------------------------------------------------------------

    def _get_yaw_direction(
        self, yaw: Optional[float], thresholds: Dict
    ) -> Tuple[str, float]:
        if yaw is None or abs(yaw) <= thresholds["yaw"]:
            return "", 0.0
        return ("right" if yaw > 0 else "left"), abs(yaw)

    def _get_pitch_direction(
        self, pitch: Optional[float], thresholds: Dict
    ) -> Tuple[str, float]:
        if pitch is None:
            return "", 0.0
        abs_pitch      = abs(pitch)
        thresh         = float(thresholds.get("pitch", 22.0))
        release_thresh = max(thresh - 2.0, 0.0)

        if abs_pitch > thresh:
            return ("up" if pitch > 0 else "down"), abs_pitch
        if abs_pitch > release_thresh and self.current_pitch_state["active"]:
            return self.current_pitch_state["direction"], abs_pitch
        return "", 0.0

    def _get_eye_h_direction(
        self, gaze_h: Optional[float], thresholds: Dict
    ) -> Tuple[str, float]:
        if gaze_h is None or abs(gaze_h) <= thresholds["eye_horizontal"]:
            return "", 0.0
        return ("right" if gaze_h > 0 else "left"), abs(gaze_h)

    def _get_eye_v_direction(
        self, gaze_v: Optional[float], thresholds: Dict
    ) -> Tuple[str, float]:
        if gaze_v is None or abs(gaze_v) <= thresholds["eye_vertical"]:
            return "", 0.0
        return ("down" if gaze_v > 0 else "up"), abs(gaze_v)

    # ------------------------------------------------------------------
    # Hysteresis
    # ------------------------------------------------------------------

    def _advance_hysteresis(self, axis: str, direction: str) -> bool:
        if direction != self._hysteresis_dir[axis]:
            self._hysteresis[axis]     = 0
            self._hysteresis_dir[axis] = direction
        self._hysteresis[axis] += 1
        return self._hysteresis[axis] >= self._HYSTERESIS_FRAMES

    def _reset_hysteresis(self, axis: str) -> None:
        self._hysteresis[axis]     = 0
        self._hysteresis_dir[axis] = ""

    # ------------------------------------------------------------------
    # Event recording
    # ------------------------------------------------------------------

    def _record_head_event(
        self,
        start_time:         float,
        end_time:           float,
        direction:          str,
        max_intensity:      float,
        confidence:         float,
        velocity:           float,
        tvt_prediction:     Optional[Dict] = None,
        tvt_prob_threshold: float = 0.85,
    ) -> None:
        gap_compensation = self._effective_gap_tolerance() * 0.5
        corrected_end    = end_time + gap_compensation
        duration         = corrected_end - start_time
        raw_duration     = end_time - start_time

        min_dur  = float(getattr(self.config, "MIN_EVENT_DURATION", 1.0))
        min_conf = float(getattr(self.config, "MIN_CONFIDENCE_THRESHOLD", 0.45))

        if duration < min_dur or not direction or confidence < min_conf:
            return

        if self._is_safe_down_look(direction, raw_duration, velocity, max_intensity):
            return

        if tvt_prediction is not None:
            prob = float(tvt_prediction.get("probability", 0.0))
            if prob > 0.5:
                head_to_tvt = {
                    "left":  "left_cheating_glance",
                    "right": "right_cheating_glance",
                    "down":  "phone_lookdown",
                }
                expected  = head_to_tvt.get(direction)
                tvt_class = tvt_prediction.get("behavior_class", "")
                if expected and (prob < tvt_prob_threshold or tvt_class != expected):
                    return

        vtype = f"head_{direction}"
        ts    = round(start_time, 2)
        self.counts[vtype] = self.counts.get(vtype, 0) + 1
        self.timestamps[vtype].append(ts)
        self.violation_events.append({
            "type": vtype, "timestamp": ts,
            "duration": round(duration, 1), "intensity": max_intensity,
            "confidence": round(confidence, 2), "velocity": velocity,
        })
        if max_intensity > self.max_intensities.get(vtype, 0.0):
            self.max_intensities[vtype] = max_intensity

    def _record_eye_event(
        self,
        start_time:         float,
        end_time:           float,
        direction:          str,
        max_intensity:      float,
        confidence:         float,
        velocity:           float,
        tvt_prediction:     Optional[Dict] = None,
        tvt_prob_threshold: float = 0.85,
    ) -> None:
        gap_compensation = self._effective_gap_tolerance() * 0.5
        corrected_end    = end_time + gap_compensation
        duration         = corrected_end - start_time

        min_dur  = float(getattr(self.config, "EYE_MIN_EVENT_DURATION", 0.8))
        min_conf = float(getattr(self.config, "EYE_MIN_CONFIDENCE_THRESHOLD", 0.35))

        if duration < min_dur or not direction or confidence < min_conf:
            return

        if tvt_prediction is not None:
            prob = float(tvt_prediction.get("probability", 0.0))
            if prob > 0.5:
                gaze_to_tvt = {
                    "left":  "left_cheating_glance",
                    "right": "right_cheating_glance",
                    "down":  "phone_lookdown",
                }
                expected  = gaze_to_tvt.get(direction)
                tvt_class = tvt_prediction.get("behavior_class", "")
                if expected and (prob < tvt_prob_threshold or tvt_class != expected):
                    return

        vtype = f"gaze_{direction}"
        ts    = round(start_time, 2)
        self.counts[vtype] = self.counts.get(vtype, 0) + 1
        self.timestamps[vtype].append(ts)
        self.violation_events.append({
            "type": vtype, "timestamp": ts,
            "duration": round(duration, 1), "intensity": max_intensity,
            "confidence": round(confidence, 2), "velocity": velocity,
        })
        if max_intensity > self.max_intensities.get(vtype, 0.0):
            self.max_intensities[vtype] = max_intensity

    def _record_face_missing_event(self, start_time: float, end_time: float) -> None:
        duration = end_time - start_time
        min_dur  = float(getattr(
            self.config, "FACE_MISSING_MIN_DURATION",
            getattr(self.config, "MIN_EVENT_DURATION", 1.0),
        ))
        if duration < min_dur:
            return
        ts = round(start_time, 2)
        self.counts["face_missing"] += 1
        self.timestamps["face_missing"].append(ts)
        self.violation_events.append({
            "type": "face_missing", "timestamp": ts,
            "duration": round(duration, 1), "intensity": 0.0,
            "confidence": 1.0, "velocity": 0.0,
        })

    def _record_face_occluded_event(
        self, start_time: float, end_time: float, max_occlusion: float
    ) -> None:
        duration = end_time - start_time
        if duration < float(getattr(self.config, "MIN_EVENT_DURATION", 1.0)):
            return
        ts = round(start_time, 2)
        self.counts["face_occluded"] += 1
        self.timestamps["face_occluded"].append(ts)
        self.violation_events.append({
            "type": "face_occluded", "timestamp": ts,
            "duration": round(duration, 1), "intensity": round(max_occlusion * 100, 1),
            "confidence": 0.85, "velocity": 0.0,
        })

    # ------------------------------------------------------------------
    # Head-turn gap bridge
    # ------------------------------------------------------------------

    def _bridge_yaw_across_missing_face(self, timestamp: float) -> None:
        bridge = float(getattr(self.config, "HEAD_TURN_BRIDGE_SECONDS", 1.0))
        if (timestamp - self._last_active_yaw_time) <= bridge:
            if self.current_yaw_state["active"]:
                self.current_yaw_state["last_update_time"] = timestamp

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def update(
        self,
        timestamp:         float,
        gaze_h:            Optional[float],
        gaze_v:            Optional[float],
        yaw:               Optional[float],
        pitch:             Optional[float],
        num_faces:         int,
        thresholds:        Dict[str, float],
        gaze_confidence:   float = 0.0,
        head_confidence:   float = 0.0,
        occlusion_ratio:   float = 0.0,
        yaw_history:       Optional[deque] = None,
        pitch_history:     Optional[deque] = None,
        eye_angle_history: Optional[deque] = None,
        timestamp_history: Optional[deque] = None,
        tvt_prediction:    Optional[Dict] = None,
    ) -> None:
        tvt_threshold = float(getattr(self.config, "TVT_PROB_THRESHOLD", 0.85))

        # Velocity estimates
        head_velocity = 0.0
        if yaw_history and pitch_history and timestamp_history:
            head_velocity = float(max(
                self._calculate_velocity(yaw_history, timestamp_history),
                self._calculate_velocity(pitch_history, timestamp_history),
            ))

        eye_velocity = 0.0
        if eye_angle_history and timestamp_history:
            eye_velocity = float(
                self._calculate_velocity(eye_angle_history, timestamp_history)
            )

        def _sc(base: float, ratio: float) -> float:
            r = float(np.clip(ratio, 0.0, 2.0))
            return float(np.clip(base * (0.6 + 0.4 * (r / 2.0)), 0.0, 1.0))

        # Direction + intensity for each axis
        yaw_dir,   yaw_int   = self._get_yaw_direction(yaw, thresholds)
        pitch_dir, pitch_int = self._get_pitch_direction(pitch, thresholds)
        eye_h_dir, eye_h_int = self._get_eye_h_direction(gaze_h, thresholds)
        eye_v_dir, eye_v_int = self._get_eye_v_direction(gaze_v, thresholds)

        yaw_thresh   = max(float(thresholds.get("yaw", 1.0)), 1e-6)
        pitch_thresh = max(float(thresholds.get("pitch", 1.0)), 1e-6)
        eyeh_thresh  = max(float(thresholds.get("eye_horizontal", 1.0)), 1e-6)
        eyev_thresh  = max(float(thresholds.get("eye_vertical", 1.0)), 1e-6)

        self._update_head_axis_state(
            self.current_yaw_state, "yaw", timestamp,
            yaw_dir, yaw_int,
            _sc(head_confidence, yaw_int / yaw_thresh if yaw_dir else 0.0),
            head_velocity, tvt_prediction, tvt_threshold, is_yaw=True,
        )
        self._update_head_axis_state(
            self.current_pitch_state, "pitch", timestamp,
            pitch_dir, pitch_int,
            _sc(head_confidence, pitch_int / pitch_thresh if pitch_dir else 0.0),
            head_velocity, tvt_prediction, tvt_threshold,
        )
        self._update_eye_axis_state(
            self.current_eye_h_state, "eye_h", timestamp,
            eye_h_dir, eye_h_int,
            _sc(gaze_confidence, eye_h_int / eyeh_thresh if eye_h_dir else 0.0),
            eye_velocity, tvt_prediction, tvt_threshold,
        )
        self._update_eye_axis_state(
            self.current_eye_v_state, "eye_v", timestamp,
            eye_v_dir, eye_v_int,
            _sc(gaze_confidence, eye_v_int / eyev_thresh if eye_v_dir else 0.0),
            eye_velocity, tvt_prediction, tvt_threshold,
        )
        min_start = float(getattr(self.config, "FACE_MISSING_MIN_START_DURATION", 0.8))

        if num_faces == 0:
            self._bridge_yaw_across_missing_face(timestamp)

            # FIX 1: Actually call the suppression check
            if self._should_suppress_face_missing(timestamp):
                # Head turn is active — suppress face_missing entirely.
                # Also clear any pending grace period so it doesn't 
                # accumulate during the turn.
                self.current_face_missing_state["first_no_face_time"] = None
            else:
                state = self.current_face_missing_state
                if not state["active"]:
                    first = state.get("first_no_face_time")
                    if first is None:
                        state["first_no_face_time"] = timestamp
                    elif (timestamp - first) >= min_start:
                        state.update({
                            "active": True, "start_time": first,
                            "last_update_time": timestamp,
                            "first_no_face_time": None,
                        })
                else:
                    state["last_update_time"] = timestamp

        else:  # num_faces > 0
            state = self.current_face_missing_state

            # FIX 2: Don't reset grace period on a single flicker frame.
            # Only reset if face has been stably present for > 1 frame.
            # Track consecutive face-present frames to distinguish 
            # genuine recovery from a detection flicker.
            # self._consecutive_face_frames = getattr(self, "_consecutive_face_frames", 0) + 1
            self._consecutive_face_frames: int = 0
            if self._consecutive_face_frames >= 2:
                state["first_no_face_time"] = None

            if state["active"]:
                # FIX 3: Use wider gap tolerance for face_missing close
                if timestamp - state["last_update_time"] > self._face_missing_gap_tolerance():
                    self._record_face_missing_event(
                        state["start_time"], state["last_update_time"]
                    )
                    state["active"] = False

        # Reset consecutive face counter when face is absent
        if num_faces == 0:
            self._consecutive_face_frames = 0

        # ── Face occlusion ────────────────────────────────────────────────
        occ_threshold = float(getattr(self.config, "FACE_OCCLUSION_THRESHOLD", 0.35))

        if num_faces > 0 and occlusion_ratio > occ_threshold:
            # FIX 4: Don't activate face_occluded if face_missing is already active
            # (prevents double-firing for the same event)
            if not self.current_face_missing_state["active"]:
                s = self.current_face_occluded_state
                if not s["active"]:
                    self.current_face_occluded_state = {
                        "active": True, "start_time": timestamp,
                        "last_update_time": timestamp, "max_occlusion": occlusion_ratio,
                    }
                else:
                    s["last_update_time"] = timestamp
                    s["max_occlusion"] = max(s["max_occlusion"], occlusion_ratio)
        else:
            s = self.current_face_occluded_state
            if s["active"]:
                if timestamp - s["last_update_time"] > self._effective_gap_tolerance():
                    self._record_face_occluded_event(
                        s["start_time"], s["last_update_time"], s["max_occlusion"]
                    )
                    s["active"] = False

        self._check_multiple_faces(timestamp, num_faces)

        # ── Face-missing ──────────────────────────────────────────────
        # min_start = float(getattr(self.config, "FACE_MISSING_MIN_START_DURATION", 0.8))
        # if num_faces == 0:
        #     self._bridge_yaw_across_missing_face(timestamp)
        #     state = self.current_face_missing_state
        #     if not state["active"]:
        #         first = state.get("first_no_face_time")
        #         if first is None:
        #             state["first_no_face_time"] = timestamp
        #         elif (timestamp - first) >= min_start:
        #             state.update({
        #                 "active": True, "start_time": first,
        #                 "last_update_time": timestamp, "first_no_face_time": None,
        #             })
        #     else:
        #         state["last_update_time"] = timestamp
        # else:
        #     state = self.current_face_missing_state
        #     state["first_no_face_time"] = None
        #     if state["active"]:
        #         if timestamp - state["last_update_time"] > self._effective_gap_tolerance():
        #             self._record_face_missing_event(
        #                 state["start_time"], state["last_update_time"]
        #             )
        #             state["active"] = False

        # # ── Face occlusion ────────────────────────────────────────────
        # occ_threshold = float(getattr(self.config, "FACE_OCCLUSION_THRESHOLD", 0.35))
        # if num_faces > 0 and occlusion_ratio > occ_threshold:
        #     s = self.current_face_occluded_state
        #     if not s["active"]:
        #         self.current_face_occluded_state = {
        #             "active": True, "start_time": timestamp,
        #             "last_update_time": timestamp, "max_occlusion": occlusion_ratio,
        #         }
        #     else:
        #         s["last_update_time"] = timestamp
        #         s["max_occlusion"]    = max(s["max_occlusion"], occlusion_ratio)
        # else:
        #     s = self.current_face_occluded_state
        #     if s["active"]:
        #         if timestamp - s["last_update_time"] > self._effective_gap_tolerance():
        #             self._record_face_occluded_event(
        #                 s["start_time"], s["last_update_time"], s["max_occlusion"]
        #             )
        #             s["active"] = False

        # self._check_multiple_faces(timestamp, num_faces)


    def _face_missing_gap_tolerance(self) -> float:
        """Wider gap tolerance specifically for face_missing — 
        head turns can cause 0.6–0.9s dropout at 18fps."""
        return max(self._effective_gap_tolerance(), 
                float(getattr(self.config, "HEAD_TURN_BRIDGE_SECONDS", 1.0)) * 0.9)

    def _should_suppress_face_missing(self, timestamp: float) -> bool:
        """
        Returns True when a face dropout is attributable to a head turn
        rather than the person genuinely leaving frame.
        
        Two conditions — either is sufficient:
        1. Yaw was actively above threshold recently (within bridge window)
        2. Yaw state machine is currently active (turn started before dropout)
        """
        bridge = float(getattr(self.config, "HEAD_TURN_BRIDGE_SECONDS", 1.0))
        if (timestamp - self._last_active_yaw_time) <= bridge:
            return True
        if self.current_yaw_state["active"]:
            return True
        return False
    
    # ------------------------------------------------------------------
    # Axis state machines
    # ------------------------------------------------------------------

    def _update_head_axis_state(
        self,
        state:              Dict,
        axis_key:           str,
        timestamp:          float,
        direction:          str,
        intensity:          float,
        confidence:         float,
        velocity:           float,
        tvt_prediction:     Optional[Dict] = None,
        tvt_prob_threshold: float = 0.85,
        is_yaw:             bool = False,
    ) -> None:
        eff_gap = self._effective_gap_tolerance()

        if direction:
            if is_yaw:
                self._last_active_yaw_time = timestamp
            ready = self._advance_hysteresis(axis_key, direction)

            if not state["active"]:
                if ready:
                    state.update({
                        "active": True, "start_time": timestamp,
                        "last_update_time": timestamp, "direction": direction,
                        "max_intensity": intensity, "confidence": confidence,
                        "velocity": velocity,
                        "confidence_samples": [float(confidence)],
                        "velocity_samples": [], "last_intensity": float(intensity),
                    })
            else:
                if direction != state["direction"]:
                    self._record_head_event(
                        state["start_time"], state["last_update_time"],
                        state["direction"], state["max_intensity"],
                        self._aggregate_event_confidence(state),
                        self._aggregate_event_velocity(state),
                        tvt_prediction, tvt_prob_threshold,
                    )
                    self._reset_hysteresis(axis_key)
                    ready = self._advance_hysteresis(axis_key, direction)
                    if ready:
                        state.update({
                            "active": True, "start_time": timestamp,
                            "last_update_time": timestamp, "direction": direction,
                            "max_intensity": intensity, "confidence": confidence,
                            "velocity": velocity,
                            "confidence_samples": [float(confidence)],
                            "velocity_samples": [], "last_intensity": float(intensity),
                        })
                    else:
                        state["active"] = False
                else:
                    # Extend existing event
                    dt = float(timestamp - state["last_update_time"])
                    if dt > 0:
                        inst_v = abs(float(intensity) - float(state.get("last_intensity", intensity))) / dt
                        if np.isfinite(inst_v):
                            vs = state.get("velocity_samples") or []
                            if len(vs) < 300:
                                vs.append(float(inst_v))
                            state["velocity_samples"] = vs
                    cs = state.get("confidence_samples") or []
                    if len(cs) < 300:
                        cs.append(float(confidence))
                    state["confidence_samples"] = cs
                    state["max_intensity"]    = max(state["max_intensity"], intensity)
                    state["last_update_time"] = timestamp
                    state["confidence"]       = max(state["confidence"], confidence)
                    state["velocity"]         = max(float(state.get("velocity", 0.0)), float(velocity))
                    state["last_intensity"]   = float(intensity)
        else:
            self._reset_hysteresis(axis_key)
            if state["active"]:
                if timestamp - state["last_update_time"] > eff_gap:
                    self._record_head_event(
                        state["start_time"], state["last_update_time"],
                        state["direction"], state["max_intensity"],
                        self._aggregate_event_confidence(state),
                        self._aggregate_event_velocity(state),
                        tvt_prediction, tvt_prob_threshold,
                    )
                    state["active"] = False

    def _update_eye_axis_state(
        self,
        state:              Dict,
        axis_key:           str,
        timestamp:          float,
        direction:          str,
        intensity:          float,
        confidence:         float,
        velocity:           float,
        tvt_prediction:     Optional[Dict] = None,
        tvt_prob_threshold: float = 0.85,
    ) -> None:
        eff_gap = self._effective_gap_tolerance()

        if direction:
            ready = self._advance_hysteresis(axis_key, direction)
            if not state["active"]:
                if ready:
                    state.update({
                        "active": True, "start_time": timestamp,
                        "last_update_time": timestamp, "direction": direction,
                        "max_intensity": intensity, "confidence": confidence,
                        "velocity": velocity,
                        "confidence_samples": [float(confidence)],
                        "velocity_samples": [], "last_intensity": float(intensity),
                    })
            else:
                if direction != state["direction"]:
                    self._record_eye_event(
                        state["start_time"], state["last_update_time"],
                        state["direction"], state["max_intensity"],
                        self._aggregate_event_confidence(state),
                        self._aggregate_event_velocity(state),
                        tvt_prediction, tvt_prob_threshold,
                    )
                    self._reset_hysteresis(axis_key)
                    ready = self._advance_hysteresis(axis_key, direction)
                    if ready:
                        state.update({
                            "active": True, "start_time": timestamp,
                            "last_update_time": timestamp, "direction": direction,
                            "max_intensity": intensity, "confidence": confidence,
                            "velocity": velocity,
                            "confidence_samples": [float(confidence)],
                            "velocity_samples": [], "last_intensity": float(intensity),
                        })
                    else:
                        state["active"] = False
                else:
                    dt = float(timestamp - state["last_update_time"])
                    if dt > 0:
                        inst_v = abs(float(intensity) - float(state.get("last_intensity", intensity))) / dt
                        if np.isfinite(inst_v):
                            vs = state.get("velocity_samples") or []
                            if len(vs) < 300:
                                vs.append(float(inst_v))
                            state["velocity_samples"] = vs
                    cs = state.get("confidence_samples") or []
                    if len(cs) < 300:
                        cs.append(float(confidence))
                    state["confidence_samples"] = cs
                    state["max_intensity"]    = max(state["max_intensity"], intensity)
                    state["last_update_time"] = timestamp
                    state["confidence"]       = max(state["confidence"], confidence)
                    state["velocity"]         = max(float(state.get("velocity", 0.0)), float(velocity))
                    state["last_intensity"]   = float(intensity)
        else:
            self._reset_hysteresis(axis_key)
            if state["active"]:
                if timestamp - state["last_update_time"] > eff_gap:
                    self._record_eye_event(
                        state["start_time"], state["last_update_time"],
                        state["direction"], state["max_intensity"],
                        self._aggregate_event_confidence(state),
                        self._aggregate_event_velocity(state),
                        tvt_prediction, tvt_prob_threshold,
                    )
                    state["active"] = False

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def _aggregate_event_confidence(self, state: Dict) -> float:
        samples = state.get("confidence_samples") or []
        if not samples:
            return float(state.get("confidence", 0.0) or 0.0)
        arr = np.array(samples, dtype=np.float64)
        return float(np.clip(np.percentile(arr, 75), 0.0, 1.0))

    def _aggregate_event_velocity(self, state: Dict) -> float:
        samples = state.get("velocity_samples") or []
        if not samples:
            return float(state.get("velocity", 0.0) or 0.0)
        arr = np.array(samples, dtype=np.float64)
        v   = float(np.percentile(arr, 90))
        return max(0.0, v) if np.isfinite(v) else float(state.get("velocity", 0.0) or 0.0)

    # ------------------------------------------------------------------
    # Multiple faces
    # ------------------------------------------------------------------

    def _check_multiple_faces(self, timestamp: float, num_faces: int) -> None:
        state = self.active_violations["multiple_faces"]
        if num_faces > 1:
            if not state["active"]:
                state.update({
                    "active": True, "start_time": timestamp,
                    "last_update_time": timestamp, "duration": 0.0,
                    "max_intensity": float(num_faces), "confidence": 0.9, "velocity": 0.0,
                })
            else:
                state["last_update_time"] = timestamp
                state["duration"]         = timestamp - float(state["start_time"])
                state["max_intensity"]    = max(float(state["max_intensity"]), float(num_faces))
        else:
            if state["active"]:
                duration = float(state["last_update_time"]) - float(state["start_time"])
                min_dur  = float(getattr(self.config, "MIN_MULTIPLE_FACE_DURATION", 0.3))
                if duration >= min_dur:
                    ts = round(float(state["start_time"]), 2)
                    self.counts["multiple_faces"] += 1
                    self.timestamps["multiple_faces"].append(ts)
                    self.violation_events.append({
                        "type": "multiple_faces", "timestamp": ts,
                        "duration": round(duration, 1), "intensity": 0.0,
                        "confidence": float(state["confidence"]), "velocity": 0.0,
                    })
                state["active"] = False

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_counts(self) -> Dict[str, int]:
        return self.counts.copy()

    def get_timestamps(self, violation_type: str) -> List[float]:
        return self.timestamps.get(violation_type, [])

    def get_all_events(self) -> List[Dict]:
        return sorted(self.violation_events, key=lambda x: x["timestamp"])

    # ------------------------------------------------------------------
    # Finalize — flush all open states at end of video
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """
        Flush every active state at end-of-stream.
        Passes last_update_time directly so _record_* methods apply
        their own single gap_compensation — no double-adding.
        """
        for state, recorder in (
            (self.current_yaw_state,   self._record_head_event),
            (self.current_pitch_state, self._record_head_event),
        ):
            if state["active"]:
                recorder(
                    state["start_time"], state["last_update_time"],
                    state["direction"],  state["max_intensity"],
                    self._aggregate_event_confidence(state),
                    self._aggregate_event_velocity(state),
                )
                state["active"] = False

        for state, recorder in (
            (self.current_eye_h_state, self._record_eye_event),
            (self.current_eye_v_state, self._record_eye_event),
        ):
            if state["active"]:
                recorder(
                    state["start_time"], state["last_update_time"],
                    state["direction"],  state["max_intensity"],
                    self._aggregate_event_confidence(state),
                    self._aggregate_event_velocity(state),
                )
                state["active"] = False

        if self.current_face_missing_state["active"]:
            self._record_face_missing_event(
                self.current_face_missing_state["start_time"],
                self.current_face_missing_state["last_update_time"],
            )
            self.current_face_missing_state["active"] = False
        self.current_face_missing_state["first_no_face_time"] = None

        if self.current_face_occluded_state["active"]:
            self._record_face_occluded_event(
                self.current_face_occluded_state["start_time"],
                self.current_face_occluded_state["last_update_time"],
                self.current_face_occluded_state["max_occlusion"],
            )
            self.current_face_occluded_state["active"] = False

        mf = self.active_violations["multiple_faces"]
        if mf["active"]:
            duration = float(mf["last_update_time"]) - float(mf["start_time"])
            min_dur  = float(getattr(self.config, "MIN_MULTIPLE_FACE_DURATION", 0.3))
            if duration >= min_dur:
                ts = round(float(mf["start_time"]), 2)
                self.counts["multiple_faces"] += 1
                self.timestamps["multiple_faces"].append(ts)
                self.violation_events.append({
                    "type": "multiple_faces", "timestamp": ts,
                    "duration": round(duration, 1), "intensity": 0.0,
                    "confidence": float(mf["confidence"]), "velocity": 0.0,
                })
            mf["active"] = False