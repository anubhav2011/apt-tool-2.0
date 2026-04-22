"""
MediaPipe Detection Client
==========================
Concrete ``BaseDetectionClient`` backed by MediaPipe FaceMesh (legacy API,
no Tasks / FaceLandmarker).  All gaze + head-pose logic is ported verbatim
from the original ``DetectionService`` so behaviour is 100% identical.

Production notes
----------------
* Uses FaceMesh (not the newer Tasks API) to avoid libGLESv2 / native-lib
  issues on headless servers.
* Kalman smoothing, hysteresis, and iris-based gaze are all retained.
* Supports the optional TVT landmark-vector path via ``get_landmark_vector``.

Switching to this client
------------------------
Set the environment variable::

    DETECTION_CLIENT=mediapipe

or leave it unset — ``mediapipe`` is the default.
"""

from __future__ import annotations

import os

# Must precede ``import mediapipe`` to suppress TFLite / absl logging.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple

from numpy.typing import NDArray

from app.client.detection.face_tracking.base import BaseDetectionClient

# Letterbox metadata: (orig_w, orig_h, square_side, offset_x, offset_y)
SquarePackMeta = Tuple[int, int, int, int, int]


class MediaPipeDetectionClient(BaseDetectionClient):
    """
    Gaze + head-pose detection backed exclusively by MediaPipe FaceMesh.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _setup(self) -> None:
        c = self.config

        # ── Landmark index lists ──────────────────────────────────────
        self.LEFT_EYE          = list(c.FACE_MESH_LEFT_EYE_INDICES)
        self.RIGHT_EYE         = list(c.FACE_MESH_RIGHT_EYE_INDICES)
        self.LEFT_IRIS         = list(c.FACE_MESH_LEFT_IRIS_INDICES)
        self.RIGHT_IRIS        = list(c.FACE_MESH_RIGHT_IRIS_INDICES)
        self.LEFT_EYE_CORNERS  = list(c.FACE_MESH_LEFT_EYE_CORNER_INDICES)
        self.RIGHT_EYE_CORNERS = list(c.FACE_MESH_RIGHT_EYE_CORNER_INDICES)
        self.landmark_indices  = list(c.HEAD_POSE_LANDMARK_INDICES)
        self.model_points      = np.array(c.HEAD_POSE_MODEL_POINTS_MM, dtype=np.float64)

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

        # ── Gaze history ──────────────────────────────────────────────
        self.gaze_history: List[Tuple[float, float]] = []
        self.max_history_size = 5

    def cleanup(self) -> None:
        for attr in ("gaze_face_mesh", "pose_face_mesh"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        self.gaze_history.clear()

    # ------------------------------------------------------------------
    # Letterbox helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _letterbox_square_rgb(
        rgb: NDArray[np.uint8],
    ) -> Tuple[NDArray[np.uint8], SquarePackMeta]:
        h, w = rgb.shape[:2]
        S    = max(w, h)
        if w == h == S:
            return rgb, (w, h, S, 0, 0)
        square          = np.zeros((S, S, 3), dtype=rgb.dtype)
        ox, oy          = (S - w) // 2, (S - h) // 2
        square[oy:oy + h, ox:ox + w] = rgb
        return square, (w, h, S, ox, oy)

    @staticmethod
    def _square_norm_to_orig_norm(
        x: float, y: float, w: int, h: int, S: int, ox: int, oy: int
    ) -> Tuple[float, float]:
        return (x * float(S) - float(ox)) / float(w), \
               (y * float(S) - float(oy)) / float(h)

    def _landmarks_orig_norm(
        self, face_landmarks, meta: SquarePackMeta
    ) -> List[Tuple[float, float]]:
        w, h, S, ox, oy = meta
        return [
            self._square_norm_to_orig_norm(lm.x, lm.y, w, h, S, ox, oy)
            for lm in face_landmarks
        ]

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
    # Landmark vector (for TVT)
    # ------------------------------------------------------------------

    def get_landmark_vector(
        self, frame: NDArray[np.uint8]
    ) -> Optional[NDArray[np.float32]]:
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sq, meta = self._letterbox_square_rgb(rgb)
        results  = self.gaze_face_mesh.process(sq)
        mlm      = getattr(results, "multi_face_landmarks", None)
        if not mlm:
            return None
        on  = self._landmarks_orig_norm(mlm[0].landmark, meta)
        out = np.zeros(936, dtype=np.float32)
        n   = min(468, len(on))
        for i in range(n):
            out[i * 2]     = on[i][0]
            out[i * 2 + 1] = on[i][1]
        return out

    # ------------------------------------------------------------------
    # Gaze detection
    # ------------------------------------------------------------------

    def detect_gaze(
        self, frame: NDArray[np.uint8]
    ) -> Tuple[
        Optional[float], Optional[float], int,
        Optional[Tuple[float, float]], float, float,
    ]:
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sq, meta = self._letterbox_square_rgb(rgb)
        results  = self.gaze_face_mesh.process(sq)
        mlm      = getattr(results, "multi_face_landmarks", None)

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

            if avg_ear < 0.05:
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

            avg_eye_width = (
                _ew(self.LEFT_EYE_CORNERS) + _ew(self.RIGHT_EYE_CORNERS)
            ) / 2.0

            if avg_eye_width < 0.8:
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

            confidence = self._calculate_gaze_confidence(
                avg_ear, iris_valid, avg_eye_width
            )
            return (
                round(h_angle, 2), round(v_angle, 2),
                num_faces, bbox_center,
                confidence, occlusion_ratio,
            )
        except (IndexError, ValueError, TypeError):
            return None, None, num_faces, bbox_center, 0.0, occlusion_ratio

    # ------------------------------------------------------------------
    # Head-pose detection
    # ------------------------------------------------------------------

    def _estimate_head_pose(
        self,
        frame: NDArray[np.uint8],
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

            # Confidence from landmark spread + rotation singularity margin
            x_coords = [on[i][0] for i in self.landmark_indices]
            y_coords = [on[i][1] for i in self.landmark_indices]
            spread_x = float(
                np.clip((max(x_coords) - min(x_coords) - 0.08) / 0.14, 0.0, 1.0)
            )
            spread_y = float(
                np.clip((max(y_coords) - min(y_coords) - 0.08) / 0.14, 0.0, 1.0)
            )
            lm_conf   = 0.5 * spread_x + 0.5 * spread_y
            sy_conf   = float(np.clip((sy - 0.08) / 0.18, 0.0, 1.0))
            head_conf = float(
                np.clip(0.15 + 0.85 * (0.6 * lm_conf + 0.4 * sy_conf), 0.0, 1.0)
            )

            return (
                round(yaw_deg, 2), round(pitch_deg, 2),
                round(roll_deg, 2), round(head_conf, 2),
            )
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
    # Identity
    # ------------------------------------------------------------------

    @property
    def client_name(self) -> str:
        return "mediapipe"
