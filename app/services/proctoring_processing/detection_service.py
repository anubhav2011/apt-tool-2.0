# detection_service.py
"""
Detection Service — Gaze detection, head pose estimation, violation tracking.

Architecture (v3.0 — pluggable detection clients):
  - Per-frame detection is fully delegated to a ``BaseDetectionClient``
    implementation selected at startup via the ``DETECTION_CLIENT`` env var.
  - Supported values: ``mediapipe`` (default), ``intel``.
  - All higher-level logic (ViolationTracker, TVT, velocity histories,
    hysteresis, gap-bridging, safe-look suppression) lives here unchanged.
  - Swapping backends requires only an env-var change; no code edits needed.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple

from numpy.typing import NDArray

from .base_proctoring_processing_service import BaseService
from app.client.detection import get_detection_client


# ======================================================================
# DetectionService
# ======================================================================

class DetectionService(BaseService):
    """
    Orchestrates per-frame detection via a pluggable client and feeds
    results into the ViolationTracker state machine.

    The active detection backend is chosen by the ``DETECTION_CLIENT``
    environment variable (default: ``mediapipe``).
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

        # ── Detection client (pluggable backend) ──────────────────────
        self._detection_client = get_detection_client(c)

        # ── Motion history (velocity computation) ─────────────────────
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

    # ------------------------------------------------------------------
    # TVT helpers
    # ------------------------------------------------------------------

    def get_landmark_vector(
        self, frame: NDArray[np.uint8]
    ) -> Optional[NDArray[np.float32]]:
        """Delegate landmark extraction to the active detection client."""
        return self._detection_client.get_landmark_vector(frame)

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
    # Gaze detection — delegates to client
    # ------------------------------------------------------------------

    def detect_gaze(self, frame: NDArray[np.uint8]) -> Tuple[
        Optional[float], Optional[float], int,
        Optional[Tuple[float, float]], float, float,
    ]:
        return self._detection_client.detect_gaze(frame)

    # ------------------------------------------------------------------
    # Head pose detection — delegates to client
    # ------------------------------------------------------------------

    def detect_head_pose(
        self, frame: NDArray[np.uint8]
    ) -> Tuple[Optional[float], Optional[float], Optional[float], float]:
        return self._detection_client.detect_head_pose(frame)

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
        # Delegate native-resource cleanup to the active detection client
        client = getattr(self, "_detection_client", None)
        if client is not None:
            try:
                client.cleanup()
            except Exception:
                pass
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
