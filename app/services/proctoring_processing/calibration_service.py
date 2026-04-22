# """
# Calibration Service
# Industry-standard calibration following PDF specification
# Variance-based thresholds with minimum floors
# """
#
# import numpy as np
# from typing import Dict, List, Tuple, Optional, TypedDict
#
# from .base_proctoring_processing_service import BaseService
#
# # ---------------------------
# # TypedDict for Baseline
# # ---------------------------
# class CalibrationBaseline(TypedDict):
#     eye_mean: float
#     eye_variance: float
#     yaw_mean: float
#     pitch_mean: float
#     roll_mean: float
#     yaw_variance: float
#     pitch_variance: float
#     face_presence_ratio: float
#     bbox_stability: float
#
#
# class CalibrationService(BaseService):
#     """
#     Service for video calibration following industry standards
#     """
#
#     def _setup(self):
#         """Initialize calibration state"""
#         self.frames: List[Dict] = []
#         self.total_frames: int = 0
#
#     def cleanup(self):
#         """Cleanup calibration resources"""
#         self.frames.clear()
#         self.log_info("Calibration service cleaned up")
#
#     def _safe_mean(self, values: List[float]) -> float:
#         if not values:
#             return 0.0
#         return float(np.mean(values))
#
#     def _safe_variance(self, values: List[float]) -> float:
#         if len(values) < 2:
#             return 0.0
#         return float(np.var(values))
#
#     # --------------------------------------------------
#     # Frame ingestion
#     # --------------------------------------------------
#     def add_frame_data(
#         self,
#         timestamp: float,
#         gaze_angle: float,
#         yaw: float,
#         pitch: float,
#         roll: float,
#         face_detected: bool,
#         bbox_center: Optional[Tuple[float, float]] = None,
#         face_presence_score: float = 1.0,
#         landmarks_stable: bool = True,
#     ):
#         """Add frame measurements during calibration window"""
#         self.total_frames += 1
#         self.frames.append({
#             "timestamp": timestamp,
#             "gaze_angle": gaze_angle,
#             "yaw": yaw,
#             "pitch": pitch,
#             "roll": roll,
#             "face_detected": face_detected,
#             "bbox_center": bbox_center,
#             "face_presence_score": face_presence_score,
#             "landmarks_stable": landmarks_stable,
#         })
#
#     # --------------------------------------------------
#     # Frame rejection with outlier trimming
#     # --------------------------------------------------
#     def _reject_unstable_frames(self) -> List[Dict]:
#         """
#         Rule 2: Trim Outlier Frames
#         Remove top 15% most unstable frames to prevent mis-calibration
#         """
#         if len(self.frames) < 2:
#             return self.frames
#
#         # Calculate stability score for each frame
#         frames_with_scores: List[Tuple[Dict, float]] = []
#
#         for i, frame in enumerate(self.frames):
#             instability_score = 0.0
#
#             # Quality penalties
#             if frame["face_presence_score"] < 0.85:
#                 instability_score += 10.0
#             if not frame["face_detected"] or not frame["landmarks_stable"]:
#                 instability_score += 20.0
#             if frame["gaze_angle"] is not None and abs(frame["gaze_angle"]) > 0.4:
#                 instability_score += abs(frame["gaze_angle"]) * 10.0
#
#             # Motion penalties
#             if i > 0:
#                 prev = self.frames[i - 1]
#                 time_delta = frame["timestamp"] - prev["timestamp"]
#
#                 if time_delta > 0:
#                     if frame["yaw"] is not None and prev["yaw"] is not None:
#                         yaw_speed = abs(frame["yaw"] - prev["yaw"]) / time_delta
#                         if yaw_speed > 10.0:
#                             instability_score += yaw_speed
#
#                     if frame["pitch"] is not None and prev["pitch"] is not None:
#                         pitch_speed = abs(frame["pitch"] - prev["pitch"]) / time_delta
#                         if pitch_speed > 10.0:
#                             instability_score += pitch_speed
#
#                     if frame["bbox_center"] and prev["bbox_center"]:
#                         shift_x = abs(frame["bbox_center"][0] - prev["bbox_center"][0])
#                         shift_y = abs(frame["bbox_center"][1] - prev["bbox_center"][1])
#                         if (shift_x + shift_y) > 0.05:
#                             instability_score += (shift_x + shift_y) * 100.0
#
#             frames_with_scores.append((frame, instability_score))
#
#         # Sort by instability score and remove top 15%
#         frames_with_scores.sort(key=lambda x: x[1])
#         trim_count = int(len(frames_with_scores) * (self.config.OUTLIER_TRIM_PERCENTAGE / 100.0))
#         stable_frames = [frame for frame, _ in frames_with_scores[:-trim_count]] if trim_count > 0 else [frame for frame, _ in frames_with_scores]
#
#         self.log_info(f"Trimmed {trim_count} outlier frames out of {len(self.frames)}")
#         return stable_frames
#
#     # --------------------------------------------------
#     # Stability window extraction
#     # --------------------------------------------------
#     def _extract_stability_window(self, stable_frames: List[Dict]) -> List[Dict]:
#         if len(stable_frames) < 10:
#             return stable_frames
#
#         best_window: Optional[List[Dict]] = None
#         best_score: float = float("inf")
#
#         for window_sec in (1.0, 2.0, 3.0):
#             for i in range(len(stable_frames)):
#                 start_time = stable_frames[i]["timestamp"]
#                 window = [f for f in stable_frames[i:] if f["timestamp"] - start_time <= window_sec]
#
#                 if len(window) < 10:
#                     continue
#
#                 gaze = [f["gaze_angle"] for f in window if f["gaze_angle"] is not None]
#                 yaw = [f["yaw"] for f in window if f["yaw"] is not None]
#                 pitch = [f["pitch"] for f in window if f["pitch"] is not None]
#
#                 if not gaze or not yaw or not pitch:
#                     continue
#
#                 eye_var = self._safe_variance(gaze)
#                 head_var = self._safe_variance(yaw) + self._safe_variance(pitch)
#                 face_presence = sum(f["face_presence_score"] for f in window) / len(window)
#
#                 bbox_positions = [f["bbox_center"] for f in window if f["bbox_center"]]
#                 bbox_score = 0.0
#                 if len(bbox_positions) > 1:
#                     xs = [p[0] for p in bbox_positions]
#                     ys = [p[1] for p in bbox_positions]
#                     bbox_score = float(np.std(xs) + np.std(ys))
#
#                 stability_score = eye_var + head_var + bbox_score - (face_presence * 10.0)
#
#                 if stability_score < best_score:
#                     best_score = stability_score
#                     best_window = window
#
#         return best_window if best_window else stable_frames
#
#     # --------------------------------------------------
#     # Baseline validation
#     # --------------------------------------------------
#     def _check_baseline_hard_limits(self, baseline: CalibrationBaseline) -> Tuple[bool, str]:
#         if baseline["yaw_mean"] > 20.0:
#             return False, "Excessive yaw movement during calibration"
#         if baseline["pitch_mean"] > 15.0:
#             return False, "Excessive pitch movement during calibration"
#         if baseline["roll_mean"] > 12.0:
#             return False, "Excessive roll movement during calibration"
#         if baseline["eye_mean"] > 0.8:
#             return False, "Excessive eye movement during calibration"
#         return True, "SUCCESS"
#
#     # --------------------------------------------------
#     # Main calibration with PDF-compliant threshold calculation
#     # --------------------------------------------------
#     def compute_calibration(self) -> Dict:
#         if self.total_frames < self.config.CALIBRATION_MIN_FRAMES:
#             return self._failed_calibration("INSUFFICIENT_FRAMES")
#
#         stable_frames = self._reject_unstable_frames()
#         if len(stable_frames) < 30:
#             return self._failed_calibration("TOO_FEW_STABLE_FRAMES")
#
#         window = self._extract_stability_window(stable_frames)
#         if len(window) < 10:
#             return self._failed_calibration("NO_STABLE_WINDOW_FOUND")
#
#         # Extract absolute values for mean calculations
#         gaze = [abs(f["gaze_angle"]) for f in window if f["gaze_angle"] is not None]
#         yaw = [abs(f["yaw"]) for f in window if f["yaw"] is not None]
#         pitch = [abs(f["pitch"]) for f in window if f["pitch"] is not None]
#         roll = [abs(f["roll"]) for f in window if f["roll"] is not None]
#
#         baseline: CalibrationBaseline = {
#             "eye_mean": self._safe_mean(gaze),
#             "eye_variance": self._safe_variance(gaze),
#             "yaw_mean": self._safe_mean(yaw),
#             "pitch_mean": self._safe_mean(pitch),
#             "roll_mean": self._safe_mean(roll),
#             "yaw_variance": self._safe_variance(yaw),      # Separate yaw variance
#             "pitch_variance": self._safe_variance(pitch),  # Separate pitch variance
#             "face_presence_ratio": sum(f["face_presence_score"] for f in window) / len(window),
#             "bbox_stability": 0.0,
#         }
#
#         bbox_positions = [f["bbox_center"] for f in window if f["bbox_center"]]
#         if len(bbox_positions) > 1:
#             xs = [p[0] for p in bbox_positions]
#             ys = [p[1] for p in bbox_positions]
#             baseline["bbox_stability"] = float(np.std(xs) + np.std(ys))
#
#         valid, reason = self._check_baseline_hard_limits(baseline)
#         if not valid:
#             return self._failed_calibration(reason)
#
#         adaptive_thresholds = {
#             "eye": max(
#                 baseline["eye_variance"] * self.config.VARIANCE_MULTIPLIER_EYE,
#                 self.config.MIN_EYE_THRESHOLD
#             ),
#             "yaw": max(
#                 baseline["yaw_variance"] * self.config.VARIANCE_MULTIPLIER_HEAD,
#                 self.config.MIN_HEAD_THRESHOLD
#             ),
#             "pitch": max(
#                 baseline["pitch_variance"] * self.config.VARIANCE_MULTIPLIER_HEAD,
#                 self.config.MIN_HEAD_THRESHOLD
#             ),
#             "roll": min(baseline["roll_mean"] * 1.2, 25.0),  # Keep existing logic for roll
#         }
#
#         baseline = {k: round(v, 2) for k, v in baseline.items()}
#         adaptive_thresholds = {k: round(v, 2) for k, v in adaptive_thresholds.items()}
#
#         self.log_info(f"Calibration SUCCESS | yaw_threshold={adaptive_thresholds['yaw']}° pitch_threshold={adaptive_thresholds['pitch']}° eye_threshold={adaptive_thresholds['eye']}°")
#         return {
#             "status": "SUCCESS",
#             "baseline": baseline,
#             "adaptive_thresholds": adaptive_thresholds,
#         }
#
#     # --------------------------------------------------
#     # Failure handler with global defaults
#     # --------------------------------------------------
#     def _failed_calibration(self, reason: str) -> Dict:
#         self.log_warning(f"Calibration failed: {reason} - Using global defaults")
#         return {
#             "status": "FAILED",
#             "baseline": {
#                 "eye_mean": 0.0,
#                 "eye_variance": self.config.DEFAULT_EYE_VARIANCE,
#                 "yaw_mean": 0.0,
#                 "pitch_mean": 0.0,
#                 "roll_mean": 0.0,
#                 "yaw_variance": self.config.DEFAULT_YAW_VARIANCE,
#                 "pitch_variance": self.config.DEFAULT_PITCH_VARIANCE,
#                 "face_presence_ratio": 0.0,
#                 "bbox_stability": 0.0,
#             },
#             "adaptive_thresholds": self.config.get_default_thresholds(),
#             "failure_reason": reason,
#         }
