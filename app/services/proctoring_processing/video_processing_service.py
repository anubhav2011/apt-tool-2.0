# video_processing_service.py
"""
Video Processing Service
Handles video download, frame extraction, and processing orchestration.

"""

import os

# OpenCV FFmpeg: increase read attempts for difficult multi-stream MP4s.
# Must be set before importing cv2 for OpenCV's FFmpeg backend to pick it up.
os.environ.setdefault(
    "OPENCV_FFMPEG_READ_ATTEMPTS",
    os.getenv("OPENCV_FFMPEG_READ_ATTEMPTS") or "20000",
)

import cv2
import numpy as np
import tempfile
import queue
import threading
import time
from typing import Dict, List, Optional, Tuple, Any

import requests
from app.utils.logger import debug_logger, is_proctoring_diagnostics_enabled
from app.utils.data_formatting import seconds_to_mmss

from app.core.config import ProctoringConfig
from app.core.exceptions import VideoDownloadError, VideoProcessingError
from .base_proctoring_processing_service import IVideoProcessingService
from .detection_service import DetectionService

_SENTINEL   = object()


class _AnalyzerError:
    """Forwarded from analyzer thread to aggregator on exception."""
    __slots__ = ("exc",)
    def __init__(self, exc: Exception) -> None:
        self.exc = exc


class VideoProcessingService(IVideoProcessingService):
    """
    Service for video processing pipeline.
    Orchestrates frame extraction and analysis.
    """

    def __init__(self, config: ProctoringConfig) -> None:
        self.config = config
        self._setup()

    def _setup(self) -> None:
        self.detection_service = DetectionService(self.config)
        self.detection_service.violation_tracker._last_active_yaw_time = -999.0
        debug_logger.debug("VideoProcessingService initialised")

    # ------------------------------------------------------------------
    # Frame helpers
    # ------------------------------------------------------------------

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame so longest edge ≤ MAX_FRAME_DIMENSION and crop left 50%."""

        if frame is None or frame.size == 0:
            return frame

        h, w = frame.shape[:2]
        if h == 0 or w == 0:
            return frame

        # -------- Resize --------
        max_dim = max(h, w)
        if max_dim > self.config.MAX_FRAME_DIMENSION:
            scale = self.config.MAX_FRAME_DIMENSION / max_dim
            frame = cv2.resize(
                frame,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_LINEAR,
            )

        # -------- Crop LEFT 50% (candidate) --------
        h, w = frame.shape[:2]
        frame = frame[:, :w // 2]

        return frame

    def _validate_frame(self, frame: np.ndarray) -> bool:
        """
        Reject corrupt or completely black frames.

        A frame is invalid when:
          - It is None or empty / too small to analyse.
          - Mean pixel value < 1.0 (true all-black = camera dropout).
          - Standard deviation < 1.0 (completely uniform solid colour).

        CHANGE: threshold lowered from 2.0 → 1.0.
        Rationale: dark interview environments (dimly lit room, evening
        lighting) can have mean luminance of 3–8 while still being valid
        frames with a detectable face. The old threshold of 2.0 was
        rejecting legitimate dim frames, causing spurious face_missing
        events and under-counting violations.
        """
        if frame is None or frame.size == 0:
            return False
        if frame.shape[0] < 4 or frame.shape[1] < 4:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(np.mean(gray))
        std  = float(np.std(gray))
        # CHANGE: 2.0 → 1.0 for mean; std unchanged
        if mean < 1.0 or std < 1.0:
            return False
        return True

    # ------------------------------------------------------------------
    # Parallel pipeline
    # ------------------------------------------------------------------

    def _run_parallel_pipeline(
        self,
        cap:        cv2.VideoCapture,
        fps:        float,
        frame_skip: int,
        thresholds: Dict[str, float],
        warmup:     float,
    ) -> Tuple[int, float]:
        """
        3-thread producer → analyser → aggregator pipeline.

        Returns (frames_processed, last_frame_timestamp).

        """
        frame_queue:   queue.Queue = queue.Queue(maxsize=60)
        results_queue: queue.Queue = queue.Queue(maxsize=60)
        frames_processed = [0]
        last_timestamp   = [0.0]
        pipeline_error   = [None]   # type: List[Optional[Exception]]

        # ── Reader thread ─────────────────────────────────────────────
        def reader() -> None:
            frame_idx = 0
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % frame_skip != 0:
                        frame_idx += 1
                        continue
                    current_time = frame_idx / fps
                    frame        = self._resize_frame(frame)
                    if not self._validate_frame(frame):
                        frame_idx += 1
                        continue
                    frame_queue.put((frame_idx, current_time, frame))
                    frame_idx += 1
            except Exception as e:
                debug_logger.exception("reader thread error: %s", e)
            finally:
                frame_queue.put(_SENTINEL)

        # ── Analyzer thread ───────────────────────────────────────────
        def analyzer() -> None:
            enable_tvt = getattr(self.config, "ENABLE_TVT", False)
            try:
                while True:
                    try:
                        item = frame_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    if item is _SENTINEL:
                        results_queue.put(_SENTINEL)
                        return
                    frame_idx, current_time, frame = item
                    try:
                        (gaze_h, gaze_v, num_faces, bbox_center,
                         gaze_confidence, occlusion_ratio) = (
                            self.detection_service.detect_gaze(frame)
                        )
                        yaw, pitch, roll, head_confidence = (
                            self.detection_service.detect_head_pose(frame)
                        )
                        landmark_vector = None
                        if enable_tvt:
                            landmark_vector = (
                                self.detection_service.get_landmark_vector(frame)
                            )
                    except Exception as frame_exc:
                        # Skip this frame; don't crash the whole pipeline
                        debug_logger.debug(
                            f"analyzer: frame {frame_idx} detection error "
                            f"(skipped): {frame_exc}"
                        )
                        continue
                    results_queue.put((frame_idx, {
                        "timestamp":       current_time,
                        "gaze_h":          gaze_h,
                        "gaze_v":          gaze_v,
                        "yaw":             yaw,
                        "pitch":           pitch,
                        "roll":            roll,
                        "num_faces":       num_faces,
                        "gaze_confidence": gaze_confidence,
                        "head_confidence": head_confidence,
                        "occlusion_ratio": occlusion_ratio,
                        "landmark_vector": landmark_vector,
                    }))
            except Exception as e:
                # CHANGE 2: forward exception to aggregator so it can stop
                debug_logger.exception("analyzer thread fatal error: %s", e)
                results_queue.put(_AnalyzerError(e))

        # ── Aggregator thread ─────────────────────────────────────────
        def aggregator() -> None:
            """
            CHANGE 1: arrival-order processing with bounded reorder buffer.

            Instead of waiting for next_expected (which stalls on gaps),
            we accumulate a small reorder buffer and flush it in timestamp
            order once it reaches REORDER_WINDOW frames or when the queue
            is drained. This preserves approximate ordering (violations
            are recorded at their start timestamp regardless) while
            keeping memory bounded.
            """
            REORDER_WINDOW    = 32    # frames to buffer before force-flush
            MAX_BUFFER_SIZE   = 512   # hard cap to prevent memory leak
            reorder_buffer: List[Tuple[int, Dict]] = []
            count = 0

            def _flush_buffer(force: bool = False) -> None:
                nonlocal count
                if not reorder_buffer:
                    return
                # Sort by frame_idx for approximate temporal ordering
                reorder_buffer.sort(key=lambda x: x[0])
                flush_count = len(reorder_buffer) if force else max(
                    0, len(reorder_buffer) - REORDER_WINDOW // 2
                )
                to_flush = reorder_buffer[:flush_count]
                del reorder_buffer[:flush_count]
                for _, r in to_flush:
                    ts = r["timestamp"]
                    if ts >= warmup:
                        try:
                            self.detection_service.update_violations(
                                ts,
                                r["gaze_h"],        r["gaze_v"],
                                r["yaw"],           r["pitch"],  r["roll"],
                                r["num_faces"],     thresholds,
                                r["gaze_confidence"],
                                r["head_confidence"],
                                r["occlusion_ratio"],
                                landmark_vector=r.get("landmark_vector"),
                            )
                        except Exception as upd_exc:
                            debug_logger.debug(
                                f"update_violations error at ts={ts:.2f}: {upd_exc}"
                            )
                    if ts > last_timestamp[0]:
                        last_timestamp[0] = ts
                    count += 1

            try:
                while True:
                    try:
                        item = results_queue.get(timeout=1.0)
                    except queue.Empty:
                        # Flush stale frames from reorder buffer on idle
                        if reorder_buffer:
                            _flush_buffer(force=False)
                        continue

                    # CHANGE 2: handle analyzer exception forwarding
                    if isinstance(item, _AnalyzerError):
                        pipeline_error[0] = item.exc
                        _flush_buffer(force=True)
                        break

                    if item is _SENTINEL:
                        _flush_buffer(force=True)
                        break

                    frame_idx, result = item
                    reorder_buffer.append((frame_idx, result))

                    # CHANGE 3: hard cap to prevent memory leak
                    if len(reorder_buffer) >= MAX_BUFFER_SIZE:
                        debug_logger.warning(
                            f"Parallel pipeline: reorder buffer reached "
                            f"{MAX_BUFFER_SIZE} — force-flushing to prevent "
                            f"memory leak (possible frame index gap)"
                        )
                        _flush_buffer(force=True)
                    elif len(reorder_buffer) >= REORDER_WINDOW:
                        _flush_buffer(force=False)

            except Exception as e:
                debug_logger.exception("aggregator thread error: %s", e)
                pipeline_error[0] = e

            frames_processed[0] = count

        t_reader     = threading.Thread(target=reader,     daemon=True)
        t_analyzer   = threading.Thread(target=analyzer,   daemon=True)
        t_aggregator = threading.Thread(target=aggregator, daemon=True)

        t_reader.start()
        t_analyzer.start()
        t_aggregator.start()

        t_reader.join()
        t_analyzer.join()
        t_aggregator.join()

        # CHANGE 2: propagate analyzer/aggregator errors
        if pipeline_error[0] is not None:
            raise VideoProcessingError(
                f"Parallel pipeline failed: {pipeline_error[0]}", 5013
            )

        return frames_processed[0], last_timestamp[0]

    # ------------------------------------------------------------------
    # Video download
    # ------------------------------------------------------------------

    def _download_video(self, video_url: str) -> str:
        tmp       = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_path = tmp.name
        tmp.close()
        try:
            debug_logger.debug(f"Downloading video: {video_url}")
            resp = requests.get(video_url, stream=True, timeout=60)
            resp.raise_for_status()
            with open(temp_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            debug_logger.debug(f"Download complete → {temp_path}")
            return temp_path
        except Exception as e:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise VideoDownloadError(
                f"Failed to download video: {e}", 5010
            )

    # ------------------------------------------------------------------
    # Main processing pipeline
    # ------------------------------------------------------------------

    def process_video(self, video_path: str, session_id: str) -> Dict:
        """
        Main video processing pipeline.

        v2.3 changes:
          - _validate_frame rejection no longer freezes last_timestamp:
            last_timestamp_raw tracks time of every frame (valid or not),
            used as fallback for video_duration when FRAME_COUNT=0.
          - video_duration uses max(cap_duration, last_frame_time) to
            handle both reliable and unreliable FRAME_COUNT sources.
          - fps_source stored separately from fps_sampled in metadata.
        """
        start_time        = time.time()
        downloaded_file   = None
        video_path_to_use = video_path

        try:
            diag  = is_proctoring_diagnostics_enabled()
            vpath = (video_path or "").strip()
            if vpath.startswith(("http://", "https://")):
                downloaded_file   = self._download_video(video_path)
                video_path_to_use = downloaded_file

            cap = cv2.VideoCapture(video_path_to_use)
            if not cap.isOpened():
                raise VideoProcessingError(
                    f"Cannot open video: {video_path_to_use}", 5011
                )

            fps_raw = cap.get(cv2.CAP_PROP_FPS)
            fps     = float(fps_raw) if fps_raw and fps_raw > 0 else float(
                self.config.TARGET_FPS
            )

            total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Primary duration from metadata
            cap_duration   = (
                total_frames / fps if fps > 0 and total_frames > 0 else 0.0
            )

            if diag:
                debug_logger.debug(
                    f"Video → FPS:{fps:.1f} | Frames:{total_frames} "
                    f"| Duration:{cap_duration:.1f}s"
                )

            frame_skip = max(1, int(round(fps / max(self.config.TARGET_FPS, 1))))
            if diag:
                debug_logger.debug(
                    f"Sampling every {frame_skip} frame(s) "
                    f"≈ {self.config.TARGET_FPS} FPS target"
                )

            thresholds = self.config.get_default_thresholds()
            if diag:
                debug_logger.debug(
                    f"Thresholds → "
                    f"eye_h={thresholds['eye_horizontal']}° "
                    f"eye_v={thresholds['eye_vertical']}° "
                    f"yaw={thresholds['yaw']}° "
                    f"pitch={thresholds['pitch']}°"
                )

            warmup           = float(getattr(self.config, "WARMUP_SECONDS", 0.0))
            frame_idx        = 0
            frames_processed = 0
            last_timestamp   = 0.0   # last timestamp of a VALID processed frame
            last_timestamp_raw = 0.0 # last timestamp of ANY frame (for duration)

            if getattr(self.config, "ENABLE_PARALLEL_PROCESSING", False):
                if diag:
                    debug_logger.debug("Using parallel 3-thread pipeline")
                frames_processed, last_timestamp = self._run_parallel_pipeline(
                    cap, fps, frame_skip, thresholds, warmup
                )
                last_timestamp_raw = last_timestamp

            else:
                # ── Sequential pipeline ───────────────────────────────────
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_idx % frame_skip != 0:
                        frame_idx += 1
                        continue

                    current_time = frame_idx / fps

                    # CHANGE: always update raw timestamp so video_duration
                    # is accurate even when last frames are black
                    last_timestamp_raw = current_time

                    frame = self._resize_frame(frame)

                    if not self._validate_frame(frame):
                        if diag:
                            debug_logger.debug(
                                f"[{current_time:.1f}s] Frame {frame_idx} "
                                f"rejected (corrupt/black)"
                            )
                        frame_idx += 1
                        continue

                    (gaze_h, gaze_v, num_faces, bbox_center,
                     gaze_confidence, occlusion_ratio) = (
                        self.detection_service.detect_gaze(frame)
                    )

                    yaw, pitch, roll, head_confidence = (
                        self.detection_service.detect_head_pose(frame)
                    )

                    landmark_vector = None
                    if getattr(self.config, "ENABLE_TVT", False):
                        landmark_vector = (
                            self.detection_service.get_landmark_vector(frame)
                        )

                    if diag and frames_processed % 60 == 0:
                        debug_logger.debug(
                            f"[{current_time:.1f}s] "
                            f"gaze=({gaze_h},{gaze_v}) "
                            f"head=({yaw},{pitch}) "
                            f"faces={num_faces} "
                            f"g_conf={gaze_confidence:.2f} "
                            f"h_conf={head_confidence:.2f} "
                            f"occ={occlusion_ratio:.2f}"
                        )

                    if current_time >= warmup:
                        self.detection_service.update_violations(
                            current_time,
                            gaze_h, gaze_v,
                            yaw, pitch, roll,
                            num_faces, thresholds,
                            gaze_confidence,
                            head_confidence,
                            occlusion_ratio,
                            landmark_vector=landmark_vector,
                        )

                    last_timestamp    = current_time
                    frames_processed += 1
                    frame_idx        += 1

            cap.release()

            # CHANGE: use max of cap-reported duration and last observed
            # frame time to handle both reliable and streaming sources.
            # cap_duration may be 0 for streaming; last_timestamp_raw may
            # be shorter if encoder added silence at end.
            video_duration = max(cap_duration, last_timestamp_raw)
            if video_duration <= 0.0 and last_timestamp > 0.0:
                video_duration = last_timestamp
            if diag and abs(cap_duration - last_timestamp_raw) > 2.0:
                debug_logger.debug(
                    f"Duration mismatch: cap_reported={cap_duration:.1f}s "
                    f"last_frame={last_timestamp_raw:.1f}s "
                    f"using={video_duration:.1f}s"
                )

            # Flush open events
            self.detection_service.violation_tracker.finalize()

            processing_time = time.time() - start_time

            report = self._generate_report(
                session_id,
                thresholds,
                processing_time,
                video_duration,
                frames_processed,
                fps_source=fps,
            )

            debug_logger.debug(
                f"Done in {processing_time:.2f}s — "
                f"{len(report['analysis']['gestures'])} gesture types | "
                f"{frames_processed} frames processed"
            )
            return report

        except (VideoDownloadError, VideoProcessingError):
            raise
        except Exception as e:
            debug_logger.exception("Error processing video: %s", e)
            raise VideoProcessingError(
                f"Video processing failed: {e}", 5012
            )
        finally:
            if downloaded_file and os.path.exists(downloaded_file):
                try:
                    os.unlink(downloaded_file)
                except Exception as cleanup_err:
                    debug_logger.warning(
                        f"Failed to delete temp file: {cleanup_err}"
                    )

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _generate_report(
        self,
        session_id:       str,
        thresholds:       Dict,
        processing_time:  float,
        video_duration:   float,
        frames_processed: int,
        fps_source:       float = 0.0,
    ) -> Dict:
        """
        Build the final gestures report from recorded violation events.

        v2.3 changes:
          - processing_metadata now has separate fps_source, fps_target,
            fps_sampled fields. Old fps_detected was misleading
            (was frames_processed/duration, not actual video FPS).
        """
        diag = is_proctoring_diagnostics_enabled()
        events = self.detection_service.get_violation_events()
        if diag:
            debug_logger.debug(f"Total violation events: {len(events)}")

        tally: Dict[str, int] = {}
        for ev in events:
            tally[ev["type"]] = tally.get(ev["type"], 0) + 1
        if diag:
            debug_logger.debug(f"Event breakdown: {tally}")

        head_events          = [e for e in events if e["type"].startswith("head_")]
        eye_events           = [e for e in events if e["type"].startswith("gaze_")]
        face_missing_events  = [e for e in events if e["type"] == "face_missing"]
        multi_face_events    = [e for e in events if e["type"] == "multiple_faces"]
        face_occluded_events = [e for e in events if e["type"] == "face_occluded"]

        report_only_high_risk = getattr(
            self.config, "REPORT_ONLY_HIGH_RISK", False
        )
        if report_only_high_risk:
            head_events = [
                e for e in head_events if self._is_high_risk_head_event(e)
            ]
            eye_events = [
                e for e in eye_events if self._is_high_risk_eye_event(e)
            ]
            if diag:
                debug_logger.debug(
                    f"After high-risk filter → "
                    f"head:{len(head_events)}, eye:{len(eye_events)}"
                )

        gestures: List[Dict] = []

        # ── Head movement ──────────────────────────────────────────────────
        if head_events:
            if diag:
                debug_logger.debug(f"Head events: {len(head_events)}")
            gestures.append({
                "name": "head_movement",
                "occurrence": [
                    {
                        "timestamp":  seconds_to_mmss(e["timestamp"]),
                        "duration":   round(float(e["duration"]), 1),
                        "direction":  e["type"].replace("head_", ""),
                        "intensity":  round(float(e["intensity"]), 1),
                        "confidence": round(float(e.get("confidence", 0.0)), 2),
                        "velocity":   round(float(e.get("velocity", 0.0)), 2),
                    }
                    for e in head_events
                ],
            })

        # ── Eye movement ───────────────────────────────────────────────────
        if eye_events:
            if diag:
                debug_logger.debug(f"Eye events: {len(eye_events)}")
            gestures.append({
                "name": "eye_movement",
                "occurrence": [
                    {
                        "timestamp":  seconds_to_mmss(e["timestamp"]),
                        "duration":   round(float(e["duration"]), 1),
                        "direction":  e["type"].replace("gaze_", ""),
                        "intensity":  round(float(e["intensity"]), 1),
                        "confidence": round(float(e.get("confidence", 0.0)), 2),
                        "velocity":   round(float(e.get("velocity", 0.0)), 2),
                    }
                    for e in eye_events
                ],
            })

        # ── Face missing ───────────────────────────────────────────────────
        if face_missing_events:
            if diag:
                debug_logger.debug(f"Face missing events: {len(face_missing_events)}")
            gestures.append({
                "name": "face_missing",
                "occurrence": [
                    {
                        "timestamp":  seconds_to_mmss(e["timestamp"]),
                        "duration":   round(float(e["duration"]), 1),
                        "direction":  None,
                        "intensity":  0.0,
                        "confidence": round(float(e.get("confidence", 1.0)), 2),
                        "velocity":   0.0,
                    }
                    for e in face_missing_events
                ],
            })

        # ── Multiple faces ─────────────────────────────────────────────────
        if multi_face_events:
            if diag:
                debug_logger.debug(f"Multiple faces events: {len(multi_face_events)}")
            gestures.append({
                "name": "multiple_faces",
                "occurrence": [
                    {
                        "timestamp":  seconds_to_mmss(e["timestamp"]),
                        "duration":   round(float(e["duration"]), 1),
                        "direction":  None,
                        "intensity":  0.0,
                        "confidence": round(float(e.get("confidence", 0.9)), 2),
                        "velocity":   0.0,
                    }
                    for e in multi_face_events
                ],
            })

        # ── Face occluded ──────────────────────────────────────────────────
        if face_occluded_events:
            if diag:
                debug_logger.debug(f"Face occluded events: {len(face_occluded_events)}")
            gestures.append({
                "name": "face_occluded",
                "occurrence": [
                    {
                        "timestamp":  seconds_to_mmss(e["timestamp"]),
                        "duration":   round(float(e["duration"]), 1),
                        "direction":  None,
                        "intensity":  round(float(e["intensity"]), 1),
                        "confidence": round(float(e.get("confidence", 0.85)), 2),
                        "velocity":   0.0,
                    }
                    for e in face_occluded_events
                ],
            })

        # CHANGE: separate fps fields for clarity
        # fps_source: actual FPS of source video
        # fps_target: what we aimed to process at (config)
        # fps_sampled: effective rate of frames actually analysed
        fps_sampled = round(
            frames_processed / max(video_duration, 1e-6), 1
        )

        return {
            "session_id": session_id,
            "status":     "success",
            "message":    "Video processed successfully",
            "analysis": {
                "gestures":        gestures,
                "thresholds_used": thresholds,
                "processing_metadata": {
                    "processing_time_sec": round(processing_time, 2),
                    "video_duration_sec":  round(video_duration, 2),
                    "frames_processed":    frames_processed,
                    # CHANGE: replaced misleading fps_detected with three clear fields
                    "fps_source":    round(fps_source, 1),
                    "fps_target":    float(self.config.TARGET_FPS),
                    "fps_sampled":   fps_sampled,
                    # backward-compat alias (same value as fps_source)
                    "fps":           round(fps_source, 1),
                },
            },
        }

    # ------------------------------------------------------------------
    # High-risk filter helpers
    # ------------------------------------------------------------------

    def _is_high_risk_head_event(self, e: Dict) -> bool:
        """intensity is a raw float — no string parsing needed."""
        intensity = float(e.get("intensity", 0.0))
        duration  = float(e.get("duration",  0.0))
        velocity  = float(e.get("velocity",  0.0))
        etype     = e.get("type", "")

        min_int = (
            float(getattr(
                self.config, "MIN_HEAD_INTENSITY_HIGH_RISK_PITCH", 22.0
            ))
            if etype in ("head_up", "head_down")
            else float(getattr(
                self.config, "MIN_HEAD_INTENSITY_HIGH_RISK", 28.0
            ))
        )
        min_dur = float(getattr(self.config, "MIN_HEAD_DURATION_HIGH_RISK", 0.8))
        fast    = float(getattr(self.config, "SUSPICIOUS_VELOCITY_THRESHOLD", 25.0))
        return intensity >= min_int or (duration >= min_dur and velocity >= fast)

    def _is_high_risk_eye_event(self, e: Dict) -> bool:
        """intensity is a raw float."""
        intensity = float(e.get("intensity", 0.0))
        duration  = float(e.get("duration",  0.0))
        velocity  = float(e.get("velocity",  0.0))
        min_int   = float(getattr(self.config, "MIN_EYE_INTENSITY_HIGH_RISK", 8.0))
        fast      = float(getattr(self.config, "SUSPICIOUS_VELOCITY_THRESHOLD", 25.0))
        return intensity >= min_int or (duration >= 0.8 and velocity >= fast)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        self.detection_service.cleanup()
        debug_logger.debug("VideoProcessingService cleaned up")