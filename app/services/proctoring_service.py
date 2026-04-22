# proctoring_service.py
"""
Proctoring Service
Main service orchestrating all proctoring business logic.

"""

from app.utils.logger import debug_logger, is_proctoring_diagnostics_enabled
from app.utils.data_formatting import (
    format_json_response,
    to_video_timestamp,
    parse_numeric_value,
    float_to_velocity_label,
    estimate_duration_from_gestures,
)
from typing import Dict, Optional, List
import asyncio
import time
import uuid
import os
import tempfile
import json

from app.core.config import ProctoringConfig, S3_CONFIG
from app.services.proctoring_processing.s3_video_queue_service import S3VideoQueueService
from app.services.proctoring_processing.scoring_service import ScoringService
from app.core.exceptions import (
    VideoProcessingError,
    DatabaseError,
    ReportNotFoundError,
    ValidationError,
)
from app.repositories.base_repository import IRepository
from app.repositories.ai_interview_status_repository import AiInterviewStatusRepository
from app.models.proctoring import ProctoringReport
from app.services.base_service import IProctoringService
from app.services.proctoring_processing import VideoProcessingService
from app.services.helper_proctoring_service import (
    EventRiskClassifier,
    record_proctoring_failure,
)


class ProctoringService(IProctoringService):
    """
    Main proctoring service — orchestrates all business logic.
    """

    def __init__(
        self,
        repository: IRepository[ProctoringReport],
        status_repository: AiInterviewStatusRepository,
    ):
        self.repository        = repository
        self.status_repository = status_repository
        self.config            = ProctoringConfig()
        self._event_risk      = EventRiskClassifier(self.config)
        debug_logger.debug("Proctoring service initialized")

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    async def process_video_upload(
        self,
        video_file,
        interview_id: Optional[str] = None,
    ) -> Dict:
        temp_path = None
        try:
            interview_id = interview_id or str(uuid.uuid4())
            debug_logger.debug(
                f"Processing uploaded video for interview: {interview_id}"
            )

            file_ext = os.path.splitext(video_file.filename)[1].lower()
            if file_ext not in self.config.ALLOWED_VIDEO_FORMATS:
                raise ValidationError(
                    f"Unsupported video format: {file_ext}", 4001
                )

            temp_file = tempfile.NamedTemporaryFile(
                delete=False, suffix=file_ext
            )
            temp_path = temp_file.name
            temp_file.close()

            content = await video_file.read()
            with open(temp_path, "wb") as f:
                f.write(content)

            file_size_mb = len(content) / (1024 * 1024)
            if file_size_mb > self.config.MAX_VIDEO_SIZE_MB:
                raise ValidationError(
                    f"Video file too large: {file_size_mb:.2f}MB exceeds "
                    f"maximum {self.config.MAX_VIDEO_SIZE_MB}MB",
                    4002,
                )

            debug_logger.debug(f"Video file size: {file_size_mb:.2f} MB")

            result           = await self.process_video_file(temp_path, interview_id)
            formatted_result = format_json_response(result)
            return json.loads(formatted_result)

        except (ValidationError, VideoProcessingError, DatabaseError):
            raise
        except Exception as e:
            debug_logger.error(
                f"Unexpected error processing video upload: {str(e)}",
                exc_info=True,
            )
            raise VideoProcessingError(
                f"Failed to process video upload: {str(e)}", 5001
            )
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                    debug_logger.debug(f"Cleaned up temp file: {temp_path}")
                except Exception as e:
                    debug_logger.warning(
                        f"Failed to cleanup temp file {temp_path}: {str(e)}"
                    )

    async def process_video_by_interview_id(self, interview_id: str) -> Dict:
        vid = (interview_id or "").strip()
        if not vid:
            raise ValidationError("interview_id is required", 4003)

        candidate_name = self.status_repository.get_candidate_name_for_interview(vid)
        if not candidate_name:
            raise ValidationError(
                "No candidate_name for this interview_id; cannot locate S3 video",
                4004,
            )

        debug_logger.debug(f"Processing S3 video for interview: {vid}")
        s3_queue = S3VideoQueueService(s3_config=S3_CONFIG)
        local_video_path: Optional[str] = None
        try:
            local_video_path = await asyncio.to_thread(
                s3_queue.download_video,
                vid,
                candidate_name,
            )
            result = await self.process_video_file(local_video_path, vid)
            formatted_result = format_json_response(result)
            return json.loads(formatted_result)
        except (ValidationError, VideoProcessingError, DatabaseError):
            raise
        except RuntimeError as e:
            debug_logger.error(
                f"S3 download failed for interview {vid}: {e}", exc_info=True
            )
            raise VideoProcessingError(f"Failed to download interview video from S3: {e}", 5001) from e
        except Exception as e:
            debug_logger.error(
                f"Unexpected error processing S3 video for interview {vid}: {e}",
                exc_info=True,
            )
            raise VideoProcessingError(
                f"Failed to process interview video from S3: {e}", 5001
            ) from e
        finally:
            if local_video_path:
                s3_queue.delete_local_file(local_video_path)

    def _compute_video_report_sync(
        self, video_path: str, interview_id: str
    ) -> Dict:
        processor = VideoProcessingService(self.config)
        report    = processor.process_video(video_path, interview_id)
        processor.cleanup()
        return report

    async def process_video_file(
        self,
        video_path: str,
        interview_id: Optional[str] = None,
    ) -> Dict:
        interview_id = interview_id or str(uuid.uuid4())
        debug_logger.debug(
            "[proctoring] interview=%s | phase=video_analysis START",
            interview_id,
        )

        self.status_repository.ensure_status_row(interview_id)
        self.status_repository.db.commit()

        vpath = (video_path or "").strip()
        if vpath.startswith(("http://", "https://")):
            self.repository.ensure_placeholder_report(interview_id)

        try:
            _t_video = time.perf_counter()
            report = await asyncio.to_thread(
                self._compute_video_report_sync, video_path, interview_id
            )
            _video_sec = time.perf_counter() - _t_video
            debug_logger.debug(
                "[proctoring] interview=%s | phase=video_analysis COMPLETE "
                "(%.1fs)",
                interview_id,
                _video_sec,
            )

            gestures          = report.get("analysis", {}).get("gestures", [])
            total_occurrences = sum(
                len(g.get("occurrence", [])) for g in gestures
            )
            debug_logger.debug(
                "[proctoring] interview=%s | extracted %s gesture types, "
                "%s occurrences",
                interview_id,
                len(gestures),
                total_occurrences,
            )

            processing_metadata = (
                report.get("analysis", {}).get("processing_metadata", {})
            )
            video_duration = processing_metadata.get("video_duration_sec", 0.0)
            fps            = processing_metadata.get("fps", 12.0)

            self._save_complete_report(
                interview_id,
                report,
                video_duration,
                fps,
                source_video_url=video_path,
            )

        except DatabaseError:
            record_proctoring_failure(self.status_repository, interview_id)
            raise
        except VideoProcessingError:
            record_proctoring_failure(self.status_repository, interview_id)
            raise
        except Exception as e:
            debug_logger.error(
                f"Error processing video for interview {interview_id}: {str(e)}",
                exc_info=True,
            )
            record_proctoring_failure(self.status_repository, interview_id)
            raise VideoProcessingError(
                f"Video processing failed: {str(e)}", 5001
            )

        try:
            self.status_repository.mark_proctoring_generated(interview_id)
        except Exception as exc:
            debug_logger.error(
                f"mark_proctoring_generated failed for interview "
                f"{interview_id}: {exc}",
                exc_info=True,
            )
            raise DatabaseError(
                f"Failed to update proctoring status: {str(exc)}", 5002
            )

        debug_logger.debug(
            "[proctoring] interview=%s | phase=status UPDATE complete "
            "(mark_proctoring_generated)",
            interview_id,
        )
        debug_logger.debug(
            "[proctoring] interview=%s | pipeline COMPLETE",
            interview_id,
        )
        return {
            "interview_id": interview_id,
            "gestures":     gestures,
        }

    def persist_processed_report(
        self,
        interview_id: str,
        report: Dict,
        source_video_url: Optional[str] = None,
    ) -> None:
        processing_metadata = (
            report.get("analysis", {}).get("processing_metadata", {})
        )
        video_duration = float(
            processing_metadata.get("video_duration_sec", 0.0) or 0.0
        )
        fps = float(processing_metadata.get("fps", 12.0) or 12.0)
        self._save_complete_report(
            interview_id,
            report,
            video_duration,
            fps,
            source_video_url=source_video_url,
        )

    def _save_complete_report(
        self,
        interview_id: str,
        report: Dict,
        video_duration: float = 0.0,
        fps: float = 12.0,
        source_video_url: Optional[str] = None,
    ) -> None:
        try:
            from datetime import date

            debug_logger.debug(
                "[proctoring] interview=%s | phase=persist_db START "
                "(risk score + report + events)",
                interview_id,
            )
            _t_db = time.perf_counter()

            analysis = report.get("analysis", {})
            gestures = analysis.get("gestures", [])

            vd = float(video_duration or 0.0)
            if vd <= 0.0:
                vd = estimate_duration_from_gestures(gestures)

            risk_score, risk_level, score_metrics = (
                self._calculate_risk_from_events(gestures, vd)
            )
            if is_proctoring_diagnostics_enabled():
                debug_logger.debug(
                    "Cheating likeliness metrics: "
                    f"HRD={score_metrics.get('hrd', 0):.4f} "
                    f"ERD={score_metrics.get('erd', 0):.4f} "
                    f"RTR={score_metrics.get('rtr', 0):.4f} "
                    f"BF={score_metrics.get('bf', 0):.4f} "
                    f"raw={score_metrics.get('raw', 0):.4f} "
                    f"face_penalty={score_metrics.get('face_penalty', 0):.4f} "
                    f"score_0_10={score_metrics.get('score_0_10', 0):.2f} "
                    f"stored={risk_score:.2f} "
                    f"head_suspicious={score_metrics.get('head_suspicious', 0):.0f} "
                    f"head_high_risk={score_metrics.get('head_high_risk', 0):.0f} "
                    f"eye_suspicious={score_metrics.get('eye_suspicious', 0):.0f} "
                    f"eye_high_risk={score_metrics.get('eye_high_risk', 0):.0f} "
                    f"bursts={score_metrics.get('bursts', 0):.0f}"
                )

            self.repository.save_proctoring_report(
                interview_id=interview_id,
                interview_date=date.today(),
                cheating_likelihood_score=risk_score,
                cheating_likelihood_level=risk_level,
            )

            self._save_events_and_summaries(interview_id, gestures)
            _db_sec = time.perf_counter() - _t_db
            debug_logger.info(
                "[proctoring] interview=%s | processing complete | report generated | "
                "risk_level=%s score=%.2f/10 | persist_db %.1fs",
                interview_id,
                risk_level,
                risk_score,
                _db_sec,
            )

        except DatabaseError:
            raise
        except Exception as e:
            debug_logger.exception(
                "Failed to save complete report for interview %s", interview_id
            )
            raise DatabaseError(
                f"Failed to save complete report: {str(e)}", 5002
            )

    def _calculate_risk_from_events(
        self, gestures: List, video_duration_sec: float
    ) -> tuple:
        cfg = self.config
        return ScoringService.compute_cheating_likelihood(
            gestures,
            video_duration_sec,
            classify_event_risk=self._event_risk.classify,
            weight_hrd=cfg.WEIGHT_HRD,
            weight_erd=cfg.WEIGHT_ERD,
            weight_rtr=cfg.WEIGHT_RTR,
            weight_bf=cfg.WEIGHT_BF,
            burst_window_sec=cfg.BURST_WINDOW_SEC,
            burst_min_events=cfg.BURST_MIN_EVENTS,
            high_risk_event_multiplier=cfg.HIGH_RISK_EVENT_MULTIPLIER,
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _save_events_and_summaries(
        self, interview_id: str, gestures: List
    ) -> None:

        for gesture in gestures:
            gesture_name = gesture.get("name", "unknown")
            occurrences = gesture.get("occurrence", [])
            total_count = len(occurrences)

            if total_count == 0:
                continue

            saved_events = 0
            for occurrence in occurrences:
                timestamp = occurrence.get("timestamp")
                event_timestamp = to_video_timestamp(timestamp)
                dur_sec = (
                    parse_numeric_value(occurrence.get("duration")) or 0.0
                )

                if gesture_name == "face_missing":
                    event_risk = "suspicious"
                elif gesture_name == "multiple_faces":
                    event_risk = "high_risk"
                elif gesture_name == "face_occluded":
                    event_risk = "suspicious"
                else:
                    event_risk = self._event_risk.classify(
                        gesture_name=gesture_name,
                        direction=occurrence.get("direction", None),
                        duration_seconds=dur_sec,
                    )

                raw_velocity = occurrence.get("velocity")
                if isinstance(raw_velocity, str) and raw_velocity.strip():
                    velocity_value: Optional[str] = raw_velocity.strip()
                elif isinstance(raw_velocity, (int, float)):
                    velocity_value = float_to_velocity_label(
                        float(raw_velocity), self.config
                    )
                else:
                    velocity_value = None

                intensity_val = parse_numeric_value(
                    occurrence.get("intensity")
                )

                ok = self.repository.save_event_log(
                    interview_id=interview_id,
                    event_type=gesture_name,
                    event_timestamp=event_timestamp,
                    duration=dur_sec,
                    direction=occurrence.get("direction", None),
                    intensity=intensity_val,
                    confidence=parse_numeric_value(
                        occurrence.get("confidence")
                    ),
                    velocity=velocity_value,
                    event_risk=event_risk,
                )
                if ok:
                    saved_events += 1

            if saved_events != total_count:
                debug_logger.warning(
                    f"Mismatch saving event logs for gesture {gesture_name}: "
                    f"expected={total_count}, saved={saved_events}"
                )
            if is_proctoring_diagnostics_enabled():
                debug_logger.debug(
                    f"Saved {saved_events} event logs for gesture: {gesture_name}"
                )

            normal_count = 0
            suspicious_count = 0
            high_risk_count = 0

            for o in occurrences:
                dur = parse_numeric_value(o.get("duration")) or 0.0
                direction = o.get("direction", None)

                if gesture_name in (
                    "face_missing", "multiple_faces", "face_occluded"
                ):
                    if gesture_name == "multiple_faces":
                        high_risk_count += 1
                    else:
                        suspicious_count += 1
                    continue

                risk = self._event_risk.classify(
                    gesture_name=gesture_name,
                    direction=direction if isinstance(direction, str) else None,
                    duration_seconds=dur,
                )
                if risk in ("ignore", None):
                    continue
                if risk == "normal":
                    normal_count += 1
                elif risk == "suspicious":
                    suspicious_count += 1
                elif risk == "high_risk":
                    high_risk_count += 1

            counted_total = normal_count + suspicious_count + high_risk_count
            if (
                counted_total == 0
                and total_count > 0
                and gesture_name in ("head_movement", "eye_movement")
            ):
                normal_count = total_count
                counted_total = total_count

            total_duration = sum(
                parse_numeric_value(o.get("duration")) or 0.0
                for o in occurrences
            )

            summary_ok = self.repository.save_event_summary(
                interview_id=interview_id,
                event_type=gesture_name,
                total_count=counted_total,
                normal_count=normal_count,
                suspicious_count=suspicious_count,
                high_risk_count=high_risk_count,
                total_duration=total_duration,
            )
            if not summary_ok:
                debug_logger.error(
                    f"Failed to save event summary for gesture {gesture_name} "
                    f"(counted_total={counted_total}, "
                    f"total_duration={total_duration:.1f}s)"
                )
            else:
                if is_proctoring_diagnostics_enabled():
                    debug_logger.debug(
                        f"Saved event summary: {gesture_name} "
                        f"({counted_total} events, {total_duration:.1f}s)"
                    )

    # ------------------------------------------------------------------
    # Report retrieval
    # ------------------------------------------------------------------

    async def get_report(self, interview_id: str) -> Optional[Dict]:
        try:
            report = self.repository.get_report_by_interview_id(interview_id)
            if not report:
                debug_logger.warning(
                    f"Report not found for interview: {interview_id}"
                )
                raise ReportNotFoundError()
            return report
        except ReportNotFoundError:
            raise
        except Exception as e:
            debug_logger.error(
                f"Error retrieving report for interview {interview_id}: {str(e)}"
            )
            raise DatabaseError(
                f"Failed to retrieve report: {str(e)}", 5003
            )

    async def delete_report(self, interview_id: str) -> Dict:
        try:
            success = self.repository.delete_report(interview_id)
            if not success:
                debug_logger.warning(
                    f"Report not found for deletion: {interview_id}"
                )
                raise ReportNotFoundError()
            debug_logger.debug(
                f"Report deleted successfully: {interview_id}"
            )
            return {
                "status":  "success",
                "message": f"Report {interview_id} deleted successfully",
            }
        except ReportNotFoundError:
            raise
        except Exception as e:
            debug_logger.error(
                f"Error deleting report for interview {interview_id}: {str(e)}"
            )
            raise DatabaseError(
                f"Failed to delete report: {str(e)}", 5005
            )