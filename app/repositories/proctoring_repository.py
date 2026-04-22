"""
Proctoring Repository
All database operations for proctoring reports using SQLAlchemy ORM.

Scheduler eligibility / retries live on ai_interview_status (see
AiInterviewStatusRepository).
"""
from app.utils.logger import debug_logger
from typing import Optional, Dict
import uuid
from datetime import date
from sqlalchemy import func
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.utils.data_formatting import normalize_interview_id
from app.models.proctoring import (
    ProctoringReport,
    ProctoringEventLog,
    ProctoringEventSummary,
)
from app.repositories.base_repository import BaseRepository
from app.repositories.ai_interview_status_repository import AiInterviewStatusRepository
from app.services.proctoring_processing.scoring_service import ScoringService


class ProctoringRepository(BaseRepository[ProctoringReport]):
    """
    Repository for proctoring data persistence.
    """

    def __init__(self, db: Session):
        super().__init__(ProctoringReport, db)
        debug_logger.debug("ProctoringRepository initialized")

    def save_report(
            self,
            session_id: str,
            candidate_id: Optional[str],
            report: Dict,
            video_duration: float = 0.0,
            fps: float = 12.0,
    ) -> None:
        try:
            # Extract and scale suspicion score (0–1 → 0–10), clamp + 2 dp
            final_score = float(
                report.get("confidence_scores", {}).get("final_suspicion_score", 0)
            )
            cheating_likelihood_score = max(
                0.0, min(10.0, round(final_score * 10.0, 2))
            )

            # Extract level
            cheating_likelihood_level = report.get("final_decision", "UNKNOWN")

            # Save report
            ok = self.save_proctoring_report(
                interview_id=session_id,
                interview_date=date.today(),
                cheating_likelihood_score=cheating_likelihood_score,
                cheating_likelihood_level=cheating_likelihood_level,
            )
            if not ok:
                # Keep callers from logging success when the write silently failed.
                raise RuntimeError(
                    f"save_proctoring_report returned False for session {session_id}"
                )

        except Exception as e:
            debug_logger.error(
                f"Failed to save report for session {session_id}: {e}"
            )
            raise

    def delete_report(self, interview_id: str) -> bool:
        try:
            iid = normalize_interview_id(interview_id)
            rows_deleted = (
                self.db.query(ProctoringReport)
                .filter(ProctoringReport.interview_id == iid)
                .delete()
            )
            self.db.commit()
            if rows_deleted > 0:
                debug_logger.info(
                    f"Deleted {rows_deleted} report(s) for interview: {interview_id}"
                )
                return True
            debug_logger.warning(f"No report found to delete for interview: {interview_id}")
            return False
        except SQLAlchemyError as e:
            self.db.rollback()
            debug_logger.error(
                f"Database error deleting report for interview {interview_id}: {e}"
            )
            return False
    def save_event_log(
        self,
        interview_id: str,
        event_type: str,
        event_timestamp: str,
        duration: Optional[float] = None,
        direction: Optional[str] = None,
        intensity: Optional[float] = None,
        confidence: Optional[float] = None,
        velocity: Optional[str] = None,
        event_risk: Optional[str] = None
    ) -> bool:
        """
        Upsert event log:
        - If (interview_id + event_type + event_timestamp) exists → UPDATE
        - Else → INSERT
        """
        try:
            # Normalize interview_id
            iid = normalize_interview_id(interview_id)

            # Check existing record
            existing = (
                self.db.query(ProctoringEventLog)
                .filter(
                    ProctoringEventLog.interview_id == iid,
                    ProctoringEventLog.event_type == event_type,
                    ProctoringEventLog.event_timestamp == event_timestamp,
                )
                .first()
            )

            if existing:
                existing.duration = duration
                existing.direction = direction
                existing.intensity = intensity
                existing.confidence = confidence
                existing.velocity = velocity
                existing.event_risk = event_risk
                existing.updated = func.now()

            else:
                event_log = ProctoringEventLog(
                    interview_id=iid,
                    event_type=event_type,
                    event_timestamp=event_timestamp,
                    duration=duration,
                    direction=direction,
                    intensity=intensity,
                    confidence=confidence,
                    velocity=velocity,
                    event_risk=event_risk,
                )
                try:
                    event_log.id = None
                except Exception:
                    pass

                self.db.add(event_log)

            # Commit transaction
            self.db.commit()
            return True

        except SQLAlchemyError as e:
            self.db.rollback()
            debug_logger.error(f"Failed to save event log: {e}")
            return False

    def save_event_summary(
        self,
        interview_id: str,
        event_type: str,
        total_count: int,
        normal_count: int,
        suspicious_count: int,
        high_risk_count: int,
        total_duration: float
    ) -> bool:
        """
        Upsert event summary:
        - If (interview_id + event_type) exists → UPDATE
        - Else → INSERT
        """
        try:
            # Normalize interview_id
            iid = normalize_interview_id(interview_id)

            # Check existing record
            existing = (
                self.db.query(ProctoringEventSummary)
                .filter(
                    ProctoringEventSummary.interview_id == iid,
                    ProctoringEventSummary.event_type == event_type,
                )
                .first()
            )

            if existing:
                existing.total_count = total_count
                existing.normal_count = normal_count
                existing.suspicious_count = suspicious_count
                existing.high_risk_count = high_risk_count
                existing.total_duration = total_duration
                existing.updated = func.now()

            else:
                summary = ProctoringEventSummary(
                    interview_id=iid,
                    event_type=event_type,
                    total_count=total_count,
                    normal_count=normal_count,
                    suspicious_count=suspicious_count,
                    high_risk_count=high_risk_count,
                    total_duration=total_duration,
                )

                try:
                    summary.id = None
                except Exception:
                    pass

                self.db.add(summary)

            self.db.commit()
            return True

        except SQLAlchemyError as e:
            self.db.rollback()
            debug_logger.error(f"Failed to save event summary: {e}")
            return False
  
    def ensure_placeholder_report(self, interview_id: str) -> bool:
        """
        Ensure a placeholder `proctoring_reports` row exists for an interview.
        Used by the scheduler so later proctoring event inserts have a parent row.
        """
        try:
            iid = normalize_interview_id(interview_id)
            existing = (
                self.db.query(ProctoringReport)
                .filter(ProctoringReport.interview_id == iid)
                .first()
            )
            if existing:
                return True

            report = ProctoringReport(
                interview_id=iid,
                interview_date=date.today(),
                cheating_likelihood_score=0.0,
                cheating_likelihood_level="PENDING",
            )
            self.db.add(report)
            status_repo = AiInterviewStatusRepository(self.db)
            status_repo.ensure_status_row(interview_id)
            self.db.commit()
            debug_logger.debug(
                f"ensure_placeholder_report: created for interview {interview_id}"
            )
            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            debug_logger.error(f"ensure_placeholder_report failed for {interview_id}: {e}")
            return False

    def save_proctoring_report(self, interview_id: str, interview_date: date,
                               cheating_likelihood_score: float,
                               cheating_likelihood_level: str) -> bool:
        """
        Upsert a proctoring report row.

        cheating_likelihood_score is clamped to 0.0–10.0 and rounded to 2 decimals.
        For finalized reports, cheating_likelihood_level follows the score
        (LESS / MODERATE / HIGH). PENDING is kept until processing completes.
        """
        try:
            iid = normalize_interview_id(interview_id)
            try:
                clamped = max(
                    0.0,
                    min(10.0, round(float(cheating_likelihood_score), 2)),
                )
            except (TypeError, ValueError):
                clamped = 0.0
            if (
                cheating_likelihood_level
                and cheating_likelihood_level.strip().upper() == "PENDING"
            ):
                level_out = cheating_likelihood_level
            else:
                level_out = ScoringService.cheating_likelihood_level_from_stored_score(
                    clamped
                )

            existing = (
                self.db.query(ProctoringReport)
                .filter(ProctoringReport.interview_id == iid)
                .first()
            )
            status_repo = AiInterviewStatusRepository(self.db)
            if existing:
                existing.interview_date            = interview_date
                existing.cheating_likelihood_score = clamped
                existing.cheating_likelihood_level = level_out
                existing.updated                   = func.now()
                status_repo.ensure_status_row(interview_id)
                self.db.commit()
            else:
                kwargs = dict(
                    interview_id=iid,
                    interview_date=interview_date,
                    cheating_likelihood_score=clamped,
                    cheating_likelihood_level=level_out,
                )
                report = ProctoringReport(**kwargs)
                self.db.add(report)
                status_repo.ensure_status_row(interview_id)
                self.db.commit()
            debug_logger.debug(
                "Saved ProctoringReport: %s (cheating_likelihood_score=%s/10) "
                "for interview %s",
                level_out,
                clamped,
                iid,
            )
            return True
        except SQLAlchemyError as e:
            self.db.rollback()
            debug_logger.error(f"Failed to save proctoring report: {e}")
            return False

    def get_report_by_interview_id(self, interview_id: str) -> Optional[Dict]:
        try:
            iid = normalize_interview_id(interview_id)
            report = (
                self.db.query(ProctoringReport)
                .filter(ProctoringReport.interview_id == iid)
                .order_by(ProctoringReport.created.desc())
                .first()
            )
            return report.to_dict() if report else None
        except SQLAlchemyError as e:
            debug_logger.error(
                f"Failed to get report for interview {interview_id}: {e}"
            )
            return None
