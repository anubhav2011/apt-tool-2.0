"""
Persistence for ai_interview_status (scheduler eligibility and retries).
"""

from typing import List, Optional, Dict, Set

from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.config import SCHEDULER_CONFIG
from app.models.ai_interview_status import AiInterviewStatus

MAX_RETRIES = SCHEDULER_CONFIG.MAX_RETRIES


class AiInterviewStatusRepository:
    """Queries and updates scheduler state on ai_interview_status."""

    def __init__(self, db: Session):
        self.db = db

    def _get_or_create(self, interview_id: str) -> AiInterviewStatus:
        row = (
            self.db.query(AiInterviewStatus)
            .filter(AiInterviewStatus.interview_id == interview_id)
            .first()
        )
        if row is None:
            row = AiInterviewStatus(
                interview_id=interview_id,
                proctoring_generated=0,
                proctoring_retry_count=0,
            )
            self.db.add(row)
            self.db.flush()
        return row

    def ensure_status_row(self, interview_id: str) -> None:
        """Ensure a row exists (0,0) so retries and scheduler scans are consistent."""
        self._get_or_create(interview_id)

    def get_candidate_name_for_interview(self, interview_id: str) -> Optional[str]:
        """
        Resolve candidate_name for S3 key building (same source as scheduler joins).

        Returns None if the interview has no ai_interview_details row or empty name.
        """
        row = self.db.execute(
            text(
                "SELECT candidate_name FROM ai_interview_details "
                "WHERE interview_id = :interview_id LIMIT 1"
            ),
            {"interview_id": interview_id},
        ).mappings().first()
        if row is None or row["candidate_name"] is None:
            return None
        name = str(row["candidate_name"]).strip()
        return name or None

    def get_pending_for_proctoring(self, limit: Optional[int] = None) -> List[str]:
        """
        interview_id values where proctoring is still needed and retries remain.

        Condition: proctoring_generated = 0 AND proctoring_retry_count < MAX_RETRIES
        """
        q = (
            self.db.query(AiInterviewStatus.interview_id)
            .filter(
                AiInterviewStatus.proctoring_generated == 0,
                AiInterviewStatus.proctoring_retry_count < MAX_RETRIES,
            )
        )
        if limit is not None:
            q = q.limit(limit)
        return [row[0] for row in q.all()]

    def get_pending_with_candidate_name(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Pending rows enriched with candidate_name from ai_interview_details.

        Eligibility condition remains:
        proctoring_generated = 0 AND proctoring_retry_count < MAX_RETRIES
        """
        min_created_date = SCHEDULER_CONFIG.MIN_INTERVIEW_CREATED_DATE
        sql = """
            SELECT
                s.interview_id AS interview_id,
                d.candidate_name AS candidate_name,
                rt.status AS recording_status
            FROM ai_interview_status s
            INNER JOIN ai_interview_details d
                ON d.interview_id = s.interview_id
            INNER JOIN company_config cc
                ON cc.company_id = d.company_id
            INNER JOIN ai_recording_tracking rt
                ON rt.interview_schedule_id = s.interview_id
            WHERE s.status = 'COMPLETED'
              AND (cc.is_proctoring_enabled = 1 OR cc.is_proctoring_enabled = TRUE)
              AND s.proctoring_generated = 0
              AND s.proctoring_retry_count < :max_retries
              AND rt.status = 'Done'
              AND d.candidate_name IS NOT NULL
              AND TRIM(d.candidate_name) <> ''
        """
        if min_created_date is not None:
            # Compare calendar dates. `created` is expected to be DATETIME/TIMESTAMP/DATE.
            sql += """
              AND s.created IS NOT NULL
              AND DATE(s.created) >= :min_created_date
            """

        # Stable order; prioritize oldest eligible interviews after the cutoff.
        sql += " ORDER BY s.created ASC, s.interview_id ASC"

        if limit is not None:
            sql += " LIMIT :limit"
            params = {"max_retries": MAX_RETRIES, "limit": int(limit)}
            if min_created_date is not None:
                params["min_created_date"] = min_created_date
            rows = self.db.execute(text(sql), params).mappings().all()
        else:
            params = {"max_retries": MAX_RETRIES}
            if min_created_date is not None:
                params["min_created_date"] = min_created_date
            rows = self.db.execute(text(sql), params).mappings().all()

        return [
            {
                "interview_id": str(r["interview_id"]),
                "candidate_name": str(r["candidate_name"]),
            }
            for r in rows
        ]

    def mark_proctoring_generated(self, interview_id: str) -> None:
        """Set proctoring_generated = 1 for a successful run and clear retry counter."""
        row = self._get_or_create(interview_id)
        row.proctoring_generated = 1
        # Reset failures so a success after prior scheduler retries does not leave count at MAX_RETRIES.
        row.proctoring_retry_count = 0
        self.db.commit()

    def increment_proctoring_retry(self, interview_id: str) -> int:
        """
        Increment proctoring_retry_count after a failed run.

        Returns:
            New retry count (always >= 1 after increment).
        """
        row = self._get_or_create(interview_id)
        row.proctoring_retry_count = int(row.proctoring_retry_count or 0) + 1
        self.db.commit()
        return row.proctoring_retry_count

    def get_generated_interview_ids(self, interview_ids: List[str]) -> Set[str]:
        """
        Return subset of interview_ids where proctoring_generated = 1.

        Used to safely cleanup old local queue artifacts without touching
        in-flight or pending jobs.
        """
        if not interview_ids:
            return set()

        rows = (
            self.db.query(AiInterviewStatus.interview_id)
            .filter(
                AiInterviewStatus.interview_id.in_(interview_ids),
                AiInterviewStatus.proctoring_generated == 1,
            )
            .all()
        )
        return {str(r[0]) for r in rows}
