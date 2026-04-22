"""
Scheduler state for proctoring video processing.

Rows are keyed by interview_id (same id as proctoring_reports.interview_id).
The scheduler scans: proctoring_generated = 0 AND proctoring_retry_count < MAX_RETRIES.
"""

from sqlalchemy import Column, String, Integer

from app.core.database import Base


class AiInterviewStatus(Base):
    """
    One row per interview; tracks whether proctoring output was generated
    and how many processing attempts failed.
    """

    __tablename__ = "ai_interview_status"

    interview_id = Column(String(36), primary_key=True)

    # 0 = not yet successfully processed; 1 = done (scheduler skips).
    proctoring_generated = Column(Integer, nullable=False, default=0)

    # Incremented on each failed VideoProcessingService run (capped by MAX_RETRIES in repo).
    proctoring_retry_count = Column(Integer, nullable=False, default=0)

    __table_args__ = ({"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4"},)
