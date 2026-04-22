"""
Background tasks for proctoring API (run after 202 response; own DB session per task).
"""

from app.core.database import get_db
from app.services.proctoring_service import ProctoringService
from app.repositories.proctoring_repository import ProctoringRepository
from app.repositories.ai_interview_status_repository import AiInterviewStatusRepository
from app.utils.logger import debug_logger


async def run_process_video_by_interview_id(interview_id: str) -> None:
    """Background task for processing video"""
    try:
        with get_db() as db:
            service = ProctoringService(
                repository=ProctoringRepository(db=db),
                status_repository=AiInterviewStatusRepository(db),
            )
            await service.process_video_by_interview_id(interview_id)

    except Exception as exc:
        debug_logger.error(
            f"[BG] process_video_by_interview_id failed interview_id={interview_id}: {exc}",
            exc_info=True,
        )

