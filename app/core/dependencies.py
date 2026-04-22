"""
Dependency Injection
Clean dependency management following FastAPI best practices
"""
import secrets

from app.utils.logger import debug_logger
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy.orm import Session

from app.core.config import ProctoringConfig, BASIC_AUTH_CONFIG
from app.core.database import get_db_session
from app.repositories.base_repository import IRepository
from app.repositories.proctoring_repository import ProctoringRepository
from app.repositories.ai_interview_status_repository import AiInterviewStatusRepository
from app.services.base_service import IProctoringService
from app.services.proctoring_service import ProctoringService
from app.models.proctoring import ProctoringReport

logger = debug_logger
security = HTTPBasic()


def verify_user(credentials: HTTPBasicCredentials = Depends(security)) -> str:
    """Verify HTTP Basic credentials for protected routes."""
    is_correct_username = secrets.compare_digest(
        credentials.username,
        BASIC_AUTH_CONFIG.USERNAME,
    )
    is_correct_password = secrets.compare_digest(
        credentials.password,
        BASIC_AUTH_CONFIG.PASSWORD,
    )

    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )

    return credentials.username

def get_config() -> ProctoringConfig:
    """
    Get configuration instance
    Returns singleton configuration instance

    Returns:
        ProctoringConfig instance
    """
    return ProctoringConfig()


def get_repository(db: Session = Depends(get_db_session)) -> IRepository[ProctoringReport]:
    """
    Get repository instance with injected database session
    Returns base interface type for better abstraction and testability
    The concrete implementation (ProctoringRepository) extends BaseRepository

    Args:
        db: Database session from FastAPI dependency

    Returns:
        Repository instance implementing IRepository interface
    """
    # BaseRepository is called through inheritance in ProctoringRepository
    # ProctoringRepository(db) -> calls BaseRepository.__init__(ProctoringReport, db)
    return ProctoringRepository(db=db)


def get_proctoring_service(
    db: Session = Depends(get_db_session),
) -> IProctoringService:
    """
    Get proctoring service instance with proper dependency injection
    All dependencies are injected via interfaces for loose coupling

    Args:
        db: Single DB session shared by proctoring and ai_interview_status repositories

    Returns:
        Proctoring service instance implementing IProctoringService
    """
    repository = ProctoringRepository(db=db)
    status_repository = AiInterviewStatusRepository(db)
    return ProctoringService(repository=repository, status_repository=status_repository)
