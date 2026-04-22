"""
Base Repository
Generic base repository interface with essential abstract methods
"""
from app.utils.logger import debug_logger
from abc import ABC
from typing import Generic, TypeVar, Type
from sqlalchemy.orm import Session

from app.core.database import Base

# Generic type for SQLAlchemy models
ModelType = TypeVar("ModelType", bound=Base)


class IRepository(ABC, Generic[ModelType]):
    """
    Interface for repository operations (optional shared method stubs).
    """

    def get_report_by_interview_id(self, interview_id):
        pass

    def save_report(self, session_id, candidate_id, report_with_metadata, video_duration, fps) -> None:
        pass

    def delete_report(self, interview_id):
        pass


class BaseRepository(IRepository[ModelType]):
    """
    Generic base repository class with common initialization.
    """

    def __init__(self, model: Type[ModelType], db: Session):
        """
        Initialize repository with model and database session

        Args:
            model: SQLAlchemy model class
            db: Database session
        """
        self.model = model
        self.db = db
        debug_logger.debug(f"{self.__class__.__name__} initialized with model {model.__name__}")
