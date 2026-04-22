"""
Base service for all proctoring processing services
Provides common utilities and logging
"""
from abc import ABC, abstractmethod
from typing import Dict
from app.utils.logger import debug_logger


class BaseService(ABC):
    """
    Base class for all processing services
    Provides configuration and logging utilities
    """

    def __init__(self, config):
        """
        Initialize base service

        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = debug_logger
        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        """Setup method to be implemented by subclasses"""
        pass

    def log_info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)

    def log_warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)

    def log_error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)

    def log_debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup method to be implemented by subclasses"""
        pass


class IVideoProcessingService(ABC):
    """Interface for video processing service"""

    @abstractmethod
    def process_video(self, video_path: str, session_id: str) -> Dict:
        """Process video and return report"""
        raise NotImplementedError

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources"""
        raise NotImplementedError
