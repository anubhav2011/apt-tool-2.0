"""
Base service interfaces
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict


class IProctoringService(ABC):
    """Interface for proctoring service"""

    @abstractmethod
    async def process_video_upload(self, video_file, interview_id: Optional[str] = None) -> Dict:
        """Process uploaded video file"""
        raise NotImplementedError

    @abstractmethod
    async def process_video_by_interview_id(self, interview_id: str) -> Dict:
        """Download interview video from S3 and process (same object layout as scheduler)."""
        raise NotImplementedError

    @abstractmethod
    async def process_video_file(self, video_path: str, interview_id: Optional[str] = None) -> Dict:
        """Process video file and return report"""
        raise NotImplementedError

    @abstractmethod
    async def get_report(self, interview_id: str) -> Optional[Dict]:
        """Get report by interview ID"""
        raise NotImplementedError

    @abstractmethod
    async def delete_report(self, interview_id: str) -> Dict:
        """Delete a report"""
        raise NotImplementedError
