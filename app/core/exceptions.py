"""
Custom exceptions for the AI Proctoring application.
"""

class AppError(Exception):
    """Base exception for proctoring-related errors"""

    def __init__(self, message: str, code: int):
        self.message = message
        self.code = code
        super().__init__(self.message)

class ConfigurationError(AppError):
    """Exception for configuration errors"""
    pass

class ValidationError(AppError):
    """Exception for input validation errors"""
    pass

class VideoProcessingError(AppError):
    """Exception for video processing errors"""
    pass

class DatabaseError(AppError):
    """Exception for database operation errors"""
    pass

class VideoDownloadError(AppError):
    """Exception for video download errors"""
    pass

class ReportNotFoundError(AppError):
    """Exception when report is not found"""
    def __init__(self):
        super().__init__("Report not found", 404)
