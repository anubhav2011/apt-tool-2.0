
"""
Pydantic models for AI Proctoring API
Aligned with PDF Specification v1.0 + Industry Standard Enhancements
"""
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from enum import Enum


class CalibrationStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    UNSTABLE = "UNSTABLE"


class BaselineType(str, Enum):
    PERSONALIZED = "PERSONALIZED"
    GLOBAL_DEFAULT = "GLOBAL_DEFAULT"


class AlertType(str, Enum):
    GAZE_AWAY = "gaze_away"
    HEAD_MOVEMENT = "head_movement"
    FACE_MISSING = "face_missing"
    MULTIPLE_FACES = "multiple_faces"


class AlertSeverity(str, Enum):
    MINOR = "MINOR"
    MAJOR = "MAJOR"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK = "HIGH_RISK"


class RiskLevel(str, Enum):
    """
    Risk Level Classifications (PDF Specification):
    - CLEAN: 0-30 (No action required)
    - MODERATE: 31-60 (Monitor for patterns)
    - SUSPICIOUS: 61-80 (Review footage recommended)
    - HIGH_RISK: 81-100 (Manual review required)
    """
    CLEAN = "CLEAN"
    MODERATE = "MODERATE"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK = "HIGH_RISK"


class Baseline(BaseModel):
    eye_mean: float = Field(..., description="Mean eye gaze angle during calibration")
    eye_variance: float = Field(..., description="Variance in eye movements")
    yaw_mean: float = Field(..., description="Mean head yaw during calibration")
    pitch_mean: float = Field(..., description="Mean head pitch during calibration")
    roll_mean: float = Field(..., description="Mean head roll during calibration")
    yaw_variance: float = Field(0.0, description="Variance in yaw movements")
    pitch_variance: float = Field(0.0, description="Variance in pitch movements")
    face_presence_ratio: float = Field(..., description="Percentage of frames with face detected")
    bbox_stability: float = Field(0.0, description="Bounding box stability score")


class AdaptiveThresholds(BaseModel):
    eye: float = Field(..., description="Adaptive threshold for eye gaze angle")
    yaw: float = Field(..., description="Adaptive threshold for head yaw")
    pitch: float = Field(..., description="Adaptive threshold for head pitch")
    roll: float = Field(..., description="Adaptive threshold for head roll")


class CalibrationResult(BaseModel):
    status: str = Field(..., description="SUCCESS or FAILED")
    yaw_threshold: float = Field(..., description="Computed yaw threshold")
    pitch_threshold: float = Field(..., description="Computed pitch threshold")
    eye_threshold: float = Field(..., description="Computed eye threshold")


class Alert(BaseModel):
    type: AlertType
    total_occurrences: int = Field(..., description="Total count of violations")
    severity: AlertSeverity
    timestamps: List[float] = Field(
        default_factory=list,
        description="Flat array of timestamps when violations occurred"
    )


class ViolationCounts(BaseModel):
    gaze_left: int = Field(0, description="Count of left gaze violations")
    gaze_right: int = Field(0, description="Count of right gaze violations")
    head_left: int = Field(0, description="Count of left head turns")
    head_right: int = Field(0, description="Count of right head turns")
    head_up: int = Field(0, description="Count of upward head tilts")
    head_down: int = Field(0, description="Count of downward head tilts")
    face_missing: int = Field(0, description="Count of frames with no face detected")
    multiple_faces: int = Field(0, description="Count of frames with multiple faces")


class DurationByRisk(BaseModel):
    """Duration in seconds by risk category"""
    mild: float = Field(0.0, description="Duration of mild violations in seconds")
    suspicious: float = Field(0.0, description="Duration of suspicious violations in seconds")
    high_risk: float = Field(0.0, description="Duration of high risk violations in seconds")


class IntensityMetrics(BaseModel):
    """Average angle deviations"""
    avg_head_angle: float = Field(0.0, description="Average head angle deviation")
    avg_eye_angle: float = Field(0.0, description="Average eye angle deviation")


class ConfidenceScores(BaseModel):
    eye_risk: float = Field(..., ge=0.0, le=1.0)
    head_risk: float = Field(..., ge=0.0, le=1.0)
    face_risk: float = Field(..., ge=0.0, le=1.0)
    final_suspicion_score: float = Field(..., ge=0.0, le=1.0)


class ProcessingMetadata(BaseModel):
    fps: float
    total_frames: int
    processing_time_sec: float
    video_duration_sec: float
    frames_processed: int


class ProctoringReport(BaseModel):
    """
    Proctoring Report matching PDF Specification
    Output: Risk Score (0-100) with detailed counts and categories
    """
    session_id: str
    calibration: CalibrationResult = Field(..., description="Calibration result with thresholds")
    counts: ViolationCounts = Field(..., description="Event occurrence counts")
    duration_sec: DurationByRisk = Field(..., description="Duration by risk category")
    intensity: IntensityMetrics = Field(..., description="Average angle deviations")
    alerts: List[Alert] = Field(..., description="Timeline of suspicious events")
    risk_score: int = Field(..., ge=0, le=100, description="Final risk score (0-100)")
    risk_level: RiskLevel = Field(..., description="Risk classification")


class ProcessVideoRequest(BaseModel):
    """Request to process the interview recording from S3."""
    interview_id: str = Field(..., description="Interview ID")

    class Config:
        json_schema_extra = {
            "example": {
                "interview_id": "00000"
            }
        }


class ProcessVideoResponse(BaseModel):
    session_id: str
    status: str
    message: str
    report: Optional[ProctoringReport] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, str]


class GestureOccurrence(BaseModel):
    """Single occurrence of a gesture event with industry-standard enhancements"""
    timestamp: str = Field(..., description="Timestamp in M:SS format")
    duration: float = Field(..., description="Duration in seconds")
    direction: str = Field("", description="Direction of movement (left/right/up/down) or empty")
    intensity: str = Field("", description="Angle intensity as string or empty")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Detection confidence (0.0-1.0)")
    velocity: str = Field("", description="Movement velocity (slow/moderate/fast/rapid) or empty")


class GestureData(BaseModel):
    """Gesture group with all occurrences"""
    name: str = Field(..., description="Gesture name: head_movement, eye_gaze, face_missing, multiple_faces, face_occluded")
    occurrence: List[GestureOccurrence] = Field(default_factory=list, description="List of gesture occurrences")


class ProcessingMetadataNew(BaseModel):
    """Processing statistics"""
    processing_time_sec: float
    video_duration_sec: float
    frames_processed: int


class ThresholdsUsed(BaseModel):
    """Thresholds used for detection"""
    eye: float = Field(..., description="Eye gaze threshold in degrees")
    yaw: float = Field(..., description="Head yaw threshold in degrees")
    pitch: float = Field(..., description="Head pitch threshold in degrees")


class VideoAnalysisResult(BaseModel):
    """
    Enhanced format response with industry-standard features
    Includes confidence scoring, velocity analysis, and comprehensive face detection
    """
    session_id: str = Field(..., description="Session identifier")
    thresholds_used: ThresholdsUsed = Field(..., description="Thresholds applied")
    processing_metadata: ProcessingMetadataNew = Field(..., description="Processing statistics")
    gestures: List[GestureData] = Field(default_factory=list, description="Detected gestures grouped by type")


class ProcessVideoResponseNew(BaseModel):
    """API response for video processing"""
    session_id: str
    status: str
    message: str
    result: VideoAnalysisResult
