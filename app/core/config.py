"""
Configuration settings for the AI Proctoring application.
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
from datetime import date
from dotenv import load_dotenv
import os

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass
class CORSConfig:
    ALLOW_ORIGINS: list = None
    ALLOW_CREDENTIALS: bool = True
    ALLOW_METHODS: list = None
    ALLOW_HEADERS: list = None

    def __post_init__(self):
        if self.ALLOW_ORIGINS is None:
            self.ALLOW_ORIGINS = ["*"]
        if self.ALLOW_METHODS is None:
            self.ALLOW_METHODS = ["*"]
        if self.ALLOW_HEADERS is None:
            self.ALLOW_HEADERS = ["*"]


@dataclass
class RateLimitConfig:
    """Rate Limiting Configuration settings"""
    REQUESTS_PER_MINUTE: int = 62
    WINDOW_SIZE_SECONDS: int = 60
    ENABLED: bool = True


@dataclass
class PathConfig:
    """Path Configuration settings (LOG_DIR follows LOG_FILE; see app.utils.logger)."""
    ROOT_DIR: str = ROOT_DIR

    @property
    def LOG_DIR(self) -> str:
        lf = (os.getenv("LOG_FILE") or "").strip()
        if lf:
            p = Path(lf)
            if not p.is_absolute():
                p = Path(str(self.ROOT_DIR)) / p
            return str(p.resolve().parent)
        return str((Path(str(self.ROOT_DIR)) / "debug_logs").resolve())

    @property
    def debug_logs_dir(self) -> str:
        return self.LOG_DIR

    @property
    def debug_logs_file(self) -> str:
        return str(Path(self.LOG_DIR) / "debug_logs_{}.log")


class ProctoringDatabaseConfig:
    """
    Database configuration for the AI Proctoring System.
    """

    @property
    def HOST(self) -> str:
        return os.getenv("MYSQL_HOST", "localhost")

    @property
    def PORT(self) -> int:
        return int(os.getenv("MYSQL_PORT", "3306") or "3306")

    @property
    def USER(self) -> str:
        return os.getenv("MYSQL_USER", "root")

    @property
    def PASSWORD(self) -> str:
        return os.getenv("MYSQL_PASSWORD", "").strip()

    @property
    def DATABASE(self) -> str:
        return os.getenv("MYSQL_DATABASE", "proctoring")

    @property
    def connection_url(self) -> str:
        return (
            f"mysql+pymysql://{self.USER}:{self.PASSWORD}"
            f"@{self.HOST}:{self.PORT}/{self.DATABASE}"
        )

    def as_dict(self) -> dict:
        return {
            "host":     self.HOST,
            "port":     self.PORT,
            "user":     self.USER,
            "password": self.PASSWORD,
            "database": self.DATABASE,
        }


class S3Config:
    """S3 configuration for video evidence storage."""

    @property
    def BUCKET_NAME(self) -> str:
        return (
            os.getenv("S3_BUCKET_NAME")
            or os.getenv("AWS_STORAGE_BUCKET_NAME")
            or ""
        ).strip()

    @property
    def REGION_NAME(self) -> str:
        return (os.getenv("AWS_REGION") or "").strip()

    @property
    def ACCESS_KEY_ID(self) -> str:
        return (os.getenv("AWS_ACCESS_KEY_ID") or "").strip()

    @property
    def SECRET_ACCESS_KEY(self) -> str:
        return (os.getenv("AWS_SECRET_ACCESS_KEY") or "").strip()

    @property
    def SESSION_TOKEN(self) -> str:
        return (os.getenv("AWS_SESSION_TOKEN") or "").strip()

    @property
    def PREFIX_ROOT(self) -> str:
        default_prefix = "data/recording-assets/ai/beam"
        return (
            os.getenv("S3_PREFIX_ROOT")
            or os.getenv("AWS_STORAGE_PATH")
            or default_prefix
        ).strip().strip("/")


@dataclass
class MediaPipeDetectionConfig:
    """MediaPipe-specific face detection and gaze estimation thresholds."""

    # ── Detection Confidence ──────────────────────────────────────────────
    MIN_DETECTION_CONFIDENCE: float      = 0.70
    MIN_TRACKING_CONFIDENCE: float       = 0.70
    MAX_FACES: int                       = 2

    # ── Gaze & Head Pose (degrees) ────────────────────────────────────────
    GAZE_HORIZONTAL_THRESHOLD: float     = 8.0
    GAZE_VERTICAL_THRESHOLD: float       = 8.0
    HEAD_YAW_THRESHOLD: float            = 28.0
    HEAD_PITCH_THRESHOLD: float          = 28.0
    HEAD_ROLL_THRESHOLD: float           = 18.0

    # ── Eye Aspect Ratio (EAR) ────────────────────────────────────────────
    EYE_ASPECT_RATIO_THRESHOLD: float    = 0.18
    MIN_IRIS_VISIBILITY: float           = 0.45
    BLINK_EAR_THRESHOLD: float           = 0.14

    # ── Confidence Multipliers ────────────────────────────────────────────
    GAZE_CONFIDENCE_MULTIPLIER: float    = 1.0
    POSE_CONFIDENCE_MULTIPLIER: float    = 1.0

    # ── Frame Quality ─────────────────────────────────────────────────────
    MIN_FACE_PRESENCE_SCORE: float       = 0.90
    MIN_CONFIDENCE_THRESHOLD: float      = 0.45
    EYE_MIN_CONFIDENCE_THRESHOLD: float  = 0.35


@dataclass
class IntelDetectionConfig:
    """Intel OpenVINO-specific face detection and gaze estimation thresholds."""

    # ── Detection Confidence ──────────────────────────────────────────────
    MIN_DETECTION_CONFIDENCE: float      = 0.65
    MIN_TRACKING_CONFIDENCE: float       = 0.60
    MAX_FACES: int                       = 2

    # ── Gaze & Head Pose (degrees) ────────────────────────────────────────
    GAZE_HORIZONTAL_THRESHOLD: float     = 8.5
    GAZE_VERTICAL_THRESHOLD: float       = 8.5
    HEAD_YAW_THRESHOLD: float            = 28.0
    HEAD_PITCH_THRESHOLD: float          = 28.0
    HEAD_ROLL_THRESHOLD: float           = 18.0

    # ── Eye Aspect Ratio (EAR) ────────────────────────────────────────────
    EYE_ASPECT_RATIO_THRESHOLD: float    = 0.16
    MIN_IRIS_VISIBILITY: float           = 0.40
    BLINK_EAR_THRESHOLD: float           = 0.12

    # ── Confidence Multipliers ────────────────────────────────────────────
    GAZE_CONFIDENCE_MULTIPLIER: float    = 1.0
    POSE_CONFIDENCE_MULTIPLIER: float    = 1.0

    # ── Frame Quality ─────────────────────────────────────────────────────
    MIN_FACE_PRESENCE_SCORE: float       = 0.85
    MIN_CONFIDENCE_THRESHOLD: float      = 0.40
    EYE_MIN_CONFIDENCE_THRESHOLD: float  = 0.30


@dataclass
class ProctoringConfig:

    # ── Video Processing ──────────────────────────────────────────────────
    MAX_FRAME_DIMENSION: int   = 1280
    TARGET_FPS: int            = 18
    WARMUP_SECONDS: float      = 3.0

    # ── Detection Client Selection ─────────────────────────────────────────
    DETECTION_CLIENT: str      = os.getenv("DETECTION_CLIENT", "mediapipe").strip().lower()

    # ── Detection Config (auto-selected based on DETECTION_CLIENT) ────────
    _detection_config: Optional[object] = None

    # ── Shared Temporal/Violation Thresholds (model-independent) ──────────
    BLINK_MIN_DURATION: float         = 0.05
    BLINK_MAX_DURATION: float         = 0.35
    NORMAL_BLINK_RATE_MIN: int        = 5
    NORMAL_BLINK_RATE_MAX: int        = 30
    ABNORMAL_BLINK_WINDOW: float      = 60.0

    # ── Temporal Consistency ──────────────────────────────────────────────
    MIN_EVENT_DURATION: float  = 1.0
    EYE_MIN_EVENT_DURATION: float  = 0.8
    EVENT_GAP_TOLERANCE: float = 0.55

    # ── Frame Rejection (model-independent) ───────────────────────────────
    MAX_HEAD_ROTATION_SPEED: float  = 10.0
    MAX_BBOX_CENTER_SHIFT: float    = 0.05
    MAX_EYE_ANGLE_VARIANCE: float   = 0.35

    # ── Movement Velocity (Cheat Pattern Detection) ────────────────────────
    MIN_VELOCITY_THRESHOLD: float        = 5.0
    SUSPICIOUS_VELOCITY_THRESHOLD: float = 25.0
    HIGH_RISK_VELOCITY_THRESHOLD: float  = 50.0
    VELOCITY_HISTORY_SIZE: int           = 10

    # ── Occlusion & Multi-Face ─────────────────────────────────────────────
    FACE_OCCLUSION_THRESHOLD: float   = 0.35
    MIN_MULTIPLE_FACE_DURATION: float = 0.2

    # ── Face Missing ──────────────────────────────────────────────────────
    FACE_MISSING_MIN_START_DURATION: float = 0.8
    FACE_MISSING_MIN_DURATION: float       = 2.0

    # ── Head-turn Bridge ──────────────────────────────────────────────────
    HEAD_TURN_BRIDGE_SECONDS: float = 1.0

    # ── Intel OpenVINO Runtime Settings ───────────────────────────────────
    INTEL_DEVICE: str = (os.getenv("INTEL_DEVICE") or "CPU").strip()
    INTEL_PRECISION: str = (os.getenv("INTEL_PRECISION") or "FP32").strip()
    INTEL_MODELS_DIR: str = "intel"
    INTEL_AUTO_DOWNLOAD: bool = True
    INTEL_DOWNLOAD_TIMEOUT: int = 300
    INTEL_MODEL_REPOSITORY_URL: str = (
        "https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/2024.5.0/models_intel"
    )

    # ── Landmark Indices & 3D Model Points (fixed for MediaPipe 468) ──────
    FACE_MESH_LEFT_EYE_INDICES: Tuple[int, ...] = (
        33, 133, 160, 159, 158, 144, 145, 153,
    )
    FACE_MESH_RIGHT_EYE_INDICES: Tuple[int, ...] = (
        362, 263, 387, 386, 385, 373, 374, 380,
    )
    FACE_MESH_LEFT_IRIS_INDICES: Tuple[int, ...] = (474, 475, 476, 477)
    FACE_MESH_RIGHT_IRIS_INDICES: Tuple[int, ...] = (469, 470, 471, 472)
    FACE_MESH_LEFT_EYE_CORNER_INDICES: Tuple[int, ...] = (33, 133)
    FACE_MESH_RIGHT_EYE_CORNER_INDICES: Tuple[int, ...] = (362, 263)

    HEAD_POSE_LANDMARK_INDICES: Tuple[int, ...] = (1, 152, 33, 263, 61, 291)
    HEAD_POSE_MODEL_POINTS_MM: Tuple[Tuple[float, float, float], ...] = (
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    )

    # ── High-risk Report Filter ───────────────────────────────────────────
    REPORT_ONLY_HIGH_RISK: bool             = False
    MIN_HEAD_INTENSITY_HIGH_RISK: float     = 28.0
    MIN_EYE_INTENSITY_HIGH_RISK: float      = 8.0
    MIN_HEAD_DURATION_HIGH_RISK: float      = 0.8

    # ── Evidence Storage ──────────────────────────────────────────────────
    STORE_EVIDENCE_FRAMES: bool          = True
    MAX_EVIDENCE_FRAMES_PER_ALERT: int   = 3

    # ── Processing Limits ─────────────────────────────────────────────────
    ENABLE_ASYNC_PROCESSING: bool  = True
    MAX_VIDEO_SIZE_MB: int         = 500
    ALLOWED_VIDEO_FORMATS: list    = None

    # ── Context-aware Proctoring ──────────────────────────────────────────
    SAFE_DOWN_MAX_VELOCITY_DEG_PER_S: float = 4.0
    SAFE_DOWN_MAX_DURATION_SEC: float       = 2.5
    SAFE_DOWN_MAX_PITCH_DEG: float          = 14.0
    MIN_HEAD_INTENSITY_HIGH_RISK_PITCH: float = 22.0

    # ── TVT (Temporal Vision Transformer) ─────────────────────────────────
    ENABLE_TVT: bool                  = False
    TVT_TEMPORAL_WINDOW: int          = 24
    TVT_CPU_OPTIMIZED: bool           = True
    TVT_PROB_THRESHOLD: float         = 0.85
    TVT_INFERENCE_INTERVAL_SEC: float = 1.0

    # ── Parallel Video Processing ─────────────────────────────────────────
    ENABLE_PARALLEL_PROCESSING: bool = False

    # ── Burst Detection ───────────────────────────────────────────────────
    BURST_WINDOW_SEC: float  = 20.0
    BURST_MIN_EVENTS: int    = 4

    # ── Cheating Likeliness Scoring Weights ───────────────────────────────
    WEIGHT_HRD: float = 0.30
    WEIGHT_ERD: float = 0.40
    WEIGHT_RTR: float = 0.20
    WEIGHT_BF:  float = 0.10
    HIGH_RISK_EVENT_MULTIPLIER: float = 2.0

    # ── Legacy Scoring ────────────────────────────────────────────────────
    SCORE_LINEAR_SCALE: float  = 3.6
    SCORE_LINEAR_OFFSET: float = 1.0
    FACE_RATIO_PENALTY_THRESHOLD: float    = 0.04
    FACE_RATIO_PENALTY_MULTIPLIER: float   = 20.0

    # ── Risk Levels ───────────────────────────────────────────────────────
    RISK_LOW_MAX: int        = 30
    RISK_MODERATE_MAX: int   = 60
    RISK_SUSPICIOUS_MAX: int = 80

    WEIGHT_GAZE: float  = 0.35
    WEIGHT_HEAD: float  = 0.40
    WEIGHT_FACE: float  = 0.25

    COUNT_WEIGHT_EYE_GAZE: float       = 1.1
    COUNT_WEIGHT_MILD_HEAD: float      = 0.6
    COUNT_WEIGHT_MAJOR_HEAD: float     = 1.3
    COUNT_WEIGHT_FACE_MISSING: float   = 4.0
    COUNT_WEIGHT_MULTIPLE_FACES: float = 6.0

    DURATION_WEIGHT_MILD: float        = 0.2
    DURATION_WEIGHT_SUSPICIOUS: float  = 0.5
    DURATION_WEIGHT_HIGH_RISK: float   = 1.0

    INTENSITY_WEIGHT_SLIGHT: float     = 0.3
    INTENSITY_WEIGHT_MODERATE: float   = 0.7
    INTENSITY_WEIGHT_EXTREME: float    = 1.4

    def __post_init__(self):
        if self.ALLOWED_VIDEO_FORMATS is None:
            self.ALLOWED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.webm', '.mkv']
        
        # Auto-select detection config based on DETECTION_CLIENT
        self._init_detection_config()

    def _init_detection_config(self) -> None:
        """Initialize the appropriate detection config based on DETECTION_CLIENT."""
        if self.DETECTION_CLIENT == "intel":
            self._detection_config = IntelDetectionConfig()
        else:
            self._detection_config = MediaPipeDetectionConfig()

    def get_detection_config(self) -> object:
        """Get the active detection config (MediaPipe or Intel)."""
        if self._detection_config is None:
            self._init_detection_config()
        return self._detection_config

    # ── Detection Config Accessors (proxy to active config) ────────────────

    @property
    def MIN_DETECTION_CONFIDENCE(self) -> float:
        """Minimum confidence for face detection (model-specific)."""
        return self.get_detection_config().MIN_DETECTION_CONFIDENCE

    @property
    def MIN_TRACKING_CONFIDENCE(self) -> float:
        """Minimum confidence for face tracking (model-specific)."""
        return self.get_detection_config().MIN_TRACKING_CONFIDENCE

    @property
    def GAZE_HORIZONTAL_THRESHOLD(self) -> float:
        """Horizontal gaze angle threshold (model-specific)."""
        return self.get_detection_config().GAZE_HORIZONTAL_THRESHOLD

    @property
    def GAZE_VERTICAL_THRESHOLD(self) -> float:
        """Vertical gaze angle threshold (model-specific)."""
        return self.get_detection_config().GAZE_VERTICAL_THRESHOLD

    @property
    def HEAD_YAW_THRESHOLD(self) -> float:
        """Head yaw rotation threshold (model-specific)."""
        return self.get_detection_config().HEAD_YAW_THRESHOLD

    @property
    def HEAD_PITCH_THRESHOLD(self) -> float:
        """Head pitch rotation threshold (model-specific)."""
        return self.get_detection_config().HEAD_PITCH_THRESHOLD

    @property
    def HEAD_ROLL_THRESHOLD(self) -> float:
        """Head roll rotation threshold (model-specific)."""
        return self.get_detection_config().HEAD_ROLL_THRESHOLD

    @property
    def EYE_ASPECT_RATIO_THRESHOLD(self) -> float:
        """Eye aspect ratio threshold for blink detection (model-specific)."""
        return self.get_detection_config().EYE_ASPECT_RATIO_THRESHOLD

    @property
    def MIN_IRIS_VISIBILITY(self) -> float:
        """Minimum iris visibility ratio (model-specific)."""
        return self.get_detection_config().MIN_IRIS_VISIBILITY

    @property
    def BLINK_EAR_THRESHOLD(self) -> float:
        """Eye aspect ratio threshold for blink (model-specific)."""
        return self.get_detection_config().BLINK_EAR_THRESHOLD

    @property
    def GAZE_CONFIDENCE_MULTIPLIER(self) -> float:
        """Multiplier for gaze confidence (model-specific)."""
        return self.get_detection_config().GAZE_CONFIDENCE_MULTIPLIER

    @property
    def POSE_CONFIDENCE_MULTIPLIER(self) -> float:
        """Multiplier for head pose confidence (model-specific)."""
        return self.get_detection_config().POSE_CONFIDENCE_MULTIPLIER

    @property
    def MIN_FACE_PRESENCE_SCORE(self) -> float:
        """Minimum face presence score (model-specific)."""
        return self.get_detection_config().MIN_FACE_PRESENCE_SCORE

    @property
    def MIN_CONFIDENCE_THRESHOLD(self) -> float:
        """Minimum confidence threshold for face (model-specific)."""
        return self.get_detection_config().MIN_CONFIDENCE_THRESHOLD

    @property
    def EYE_MIN_CONFIDENCE_THRESHOLD(self) -> float:
        """Minimum confidence threshold for eyes (model-specific)."""
        return self.get_detection_config().EYE_MIN_CONFIDENCE_THRESHOLD

    def get_default_thresholds(self) -> Dict[str, float]:
        """Return thresholds for the active detection client."""
        cfg = self.get_detection_config()
        return {
            'eye_horizontal':     cfg.GAZE_HORIZONTAL_THRESHOLD,
            'eye_vertical':       cfg.GAZE_VERTICAL_THRESHOLD,
            'yaw':                cfg.HEAD_YAW_THRESHOLD,
            'pitch':              cfg.HEAD_PITCH_THRESHOLD,
            'roll':               cfg.HEAD_ROLL_THRESHOLD,
            'min_event_duration': self.MIN_EVENT_DURATION,
        }


class SchedulerConfig:
    """APScheduler settings for the proctoring background job."""
    MAX_RETRIES: int = 3

    @property
    def MIN_INTERVIEW_CREATED_DATE(self) -> Optional[str]:
        """
        Optional cutoff date (YYYY-MM-DD) used by the scheduler eligibility query.

        If set, the scheduler will only pick interviews whose `ai_interview_status.created`
        is on/after this date (date comparison).
        """
        raw = (os.getenv("SCHEDULER_MIN_INTERVIEW_CREATED_DATE") or "").strip()
        if not raw:
            return None
        # Validate format early so we fail fast on misconfiguration.
        date.fromisoformat(raw)  # raises ValueError if invalid
        return raw

    @property
    def INTERVAL_SECONDS(self) -> int:
        raw = os.getenv("SCHEDULER_INTERVAL_SECONDS", "3600") or "3600"
        return max(1, int(raw))

    @property
    def TIMEZONE(self) -> str:
        return (os.getenv("SCHEDULER_TIMEZONE", "UTC") or "UTC").strip()

    @property
    def JOB_COALESCE(self) -> bool:
        v = (os.getenv("SCHEDULER_JOB_COALESCE", "true") or "true").strip().lower()
        return v in ("1", "true", "yes", "y")

    @property
    def JOB_MAX_INSTANCES(self) -> int:
        raw = os.getenv("SCHEDULER_JOB_MAX_INSTANCES", "1") or "1"
        return max(1, int(raw))


class BasicAuthConfig:
    """HTTP Basic authentication settings."""

    @property
    def USERNAME(self) -> str:
        return (os.getenv("AUTH_USERNAME") or "admin").strip()

    @property
    def PASSWORD(self) -> str:
        return (os.getenv("AUTH_PASSWORD") or "strong_password").strip()


# ── Singleton Instances ───────────────────────────────────────────────────────

PATH_CONFIG          = PathConfig()
CORS_CONFIG          = CORSConfig()
RATE_LIMIT_CONFIG    = RateLimitConfig()
PROCTORING_CONFIG    = ProctoringConfig()
S3_CONFIG            = S3Config()
PROCTORING_DB_CONFIG = ProctoringDatabaseConfig()
SCHEDULER_CONFIG     = SchedulerConfig()
BASIC_AUTH_CONFIG    = BasicAuthConfig()
