# config.py
"""
Configuration settings for the AI Interviewer application.
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
class ProctoringConfig:

    # ── Video Processing ──────────────────────────────────────────────────
    MAX_FRAME_DIMENSION: int   = 1280
    TARGET_FPS: int            = 18
    WARMUP_SECONDS: float      = 3.0

    # ── Detection Client Selection ─────────────────────────────────────────
    # Options: "mediapipe" (default) or "intel"
    DETECTION_CLIENT: str      = os.getenv("DETECTION_CLIENT", "mediapipe").strip().lower()

    # ── Detection Accuracy Tuning (Both MediaPipe & Intel) ────────────────
    # These parameters fine-tune face detection, iris tracking, and pose
    # estimation for accuracy across different lighting/environment conditions
    
    DETECTION_MIN_CONFIDENCE: float = float(
        os.getenv("DETECTION_MIN_CONFIDENCE", "0.65") or "0.65"
    )
    """
    Minimum confidence threshold for face detection (0.0 - 1.0).
    Higher values reduce false positives but increase misses in poor lighting.
    MediaPipe: 0.6-0.7, Intel: 0.5-0.7. Balanced default: 0.65
    """

    DETECTION_MIN_TRACKING_CONFIDENCE: float = float(
        os.getenv("DETECTION_MIN_TRACKING_CONFIDENCE", "0.65") or "0.65"
    )
    """
    Minimum confidence for face tracking frame-to-frame (0.0 - 1.0).
    Higher values stabilize tracking but slow occlusion recovery.
    MediaPipe: 0.7, Intel: 0.5-0.6. Balanced: 0.65
    """

    GAZE_CONFIDENCE_MULTIPLIER: float = float(
        os.getenv("GAZE_CONFIDENCE_MULTIPLIER", "1.0") or "1.0"
    )
    """
    Multiplier for gaze detection confidence (Intel-specific).
    > 1.0: boost confidence (trust gaze more), < 1.0: reduce (stricter).
    Use for accuracy tuning when Intel gaze is over/under-confident vs MediaPipe.
    """

    POSE_CONFIDENCE_MULTIPLIER: float = float(
        os.getenv("POSE_CONFIDENCE_MULTIPLIER", "1.0") or "1.0"
    )
    """
    Multiplier for head pose confidence (Intel-specific).
    > 1.0: boost, < 1.0: reduce. Tune if Intel head pose differs from MediaPipe.
    """

    # ── Gaze & Pose Thresholds (Calibration-Free) ─────────────────────────
    FIXED_EYE_HORIZONTAL_THRESHOLD: float = 8.0
    FIXED_EYE_VERTICAL_THRESHOLD: float   = 8.0
    FIXED_HEAD_YAW_THRESHOLD: float       = 28.0
    FIXED_HEAD_PITCH_THRESHOLD: float     = 28.0
    FIXED_HEAD_ROLL_THRESHOLD: float      = 18.0

    # ── EAR & Blink Detection ─────────────────────────────────────────────
    EYE_ASPECT_RATIO_THRESHOLD: float = float(
        os.getenv("EYE_ASPECT_RATIO_THRESHOLD", "0.18") or "0.18"
    )
    """
    Eye aspect ratio (EAR) threshold for open/closed eye detection (0.05-0.30).
    Lower = more sensitive to blinks, higher = stricter.
    Both MediaPipe & Intel benefit from this tuning.
    """
    
    MIN_IRIS_VISIBILITY: float        = 0.45
    BLINK_EAR_THRESHOLD: float        = 0.14
    MIN_BLINK_DURATION: float         = 0.05
    MAX_BLINK_DURATION: float         = 0.35
    NORMAL_BLINK_RATE_MIN: int        = 5
    NORMAL_BLINK_RATE_MAX: int        = 30
    ABNORMAL_BLINK_WINDOW: float      = 60.0

    # ── Temporal Consistency ──────────────────────────────────────────────
    MIN_EVENT_DURATION: float  = 1.0
    EYE_MIN_EVENT_DURATION: float  = 0.8
    EVENT_GAP_TOLERANCE: float = 0.55

    # ── Frame Rejection ───────────────────────────────────────────────────
    MAX_HEAD_ROTATION_SPEED: float  = 10.0
    MAX_BBOX_CENTER_SHIFT: float    = 0.05
    MAX_EYE_ANGLE_VARIANCE: float   = 0.35
    MIN_FACE_PRESENCE_SCORE: float  = 0.90
    MIN_CONFIDENCE_THRESHOLD: float = 0.45
    EYE_MIN_CONFIDENCE_THRESHOLD: float = 0.35

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

    # ── MediaPipe Settings ────────────────────────────────────────────────
    MEDIAPIPE_MAX_FACES: int                    = 2
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float   = 0.7
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float    = 0.7

    # ── Intel OpenVINO Settings ──────────────────────────────────────────
    INTEL_DEVICE: str = (os.getenv("INTEL_DEVICE") or "CPU").strip()
    """Device for OpenVINO: CPU (default), GPU, AUTO, etc."""
    
    INTEL_PRECISION: str = (os.getenv("INTEL_PRECISION") or "FP32").strip()
    """Model precision: FP32 (best accuracy), FP16 (faster), INT8 (fastest)"""
    
    INTEL_MODELS_DIR: str = (os.getenv("INTEL_MODELS_DIR") or "intel").strip()
    """Directory for downloaded Intel OMZ models (relative to app/client/detection/)"""
    
    INTEL_AUTO_DOWNLOAD: bool = (
        (os.getenv("INTEL_AUTO_DOWNLOAD") or "true").strip().lower() in ("1", "true", "yes")
    )
    """Auto-download models on server startup when DETECTION_CLIENT=intel"""
    
    INTEL_DOWNLOAD_TIMEOUT: int = int(os.getenv("INTEL_DOWNLOAD_TIMEOUT") or "300")
    """Timeout (seconds) for model downloads"""
    
    INTEL_MODEL_REPOSITORY_URL: str = (
        os.getenv("INTEL_MODEL_REPOSITORY_URL")
        or "https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/2024.5.0/models_intel"
    ).strip()
    """Fallback repository for manual model downloads"""

    # Face mesh landmarks (468 model) & 6-point head-pose 3D model (mm) ─
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


    def __post_init__(self):
        if self.ALLOWED_VIDEO_FORMATS is None:
            self.ALLOWED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.webm', '.mkv']

    # ── Context-aware Proctoring ──────────────────────────────────────────
    SAFE_DOWN_MAX_VELOCITY_DEG_PER_S: float = 4.0

    SAFE_DOWN_MAX_DURATION_SEC: float       = 2.5
    SAFE_DOWN_MAX_PITCH_DEG: float          = 14.0

    # ── High-risk Filter: lower bar for head up/down ──────────────────────
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

    # ── Cheating Likeliness Scoring Weights (v2.4) ────────────────────────

    WEIGHT_HRD: float = 0.30
    WEIGHT_ERD: float = 0.40
    WEIGHT_RTR: float = 0.20
    WEIGHT_BF:  float = 0.10

    # CHANGE 5: multiplier raised 1.8 → 2.0.

    HIGH_RISK_EVENT_MULTIPLIER: float = 2.0

    # Legacy callers — two-stage piecewise linear now used in ScoringService
    SCORE_LINEAR_SCALE: float  = 3.6
    SCORE_LINEAR_OFFSET: float = 1.0

    # ── Face Penalty (ratio-based) ────────────────────────────────────────
    FACE_RATIO_PENALTY_THRESHOLD: float    = 0.04   # updated: was 0.06; matches new smooth curve anchor
    FACE_RATIO_PENALTY_MULTIPLIER: float   = 20.0   # legacy caller compat

    # ── Legacy PDF Scoring ────────────────────────────────────────────────
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

    # ── Threshold Accessors ───────────────────────────────────────────────

    def get_default_thresholds(self) -> Dict[str, float]:
        """Return fixed industry-standard thresholds (no calibration)."""
        return {
            'eye_horizontal':     self.FIXED_EYE_HORIZONTAL_THRESHOLD,
            'eye_vertical':       self.FIXED_EYE_VERTICAL_THRESHOLD,
            'yaw':                self.FIXED_HEAD_YAW_THRESHOLD,
            'pitch':              self.FIXED_HEAD_PITCH_THRESHOLD,
            'roll':               self.FIXED_HEAD_ROLL_THRESHOLD,
            'min_event_duration': self.MIN_EVENT_DURATION,
        }


class SchedulerConfig:
    """APScheduler settings for the proctoring background job."""
     # Failed proctoring attempts before eligibility ends (scheduler scans + status repo).
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
