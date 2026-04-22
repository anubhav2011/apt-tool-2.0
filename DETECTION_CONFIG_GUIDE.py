"""
DETECTION CONFIG ARCHITECTURE GUIDE
====================================

This document explains how the new config architecture works and how to use it
when switching between MediaPipe and Intel detection clients.

═════════════════════════════════════════════════════════════════════════════
1. ARCHITECTURE OVERVIEW
═════════════════════════════════════════════════════════════════════════════

The detection system has THREE layers of configuration:

1. RUNTIME SETTINGS (.env file)
   └─ DETECTION_CLIENT, INTEL_DEVICE, INTEL_PRECISION
      (These control which backend runs and how)

2. BACKEND-SPECIFIC THRESHOLDS (config.py dataclasses)
   ├─ MediaPipeDetectionConfig (for DETECTION_CLIENT=mediapipe)
   └─ IntelDetectionConfig (for DETECTION_CLIENT=intel)

3. MODEL-INDEPENDENT LOGIC (config.py ProctoringConfig)
   └─ Violation tracking, scoring, temporal thresholds
      (These are the same regardless of backend)

═════════════════════════════════════════════════════════════════════════════
2. HOW IT WORKS AUTOMATICALLY
═════════════════════════════════════════════════════════════════════════════

When ProctoringConfig is initialized:

   __post_init__() called
        │
        └─> _init_detection_config()
             │
             └─> Checks DETECTION_CLIENT env var
                  │
                  ├─ If "intel"     → creates IntelDetectionConfig()
                  │
                  └─ Otherwise      → creates MediaPipeDetectionConfig()

Then when any code accesses config.MIN_DETECTION_CONFIDENCE:

   ProctoringConfig.MIN_DETECTION_CONFIDENCE
        │
        └─> @property calls get_detection_config()
             │
             └─> Returns the active detection config (Intel or MediaPipe)
                  │
                  └─> Returns its MIN_DETECTION_CONFIDENCE value

═════════════════════════════════════════════════════════════════════════════
3. SWITCHING CLIENTS (ZERO CODE CHANGES REQUIRED)
═════════════════════════════════════════════════════════════════════════════

To use MediaPipe (default):
   $ export DETECTION_CLIENT=mediapipe  (or leave unset)
   $ uvicorn app.main:app

   Result:
   ✓ MediaPipeDetectionConfig() activated
   ✓ config.MIN_DETECTION_CONFIDENCE = 0.70
   ✓ config.GAZE_HORIZONTAL_THRESHOLD = 8.0
   ✓ config.EYE_ASPECT_RATIO_THRESHOLD = 0.18
   ✓ All other MediaPipe thresholds used

To use Intel OpenVINO:
   $ export DETECTION_CLIENT=intel
   $ export INTEL_DEVICE=CPU
   $ export INTEL_PRECISION=FP32
   $ uvicorn app.main:app

   Result:
   ✓ IntelDetectionConfig() activated
   ✓ config.MIN_DETECTION_CONFIDENCE = 0.65
   ✓ config.GAZE_HORIZONTAL_THRESHOLD = 8.5
   ✓ config.EYE_ASPECT_RATIO_THRESHOLD = 0.16
   ✓ All other Intel thresholds used
   ✓ Models auto-downloaded on first startup

═════════════════════════════════════════════════════════════════════════════
4. THRESHOLDS COMPARISON
═════════════════════════════════════════════════════════════════════════════

PARAMETER                          MEDIAPIPE    INTEL
─────────────────────────────────────────────────────
MIN_DETECTION_CONFIDENCE           0.70         0.65
MIN_TRACKING_CONFIDENCE            0.70         0.60
GAZE_HORIZONTAL_THRESHOLD          8.0          8.5
GAZE_VERTICAL_THRESHOLD            8.0          8.5
HEAD_YAW_THRESHOLD                 28.0         28.0
HEAD_PITCH_THRESHOLD               28.0         28.0
HEAD_ROLL_THRESHOLD                18.0         18.0
EYE_ASPECT_RATIO_THRESHOLD         0.18         0.16
MIN_IRIS_VISIBILITY                0.45         0.40
BLINK_EAR_THRESHOLD                0.14         0.12
MIN_FACE_PRESENCE_SCORE            0.90         0.85
MIN_CONFIDENCE_THRESHOLD           0.45         0.40
EYE_MIN_CONFIDENCE_THRESHOLD       0.35         0.30

Key Differences:
─ Intel typically needs LOWER detection confidence (better at finding faces)
─ Intel has LOWER eye thresholds (better at tracking iris movement)
─ Intel is more resilient in poor lighting
─ Intel trades speed for accuracy (FP32 > FP16 > INT8 in accuracy)

═════════════════════════════════════════════════════════════════════════════
5. MODIFYING THRESHOLDS FOR ACCURACY TUNING
═════════════════════════════════════════════════════════════════════════════

To tune thresholds FOR A SPECIFIC CLIENT, edit the config class:

Example: Make Intel more strict about face detection
──────────────────────────────────────────────────
File: app/core/config.py

   @dataclass
   class IntelDetectionConfig:
       MIN_DETECTION_CONFIDENCE: float = 0.65  # ← Change to 0.72 for stricter

Result: Only when DETECTION_CLIENT=intel will use stricter threshold
         MediaPipe config unaffected

Example: Make MediaPipe more sensitive to blinks
──────────────────────────────────────────────────
File: app/core/config.py

   @dataclass
   class MediaPipeDetectionConfig:
       EYE_ASPECT_RATIO_THRESHOLD: float = 0.18  # ← Change to 0.15 for more sensitive

Result: Only when DETECTION_CLIENT=mediapipe will detect blinks earlier
         Intel config unaffected

═════════════════════════════════════════════════════════════════════════════
6. ENVIRONMENT VARIABLES (RUNTIME-ONLY)
═════════════════════════════════════════════════════════════════════════════

Essential .env settings:

DETECTION_CLIENT=mediapipe           # Which backend to use
INTEL_DEVICE=CPU                     # Hardware device (CPU/GPU/AUTO)
INTEL_PRECISION=FP32                 # Model precision (FP32/FP16/INT8)

DO NOT add to .env:
✗ DETECTION_MIN_CONFIDENCE (use config class instead)
✗ EYE_ASPECT_RATIO_THRESHOLD (use config class instead)
✗ Any accuracy thresholds (all in config.py now)

═════════════════════════════════════════════════════════════════════════════
7. HOW DETECTION SERVICE USES IT
═════════════════════════════════════════════════════════════════════════════

File: app/services/proctoring_processing/detection_service.py

   def _setup(self):
       # Get the active detection client (MediaPipe or Intel)
       self._detection_client = get_detection_client(self.config)
       
       # All per-frame work delegated to the client:
       # - detect_gaze() 
       # - detect_head_pose()
       # - get_landmark_vector()

   def detect_gaze(self, frame):
       # Calls the active backend's gaze detection
       return self._detection_client.detect_gaze(frame)

   # ViolationTracker, TVT, scoring, velocity tracking all stay here
   # (model-independent logic)

The config thresholds are used by both clients:
├─ MediaPipeDetectionClient reads MediaPipeDetectionConfig via config proxy
└─ IntelDetectionClient reads IntelDetectionConfig via config proxy

═════════════════════════════════════════════════════════════════════════════
8. CLIENT INITIALIZATION FLOW
═════════════════════════════════════════════════════════════════════════════

app.main.py startup
  │
  └─> ProctoringConfig created
       │
       └─> Calls __post_init__()
            │
            └─> Calls _init_detection_config()
                 │
                 └─> Reads DETECTION_CLIENT env var
                      │
                      ├─ "intel" → IntelDetectionConfig()
                      └─ else    → MediaPipeDetectionConfig()
  │
  └─> If DETECTION_CLIENT=intel
       │
       └─> Calls ensure_intel_models()
            │
            └─> Auto-downloads 4 Intel OMZ models to
                app/client/detection/face_tracking/intel/
  │
  └─> DetectionService._setup() called
       │
       └─> Calls get_detection_client(config)
            │
            └─> Factory reads config.DETECTION_CLIENT
                 │
                 ├─ Creates MediaPipeDetectionClient (if mediapipe)
                 └─ Creates IntelDetectionClient (if intel)

═════════════════════════════════════════════════════════════════════════════
9. ADDING A NEW DETECTION BACKEND
═════════════════════════════════════════════════════════════════════════════

Step 1: Create threshold config class
────────────────────────────────────
File: app/core/config.py

   @dataclass
   class YourModelDetectionConfig:
       MIN_DETECTION_CONFIDENCE: float = 0.68
       # ... all your thresholds

Step 2: Update ProctoringConfig initialization
─────────────────────────────────────────────
File: app/core/config.py

   def _init_detection_config(self) -> None:
       if self.DETECTION_CLIENT == "intel":
           self._detection_config = IntelDetectionConfig()
       elif self.DETECTION_CLIENT == "yourmodel":      # ← Add
           self._detection_config = YourModelDetectionConfig()  # ← Add
       else:
           self._detection_config = MediaPipeDetectionConfig()

Step 3: Create client implementation
──────────────────────────────────
File: app/client/detection/face_tracking/yourmodel_client.py

   from app.client.detection.face_tracking.base import BaseDetectionClient
   
   class YourModelDetectionClient(BaseDetectionClient):
       def detect_gaze(self, frame):
           # Your implementation
       # ... other methods

Step 4: Register in factory
──────────────────────────
File: app/client/detection/factory.py

   def _load_yourmodel_class() -> Type[BaseDetectionClient]:
       from app.client.detection.face_tracking.yourmodel_client import (
           YourModelDetectionClient,
       )
       return YourModelDetectionClient

   _REGISTRY = {
       "mediapipe": MediaPipeDetectionClient,
       "intel": IntelDetectionClient,
       "yourmodel": YourModelDetectionClient,  # ← Add
   }

Step 5: Use it
─────────────
   $ export DETECTION_CLIENT=yourmodel
   $ uvicorn app.main:app

═════════════════════════════════════════════════════════════════════════════
10. QUICK REFERENCE
═════════════════════════════════════════════════════════════════════════════

SWITCH TO MEDIAPIPE:
   export DETECTION_CLIENT=mediapipe
   uvicorn app.main:app

SWITCH TO INTEL:
   export DETECTION_CLIENT=intel
   export INTEL_DEVICE=CPU
   export INTEL_PRECISION=FP32
   uvicorn app.main:app

EDIT MEDIAPIPE THRESHOLDS:
   File: app/core/config.py
   Class: MediaPipeDetectionConfig
   Change any threshold value

EDIT INTEL THRESHOLDS:
   File: app/core/config.py
   Class: IntelDetectionConfig
   Change any threshold value

GET ACTIVE THRESHOLDS PROGRAMMATICALLY:
   from app.core.config import PROCTORING_CONFIG
   cfg = PROCTORING_CONFIG.get_detection_config()
   print(cfg.MIN_DETECTION_CONFIDENCE)  # Returns active value

═════════════════════════════════════════════════════════════════════════════
"""
