"""
ERROR RESOLUTION SUMMARY
========================

ERROR MESSAGE:
  ModuleNotFoundError: No module named 'app.client.detection.intel_client'

SOURCE:
  File "app/client/detection/factory.py", line 59, in _load_intel_class
    from app.client.detection.intel_client import IntelDetectionClient

ROOT CAUSE:
  Code restructuring: Detection clients moved to face_tracking subfolder
  Your local machine hasn't synced the latest changes

THE PROBLEM IN ONE IMAGE:

  Your Local Machine:            V0 VM (Current):
  ========================       ========================
  app/client/detection/          app/client/detection/
  ├── intel_client.py ❌         ├── factory.py ✓
  ├── mediapipe_client.py ❌     ├── __init__.py ✓
  ├── factory.py ❌              └── face_tracking/ ✓
  └── __init__.py ❌                ├── intel_client.py ✓
                                    ├── mediapipe_client.py ✓
                                    ├── base.py ✓
                                    ├── model_downloader.py ✓
                                    └── __init__.py ✓

IMMEDIATE FIX (Copy-Paste):

  git fetch origin
  git pull origin v0/anubhavvaish-8359-6e686d79
  find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
  pip install -r requirements.txt

THEN TEST:

  # Test MediaPipe
  set DETECTION_CLIENT=mediapipe
  uvicorn app.main:app

  # Then test Intel (in separate terminal)
  set DETECTION_CLIENT=intel
  uvicorn app.main:app

INTEL MODEL DOWNLOAD LOCATION:
  app/client/detection/face_tracking/intel/
  └── Auto-downloaded on first Intel startup

See ERROR_FIX_COMPLETE_GUIDE.py for full step-by-step instructions.
"""
