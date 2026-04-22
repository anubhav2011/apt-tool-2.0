"""
RESOLUTION: ModuleNotFoundError: No module named 'app.client.detection.intel_client'
===================================================================================

THE ERROR (From your error log):
  Traceback shows:
    File "app/client/detection/factory.py", line 59, in _load_intel_class
      from app.client.detection.intel_client import IntelDetectionClient
    ModuleNotFoundError: No module named 'app.client.detection.intel_client'

WHY IT HAPPENED:
  
  The code was restructured on the V0 VM to reorganize detection clients into a
  subfolder for better organization. Your local machine still has the OLD code
  structure and Python is trying to import from the wrong path.

  OLD STRUCTURE (on your machine):
    app/client/detection/
    ├── intel_client.py              ← Tried to import from here
    ├── mediapipe_client.py
    ├── factory.py
    └── __init__.py

  NEW STRUCTURE (on V0 VM - current):
    app/client/detection/
    ├── factory.py
    ├── __init__.py
    └── face_tracking/               ← New subfolder!
        ├── intel_client.py          ← Should import from HERE
        ├── mediapipe_client.py
        ├── base.py
        ├── model_downloader.py
        └── __init__.py

THE FIX (3 simple steps):

  Step 1: Update code from Git
    git fetch origin
    git pull origin v0/anubhavvaish-8359-6e686d79

  Step 2: Clear Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete

  Step 3: Reinstall (optional but recommended)
    pip install -r requirements.txt --force-reinstall

VERIFY THE FIX:

  After pulling, verify these files exist:
    ✓ app/client/detection/face_tracking/intel_client.py
    ✓ app/client/detection/face_tracking/mediapipe_client.py
    ✓ app/client/detection/face_tracking/base.py
    ✓ app/client/detection/face_tracking/model_downloader.py

  Then test:
    DETECTION_CLIENT=mediapipe uvicorn app.main:app
    # Should work without errors

WHAT CHANGED IN CODE:

  1. Clients moved to face_tracking subfolder:
     app/client/detection/face_tracking/
     ├── intel_client.py (moved from app/client/detection/intel_client.py)
     └── mediapipe_client.py (moved from app/client/detection/mediapipe_client.py)

  2. factory.py updated to import from new location:
     OLD: from app.client.detection.intel_client import IntelDetectionClient
     NEW: from app.client.detection.face_tracking.intel_client import IntelDetectionClient

  3. config.py added two detection config classes:
     - MediaPipeDetectionConfig (MediaPipe-specific thresholds)
     - IntelDetectionConfig (Intel-specific thresholds)

  4. Auto-selection on startup:
     ProctoringConfig.__post_init__() reads DETECTION_CLIENT env var
     and auto-selects MediaPipeDetectionConfig or IntelDetectionConfig

INTEL MODEL DOWNLOAD PATH:
  
  Location: app/client/detection/face_tracking/intel/
  
  When you set DETECTION_CLIENT=intel:
    1. app/main.py calls ensure_intel_models()
    2. Models auto-download to: face_tracking/intel/*/FP32/
    3. Directory structure created:
       app/client/detection/face_tracking/intel/
       ├── face-detection-adas-0001/FP32/
       ├── facial-landmarks-35-adas-0002/FP32/
       ├── head-pose-estimation-adas-0001/FP32/
       └── gaze-estimation-adas-0002/FP32/

TESTING AFTER FIX:

  # Test 1: Start with MediaPipe (should work immediately)
  set DETECTION_CLIENT=mediapipe
  uvicorn app.main:app
  
  Expected output:
    "Detection client initialised: MediaPipeDetectionClient (DETECTION_CLIENT='mediapipe')"
    Server starts on http://127.0.0.1:8000

  # Test 2: Switch to Intel (models auto-download)
  set DETECTION_CLIENT=intel
  uvicorn app.main:app
  
  Expected output:
    "DETECTION_CLIENT=intel detected — ensuring Intel models are present …"
    "Intel models ready."
    "Detection client initialised: IntelDetectionClient (DETECTION_CLIENT='intel')"
    Server starts on http://127.0.0.1:8000
    
    Models downloaded to: app/client/detection/face_tracking/intel/

IF IT STILL FAILS:

  1. Delete and recreate virtual environment:
     python -m venv venv_fresh
     venv_fresh\Scripts\activate
     pip install -r requirements.txt

  2. Or do a clean clone:
     cd ..
     git clone -b v0/anubhavvaish-8359-6e686d79 https://github.com/anubhav2011/apt-tool-2.0.git apt-tool-fresh
     cd apt-tool-fresh
     python -m venv venv
     venv\Scripts\activate
     pip install -r requirements.txt
     DETECTION_CLIENT=mediapipe uvicorn app.main:app

SUMMARY:
  Your code was out of sync. Pull the latest changes, clear Python cache, and
  you're done. The system now auto-selects the correct detection config based
  on DETECTION_CLIENT environment variable, and Intel models auto-download
  on first startup.
"""
