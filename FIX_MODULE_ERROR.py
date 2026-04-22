"""
RESOLUTION: No module named 'app.client.detection.intel_client'

The error occurs because your local Git repository is out of sync with the latest changes.
The detection clients have been reorganized into a face_tracking subfolder.

OLD STRUCTURE (outdated on your machine):
  app/client/detection/
  ├── intel_client.py          ❌ OLD
  ├── mediapipe_client.py      ❌ OLD
  └── factory.py

NEW STRUCTURE (correct):
  app/client/detection/
  ├── factory.py
  └── face_tracking/
      ├── intel_client.py      ✅ CORRECT
      ├── mediapipe_client.py  ✅ CORRECT
      ├── base.py
      ├── model_downloader.py
      └── __init__.py

SOLUTION:

Option 1 - Pull latest code from Git:
    git pull origin v0/anubhavvaish-8359-6e686d79
    pip install -r requirements.txt
    
Option 2 - Manual cleanup and sync:
    1. Delete __pycache__ folders:
       find . -type d -name "__pycache__" -exec rm -rf {} +
    2. Delete .pyc files:
       find . -type f -name "*.pyc" -delete
    3. Manually reorganize your app/client/detection/ folder structure to match the NEW STRUCTURE above
    4. Ensure factory.py imports from face_tracking subfolder
    5. Restart your server

Option 3 - Clean environment (recommended):
    1. Delete all Python cache: find . -type d -name "__pycache__" -exec rm -rf {} +
    2. Delete .pyc files: find . -type f -name "*.pyc" -delete
    3. Delete your virtual environment and create a fresh one:
       python -m venv venv
       source venv/bin/activate  # On Windows: venv\Scripts\activate
    4. Install dependencies: pip install -r requirements.txt
    5. Restart server: DETECTION_CLIENT=mediapipe uvicorn app.main:app
    6. Then test Intel: DETECTION_CLIENT=intel uvicorn app.main:app

VERIFY THE FIX:

After syncing your code, verify these files exist:
  ✓ app/client/detection/face_tracking/intel_client.py
  ✓ app/client/detection/face_tracking/mediapipe_client.py
  ✓ app/client/detection/face_tracking/base.py
  ✓ app/client/detection/face_tracking/model_downloader.py
  ✓ app/client/detection/factory.py
  ✓ app/client/detection/__init__.py

And that factory.py imports from:
  ✓ from app.client.detection.face_tracking.intel_client import IntelDetectionClient
  ✓ from app.client.detection.face_tracking.mediapipe_client import MediaPipeDetectionClient

TESTING:

1. Start with MediaPipe (default):
   python -m pytest tests/ -v  # or just start the server
   
2. Switch to Intel:
   export DETECTION_CLIENT=intel
   uvicorn app.main:app
   # Models auto-download to: app/client/detection/face_tracking/intel/

3. Verify in logs:
   "Detection client initialised: MediaPipeDetectionClient"
   or
   "Detection client initialised: IntelDetectionClient"
"""
