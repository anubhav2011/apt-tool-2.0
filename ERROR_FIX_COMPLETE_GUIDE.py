"""
COMPLETE DIAGNOSTIC AND FIX FOR: ModuleNotFoundError: No module named 'app.client.detection.intel_client'

ROOT CAUSE:
  Your local machine has out-of-date code that was restructured here on the V0 VM.
  The detection clients were moved from app/client/detection/ into 
  app/client/detection/face_tracking/ subfolder for better organization.

FILES THAT CHANGED:
  
  1. Created new structure:
     app/client/detection/face_tracking/
     ├── __init__.py
     ├── base.py                    (Abstract BaseDetectionClient)
     ├── mediapipe_client.py        (MediaPipe implementation)
     ├── intel_client.py            (Intel OpenVINO implementation)
     └── model_downloader.py        (Auto-download Intel OMZ models)

  2. Updated factory.py:
     - Now imports from: app.client.detection.face_tracking.*
     - Old imports from app.client.detection.* removed

  3. Updated detection/__init__.py:
     - Re-exports from face_tracking subfolder

  4. Updated config.py:
     - Added MediaPipeDetectionConfig class
     - Added IntelDetectionConfig class
     - Auto-selection based on DETECTION_CLIENT env var

  5. Updated main.py:
     - Imports ensure_intel_models from face_tracking.model_downloader
     - Calls model downloader at startup when DETECTION_CLIENT=intel

EXACT STEPS TO FIX YOUR LOCAL MACHINE:

Step 1: Clean Python Cache
  Windows (PowerShell as Administrator):
    Get-ChildItem -Path . -Include __pycache__ -Recurse -Force | Remove-Item -Recurse -Force
    Get-ChildItem -Path . -Include *.pyc -Recurse -Force | Remove-Item -Force

  Linux/Mac:
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete

Step 2: Pull Latest Code
  git fetch origin
  git pull origin v0/anubhavvaish-8359-6e686d79

Step 3: Verify New Structure
  Check that these files exist in your local repo:
    ✓ app/client/detection/face_tracking/base.py
    ✓ app/client/detection/face_tracking/mediapipe_client.py
    ✓ app/client/detection/face_tracking/intel_client.py
    ✓ app/client/detection/face_tracking/model_downloader.py
    ✓ app/client/detection/factory.py
    ✓ app/client/detection/__init__.py

Step 4: Delete Virtual Environment (if problematic)
  Windows:
    rmdir /s venv
  Linux/Mac:
    rm -rf venv

Step 5: Create Fresh Virtual Environment
  python -m venv venv
  # Windows:
  venv\Scripts\activate
  # Linux/Mac:
  source venv/bin/activate

Step 6: Install Dependencies
  pip install -r requirements.txt

Step 7: Clear Import Cache
  python -c "import py_compile; import os; [os.remove(f) for f in __import__('glob').glob('**/*.pyc', recursive=True)]"

Step 8: Test Server Start (MediaPipe - default)
  export DETECTION_CLIENT=mediapipe
  # or Windows: set DETECTION_CLIENT=mediapipe
  
  uvicorn app.main:app --reload
  
  Expected log:
    "Detection client initialised: MediaPipeDetectionClient (DETECTION_CLIENT='mediapipe')"

Step 9: Test Intel Client (with auto-download)
  export DETECTION_CLIENT=intel
  # or Windows: set DETECTION_CLIENT=intel
  
  uvicorn app.main:app --reload
  
  Expected logs:
    "DETECTION_CLIENT=intel detected — ensuring Intel models are present …"
    "Intel models ready."
    "Detection client initialised: IntelDetectionClient (DETECTION_CLIENT='intel')"
    Models downloaded to: app/client/detection/face_tracking/intel/
    
Step 10: Verify Model Download Location
  List the models:
    Windows PowerShell: Get-ChildItem -Path "app\client\detection\face_tracking\intel" -Recurse
    Linux/Mac: find app/client/detection/face_tracking/intel -type f
    
  Expected structure:
    app/client/detection/face_tracking/intel/
    ├── face-detection-adas-0001/FP32/
    │   ├── face-detection-adas-0001.xml
    │   └── face-detection-adas-0001.bin
    ├── facial-landmarks-35-adas-0002/FP32/
    │   ├── facial-landmarks-35-adas-0002.xml
    │   └── facial-landmarks-35-adas-0002.bin
    ├── head-pose-estimation-adas-0001/FP32/
    │   ├── head-pose-estimation-adas-0001.xml
    │   └── head-pose-estimation-adas-0001.bin
    └── gaze-estimation-adas-0002/FP32/
        ├── gaze-estimation-adas-0002.xml
        └── gaze-estimation-adas-0002.bin

WHAT AUTO-LOADS ON EACH CLIENT:

MediaPipe (DETECTION_CLIENT=mediapipe):
  - Config: MediaPipeDetectionConfig
  - Thresholds:
      MIN_DETECTION_CONFIDENCE = 0.70
      MIN_TRACKING_CONFIDENCE = 0.70
      EYE_ASPECT_RATIO_THRESHOLD = 0.18
  - No model files needed (uses local MediaPipe models)

Intel (DETECTION_CLIENT=intel):
  - Config: IntelDetectionConfig
  - Thresholds:
      MIN_DETECTION_CONFIDENCE = 0.65
      MIN_TRACKING_CONFIDENCE = 0.60
      EYE_ASPECT_RATIO_THRESHOLD = 0.16
  - Auto-downloads 4 Intel OMZ models on first startup
  - Models cached in: app/client/detection/face_tracking/intel/
  - Uses INTEL_DEVICE env var (default: CPU)
  - Uses INTEL_PRECISION env var (default: FP32)

ENVIRONMENT VARIABLES TO SET:

Required:
  DETECTION_CLIENT=mediapipe  # or intel
  
Optional (Intel-specific):
  INTEL_DEVICE=CPU            # CPU, GPU, AUTO
  INTEL_PRECISION=FP32        # FP32 (best), FP16 (faster), INT8 (fastest)

IF PROBLEM PERSISTS:

1. Check Python version compatibility:
   python --version
   # Should be 3.8+

2. Verify openvino is installed (for Intel):
   pip list | grep openvino
   # Should show: openvino, openvino-dev

3. Check if old app/client/detection/intel_client.py exists:
   ls -la app/client/detection/
   # Should NOT show intel_client.py or mediapipe_client.py in this directory
   # They should ONLY be in face_tracking/ subfolder

4. Force reimport:
   python -c "from app.client.detection import get_detection_client; print('OK')"
   # Should print: OK

5. Check import path:
   python -c "from app.client.detection.face_tracking.intel_client import IntelDetectionClient; print(IntelDetectionClient)"
   # Should print the class

6. If all fails, do a clean clone:
   cd ..
   rm -rf apt-tool
   git clone -b v0/anubhavvaish-8359-6e686d79 https://github.com/anubhav2011/apt-tool-2.0.git apt-tool
   cd apt-tool
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   DETECTION_CLIENT=mediapipe uvicorn app.main:app
"""
