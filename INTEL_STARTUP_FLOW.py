"""
INTEL DETECTION CLIENT - COMPLETE STARTUP FLOW
===============================================

Exact sequence when you run:
  export DETECTION_CLIENT=intel
  uvicorn app.main:app

================================================================================
SECOND 0: STARTUP (app/main.py)
================================================================================

Line 32-38: Imports
  ├─ from app.client.detection.face_tracking.model_downloader import ensure_intel_models
  └─ (does NOT import IntelDetectionClient yet - lazy loading)

Line 40: config = ProctoringConfig()
  ├─ Dataclass initialization
  ├─ Line 311: DETECTION_CLIENT = os.getenv("DETECTION_CLIENT", "mediapipe").strip().lower()
  │  └─ Value: "intel"
  └─ Line 321-322: __post_init__() → _init_detection_config()
     ├─ Line 326: if self.DETECTION_CLIENT == "intel":
     ├─ Line 327: self._detection_config = IntelDetectionConfig()
     │  └─ IntelDetectionConfig created with 14 thresholds:
     │     - MIN_DETECTION_CONFIDENCE = 0.65
     │     - MIN_TRACKING_CONFIDENCE = 0.60
     │     - EYE_ASPECT_RATIO_THRESHOLD = 0.16
     │     - etc. (all optimized for Intel)
     └─ Now config property accesses return Intel values

================================================================================
SECOND 1: LIFESPAN STARTUP (app/main.py lines 49-101)
================================================================================

Line 49-101: @asynccontextmanager async def lifespan(app: FastAPI)
  │
  └─ STARTUP PHASE:
     │
     ├─ Line 65: init_database(PROCTORING_DB_CONFIG)
     │  └─ Connects to MySQL, tables already exist
     │
     ├─ Line 78: _active_client = (os.getenv("DETECTION_CLIENT") or "mediapipe").strip().lower()
     │  └─ Value: "intel"
     │
     ├─ Line 79-91: if _active_client == "intel":
     │  │
     │  ├─ Line 83: _models_ok = ensure_intel_models()
     │  │  │
     │  │  └─ CALL ensure_intel_models() in model_downloader.py:
     │  │     │
     │  │     ├─ Line 230-235: _all_present()
     │  │     │  ├─ Check: Does intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml exist?
     │  │     │  ├─ Check: Does intel/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002.xml exist?
     │  │     │  ├─ Check: Does intel/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml exist?
     │  │     │  ├─ Check: Does intel/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml exist?
     │  │     │  └─ If all 4 exist: Return True → Line 231-234 → Log "already present" → Return True
     │  │     │
     │  │     ├─ If NOT all present:
     │  │     │  │
     │  │     │  ├─ Line 237: precision = _precision() → os.getenv("INTEL_PRECISION") → "FP32"
     │  │     │  ├─ Line 238: output_root = _models_root()
     │  │     │  │  ├─ os.getenv("INTEL_MODELS_DIR") → None (not set)
     │  │     │  │  └─ return Path(__file__).parent
     │  │     │  │     └─ = /vercel/share/v0-project/app/client/detection/face_tracking/
     │  │     │  │
     │  │     │  ├─ Line 245-253: Try omz_downloader CLI
     │  │     │  │  ├─ _omz_cli_available() → shutil.which("omz_downloader")
     │  │     │  │  │  └─ If installed (from openvino-dev): True
     │  │     │  │  │  └─ If not installed: False → Skip to HTTPS fallback
     │  │     │  │  │
     │  │     │  │  └─ IF available:
     │  │     │  │     ├─ For each model in _MODELS:
     │  │     │  │     │  ├─ Run: omz_downloader --name face-detection-adas-0001 --precision FP32 --output_dir /...
     │  │     │  │     │  ├─ Run: omz_downloader --name facial-landmarks-35-adas-0002 --precision FP32 --output_dir /...
     │  │     │  │     │  ├─ Run: omz_downloader --name head-pose-estimation-adas-0001 --precision FP32 --output_dir /...
     │  │     │  │     │  └─ Run: omz_downloader --name gaze-estimation-adas-0002 --precision FP32 --output_dir /...
     │  │     │  │     │
     │  │     │  │     ├─ If all succeed and _all_present(): Return True
     │  │     │  │     └─ Else: Continue to fallback
     │  │     │  │
     │  │     │  ├─ Line 256-259: Direct HTTPS fallback
     │  │     │  │  ├─ For each model:
     │  │     │  │  │  ├─ URL: https://storage.openvinotoolkit.org/repositories/open_model_zoo/models_slides_demos/models/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml
     │  │     │  │  │  ├─ Download to: /vercel/share/v0-project/app/client/detection/face_tracking/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml
     │  │     │  │  │  └─ (repeat for .bin and other 3 models)
     │  │     │  │  │
     │  │     │  │  └─ If all download successfully and _all_present(): Return True
     │  │     │  │
     │  │     │  └─ If any failure: Log error with manual install commands → Return False
     │  │     │
     │  │     └─ RESULT: True if all models present, False otherwise
     │  │
     │  └─ Back in main.py:
     │     ├─ If _models_ok: Log "Intel models ready."
     │     └─ Else: Log "One or more Intel models could not be downloaded."
     │        └─ Application CONTINUES anyway (Intel client will return None on gaze calls)
     │
     ├─ Line 94: start_scheduler()
     │  └─ Start background scheduler for processing queue
     │
     ├─ Line 97: yield ← APPLICATION READY FOR REQUESTS
     │
     └─ Line 100-101: SHUTDOWN (when server stops)
        └─ stop_scheduler()

================================================================================
SECOND 2: FIRST VIDEO PROCESSING REQUEST
================================================================================

Request: POST /v1/proctoring/detect with video file
  │
  └─ VIDEO PROCESSING SERVICE INITIALIZATION:
     │
     ├─ VideoProcessingService.__init__(config)
     │  ├─ self.config = config
     │  └─ self._detection_service = DetectionService(config)
     │     └─ DetectionService.__init__(config)
     │        ├─ Line 37-40: super().__init__(config)
     │        └─ Line 46: _setup()
     │           │
     │           ├─ Line 50: self._detection_client = get_detection_client(c)
     │           │  │
     │           │  └─ CALL get_detection_client(config) in factory.py:
     │           │     │
     │           │     ├─ Line 77: raw = (os.getenv("DETECTION_CLIENT") or "mediapipe").strip().lower()
     │           │     │  └─ Value: "intel"
     │           │     │
     │           │     ├─ Line 78: loader = _REGISTRY.get(raw)
     │           │     │  ├─ _REGISTRY["intel"] = _load_intel_class (line 65)
     │           │     │  └─ loader = function reference
     │           │     │
     │           │     ├─ Line 82: client_class = loader()
     │           │     │  ├─ CALL _load_intel_class():
     │           │     │  │  └─ from app.client.detection.face_tracking.intel_client import IntelDetectionClient
     │           │     │  │     └─ IMPORTS IntelDetectionClient CLASS HERE (lazy load)
     │           │     │  └─ return IntelDetectionClient
     │           │     │
     │           │     └─ Line 83: client = client_class(config)
     │           │        ├─ CALL IntelDetectionClient(config):
     │           │        │
     │           │        ├─ BaseDetectionClient.__init__(config) (line 38-40 in base.py):
     │           │        │  ├─ Line 39: self.config = config
     │           │        │  └─ Line 40: self._setup()
     │           │        │
     │           │        └─ IntelDetectionClient._setup() (lines 267-355 in intel_client.py):
     │           │           │
     │           │           ├─ Line 269-270: Core initialization
     │           │           │  ├─ from openvino import Core
     │           │           │  └─ core = Core()
     │           │           │
     │           │           ├─ Line 273: model_paths = get_model_paths()
     │           │           │  ├─ CALL get_model_paths() in model_downloader.py (lines 276-298):
     │           │           │  │  ├─ For each of 4 models:
     │           │           │  │  │  ├─ xml = model_xml("face-detection-adas-0001")
     │           │           │  │  │  │  └─ = _models_root() / "intel" / "face-detection-adas-0001" / "FP32" / "face-detection-adas-0001.xml"
     │           │           │  │  │  │  └─ = /vercel/.../face_tracking/intel/face-detection-adas-0001/FP32/face-detection-adas-0001.xml
     │           │           │  │  │  │
     │           │           │  │  │  ├─ if NOT xml.exists():
     │           │           │  │  │  │  └─ Log error → return None
     │           │           │  │  │  │
     │           │           │  │  │  └─ paths[key] = str(xml)
     │           │           │  │  │
     │           │           │  │  └─ return {"face_det": "/...xml", "landmarks": "/.../xml", "head_pose": "/...xml", "gaze": "/...xml"}
     │           │           │  │
     │           │           │  └─ IF model_paths is None:
     │           │           │     └─ Log error, set self._available = False, return
     │           │           │
     │           │           ├─ Line 312-346: Load and compile models
     │           │           │  │
     │           │           │  ├─ Line 312-314: Face detector
     │           │           │  │  ├─ model = core.read_model(model=model_paths["face_det"])
     │           │           │  │  ├─ compiled = core.compile_model(model=model, device_name="CPU")
     │           │           │  │  └─ self._face_detector = _FaceDetector(core, model_paths["face_det"], "CPU", 0.5)
     │           │           │  │
     │           │           │  ├─ Line 318-320: Landmarks detector
     │           │           │  │  └─ self._landmarks_detector = _FacialLandmarksDetector(...)
     │           │           │  │
     │           │           │  ├─ Line 324-326: Head pose estimator
     │           │           │  │  └─ self._head_pose_estimator = _HeadPoseEstimator(...)
     │           │           │  │
     │           │           │  └─ Line 330-332: Gaze estimator
     │           │           │     └─ self._gaze_estimator = _GazeEstimator(...)
     │           │           │
     │           │           ├─ Line 333: self._available = True
     │           │           │  └─ Client fully initialized, ready to use
     │           │           │
     │           │           └─ Line 340-345: Initialize Kalman smoothing
     │           │              └─ self._init_kalman_filters()
     │           │
     │           └─ Back to factory.py line 84-85:
     │              └─ Log "Detection client initialised: IntelDetectionClient"
     │              └─ return client
     │
     ├─ Back to DetectionService._setup() line 50
     │  └─ self._detection_client is now fully initialized IntelDetectionClient
     │
     └─ VideoProcessingService now ready to process video frames

================================================================================
SECOND 3+: FRAME-BY-FRAME GAZE DETECTION
================================================================================

For each video frame:
  │
  └─ VideoProcessingService._process_frame(frame):
     │
     ├─ Line (from detection_service.py 115): gaze_result = detection_service.detect_gaze(frame)
     │  │
     │  └─ DetectionService.detect_gaze(frame):
     │     │
     │     └─ Line 115: return self._detection_client.detect_gaze(frame)
     │        │
     │        └─ IntelDetectionClient.detect_gaze(frame) (lines 436-545):
     │           │
     │           ├─ Line 442: if not self._available: return None, None, 0, None, 0.0, 0.0
     │           │  └─ (If models weren't loaded, return gracefully)
     │           │
     │           ├─ Line 449: boxes = self._face_detector.detect(frame)
     │           │  ├─ Runs inference on frame with face-detection-adas-0001
     │           │  ├─ threshold = 0.5 (hardcoded in _FaceDetector.__init__)
     │           │  └─ Returns list of (x1, y1, x2, y2) bounding boxes
     │           │
     │           ├─ Line 454-456: if num_faces == 0: return None results
     │           │
     │           ├─ Line 468: yaw_d, pitch_d, roll_d = self._head_pose_estimator.estimate(face_crop)
     │           │  └─ Runs inference on face crop with head-pose-estimation-adas-0001
     │           │
     │           ├─ Line 485: le_center, re_center, _ = self._landmarks_detector.detect(face_crop)
     │           │  └─ Runs inference on face crop with facial-landmarks-35-adas-0002
     │           │
     │           ├─ Line 499-500: gaze_vec = self._gaze_estimator.estimate(le_img, re_img, head_pose_deg)
     │           │  └─ Runs inference on eye crops with gaze-estimation-adas-0002
     │           │  └─ Returns gaze vector (x, y, z)
     │           │
     │           ├─ Line 507-520: Convert gaze vector to degrees using _gaze_vec_to_degrees()
     │           │  ├─ h_deg = atan2(gx, -gz) in degrees
     │           │  ├─ v_deg = atan2(-gy, sqrt(gx²+gz²)) in degrees
     │           │  └─ conf = clipped norm of gaze_vec[:2]
     │           │
     │           ├─ Line 521-527: Apply Kalman smoothing
     │           │  ├─ h_angle, v_angle = self._smooth_gaze_with_kalman(h_angle, v_angle)
     │           │  └─ Returns smoothed gaze angles
     │           │
     │           └─ Line 540: return (h_angle, v_angle, num_faces, bbox_center, confidence, occlusion)
     │              └─ Example: (5.23, 3.45, 1, (640, 480), 0.92, 0.05)
     │
     ├─ Back to VideoProcessingService:
     │  └─ violation_tracker.update(gaze_result, config thresholds)
     │     ├─ Compare gaze angles against config.GAZE_HORIZONTAL_THRESHOLD (8.5 for Intel)
     │     ├─ Compare gaze angles against config.GAZE_VERTICAL_THRESHOLD (8.5 for Intel)
     │     └─ Track violations
     │
     └─ Store results in database

================================================================================
RESULT:
Accurate face detection, head pose, and gaze estimation using Intel models
with thresholds optimized for Intel adas models (IntelDetectionConfig).

All models loaded from: app/client/detection/face_tracking/intel/*/FP32/
All thresholds dynamically selected based on config.DETECTION_CLIENT = "intel"
"""
