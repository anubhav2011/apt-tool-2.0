"""
FastAPI Backend for AI Proctoring System
Production-ready, CPU-optimized, enterprise-grade
Restructured with clean architecture
"""
import sys
import os
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from dotenv import load_dotenv
load_dotenv()

# Before scheduler → mediapipe / TensorFlow Lite: cut native C++ stderr noise
# (xnnpack default, inference_feedback_manager, TFLite delegate INFO, etc.).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

# OpenCV FFmpeg: avoid premature decode failure on some MP4s/multi-stream files.
# Must be set before any module imports `cv2`.
os.environ.setdefault("OPENCV_FFMPEG_READ_ATTEMPTS", os.getenv("OPENCV_FFMPEG_READ_ATTEMPTS") or "20000")

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from contextlib import asynccontextmanager
import uvicorn

from app.utils.logger import debug_logger, is_debug_env
from app.core.config import ProctoringConfig, PROCTORING_DB_CONFIG
from app.core.database import init_database
from app.api.v1.router import api_router
from app.core.scheduler import start_scheduler, stop_scheduler
from app.core.dependencies import verify_user
from app.client.detection.face_tracking.model_downloader import ensure_intel_models

config = ProctoringConfig()

APP_VERSION = (os.getenv("APP_VERSION") or "1.0.0").strip()


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan (startup + shutdown)
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    # ── Startup ──────────────────────────────────────────────────────────
    debug_logger.info(
        "AI Proctoring System v2.0 starting up… | version=%s | debug=%s",
        APP_VERSION,
        is_debug_env(),
    )
    debug_logger.info(
        f"CPU-optimized | FPS={config.TARGET_FPS} | "
        f"Max Frame={config.MAX_FRAME_DIMENSION}px"
    )

    # Database — engine + sessions only; no CREATE DATABASE / create_all (tables already exist)
    try:
        init_database(PROCTORING_DB_CONFIG)
        # from app.models.proctoring import (
        #     ProctoringReport,
        #     ProctoringEventLog,
        #     ProctoringEventSummary,
        # )
        # create_tables()
        debug_logger.info("Database connection ready (ORM fetch/update)")
    except Exception:
        debug_logger.exception("Database initialization failed")
        raise

    # Intel OpenVINO models — download on first start when selected
    _active_client = (os.getenv("DETECTION_CLIENT") or "mediapipe").strip().lower()
    if _active_client == "intel":
        debug_logger.info(
            "DETECTION_CLIENT=intel detected — ensuring Intel models are present …"
        )
        _models_ok = ensure_intel_models()
        if _models_ok:
            debug_logger.info("Intel models ready.")
        else:
            debug_logger.warning(
                "One or more Intel models could not be downloaded. "
                "The Intel detection client will return None values until "
                "all model files are available."
            )

    # Scheduler (failures logged with traceback in debug_logs via start_scheduler)
    start_scheduler()
    # debug_logger.info("Background scheduler started")

    yield  # ── application runs here ──

    # ── Shutdown ─────────────────────────────────────────────────────────
    debug_logger.info("AI Proctoring System shutting down…")
    stop_scheduler()


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Proctoring System API",
    description=(
        "CPU-optimized video proctoring with count-based detection, "
        "database storage, and automatic retry scheduler."
    ),
    version=APP_VERSION,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ───────────────────────────────────────────────────────────────────
app.include_router(api_router, prefix="/api/v1", dependencies=[Depends(verify_user)])


@app.get("/openapi.json", include_in_schema=False, dependencies=[Depends(verify_user)])
async def openapi_json():
    return app.openapi()


@app.get("/docs", include_in_schema=False, dependencies=[Depends(verify_user)])
async def docs():
    return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{app.title} - Docs")


@app.get("/redoc", include_in_schema=False, dependencies=[Depends(verify_user)])
async def redoc():
    return get_redoc_html(openapi_url="/openapi.json", title=f"{app.title} - ReDoc")

# ── Uvicorn entry ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("APP_PORT", "8012"))
    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=port,
        reload=False,
        log_level="debug" if is_debug_env() else "info",
    )
