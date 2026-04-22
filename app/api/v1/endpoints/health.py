"""
Health Check Endpoints
"""
from fastapi import APIRouter, Depends
from app.core.scheduler import get_scheduler_status

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check — includes scheduler status."""
    scheduler = get_scheduler_status()
    return {
        "status": "healthy",
        "version": "2.0.0",
        "components": {
            "api":              "operational",
            "video_processor":  "operational",
            "database":         "operational",
            "mediapipe":        "operational",
            "opencv":           "operational",
            "scheduler":        "running" if scheduler["running"] else "stopped",
        },
        "scheduler": scheduler,
    }


@router.get("/")
async def root():
    """Root endpoint."""
    scheduler = get_scheduler_status()
    return {
        "status": "operational",
        "version": "2.0.0",
        "components": {
            "api":               "healthy",
            "video_processor":   "ready",
            "database":          "configured",
            "detection_engine":  "count_based_ready",
            "scheduler":         "running" if scheduler["running"] else "stopped",
        },
        "scheduler": scheduler,
    }