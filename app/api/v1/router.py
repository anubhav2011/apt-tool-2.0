"""
Central router configuration
Consolidates all API v1 endpoints
"""
from fastapi import APIRouter

from .endpoints import health, proctoring, reports

# Create main API router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(proctoring.router, prefix="/proctoring", tags=["Proctoring"])
api_router.include_router(reports.router, prefix="/reports", tags=["Reports"])
