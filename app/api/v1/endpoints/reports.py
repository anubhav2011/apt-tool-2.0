"""
Report Management Endpoints
Request/response handling only - all logic in services
"""
from fastapi import APIRouter, HTTPException, Depends

from app.services.base_service import IProctoringService
from app.core.dependencies import get_proctoring_service
from app.core.exceptions import ReportNotFoundError, DatabaseError

router = APIRouter()


@router.get("/{interview_id}")
async def get_report(
    interview_id: str,
    proctoring_service: IProctoringService = Depends(get_proctoring_service)
):
    """
    Retrieve proctoring report by interview ID
    """
    try:
        result = await proctoring_service.get_report(interview_id)
        return result

    except ReportNotFoundError as e:
        raise HTTPException(status_code=404, detail={"response": e.message, "code": e.code})
    except DatabaseError as e:
        raise HTTPException(status_code=500, detail={"response": e.message, "code": e.code})
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "response": "An unexpected error occurred",
            "code": 5000,
            "error": str(e)
        })


@router.delete("/{interview_id}")
async def delete_report(
    interview_id: str,
    proctoring_service: IProctoringService = Depends(get_proctoring_service)
):
    """
    Delete proctoring report (GDPR compliance)
    """
    try:
        result = await proctoring_service.delete_report(interview_id)
        return result

    except ReportNotFoundError as e:
        raise HTTPException(status_code=404, detail={"response": e.message, "code": e.code})
    except DatabaseError as e:
        raise HTTPException(status_code=500, detail={"response": e.message, "code": e.code})
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "response": "An unexpected error occurred",
            "code": 5000,
            "error": str(e)
        })
