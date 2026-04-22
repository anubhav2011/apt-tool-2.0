"""
Proctoring Endpoints
Request/response handling only - all logic in services
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks

from app.schemas.models import ProcessVideoRequest
from app.services.base_service import IProctoringService
from app.services.proctoring_service import ProctoringService
from app.core.dependencies import get_proctoring_service
from app.core.config import S3_CONFIG
from app.core.exceptions import VideoProcessingError, DatabaseError, ValidationError
from app.services.proctoring_processing.s3_video_queue_service import S3VideoQueueService

from app.services.background_task import run_process_video_by_interview_id

router = APIRouter()

@router.post("/process-video")
async def process_video_file(
    video: UploadFile = File(...),
    interview_id: str = Form(..., description="Interview ID"),
    proctoring_service: IProctoringService = Depends(get_proctoring_service)
):
    """
    Process uploaded video file for proctoring analysis

    """
    
    vid = interview_id.strip()
    if not vid:
        raise HTTPException(
            status_code=400,
            detail={"response": "Interview ID is not provided", "code": 4003},
        )
    try:
        result = await proctoring_service.process_video_upload(
            video_file=video,
            interview_id=vid,
        )
        return result

    except ValidationError as e:
        raise HTTPException(status_code=400, detail={"response": e.message, "code": e.code})
    except VideoProcessingError as e:
        raise HTTPException(status_code=500, detail={"response": e.message, "code": e.code})
    except DatabaseError as e:
        raise HTTPException(status_code=500, detail={"response": e.message, "code": e.code})
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "response": "An unexpected error occurred",
            "code": 5000,
            "error": str(e)
        })



@router.post("/process-video-by-interview-id", status_code=202)
async def process_video_by_interview_id(
    request: ProcessVideoRequest,
    background_tasks: BackgroundTasks,
    proctoring_service: ProctoringService = Depends(get_proctoring_service),
):
    vid = request.interview_id

    if not vid:
        raise HTTPException(
            status_code=400,
            detail={"response": "interview_id is required", "code": 4003},
        )

    candidate_name = proctoring_service.status_repository.get_candidate_name_for_interview(vid)

    if not candidate_name:
        raise HTTPException(
            status_code=400,
            detail={"response": "No candidate_name for this interview_id", "code": 4004},
        )

    s3_queue = S3VideoQueueService(s3_config=S3_CONFIG)
    if not s3_queue.object_exists(vid, candidate_name):
        raise HTTPException(
            status_code=404,
            detail={"response": "Video file not found in S3", "code": 4005},
        )

    background_tasks.add_task(run_process_video_by_interview_id, vid)

    return {"message": "Processing started"}