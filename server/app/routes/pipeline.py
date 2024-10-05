from fastapi import APIRouter, HTTPException, Depends
from server.app.models import PipelineRequest
from server.app.services.pipeline_service import run_pipeline_service

router = APIRouter()


@router.post("/run_pipeline")
async def run_pipeline(request: PipelineRequest):
    try:
        result = run_pipeline_service(request.yaml_config)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
