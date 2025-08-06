from fastapi import APIRouter, Body, Depends
from app.api.v1.schemas import RunRequest, RunResponse
from app.core.security import get_api_key
from app.services.query_processor import QueryProcessor

router = APIRouter()
query_processor = QueryProcessor()

@router.post("/hackrx/run", tags=["hackrx"]) # Removed response_model to strictly match problem statement's string list output
async def run_submission( # Made the endpoint function async
    request: RunRequest,
    api_key: str = Depends(get_api_key)
):
    """
    Processes documents and answers questions using the LLM-powered retrieval system.
    """
    
    # Run the entire pipeline, which is now an async operation
    results = await query_processor.run_pipeline(
        document_url=str(request.documents), 
        questions=request.questions
    )
    
    # The query_processor.run_pipeline now returns the simplified answers directly
    return results

