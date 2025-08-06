from fastapi import FastAPI
from app.api.v1.endpoints import router as api_router_v1
from app.core.config import settings

def create_app():
    app = FastAPI(
        title="LLM-Powered Query-Retrieval System",
        description="A system for processing documents and answering questions.",
        version="1.0.0",
        openapi_url="/api/v1/openapi.json"
    )

    app.include_router(api_router_v1, prefix="/api/v1")

    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)