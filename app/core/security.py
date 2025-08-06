from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from app.core.config import settings

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if not api_key or not api_key.startswith("Bearer "):
        raise HTTPException(
            status_code=403, detail="Invalid or missing Authorization header"
        )
    token = api_key.split("Bearer ")[1]
    if token == settings.BEARER_TOKEN:
        return token
    else:
        raise HTTPException(status_code=403, detail="Could not validate credentials")