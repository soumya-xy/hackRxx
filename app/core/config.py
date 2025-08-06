import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GOOGLE_API_KEY: str # Changed from OPENAI_API_KEY
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str
    BEARER_TOKEN: str
    DATABASE_URL: str

    class Config:
        env_file = ".env"

settings = Settings()