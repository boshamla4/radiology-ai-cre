from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite+aiosqlite:///./radiology.db"
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llava"
    CHROMA_PATH: str = "./chroma_db"
    MODELS_PATH: str = "../models"
    DATASET_PATH: str = "../DATASET"
    ENVIRONMENT: str = "development"
    SECRET_KEY: str = "change-me-in-production"

    class Config:
        env_file = ".env"


settings = Settings()
BASE_DIR = Path(__file__).resolve().parent.parent
