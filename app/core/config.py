import os
from typing import Optional

try:
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        from pydantic import BaseSettings
    except ImportError:
        class BaseSettings:
            pass


class Settings(BaseSettings):
    PROJECT_NAME: str = "OCR Document Intelligence"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./ocr_app.db")
    POPPLER_PATH: Optional[str] = None

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()