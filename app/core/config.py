import os
from typing import Optional

try:
    # Pydantic v2 standard import
    from pydantic_settings import BaseSettings
except ImportError:
    try:
        # Pydantic v1 fallback
        from pydantic import BaseSettings
    except ImportError:
        raise ImportError(
            "pydantic-settings or pydantic is required. "
            "Run 'pip install pydantic-settings' to fix this."
        )


class Settings(BaseSettings):
    PROJECT_NAME: str = "OCR Document Intelligence"
    
    # Unified SQLite database filename across all modules
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./ocr_platform.db")
    
    # Executable dependencies path overrides
    POPPLER_PATH: Optional[str] = os.getenv("POPPLER_PATH", None)

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()