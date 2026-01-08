from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    APP_NAME: str = "RAG Chatbot API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Groq
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = ""

    # CORS
    ALLOWED_ORIGINS: List[str] = ["*"]

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    PDF_DIRECTORY: Path = BASE_DIR / "data" / "pdfs"
    VECTOR_DB_PATH: Path = BASE_DIR / "vectordb"
    CHAT_HISTORY_DB_PATH: Path = DATA_DIR / "chat_history.db"

    # Vector Store
    COLLECTION_NAME: str = "pdf_documents"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # RAG
    SIMILARITY_TOP_K: int = 3
    SIMILARITY_THRESHOLD: float = 0.35


    # Dynamic threshold settings
    SHORT_QUERY_THRESHOLD:float = 0.7
    COMPLEX_QUERY_THRESHOLD:float  = 0.5
    FACT_QUERY_THRESHOLD:float  = 0.8
    DEFINITION_QUERY_THRESHOLD :float = 0.7

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True
    )


settings = Settings()

settings.PDF_DIRECTORY.mkdir(parents=True, exist_ok=True)
settings.VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
