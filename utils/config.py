"""Application configuration via pydantic-settings."""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4"
    embedding_model: str = "text-embedding-3-small"

    # Database
    database_url: str = "sqlite:///data/research.db"

    # Paths
    faiss_index_path: str = "data/faiss_index"
    upload_dir: str = "data/uploads"
    export_dir: str = "data/exports"

    # RAG
    chunk_size: int = 1000
    chunk_overlap: int = 200
    rag_top_k: int = 5

    # Logging
    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        """Create required data directories if they don't exist."""
        for path in [self.faiss_index_path, self.upload_dir, self.export_dir]:
            Path(path).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
