from dotenv import load_dotenv
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# load .env early; pydantic-settings will also read it
load_dotenv()


class Settings(BaseSettings):
    # configuration for pydantic-settings v2
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # general
    ENV: str = "development"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # paths
    MODEL_PATH: str = "./models/best.pt"
    DATA_YAML_PATH: str = "./data/data.yaml"
    KB_PATH: str = "data/kb.json"

    # OpenAI / LLM settings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_LLM_MODEL: str = Field(
        "gpt-4.1-mini",
        env=("OPENAI_LLM_MODEL", "OPENAI_MODEL"),
    )
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # rag flags
    RAG_ENABLED: bool = True
    RAG_TOP_K: int = 6

    # detector thresholds
    CONFIDENCE_THRESHOLD: float = 0.35
    IOU_DUPLICATE_THRESHOLD: float = 0.5
    MAX_IMAGE_MB: int = 10

    def require_openai_key(self) -> str:
        """Return a non-empty OpenAI API key or raise a clear error.
        The key is stripped of whitespace.  This method allows imports to
        succeed even when the environment variable is missing; callers can
        invoke it when the key is actually needed.
        """
        key = self.OPENAI_API_KEY
        if not key or (isinstance(key, str) and not key.strip()):
            raise ValueError("OPENAI_API_KEY não configurada. Defina no arquivo .env")
        return key.strip()


# singleton instance
settings = Settings()
