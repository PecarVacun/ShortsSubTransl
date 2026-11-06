"""Configuration for the subtitle localization pipeline."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Runtime settings shared by services and the business-logic pipeline."""

    deepl_api_key: str
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:70b-instruct-q4_K_M"
    ollama_context_tokens: int = 8192
    ollama_output_tokens: int = 2048
    ollama_input_bytes: int = 4096
    request_timeout_seconds: int = 300
    max_line_characters: int = 38
    max_lines: int = 2
    minimum_duration_ms: int = 1000

    @classmethod
    def from_env(cls, env_file: Path = Path(".env")) -> "Settings":
        """Load .env values without overriding variables already in the environment."""
        load_dotenv(dotenv_path=env_file, override=False)
        return cls(
            deepl_api_key=os.getenv("DEEPL_API_KEY", "").strip(),
            ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
            ollama_model=os.getenv(
                "OLLAMA_MODEL", "llama3.1:70b-instruct-q4_K_M"
            ).strip(),
        )
