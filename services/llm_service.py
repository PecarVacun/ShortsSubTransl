"""Create screenplay metadata and character cards using the Ollama API."""

import json
from dataclasses import dataclass
from typing import Any

import requests

from config import Settings


class LLMServiceError(RuntimeError):
    """Ollama failed or returned incomplete or invalid data."""


@dataclass(frozen=True)
class ScriptAnalysis:
    """Analysis of each script chunk, retained to avoid losing script content."""

    chunks: list[dict[str, Any]]

    def to_json(self) -> str:
        """Serialize all chunk analyses for storage and translation context."""
        return json.dumps({"chunks": self.chunks}, ensure_ascii=False, indent=2)


_INSTRUCTIONS = (
    "Analyze this screenplay excerpt. Return a JSON object with keys "
    '"metadata" (an object) and "characters" (an array of objects). '
    "Include only facts supported by the excerpt; preserve original names. "
    "Character cards should include name, description, relationships and "
    "relevant terminology when available. Do not invent missing information.\n\n"
    "SCREENPLAY EXCERPT:\n"
)


def _chunks_by_utf8_bytes(text: str, limit: int) -> list[str]:
    """Split text at Unicode character boundaries without dropping content."""
    if limit < 4:
        raise ValueError("Script chunk limit must be at least four bytes.")
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for character in text:
        char_size = len(character.encode("utf-8"))
        if size + char_size > limit:
            chunks.append("".join(current))
            current, size = [], 0
        current.append(character)
        size += char_size
    if current:
        chunks.append("".join(current))
    return chunks


class LLMService:
    """Analyze the entire script in bounded chunks using a local Ollama model."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def analyze_script(self, text: str) -> ScriptAnalysis:
        """Return validated per-chunk analysis or raise instead of truncating."""
        if not text.strip():
            raise LLMServiceError("Cannot analyze an empty script.")
        settings = self.settings
        available_prompt_bytes = settings.ollama_context_tokens - settings.ollama_output_tokens
        if (
            settings.ollama_context_tokens <= 0
            or settings.ollama_output_tokens <= 0
            or settings.ollama_input_bytes <= 0
            or settings.ollama_output_tokens >= settings.ollama_context_tokens
        ):
            raise LLMServiceError("Ollama context and output token budgets are invalid.")
        input_limit = min(
            settings.ollama_input_bytes,
            available_prompt_bytes - len(_INSTRUCTIONS.encode("utf-8")),
        )
        if input_limit < 4:
            raise LLMServiceError("Ollama input budget is too small for the prompt.")
        chunks = _chunks_by_utf8_bytes(text, input_limit)
        results: list[dict[str, Any]] = []
        for index, chunk in enumerate(chunks, 1):
            payload = {
                "model": settings.ollama_model,
                "messages": [{"role": "user", "content": _INSTRUCTIONS + chunk}],
                "stream": False,
                "format": "json",
                "options": {
                    "num_ctx": settings.ollama_context_tokens,
                    "num_predict": settings.ollama_output_tokens,
                    "temperature": 0.2,
                },
            }
            try:
                response = requests.post(
                    f"{settings.ollama_url}/api/chat",
                    json=payload,
                    timeout=settings.request_timeout_seconds,
                )
                response.raise_for_status()
                data = response.json()
                if data.get("done") is not True or data.get("done_reason") == "length":
                    raise LLMServiceError(f"Ollama output for chunk {index} was incomplete.")
                prompt_tokens = data.get("prompt_eval_count")
                output_tokens = data.get("eval_count")
                if not isinstance(prompt_tokens, int) or not isinstance(output_tokens, int):
                    raise LLMServiceError(
                        f"Ollama did not confirm token counts for chunk {index}."
                    )
                if prompt_tokens + output_tokens > settings.ollama_context_tokens:
                    raise LLMServiceError(
                        f"Ollama context limit was exceeded for chunk {index}."
                    )
                result = json.loads(data["message"]["content"])
                if (
                    not isinstance(result, dict)
                    or not isinstance(result.get("metadata"), dict)
                    or not isinstance(result.get("characters"), list)
                    or not all(isinstance(card, dict) for card in result["characters"])
                ):
                    raise LLMServiceError(f"Invalid analysis structure for chunk {index}.")
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                raise LLMServiceError(
                    f"Ollama failed on script chunk {index}: {exc}"
                ) from exc
            results.append(result)
        return ScriptAnalysis(chunks=results)
