"""Tests for lossless bounded screenplay chunking."""

from unittest.mock import Mock, patch

import pytest

from config import Settings
from services.llm_service import LLMService, LLMServiceError, _chunks_by_utf8_bytes


def test_chunking_preserves_unicode_text() -> None:
    text = "Актёр 🙂 speaks. " * 10
    chunks = _chunks_by_utf8_bytes(text, 20)
    assert "".join(chunks) == text
    assert all(len(chunk.encode("utf-8")) <= 20 for chunk in chunks)


def test_rejects_empty_script_without_calling_ollama() -> None:
    service = LLMService(Settings(deepl_api_key=""))
    with patch("services.llm_service.requests.post") as post:
        with pytest.raises(LLMServiceError, match="empty script"):
            service.analyze_script(" \n ")
    post.assert_not_called()


def test_keeps_all_chunk_analyses() -> None:
    service = LLMService(
        Settings(
            deepl_api_key="",
            ollama_input_bytes=300,
            ollama_context_tokens=8192,
            ollama_output_tokens=2048,
        )
    )
    response = Mock()
    response.json.return_value = {
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 100,
        "eval_count": 20,
        "message": {"content": '{"metadata": {}, "characters": []}'},
    }
    response.raise_for_status.return_value = None
    with patch("services.llm_service.requests.post", return_value=response) as post:
        analysis = service.analyze_script("A" * 500)
    assert len(analysis.chunks) > 1
    assert len(analysis.chunks) == post.call_count
