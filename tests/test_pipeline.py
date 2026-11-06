"""Tests for end-to-end orchestration without external API calls."""

from pathlib import Path
from unittest.mock import Mock

import pysrt
import pytest

from config import Settings
from core.pipeline import PipelineError, run_localization_pipeline


def test_pipeline_writes_localized_srt_only_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = tmp_path / "script.pdf"
    source = tmp_path / "source.srt"
    script.touch()
    source.write_text(
        "1\n00:00:00,000 --> 00:00:00,500\nHello there\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "core.pipeline.Settings.from_env",
        lambda: Settings(deepl_api_key="test-key"),
    )
    monkeypatch.setattr("core.pipeline.extract_script_text", lambda _: "Screenplay")
    analysis = Mock()
    analysis.to_json.return_value = '{"chunks": []}'
    monkeypatch.setattr("core.pipeline.LLMService.analyze_script", lambda _, text: analysis)
    monkeypatch.setattr(
        "core.pipeline.TranslatorService.translate",
        lambda _, text, language, context: "Bonjour",
    )

    events = list(run_localization_pipeline("", script, source, "fr"))
    output = tmp_path / "source.FR.srt"
    parsed = pysrt.open(str(output), encoding="utf-8")

    assert [event.stage for event in events][-1] == "complete"
    assert events[-1].output_path == output
    assert parsed[0].text == "Bonjour"
    assert parsed[0].end.ordinal == 1000


def test_pipeline_fails_before_services_when_api_key_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, source = tmp_path / "script.pdf", tmp_path / "source.srt"
    script.touch()
    source.touch()
    monkeypatch.setattr(
        "core.pipeline.Settings.from_env",
        lambda: Settings(deepl_api_key=""),
    )
    extract = Mock()
    monkeypatch.setattr("core.pipeline.extract_script_text", extract)

    with pytest.raises(PipelineError, match="DEEPL_API_KEY"):
        list(run_localization_pipeline("", script, source, "RU"))
    extract.assert_not_called()


def test_pipeline_does_not_write_output_if_formatting_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script, source = tmp_path / "script.pdf", tmp_path / "source.srt"
    script.touch()
    source.write_text(
        "1\n00:00:00,000 --> 00:00:01,500\nHello there\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "core.pipeline.Settings.from_env",
        lambda: Settings(deepl_api_key="test-key"),
    )
    monkeypatch.setattr("core.pipeline.extract_script_text", lambda _: "Screenplay")
    analysis = Mock()
    analysis.to_json.return_value = "{}"
    monkeypatch.setattr("core.pipeline.LLMService.analyze_script", lambda _, text: analysis)
    monkeypatch.setattr(
        "core.pipeline.TranslatorService.translate",
        lambda _, text, language, context: "x" * 39,
    )

    with pytest.raises(PipelineError, match="Subtitle formatting failed"):
        list(run_localization_pipeline("", script, source, "RU"))
    assert not (tmp_path / "source.RU.srt").exists()
