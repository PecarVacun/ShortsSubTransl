"""Orchestrate script analysis, subtitle translation and SRT formatting."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import pysrt

from config import Settings
from core.subtitle_formatter import SubtitleFormatError, format_subtitles
from services.llm_service import LLMService, LLMServiceError
from services.pdf_extractor import PDFExtractionError, extract_script_text
from services.translator_service import TranslationError, TranslatorService


class PipelineError(RuntimeError):
    """The localization pipeline could not complete safely."""


@dataclass(frozen=True)
class PipelineProgress:
    """A UI-independent progress event emitted by the pipeline."""

    stage: str
    progress: float
    message: str
    output_path: Path | None = None


def _atomic_save(subtitles: pysrt.SubRipFile, output_path: Path) -> None:
    """Write a complete SRT atomically so a failed run leaves no partial output."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".srt",
            prefix=f".{output_path.stem}.",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        subtitles.save(str(temporary_path), encoding="utf-8")
        os.replace(temporary_path, output_path)
    except OSError as exc:
        raise PipelineError(f"Could not write localized subtitle file: {output_path}") from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def run_localization_pipeline(
    video_path: str | Path,
    script_path: str | Path,
    srt_path: str | Path,
    target_lang: str,
) -> Generator[PipelineProgress, None, None]:
    """Yield progress while creating a translated, broadcast-formatted SRT.

    ``video_path`` is accepted for UI/workflow compatibility; this pipeline
    does not inspect video content. Screenplay text is processed in bounded
    chunks by Ollama, then all chunks' character cards and metadata are sent
    to DeepL as advisory context. DeepL context is not a terminology glossary.
    """
    video = Path(video_path) if video_path else None
    script, source_srt = Path(script_path), Path(srt_path)
    language = target_lang.strip().upper()
    if not re.fullmatch(r"[A-Z0-9_-]{2,12}", language):
        raise PipelineError("Target language must be a language code such as RU or EN-US.")
    if not script.is_file():
        raise PipelineError(f"Script PDF not found: {script}")
    if not source_srt.is_file():
        raise PipelineError(f"Subtitle file not found: {source_srt}")

    settings = Settings.from_env()
    if not settings.deepl_api_key:
        raise PipelineError("DEEPL_API_KEY is missing from the environment or .env.")
    video_context = f" for {video.name}" if video else ""
    yield PipelineProgress("script", 0.05, f"Extracting screenplay text{video_context}.")
    try:
        script_text = extract_script_text(script)
    except PDFExtractionError as exc:
        raise PipelineError(f"Script PDF processing failed: {exc}") from exc

    yield PipelineProgress("analysis", 0.15, "Analyzing the screenplay with local Ollama.")
    try:
        analysis = LLMService(settings).analyze_script(script_text)
    except LLMServiceError as exc:
        raise PipelineError(f"Screenplay analysis failed: {exc}") from exc
    glossary_context = analysis.to_json()

    try:
        subtitles = pysrt.open(str(source_srt), encoding="utf-8-sig")
    except (OSError, UnicodeError, pysrt.Error) as exc:
        raise PipelineError(f"Could not read SRT subtitles: {source_srt}") from exc
    if not subtitles:
        raise PipelineError("The SRT file contains no subtitle cues.")

    translator = TranslatorService(settings.deepl_api_key)
    cue_count = len(subtitles)
    for index, cue in enumerate(subtitles, 1):
        try:
            cue.text = translator.translate(cue.text, language, glossary_context)
        except TranslationError as exc:
            # Preserve the cue number to make large-file failures actionable.
            raise PipelineError(f"Translation failed at subtitle {cue.index}: {exc}") from exc
        yield PipelineProgress(
            "translation",
            0.2 + 0.65 * index / cue_count,
            f"Translated subtitle {index} of {cue_count}.",
        )

    yield PipelineProgress("formatting", 0.9, "Validating broadcast line and timing limits.")
    try:
        format_subtitles(
            subtitles,
            max_characters=settings.max_line_characters,
            max_lines=settings.max_lines,
            minimum_duration_ms=settings.minimum_duration_ms,
        )
    except SubtitleFormatError as exc:
        raise PipelineError(f"Subtitle formatting failed: {exc}") from exc
    output_path = source_srt.with_name(f"{source_srt.stem}.{language}.srt")
    _atomic_save(subtitles, output_path)
    yield PipelineProgress("complete", 1.0, "Localization complete.", output_path)
