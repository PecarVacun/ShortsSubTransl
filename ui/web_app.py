"""Gradio interface; all localization logic lives in the pipeline."""

from __future__ import annotations

from collections.abc import Iterator

import gradio as gr

from core.pipeline import PipelineError, PipelineProgress, run_localization_pipeline


def localize(
    video_path: str | None,
    script_path: str | None,
    srt_path: str | None,
    target_lang: str,
) -> Iterator[tuple[str, str | None]]:
    """Adapt pipeline events to the UI without importing service modules."""
    if not script_path or not srt_path:
        yield "Select both a screenplay PDF and an SRT file.", None
        return
    try:
        events: Iterator[PipelineProgress] = run_localization_pipeline(
            video_path or "",
            script_path,
            srt_path,
            target_lang,
        )
        for event in events:
            yield event.message, str(event.output_path) if event.output_path else None
    except PipelineError as exc:
        yield f"Localization failed: {exc}", None


def create_app() -> gr.Blocks:
    """Build the web UI and wire its single action to the pipeline."""
    with gr.Blocks(title="Subtitle localization") as app:
        gr.Markdown("# Subtitle localization")
        gr.Markdown(
            "The video is accepted for workflow context but is not analyzed. "
            "The screenplay and SRT are processed independently."
        )
        video = gr.File(label="Video (optional)", file_types=["video"], type="filepath")
        script = gr.File(label="Screenplay PDF", file_types=[".pdf"], type="filepath")
        subtitles = gr.File(label="Source SRT", file_types=[".srt"], type="filepath")
        language = gr.Dropdown(
            choices=["RU", "EN-US", "EN-GB", "ES", "FR", "DE", "IT", "JA", "ZH"],
            value="RU",
            label="Target language",
        )
        run_button = gr.Button("Localize subtitles", variant="primary")
        status = gr.Textbox(label="Progress", interactive=False)
        output = gr.File(label="Localized SRT")
        run_button.click(
            fn=localize,
            inputs=[video, script, subtitles, language],
            outputs=[status, output],
        )
    return app


if __name__ == "__main__":
    create_app().queue().launch()
