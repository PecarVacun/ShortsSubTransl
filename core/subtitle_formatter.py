"""Validate and format SRT cues without discarding text."""

import textwrap

import pysrt


class SubtitleFormatError(ValueError):
    """A cue cannot meet formatting or timing requirements without data loss."""


def normalize_text(text: str) -> str:
    """Replace incidental whitespace and line breaks with single spaces."""
    return " ".join(text.split())


def format_subtitles(
    subtitles: pysrt.SubRipFile,
    max_characters: int = 38,
    max_lines: int = 2,
    minimum_duration_ms: int = 1000,
) -> pysrt.SubRipFile:
    """Format cues and extend short durations only when no overlap is created.

    All cues are validated before the input object is changed. Invalid or
    overlapping source cues, unbreakable words, and text that does not fit
    require manual correction rather than silent truncation.
    """
    if max_characters <= 0 or max_lines <= 0 or minimum_duration_ms <= 0:
        raise SubtitleFormatError("Formatting limits must be positive.")
    previous_end = 0
    for cue in subtitles:
        start, end = cue.start.ordinal, cue.end.ordinal
        if start < 0 or end <= start or start < previous_end:
            raise SubtitleFormatError(
                f"Subtitle {cue.index} has invalid or overlapping timing."
            )
        previous_end = end

    formatted: list[tuple[str, int]] = []
    for index, cue in enumerate(subtitles):
        start, end = cue.start.ordinal, cue.end.ordinal
        label = f"Subtitle {cue.index}"
        text = normalize_text(cue.text)
        if not text:
            raise SubtitleFormatError(f"{label} has no text.")
        lines = textwrap.wrap(
            text,
            width=max_characters,
            break_long_words=False,
            break_on_hyphens=False,
        )
        if len(lines) > max_lines or any(len(line) > max_characters for line in lines):
            raise SubtitleFormatError(
                f"{label} cannot fit in {max_lines} lines of {max_characters} characters."
            )
        new_end = max(end, start + minimum_duration_ms)
        if index + 1 < len(subtitles) and new_end > subtitles[index + 1].start.ordinal:
            raise SubtitleFormatError(
                f"{label} cannot be extended without overlapping the next cue."
            )
        formatted.append(("\n".join(lines), new_end))
    for cue, (text, end) in zip(subtitles, formatted):
        cue.text = text
        cue.end = pysrt.SubRipTime(milliseconds=end)
    return subtitles
