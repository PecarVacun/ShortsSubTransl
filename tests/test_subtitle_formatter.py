"""Tests for subtitle layout and timing validation."""

import pysrt
import pytest

from core.subtitle_formatter import SubtitleFormatError, format_subtitles, normalize_text


def cue(index: int, start: int, end: int, text: str) -> pysrt.SubRipItem:
    """Create a cue with times in milliseconds."""
    return pysrt.SubRipItem(
        index=index,
        start=pysrt.SubRipTime(milliseconds=start),
        end=pysrt.SubRipTime(milliseconds=end),
        text=text,
    )


def test_normalizes_random_line_breaks() -> None:
    assert normalize_text("One\n   two\r\nthree") == "One two three"


def test_wraps_without_exceeding_limit_and_extends_short_duration() -> None:
    subtitles = pysrt.SubRipFile(
        [cue(1, 0, 500, "This is a subtitle with enough words to wrap across lines.")]
    )
    format_subtitles(subtitles)
    assert len(subtitles[0].text.splitlines()) <= 2
    assert all(len(line) <= 38 for line in subtitles[0].text.splitlines())
    assert subtitles[0].end.ordinal == 1000


def test_rejects_text_that_does_not_fit_without_mutating_input() -> None:
    subtitles = pysrt.SubRipFile([cue(1, 0, 1500, "word " * 20)])
    original = subtitles[0].text
    with pytest.raises(SubtitleFormatError, match="cannot fit"):
        format_subtitles(subtitles)
    assert subtitles[0].text == original


def test_rejects_unbreakable_word() -> None:
    subtitles = pysrt.SubRipFile([cue(1, 0, 1500, "x" * 39)])
    with pytest.raises(SubtitleFormatError, match="cannot fit"):
        format_subtitles(subtitles)


def test_short_duration_cannot_extend_into_next_cue() -> None:
    subtitles = pysrt.SubRipFile(
        [cue(1, 0, 500, "First"), cue(2, 700, 1800, "Second")]
    )
    with pytest.raises(SubtitleFormatError, match="overlapping the next cue"):
        format_subtitles(subtitles)
    assert subtitles[0].end.ordinal == 500


def test_rejects_source_timing_overlap() -> None:
    subtitles = pysrt.SubRipFile(
        [cue(1, 0, 1200, "First"), cue(2, 1000, 2000, "Second")]
    )
    with pytest.raises(SubtitleFormatError, match="overlapping timing"):
        format_subtitles(subtitles)
