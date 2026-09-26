"""Tests for SubtitleGenerator: timestamps, file formats, and validation."""
import subprocess
import sys
from pathlib import Path

import pytest

from src.subtitle_generator import SubtitleGenerator
from src.translator_module import TranslatedSegment

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def generator(tmp_path):
    return SubtitleGenerator(output_dir=str(tmp_path / "subtitles"))


@pytest.fixture
def sample_segments():
    return [
        TranslatedSegment(0.0, 3.5, "Hello everyone", "سلام سب کو", "en"),
        TranslatedSegment(3.5, 7.0, "Welcome to this lecture", "اس لیکچر میں خوش آمدید", "en"),
    ]


class TestFormatTimestamp:
    @pytest.mark.parametrize(
        "seconds,expected",
        [
            (0, "00:00:00,000"),
            (1.5, "00:00:01,500"),
            (61.25, "00:01:01,250"),
            (3661.789, "01:01:01,789"),
        ],
    )
    def test_formats_to_srt_convention(self, generator, seconds, expected):
        assert generator.format_timestamp(seconds) == expected


class TestSrt:
    def test_starts_at_index_one(self, generator, sample_segments):
        assert generator.create_srt_content(sample_segments).startswith("1\n")

    def test_uses_the_arrow_timing_separator(self, generator, sample_segments):
        assert "-->" in generator.create_srt_content(sample_segments)

    def test_writes_one_entry_per_segment(self, generator, sample_segments):
        assert generator.create_srt_content(sample_segments).count("-->") == 2

    def test_separates_entries_with_a_blank_line(self, generator, sample_segments):
        assert "\n\n" in generator.create_srt_content(sample_segments)

    def test_contains_the_translated_text(self, generator, sample_segments):
        assert "سلام سب کو" in generator.create_srt_content(sample_segments)

    def test_writes_utf8_to_disk(self, generator, sample_segments):
        path = Path(generator.generate_srt(sample_segments, "lecture"))
        assert "سلام سب کو" in path.read_text(encoding="utf-8")


class TestVtt:
    def test_has_the_webvtt_header(self, generator, sample_segments):
        path = generator.generate_vtt(sample_segments, "lecture")
        assert Path(path).read_text(encoding="utf-8").startswith("WEBVTT")

    def test_uses_dots_for_milliseconds_not_commas(self, generator, sample_segments):
        content = Path(generator.generate_vtt(sample_segments, "lecture")).read_text(encoding="utf-8")
        # WEBVTT, blank, index, then the timing line.
        timing_line = content.split("\n")[3]
        assert "-->" in timing_line
        assert "." in timing_line
        assert "," not in timing_line


class TestDualLanguage:
    def test_includes_both_the_original_and_the_translation(self, generator, sample_segments):
        path = generator.generate_dual_language_srt(sample_segments, "lecture")
        content = Path(path).read_text(encoding="utf-8")
        assert "Hello everyone" in content
        assert "سلام سب کو" in content


class TestValidateTiming:
    def test_accepts_valid_timings(self, generator, sample_segments):
        assert generator.validate_timing(sample_segments) is True

    def test_accepts_an_empty_list(self, generator):
        assert generator.validate_timing([]) is True

    def test_rejects_overlaps(self, generator):
        overlap = [
            TranslatedSegment(0.0, 5.0, "a", "a", "en"),
            TranslatedSegment(3.0, 7.0, "b", "b", "en"),
        ]
        assert generator.validate_timing(overlap) is False

    def test_rejects_a_negative_duration_on_the_final_segment(self, generator):
        """A `len-1` loop skipped the last segment, so a bad final cue slipped through."""
        inverted = [
            TranslatedSegment(0.0, 5.0, "a", "a", "en"),
            TranslatedSegment(9.0, 7.0, "b", "b", "en"),
        ]
        assert generator.validate_timing(inverted) is False

    def test_rejects_a_zero_length_segment(self, generator):
        assert generator.validate_timing([TranslatedSegment(2.0, 2.0, "a", "a", "en")]) is False


class TestOptimizeTiming:
    @pytest.fixture
    def long_segment(self):
        return [TranslatedSegment(0.0, 5.0, "x" * 100, "word " * 30, "en")]

    def test_wraps_long_text_onto_multiple_lines(self, generator, long_segment):
        wrapped = generator.optimize_timing(long_segment, max_chars_per_line=42)
        assert len(wrapped[0].translated_text.split("\n")) > 1

    def test_no_line_exceeds_the_limit(self, generator, long_segment):
        wrapped = generator.optimize_timing(long_segment, max_chars_per_line=42)
        assert all(len(line) <= 42 for line in wrapped[0].translated_text.split("\n"))

    def test_wrapping_preserves_every_word(self, generator, long_segment):
        wrapped = generator.optimize_timing(long_segment, max_chars_per_line=42)
        assert (
            wrapped[0].translated_text.replace("\n", " ")
            == long_segment[0].translated_text.strip()
        )

    def test_short_text_is_left_alone(self, generator, sample_segments):
        wrapped = generator.optimize_timing(sample_segments, max_chars_per_line=42)
        assert wrapped[0].translated_text == "سلام سب کو"

    def test_timing_is_carried_over(self, generator, long_segment):
        wrapped = generator.optimize_timing(long_segment)
        assert (wrapped[0].start, wrapped[0].end) == (0.0, 5.0)


class TestRunsAsAScript:
    """`python src/subtitle_generator.py` used to fail with an ImportError."""

    def test_exits_cleanly_and_writes_a_file(self, tmp_path):
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / "src" / "subtitle_generator.py")],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),
        )
        assert proc.returncode == 0, proc.stderr
        assert "ImportError" not in proc.stderr
        assert "ModuleNotFoundError" not in proc.stderr
        assert "Generated:" in proc.stdout
