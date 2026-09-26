"""Tests for UrduTranslator.

No test here contacts OpenAI. The client is replaced with a fake so the
batching, numbering-cleanup and repair logic can be verified offline.
"""
import os

import pytest

from src.speech_to_text import TranscriptionSegment
from src.translator_module import TranslatedSegment, UrduTranslator
from tests.fake_openai import FakeClient


def segments(n, offset=0):
    return [TranscriptionSegment(i * 1.0, i * 1.0 + 1, f"seg{i}", "en") for i in range(offset, offset + n)]


class TestApiKey:
    def test_rejects_a_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OpenAI API key required"):
            UrduTranslator()

    def test_reads_the_key_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        assert UrduTranslator().api_key == "sk-from-env"

    def test_explicit_key_wins_over_the_environment(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        assert UrduTranslator(api_key="sk-explicit").api_key == "sk-explicit"


class TestTranslateText:
    def test_returns_the_stripped_translation(self):
        t = UrduTranslator(api_key="sk-test")
        t.client = FakeClient("  خوش آمدید  ")
        assert t.translate_text("Welcome") == "خوش آمدید"

    @pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
    def test_blank_input_short_circuits_without_calling_the_api(self, blank):
        t = UrduTranslator(api_key="sk-test")
        t.client = FakeClient("unused")
        assert t.translate_text(blank) == ""
        assert t.client.chat.completions.calls == 0


class TestTranslateSegments:
    def test_returns_one_segment_per_input(self):
        t = UrduTranslator(api_key="sk-test")
        t.client = FakeClient("1. پہلا\n2. دوسرا")
        res = t.translate_segments(segments(2), batch_size=10)
        assert len(res) == 2

    def test_strips_the_numbering_the_model_echoes_back(self):
        t = UrduTranslator(api_key="sk-test")
        t.client = FakeClient("1. پہلا\n2. دوسرا")
        res = t.translate_segments(segments(2), batch_size=10)
        assert res[0].translated_text == "پہلا"
        assert res[1].translated_text == "دوسرا"

    def test_preserves_timing_and_metadata(self):
        t = UrduTranslator(api_key="sk-test")
        t.client = FakeClient("1. پہلا\n2. دوسرا")
        res = t.translate_segments(segments(2), batch_size=10)
        assert (res[0].start, res[0].end) == (0.0, 1.0)
        assert res[0].original_text == "seg0"
        assert res[0].source_lang == "en"
        assert res[0].target_lang == "ur"

    def test_no_segment_is_dropped_when_the_model_under_delivers(self):
        """A short batch reply used to silently drop the tail of the subtitles.

        The translator now re-requests any segment the batch did not cover, so
        every input still produces an output.
        """
        t = UrduTranslator(api_key="sk-test")
        t.client = FakeClient("only one line")  # 1 reply for 5 segments
        res = t.translate_segments(segments(5), batch_size=10)
        assert len(res) == 5
        assert [r.original_text for r in res] == [f"seg{i}" for i in range(5)]

    def test_uses_the_numbered_batch_format(self):
        t = UrduTranslator(api_key="sk-test")
        client = FakeClient("1. a\n2. b\n3. c")
        t.client = client
        t.translate_segments(segments(3), batch_size=10)
        assert client.chat.completions.batches[0] == [
            "1. seg0", "2. seg1", "3. seg2",
        ]

    def test_splits_into_batches(self):
        t = UrduTranslator(api_key="sk-test")
        client = FakeClient("ok")
        t.client = client
        t.translate_segments(segments(25), batch_size=10)
        assert [len(b) for b in client.chat.completions.batches] == [10, 10, 5]

    def test_preserves_order_across_batches(self):
        t = UrduTranslator(api_key="sk-test")
        t.client = FakeClient("1. one\n2. two\n3. three")
        res = t.translate_segments(segments(25), batch_size=3)
        assert len(res) == 25
        assert [r.original_text for r in res] == [f"seg{i}" for i in range(25)]

    def test_falls_back_to_the_original_text_when_a_repair_fails(self):
        t = UrduTranslator(api_key="sk-test")
        t.client = FakeClient("")  # empty batch reply, and the repair also fails

        def boom(text, source_lang="en"):
            raise RuntimeError("api down")

        t.translate_text = boom
        res = t.translate_segments(segments(2), batch_size=10)
        assert [r.translated_text for r in res] == ["seg0", "seg1"]

    def test_repairs_a_blank_line_in_the_reply(self):
        """A blank line used to become an empty cue, leaving a gap in the file."""
        t = UrduTranslator(api_key="sk-test")
        t.client = FakeClient("1. پہلا\n\n3. تیسرا")  # line 2 came back empty
        t.translate_text = lambda text, source_lang="en": f"repaired:{text}"
        res = t.translate_segments(segments(3), batch_size=10)

        assert [r.translated_text for r in res] == [
            "پہلا", "repaired:seg1", "تیسرا",
        ]
        assert all(r.translated_text.strip() for r in res)

    def test_never_emits_an_empty_cue(self):
        """Even a blank repair must not leave a blank subtitle."""
        t = UrduTranslator(api_key="sk-test")
        t.client = FakeClient("1. ایک\n2. دو")
        t.translate_text = lambda text, source_lang="en": "   "
        res = t.translate_segments(segments(3), batch_size=10)
        assert all(r.translated_text.strip() for r in res)


class TestTranslatedSegment:
    def test_target_language_defaults_to_urdu(self):
        seg = TranslatedSegment(0.0, 1.0, "hello", "ہیلو", "en")
        assert seg.target_lang == "ur"
