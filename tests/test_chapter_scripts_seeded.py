"""Tests for scripts/chapters/detect_chapters.py — the seeded chapter detection centerpiece.

Patches BOTH the LLM client construction (`_build_llm_client`) and the
fallback detector path (`_fallback_llm_chapters`) so tests run without any
real LLM endpoint. The fake records every prompt sent and returns
deterministic chapter JSON.
"""

from __future__ import annotations

from typing import Any, List

import pytest

from scripts.chapters import detect_chapters as dc_mod
from scripts.chapters.detect_chapters import detect


# Synthetic 5-item listicle
LISTICLE_SRT = """\
1
00:00:00.000 --> 00:00:02.000
Top five reasons you should move to Portugal.

2
00:00:02.500 --> 00:00:07.000
First, the climate is mild all year round.

3
00:00:07.500 --> 00:00:12.000
Second, the cost of living is much lower.

4
00:00:12.500 --> 00:00:17.000
Third, the visa process has been streamlined.

5
00:00:17.500 --> 00:00:22.000
Fourth, the healthcare is excellent and affordable.

6
00:00:22.500 --> 00:00:27.000
And fifth, finally, the food and wine are world class.

7
00:00:27.500 --> 00:00:30.000
That is why Portugal should be your next home.
"""

# Non-listicle narrative
NARRATIVE_SRT = """\
1
00:00:00.000 --> 00:00:03.000
Welcome to our exploration of ancient Rome.

2
00:00:03.500 --> 00:00:08.000
The city was founded on seven hills along the Tiber river.

3
00:00:08.500 --> 00:00:13.000
Its empire at its peak stretched across three continents.

4
00:00:13.500 --> 00:00:18.000
The colosseum still stands as a marvel of engineering.

5
00:00:18.500 --> 00:00:22.000
Today Rome is a vibrant capital with deep historical roots.
"""


class _FakeResponse:
    def __init__(self, data: Any):
        self.parsed_data = data


class _FakeLLM:
    """Records every prompt and returns chapters from a configurable list of responses.

    Each queued "response" is a list of chapter dicts (i.e., what the LLM would
    return as a JSON array). To queue multiple responses for successive calls,
    pass multiple arguments.
    """

    def __init__(self):
        self.prompts: List[str] = []
        self.call_count = 0
        self._responses: List[Any] = [[]]

    def queue_responses(self, *responses):
        # Each response is a list of chapters; flatten the variadic wrapper.
        self._responses = list(responses) if responses else [[]]

    def generate(self, request):
        self.prompts.append(request.prompt)
        self.call_count += 1
        idx = min(self.call_count - 1, len(self._responses) - 1)
        data = self._responses[idx]
        return _FakeResponse(data)


def _shim(d):
    """Object exposing .to_dict() (matches ChapterCandidate interface)."""
    return _DictShim(d)


class _DictShim:
    def __init__(self, d):
        self._d = d

    def to_dict(self):
        return self._d


@pytest.fixture
def fake_llm():
    return _FakeLLM()


@pytest.fixture
def patched_detect(monkeypatch, fake_llm):
    """Patch both LLM construction and fallback paths."""
    monkeypatch.setattr(dc_mod, "_build_llm_client", lambda config: fake_llm)
    # Default fallback stub returns one synthetic chapter; tests can override.
    monkeypatch.setattr(
        dc_mod, "_fallback_llm_chapters",
        lambda *a, **k: [_shim({
            "chapter_id": 0, "start_segment_idx": 0, "end_segment_idx": 99,
            "title": "Fallback", "topics": [], "confidence": "medium",
        })],
    )
    return fake_llm


@pytest.fixture
def listicle_srt(tmp_path):
    p = tmp_path / "portugal.srt"
    p.write_text(LISTICLE_SRT, encoding="utf-8")
    return p


@pytest.fixture
def narrative_srt(tmp_path):
    p = tmp_path / "rome.srt"
    p.write_text(NARRATIVE_SRT, encoding="utf-8")
    return p


@pytest.fixture
def dummy_config(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(
        "matching:\n  chapter_detection: {}\n"
        "llm:\n  provider: ollama\n  model: gemma3:4b\n",
        encoding="utf-8",
    )
    return p


class TestSeedBlockInjection:
    def test_listicle_input_triggers_seeding(self, listicle_srt, dummy_config, patched_detect):
        fake = patched_detect
        fake.queue_responses([
            {"chapter_id": 0, "start_segment_idx": 2, "end_segment_idx": 3,
             "title": "Climate", "topics": ["weather"], "confidence": "high"},
            {"chapter_id": 1, "start_segment_idx": 4, "end_segment_idx": 5,
             "title": "Cost", "topics": ["money"], "confidence": "high"},
        ])
        payload = detect(listicle_srt, config_path=dummy_config)
        assert payload["seeded"] is True
        assert payload["seed_group_count"] >= 2
        # Every LLM prompt contains the seed marker
        assert fake.call_count >= 1, f"expected >= 1 LLM call, got {fake.call_count}"
        for prompt in fake.prompts:
            assert "PRE-DETECTED LIST STRUCTURE" in prompt
        # Output shape is correct
        assert "llm_chapters" in payload
        assert "unified_chapters" in payload

    def test_narrative_input_no_seeding(self, narrative_srt, dummy_config, patched_detect):
        fake = patched_detect
        payload = detect(narrative_srt, config_path=dummy_config)
        assert payload["seeded"] is False
        assert payload["seed_group_count"] == 0
        # No LLM call when listicle is empty
        assert fake.call_count == 0


class TestFallbackPaths:
    def test_no_llm_client_uses_fallback(self, listicle_srt, dummy_config, monkeypatch):
        # `_build_llm_client` returns None → fallback path kicks in
        monkeypatch.setattr(dc_mod, "_build_llm_client", lambda config: None)
        monkeypatch.setattr(
            dc_mod, "_fallback_llm_chapters",
            lambda *a, **k: [_shim({
                "chapter_id": 0, "start_segment_idx": 0, "end_segment_idx": 6,
                "title": "Whole list", "topics": [], "confidence": "medium",
            })],
        )
        payload = detect(listicle_srt, config_path=dummy_config)
        assert "llm_chapters" in payload
        assert payload["seeded"] is False
        assert payload["llm_chapters"][0]["title"] == "Whole list"

    def test_llm_empty_falls_back(self, listicle_srt, dummy_config, patched_detect):
        fake = patched_detect
        # LLM is patched but returns empty list
        payload = detect(listicle_srt, config_path=dummy_config)
        # Seed attempted but LLM returned nothing → fallback path used
        assert fake.call_count >= 1
        assert payload["seeded"] is False
        assert payload["llm_chapters"]


class TestPromptShape:
    def test_seeded_prompt_includes_rules(self, listicle_srt, dummy_config, patched_detect):
        fake = patched_detect
        fake.queue_responses([
            {"chapter_id": 0, "start_segment_idx": 0, "end_segment_idx": 6,
             "title": "All", "topics": [], "confidence": "high"},
        ])
        detect(listicle_srt, config_path=dummy_config)
        prompt = fake.prompts[0]
        assert "Treat each item's end_segment_idx" in prompt
        assert "Do NOT introduce a boundary in the MIDDLE" in prompt
        assert "Expected total items per header" in prompt
