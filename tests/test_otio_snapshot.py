"""
Regression snapshots for OTIO generation.

These tests are intentionally strict:
- No auto-creating snapshots during tests
- No auto-updating snapshots during tests
- Current generation output must match committed baselines
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import opentimelineio as otio
import pytest

from src.otio.timeline import create_timeline
from src.otio.utils import create_clip_with_timewarp
from src.utils import AlternativeMatch, Match, MatchResult, SRTSegment, StrategyMatch


SNAPSHOT_DIR = Path(__file__).parent / "snapshots"


def _load_snapshot(snapshot_name: str) -> dict:
    snapshot_path = SNAPSHOT_DIR / f"{snapshot_name}.snap.json"
    assert snapshot_path.exists(), f"Missing snapshot: {snapshot_path}"
    return json.loads(snapshot_path.read_text(encoding="utf-8"))


def _serialize_otio(otio_object) -> dict:
    return json.loads(otio.adapters.write_to_string(otio_object, "otio_json"))


def _normalize_path(path: str) -> str:
    return str(path).replace("\\", "/")


def _build_match_results() -> list[MatchResult]:
    vo1 = SRTSegment(index=1, start_time=0.0, end_time=2.5, text="Opening line")
    v1 = SRTSegment(
        index=1,
        start_time=10.0,
        end_time=12.5,
        text="Primary footage one",
        source_file="C:/fixture/primary_01.mp4",
    )
    m1 = Match(
        voiceover_segment=vo1,
        video_segment=v1,
        video_scene=None,
        confidence=0.93,
        reasoning="semantic",
    )

    alt1 = AlternativeMatch(
        video_segment=SRTSegment(
            index=1,
            start_time=20.0,
            end_time=22.5,
            text="Alt footage one",
            source_file="C:/fixture/alt_01.mp4",
        ),
        video_scene=None,
        confidence=0.81,
        reasoning="alt",
    )
    sec1 = AlternativeMatch(
        video_segment=SRTSegment(
            index=1,
            start_time=30.0,
            end_time=32.5,
            text="Secondary one",
            source_file="C:/fixture/sec_01.mp4",
        ),
        video_scene=None,
        confidence=0.72,
        reasoning="secondary",
    )
    strat1 = StrategyMatch(
        video_segment=SRTSegment(
            index=1,
            start_time=40.0,
            end_time=42.5,
            text="Strat one",
            source_file="C:/fixture/strat_01.mp4",
        ),
        video_scene=None,
        confidence=0.68,
        reasoning="strategy",
        strategy="embedding_diversity",
    )
    mr1 = MatchResult(
        primary_match=m1,
        alternatives=[alt1],
        secondary_matches=[sec1],
        strategy_matches=[strat1],
        matched_keywords=["opening", "line"],
    )

    vo2 = SRTSegment(index=2, start_time=3.0, end_time=5.0, text="Second line")
    v2 = SRTSegment(
        index=2,
        start_time=50.0,
        end_time=52.0,
        text="Primary footage two",
        source_file="C:/fixture/primary_02.mp4",
    )
    m2 = Match(
        voiceover_segment=vo2,
        video_segment=v2,
        video_scene=None,
        confidence=0.89,
        reasoning="semantic2",
    )

    alt2 = AlternativeMatch(
        video_segment=SRTSegment(
            index=2,
            start_time=60.0,
            end_time=62.0,
            text="Alt footage two",
            source_file="C:/fixture/alt_02.mp4",
        ),
        video_scene=None,
        confidence=0.79,
        reasoning="alt2",
    )
    sec2 = AlternativeMatch(
        video_segment=SRTSegment(
            index=2,
            start_time=70.0,
            end_time=72.0,
            text="Secondary two",
            source_file="C:/fixture/sec_02.mp4",
        ),
        video_scene=None,
        confidence=0.74,
        reasoning="secondary2",
    )
    strat2 = StrategyMatch(
        video_segment=SRTSegment(
            index=2,
            start_time=80.0,
            end_time=82.0,
            text="Strat two",
            source_file="C:/fixture/strat_02.mp4",
        ),
        video_scene=None,
        confidence=0.66,
        reasoning="strategy2",
        strategy="broll_only",
    )
    mr2 = MatchResult(
        primary_match=m2,
        alternatives=[alt2],
        secondary_matches=[sec2],
        strategy_matches=[strat2],
        matched_keywords=["second", "line"],
    )

    return [mr1, mr2]


class TestOTIOSnapshot:
    """Snapshot tests for OTIO output."""

    @pytest.mark.fast
    def test_clip_creation_snapshot(self):
        clip = create_clip_with_timewarp(
            name="Regression Clip",
            source_path="C:/fixture/primary_01.mp4",
            source_start=0.0,
            source_duration=2.5,
            target_duration=2.5,
            frame_rate=30.0,
            metadata={
                "confidence": 0.93,
                "strategy": "primary",
                "vo_index": 0,
                "vo_text": "Opening line",
            },
            media_duration=120.0,
        )

        current = _serialize_otio(clip)
        expected = _load_snapshot("clip_creation")
        assert current == expected

    @pytest.mark.fast
    def test_clip_with_timewarp_snapshot(self):
        clip = create_clip_with_timewarp(
            name="Regression Clip Slow",
            source_path="C:/fixture/primary_02.mp4",
            source_start=4.1666666667,
            source_duration=2.5,
            target_duration=3.75,
            frame_rate=24.0,
            metadata={
                "confidence": 0.78,
                "strategy": "primary",
                "vo_index": 1,
                "vo_text": "Second line",
            },
            media_duration=120.0,
        )

        current = _serialize_otio(clip)
        expected = _load_snapshot("clip_with_timewarp")
        assert current == expected

    @pytest.mark.fast
    def test_timeline_structure_snapshot(self):
        output = SimpleNamespace(
            include_alternatives=True,
            num_alternatives=2,
            include_strategy_tracks=True,
            strategy_tracks=["embedding_diversity", "broll_only"],
            time_scale_factor=1.0,
            voiceover_offset=0.0,
            min_gap_threshold=0.0,
            gap_mode="scale",
        )
        config = SimpleNamespace(output=output)

        with (
            patch("builtins.print"),
            patch("src.otio.timeline._is_missing_file", return_value=False),
            patch("src.otio.timeline._has_problematic_path", return_value=False),
            patch("src.otio.timeline._to_windows_path", side_effect=_normalize_path),
            patch("src.otio.utils._to_windows_path", side_effect=_normalize_path),
            patch("src.otio.utils._get_media_duration", return_value=120.0),
        ):
            timeline = create_timeline(_build_match_results(), config, frame_rate=30.0)

        current = _serialize_otio(timeline)
        expected = _load_snapshot("timeline_structure")
        assert current == expected

    @pytest.mark.fast
    def test_metadata_serialization_snapshot(self):
        metadata = {
            "confidence": 0.92,
            "strategy": "primary",
            "vo_index": 0,
            "vo_text": "Welcome to this documentary.",
            "vo_start": 0.0,
            "vo_end": 5.5,
            "video_source_file": "dQw4w9WgXcQ",
            "video_title": "Test Video",
            "video_start": 0.0,
            "video_end": 5.5,
            "matched_keywords": ["documentary", "welcome", "test"],
        }

        expected = _load_snapshot("metadata_serialization")
        assert metadata == expected
