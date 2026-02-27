#!/usr/bin/env python3
"""
Regenerate OTIO and SRT generation regression snapshots.

Usage:
    python scripts/update_generation_snapshots.py
    python scripts/update_generation_snapshots.py --otio-only
    python scripts/update_generation_snapshots.py --srt-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import opentimelineio as otio

# Add project root and scripts directory to import path.
_script_path = os.path.abspath(__file__)
project_root = Path(_script_path).parent.parent
scripts_dir = Path(_script_path).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

# Ensure relative paths resolve from repository root.
os.chdir(project_root)

from script_utils import print_error, print_header, print_ok
from src.otio.timeline import create_timeline
from src.otio.utils import create_clip_with_timewarp
from src.transcription.utils import write_srt
from src.utils import AlternativeMatch, Match, MatchResult, SRTSegment, StrategyMatch, write_srt_file


SNAPSHOT_DIR = project_root / "tests" / "snapshots"


def _serialize_otio(otio_object) -> dict:
    return json.loads(otio.adapters.write_to_string(otio_object, "otio_json"))


def _normalize_path(path: str) -> str:
    return str(path).replace("\\", "/")


def _write_json_snapshot(filename: str, payload: dict) -> Path:
    path = SNAPSHOT_DIR / filename
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_text_snapshot(filename: str, payload: str) -> Path:
    path = SNAPSHOT_DIR / filename
    path.write_text(payload, encoding="utf-8")
    return path


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


def update_otio_snapshots() -> list[Path]:
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
    clip_slow = create_clip_with_timewarp(
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

    with (
        patch("builtins.print"),
        patch("src.otio.timeline._is_missing_file", return_value=False),
        patch("src.otio.timeline._has_problematic_path", return_value=False),
        patch("src.otio.timeline._to_windows_path", side_effect=_normalize_path),
        patch("src.otio.utils._to_windows_path", side_effect=_normalize_path),
        patch("src.otio.utils._get_media_duration", return_value=120.0),
    ):
        timeline = create_timeline(_build_match_results(), config, frame_rate=30.0)

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

    return [
        _write_json_snapshot("clip_creation.snap.json", _serialize_otio(clip)),
        _write_json_snapshot("clip_with_timewarp.snap.json", _serialize_otio(clip_slow)),
        _write_json_snapshot("timeline_structure.snap.json", _serialize_otio(timeline)),
        _write_json_snapshot("metadata_serialization.snap.json", metadata),
    ]


def update_srt_snapshots() -> list[Path]:
    segments = [
        {"start": 0.0, "end": 1.234, "text": "  First line  "},
        {"start": 2.5, "end": 4.0, "text": "Second line\nwith break"},
        {"start": 4.75, "end": 6.0, "text": "Third line"},
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        default_path = temp_dir_path / "transcription_default.srt"
        contiguous_path = temp_dir_path / "transcription_contiguous.srt"
        utils_path = temp_dir_path / "utils_writer.srt"

        write_srt(segments, str(default_path), force_contiguous_timing=False)
        write_srt(segments, str(contiguous_path), force_contiguous_timing=True)

        write_srt_file(
            [
                SRTSegment(index=10, start_time=0.0, end_time=1.5, text="Utils one"),
                SRTSegment(index=20, start_time=1.5, end_time=3.25, text="Utils two"),
            ],
            str(utils_path),
        )

        return [
            _write_text_snapshot("write_srt_default.snap.srt", default_path.read_text(encoding="utf-8")),
            _write_text_snapshot("write_srt_contiguous.snap.srt", contiguous_path.read_text(encoding="utf-8")),
            _write_text_snapshot("write_srt_file_default.snap.srt", utils_path.read_text(encoding="utf-8")),
        ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Update OTIO/SRT generation snapshots in tests/snapshots.")
    parser.add_argument("--otio-only", action="store_true", help="Update only OTIO snapshots.")
    parser.add_argument("--srt-only", action="store_true", help="Update only SRT snapshots.")
    args = parser.parse_args()

    if args.otio_only and args.srt_only:
        print_error("Choose at most one of --otio-only or --srt-only.", exit_code=1)
        return

    if not SNAPSHOT_DIR.exists():
        print_error(f"Snapshot directory not found: {SNAPSHOT_DIR}", exit_code=1)
        return

    updated_paths: list[Path] = []
    if not args.srt_only:
        print_header("Updating OTIO snapshots")
        updated_paths.extend(update_otio_snapshots())

    if not args.otio_only:
        print_header("Updating SRT snapshots")
        updated_paths.extend(update_srt_snapshots())

    print_header("Updated Files")
    for path in updated_paths:
        print_ok(str(path.relative_to(project_root)))


if __name__ == "__main__":
    main()
