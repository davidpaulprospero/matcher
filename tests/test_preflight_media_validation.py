"""
Tests for pre-flight media validation in OTIO timeline assembly.

Story: US-56-006 - Add pre-flight media validation before timeline assembly
"""

import pytest
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from unittest.mock import patch, MagicMock

from src.otio.timeline import validate_media_paths


# ---------------------------------------------------------------------------
# Lightweight fakes — we only need .video_segment.source_file on each match
# ---------------------------------------------------------------------------

@dataclass
class FakeSegment:
    source_file: str = ""
    index: int = 0
    start_time: float = 0.0
    end_time: float = 5.0
    text: str = ""


@dataclass
class FakeMatch:
    video_segment: FakeSegment = field(default_factory=FakeSegment)
    voiceover_segment: FakeSegment = field(default_factory=FakeSegment)
    confidence: float = 0.8
    reasoning: str = ""


@dataclass
class FakeAltMatch:
    video_segment: FakeSegment = field(default_factory=FakeSegment)
    confidence: float = 0.7
    reasoning: str = ""
    diversity_score: float = 0.0


@dataclass
class FakeStrategyMatch:
    video_segment: FakeSegment = field(default_factory=FakeSegment)
    confidence: float = 0.6
    reasoning: str = ""
    strategy: str = "embedding_diversity"


@dataclass
class FakeMatchResult:
    primary_match: FakeMatch = field(default_factory=FakeMatch)
    alternatives: list = field(default_factory=list)
    secondary_matches: list = field(default_factory=list)
    strategy_matches: list = field(default_factory=list)
    has_gap: bool = False
    gap_reason: str = ""


@dataclass
class FakeDownloadedSegment:
    video_id: str = ""
    file: str = ""
    original_start: float = 0.0
    original_end: float = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_match(source_file: str, alts=None, secs=None, strats=None):
    """Create a FakeMatchResult with specified source files."""
    mr = FakeMatchResult(
        primary_match=FakeMatch(video_segment=FakeSegment(source_file=source_file)),
        alternatives=[
            FakeAltMatch(video_segment=FakeSegment(source_file=f))
            for f in (alts or [])
        ],
        secondary_matches=[
            FakeAltMatch(video_segment=FakeSegment(source_file=f))
            for f in (secs or [])
        ],
        strategy_matches=[
            FakeStrategyMatch(video_segment=FakeSegment(source_file=f))
            for f in (strats or [])
        ],
    )
    return mr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestValidateMediaPaths:
    """Tests for validate_media_paths pre-flight validation."""

    @patch("src.otio.timeline._is_missing_file", return_value=False)
    def test_all_valid(self, mock_missing):
        """All valid video files produce correct valid count."""
        matches = [
            _make_match("E:/videos/clip1.mp4"),
            _make_match("E:/videos/clip2.mp4"),
            _make_match("E:/videos/clip3.mp4"),
        ]
        summary = validate_media_paths(matches)
        assert summary['valid'] == 3
        assert summary['total'] == 3
        assert summary['audio_only'] == []
        assert summary['missing'] == []
        assert summary['problematic_path'] == []
        assert summary['non_media'] == []

    @patch("src.otio.timeline._is_missing_file", return_value=False)
    def test_audio_only_detected(self, mock_missing):
        """Audio-only files (mp3, wav) are flagged."""
        matches = [
            _make_match("E:/videos/clip1.mp4"),
            _make_match("E:/audio/track.mp3"),
            _make_match("E:/audio/track2.wav"),
        ]
        summary = validate_media_paths(matches)
        assert summary['valid'] == 1
        assert len(summary['audio_only']) == 2
        assert "S001" in summary['audio_only']
        assert "S002" in summary['audio_only']

    @patch("src.otio.timeline._is_missing_file", return_value=True)
    def test_missing_file_detected(self, mock_missing):
        """Missing files are flagged when _is_missing_file returns True."""
        matches = [
            _make_match("E:/videos/does_not_exist.mp4"),
        ]
        summary = validate_media_paths(matches)
        assert len(summary['missing']) == 1
        assert "S000" in summary['missing']
        assert summary['valid'] == 0

    @patch("src.otio.timeline._is_missing_file", return_value=False)
    def test_non_media_detected(self, mock_missing):
        """Non-media files like .srt or .txt are flagged."""
        matches = [
            _make_match("E:/subs/subtitle.srt"),
            _make_match("E:/docs/notes.txt"),
            _make_match("E:/videos/clip.mp4"),
        ]
        summary = validate_media_paths(matches)
        assert len(summary['non_media']) == 2
        assert summary['valid'] == 1

    @patch("src.otio.timeline._is_missing_file", return_value=False)
    def test_problematic_path_detected(self, mock_missing):
        """Paths with problematic unicode chars are flagged."""
        # \ufffd is the replacement character which triggers _has_problematic_path
        matches = [
            _make_match("E:/videos/clip\ufffd.mp4"),
        ]
        summary = validate_media_paths(matches)
        assert len(summary['problematic_path']) == 1
        assert summary['valid'] == 0

    @patch("src.otio.timeline._is_missing_file", return_value=False)
    def test_mixed_issues(self, mock_missing):
        """Mixed valid and invalid paths all categorized correctly."""
        matches = [
            _make_match("E:/videos/good.mp4"),      # S000 - valid
            _make_match("E:/audio/song.mp3"),         # S001 - audio-only
            _make_match("E:/subs/captions.srt"),      # S002 - non-media
            _make_match("E:/videos/ok\ufffd.mp4"),    # S003 - problematic
        ]
        summary = validate_media_paths(matches)
        assert summary['valid'] == 1
        assert summary['total'] == 4
        assert len(summary['audio_only']) == 1
        assert len(summary['non_media']) == 1
        assert len(summary['problematic_path']) == 1

    @patch("src.otio.timeline._is_missing_file", return_value=False)
    def test_alternatives_validated(self, mock_missing):
        """Alternative matches (V2-V3) are also validated."""
        matches = [
            _make_match(
                "E:/videos/good.mp4",
                alts=["E:/audio/alt.mp3", "E:/videos/alt2.mp4"],
            ),
        ]
        summary = validate_media_paths(matches)
        # 1 primary valid + 1 alt audio-only + 1 alt valid
        assert summary['valid'] == 2
        assert len(summary['audio_only']) == 1
        assert "S000-alt0" in summary['audio_only']
        assert summary['total'] == 3

    @patch("src.otio.timeline._is_missing_file", return_value=False)
    def test_secondary_matches_validated(self, mock_missing):
        """Secondary matches (V4-V6) are also validated."""
        matches = [
            _make_match(
                "E:/videos/good.mp4",
                secs=["E:/audio/sec.wav"],
            ),
        ]
        summary = validate_media_paths(matches)
        assert len(summary['audio_only']) == 1
        assert "S000-sec0" in summary['audio_only']

    @patch("src.otio.timeline._is_missing_file", return_value=False)
    def test_strategy_matches_validated(self, mock_missing):
        """Strategy matches (V7+) are also validated."""
        matches = [
            _make_match(
                "E:/videos/good.mp4",
                strats=["E:/subs/strat.srt"],
            ),
        ]
        summary = validate_media_paths(matches)
        assert len(summary['non_media']) == 1
        assert "S000-strat0" in summary['non_media']

    @patch("src.otio.timeline._is_missing_file", return_value=False)
    def test_downloaded_segments_resolve_ids(self, mock_missing):
        """Video IDs are resolved through downloaded_segments lookup."""
        matches = [
            _make_match("abc123"),  # video ID, not a path
        ]
        segments = [
            FakeDownloadedSegment(video_id="abc123", file="E:/downloads/abc123.mp4"),
        ]
        summary = validate_media_paths(matches, downloaded_segments=segments)
        assert summary['valid'] == 1
        assert summary['total'] == 1

    @patch("src.otio.timeline._is_missing_file", return_value=False)
    def test_empty_matches_list(self, mock_missing):
        """Empty matches list returns zeroed summary."""
        summary = validate_media_paths([])
        assert summary['valid'] == 0
        assert summary['total'] == 0
        assert summary['audio_only'] == []
        assert summary['missing'] == []

    @patch("src.otio.timeline._is_missing_file", return_value=False)
    def test_summary_structure(self, mock_missing):
        """Summary dict has all required keys."""
        summary = validate_media_paths([])
        assert set(summary.keys()) == {
            'valid', 'audio_only', 'missing',
            'problematic_path', 'non_media', 'total',
        }

    def test_empty_source_file_flagged_as_missing(self):
        """Empty source_file string is flagged as missing."""
        matches = [_make_match("")]
        summary = validate_media_paths(matches)
        assert len(summary['missing']) == 1
        assert summary['valid'] == 0
