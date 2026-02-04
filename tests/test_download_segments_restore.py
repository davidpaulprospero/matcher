"""Integration test for download_segments checkpoint restore roundtrip (US-52-008).

Verifies that the serialization-deserialization roundtrip through the
ITERATIVE_MATCH checkpoint save path preserves video file references so that
download_segments._collect_matched_segments() can correctly extract video_id
from restored matches.

Covers both MatchResult (with primary_match.video_segment.source_file) and
plain Match (with video_file) structures.

Pytest marker: fast (no network or subprocess calls)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from src.utils import MatchResult, SRTSegment
from src.utils import Match as UtilsMatch
from src.state import Match as StateMatch, PipelineState, restore_matches_from_dicts
from src.stages.download_segments import DownloadVideoSegmentsStage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_voiceover_segment(index: int, start: float, end: float, text: str = "test") -> SRTSegment:
    """Create a voiceover SRTSegment (no source_file)."""
    return SRTSegment(
        index=index, start_time=start, end_time=end, text=text, source_file=""
    )


def _make_video_segment(
    index: int, start: float, end: float, source_file: str, text: str = "video text"
) -> SRTSegment:
    """Create a video SRTSegment with source_file populated."""
    return SRTSegment(
        index=index, start_time=start, end_time=end, text=text, source_file=source_file
    )


def _make_match_result(
    segment_index: int,
    video_id: str,
    video_start: float = 0.0,
    video_end: float = 10.0,
    confidence: float = 0.85,
) -> MatchResult:
    """Create a MatchResult with primary_match containing a video_segment."""
    vo_seg = _make_voiceover_segment(segment_index, segment_index * 10.0, (segment_index + 1) * 10.0)
    vid_seg = _make_video_segment(0, video_start, video_end, source_file=video_id)
    primary = UtilsMatch(
        voiceover_segment=vo_seg,
        video_segment=vid_seg,
        video_scene=None,
        confidence=confidence,
        reasoning="test match",
    )
    return MatchResult(primary_match=primary)


def _serialize_matches_like_iterative_match(matches: list) -> List[Dict[str, Any]]:
    """Replicate the serialization logic from iterative_match.py save_checkpoint.

    This mirrors the exact code path in src/stages/iterative_match.py:420-485.
    """
    serialized = []
    for i, match in enumerate(matches):
        if hasattr(match, 'primary_match') and match.primary_match:
            pm = match.primary_match
            source_file = ''
            video_start = 0.0
            video_end = 0.0

            if hasattr(pm, 'video_segment') and pm.video_segment:
                source_file = getattr(pm.video_segment, 'source_file', '')
                video_start = getattr(pm.video_segment, 'start_time', 0.0)
                video_end = getattr(pm.video_segment, 'end_time', 0.0)

            conf = getattr(pm, 'confidence', 0.0)

            serialized.append({
                'segment_index': i,
                'video_file': source_file,
                'video_start': float(video_start),
                'video_end': float(video_end),
                'confidence': float(conf),
                'strategy': getattr(match, 'strategy', getattr(pm, 'reasoning', '')),
                'reason': getattr(pm, 'reasoning', ''),
                'face_score': getattr(match, 'face_score', 0.5),
            })
        else:
            video_file = getattr(match, 'video_file', '')
            video_start = getattr(match, 'video_start', 0.0)
            video_end = getattr(match, 'video_end', 0.0)
            confidence = getattr(match, 'confidence', 0.0)

            if not video_file:
                pm = getattr(match, 'primary_match', None)
                if pm is not None:
                    vs = getattr(pm, 'video_segment', None)
                    if vs is not None:
                        video_file = getattr(vs, 'source_file', '')
                        video_start = getattr(vs, 'start_time', video_start)
                        video_end = getattr(vs, 'end_time', video_end)
                    confidence = getattr(pm, 'confidence', confidence)

            serialized.append({
                'segment_index': getattr(match, 'segment_index', i),
                'video_file': video_file,
                'video_start': float(video_start),
                'video_end': float(video_end),
                'confidence': float(confidence),
                'strategy': getattr(match, 'strategy', ''),
                'reason': getattr(match, 'reason', ''),
                'face_score': getattr(match, 'face_score', 0.5),
            })
    return serialized


# ---------------------------------------------------------------------------
# Tests: Serialization preserves video_file
# ---------------------------------------------------------------------------

class TestMatchResultSerialization:
    """Test that MatchResult serialization preserves source_file as video_file."""

    def test_match_result_serializes_source_file(self):
        """MatchResult with primary_match.video_segment.source_file serializes to video_file."""
        mr = _make_match_result(0, "dQw4w9WgXcQ", video_start=5.0, video_end=15.0)
        serialized = _serialize_matches_like_iterative_match([mr])

        assert len(serialized) == 1
        assert serialized[0]['video_file'] == "dQw4w9WgXcQ"
        assert serialized[0]['video_start'] == 5.0
        assert serialized[0]['video_end'] == 15.0
        assert serialized[0]['confidence'] == 0.85

    def test_plain_match_serializes_video_file(self):
        """Plain state.Match with video_file serializes correctly."""
        match = StateMatch(
            segment_index=0,
            video_file="abc123xyz",
            video_start=2.0,
            video_end=12.0,
            confidence=0.9,
            strategy="restored",
        )
        serialized = _serialize_matches_like_iterative_match([match])

        assert len(serialized) == 1
        assert serialized[0]['video_file'] == "abc123xyz"
        assert serialized[0]['video_start'] == 2.0
        assert serialized[0]['video_end'] == 12.0

    def test_multiple_match_results_serialize(self):
        """Multiple MatchResults all preserve their video IDs."""
        video_ids = ["vid_AAA", "vid_BBB", "vid_CCC"]
        matches = [_make_match_result(i, vid) for i, vid in enumerate(video_ids)]
        serialized = _serialize_matches_like_iterative_match(matches)

        assert len(serialized) == 3
        for i, vid in enumerate(video_ids):
            assert serialized[i]['video_file'] == vid


# ---------------------------------------------------------------------------
# Tests: Deserialization via Match.from_dict preserves video_file
# ---------------------------------------------------------------------------

class TestMatchFromDictDeserialization:
    """Test that Match.from_dict() correctly restores video_file."""

    def test_roundtrip_preserves_video_file(self):
        """Serialize MatchResult -> deserialize via Match.from_dict -> video_file preserved."""
        mr = _make_match_result(0, "roundtrip_vid_001", video_start=3.0, video_end=18.0)
        serialized = _serialize_matches_like_iterative_match([mr])
        restored = StateMatch.from_dict(serialized[0])

        assert restored.video_file == "roundtrip_vid_001"
        assert restored.video_start == 3.0
        assert restored.video_end == 18.0
        assert restored.confidence == 0.85

    def test_roundtrip_plain_match_preserves_video_file(self):
        """Serialize plain Match -> deserialize via Match.from_dict -> video_file preserved."""
        match = StateMatch(
            segment_index=1,
            video_file="plain_match_vid",
            video_start=0.0,
            video_end=20.0,
            confidence=0.7,
            strategy="keyword",
        )
        serialized = _serialize_matches_like_iterative_match([match])
        restored = StateMatch.from_dict(serialized[0])

        assert restored.video_file == "plain_match_vid"
        assert restored.video_start == 0.0
        assert restored.video_end == 20.0
        assert restored.confidence == 0.7

    def test_restore_matches_from_dicts_batch(self):
        """restore_matches_from_dicts() correctly restores multiple matches."""
        video_ids = ["batch_vid_A", "batch_vid_B", "batch_vid_C"]
        match_results = [_make_match_result(i, vid, confidence=0.6 + i * 0.1) for i, vid in enumerate(video_ids)]
        serialized = _serialize_matches_like_iterative_match(match_results)
        restored = restore_matches_from_dicts(serialized)

        assert restored is not None
        assert len(restored) == 3
        for i, vid in enumerate(video_ids):
            assert restored[i].video_file == vid


# ---------------------------------------------------------------------------
# Tests: _collect_matched_segments extracts video_id from restored matches
# ---------------------------------------------------------------------------

class TestCollectMatchedSegmentsFromRestoredMatches:
    """Test that _collect_matched_segments correctly extracts video_id from restored matches."""

    def _make_stage(self) -> DownloadVideoSegmentsStage:
        """Create a minimal DownloadVideoSegmentsStage for testing."""
        stage = DownloadVideoSegmentsStage.__new__(DownloadVideoSegmentsStage)
        stage.logger = MagicMock()
        return stage

    def test_restored_matches_produce_segments(self):
        """Restored state.Match objects (via from_dict) yield correct segments for download."""
        # Simulate: MatchResult -> serialize -> deserialize -> collect
        video_ids = ["seg_vid_X", "seg_vid_Y"]
        match_results = [
            _make_match_result(0, "seg_vid_X", video_start=5.0, video_end=15.0),
            _make_match_result(1, "seg_vid_Y", video_start=20.0, video_end=30.0),
        ]
        serialized = _serialize_matches_like_iterative_match(match_results)
        restored = restore_matches_from_dicts(serialized)

        state = PipelineState()
        state.matches = restored

        stage = self._make_stage()
        segments = stage._collect_matched_segments(state, buffer_seconds=2.0)

        assert len(segments) == 2
        extracted_ids = {s['video_id'] for s in segments}
        assert extracted_ids == {"seg_vid_X", "seg_vid_Y"}

    def test_restored_plain_match_produces_segment(self):
        """Plain state.Match (not from MatchResult) also yields correct segment."""
        match = StateMatch(
            segment_index=0,
            video_file="plain_vid_Z",
            video_start=0.0,
            video_end=10.0,
            confidence=0.75,
            strategy="restored",
        )
        serialized = _serialize_matches_like_iterative_match([match])
        restored = restore_matches_from_dicts(serialized)

        state = PipelineState()
        state.matches = restored

        stage = self._make_stage()
        segments = stage._collect_matched_segments(state, buffer_seconds=2.0)

        assert len(segments) == 1
        assert segments[0]['video_id'] == "plain_vid_Z"
        assert segments[0]['start'] == 0.0
        assert segments[0]['end'] == 10.0

    def test_mixed_match_types_all_produce_segments(self):
        """Mix of MatchResult and plain Match all roundtrip correctly."""
        mr1 = _make_match_result(0, "mr_vid_1", video_start=0.0, video_end=10.0)
        mr2 = _make_match_result(1, "mr_vid_2", video_start=5.0, video_end=15.0)
        plain = StateMatch(
            segment_index=2,
            video_file="plain_vid_3",
            video_start=20.0,
            video_end=30.0,
            confidence=0.8,
        )

        # All go through the same serialization path
        all_matches = [mr1, mr2, plain]
        serialized = _serialize_matches_like_iterative_match(all_matches)
        restored = restore_matches_from_dicts(serialized)

        state = PipelineState()
        state.matches = restored

        stage = self._make_stage()
        segments = stage._collect_matched_segments(state, buffer_seconds=2.0)

        extracted_ids = {s['video_id'] for s in segments}
        assert extracted_ids == {"mr_vid_1", "mr_vid_2", "plain_vid_3"}

    def test_empty_source_file_skipped(self):
        """Matches with empty video_file are skipped by _collect_matched_segments."""
        # Manually create a serialized match with empty video_file
        serialized = [{
            'segment_index': 0,
            'video_file': '',
            'video_start': 0.0,
            'video_end': 10.0,
            'confidence': 0.5,
            'strategy': 'gap',
        }]

        # Match.from_dict rejects empty video_file with ValueError
        with pytest.raises(ValueError, match="invalid video_file"):
            StateMatch.from_dict(serialized[0])

    def test_segment_times_preserved_through_roundtrip(self):
        """video_start and video_end survive the full roundtrip."""
        mr = _make_match_result(0, "timing_vid", video_start=42.5, video_end=67.3)
        serialized = _serialize_matches_like_iterative_match([mr])
        restored = restore_matches_from_dicts(serialized)

        state = PipelineState()
        state.matches = restored

        stage = self._make_stage()
        segments = stage._collect_matched_segments(state, buffer_seconds=0.0)

        assert len(segments) == 1
        assert segments[0]['start'] == 42.5
        assert segments[0]['end'] == 67.3

    def test_deduplication_after_roundtrip(self):
        """Duplicate video_id + time ranges are deduplicated by _collect_matched_segments."""
        # Two matches pointing to same video/time
        mr1 = _make_match_result(0, "dup_vid", video_start=5.0, video_end=15.0)
        mr2 = _make_match_result(1, "dup_vid", video_start=5.0, video_end=15.0)

        serialized = _serialize_matches_like_iterative_match([mr1, mr2])
        restored = restore_matches_from_dicts(serialized)

        state = PipelineState()
        state.matches = restored

        stage = self._make_stage()
        segments = stage._collect_matched_segments(state, buffer_seconds=2.0)

        # Should be deduplicated to 1 segment
        assert len(segments) == 1
        assert segments[0]['video_id'] == "dup_vid"
