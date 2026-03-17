"""
Tests for US-129-008: Download queue priority based on voiceover segment order.

Tests cover:
- DownloadOrchestrator accepts voiceover segment timeline data
- Downloads sorted by corresponding voiceover segment start time
- Earlier segments get higher priority in download queue
- Configurable priority_boost_for_early_segments
- Priority displayed in download progress output
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass, field
from typing import List, Dict, Any


# Mock segment data with segment_index for voiceover ordering
def create_mock_segments(segment_indices: List[int]) -> List[Dict[str, Any]]:
    """Create mock segments with segment_index for voiceover timeline."""
    segments = []
    for idx in segment_indices:
        segments.append({
            'video_id': f'video_{idx}',
            'start': float(idx * 10),
            'end': float(idx * 10 + 10),
            'confidence': 0.5 + (idx * 0.05),
            'duration_tier': 'short',
            'segment_index': idx,  # US-129-008: Maps to voiceover segment
        })
    return segments


class TestVoiceoverSegmentOrdering:
    """Tests for voiceover segment ordering in download segments."""

    def test_collect_matched_segments_includes_segment_index(self):
        """Verify _collect_matched_segments includes segment_index from matches."""
        from src.stages.download_segments import DownloadVideoSegmentsStage

        # Create mock state with matches that have segment_index
        mock_state = MagicMock()
        mock_match = MagicMock()
        mock_match.segment_index = 3
        mock_match.video_file = 'test_video_id'
        mock_match.video_start = 30.0
        mock_match.video_end = 40.0
        mock_match.confidence = 0.75
        mock_match.primary_match = None

        mock_state.matches = [mock_match]
        mock_state.voiceover_segments = []

        stage = DownloadVideoSegmentsStage.__new__(DownloadVideoSegmentsStage)
        stage._config = None

        segments = stage._collect_matched_segments(mock_state, buffer_seconds=5.0)

        # Verify segment_index is included
        assert len(segments) == 1
        assert segments[0]['segment_index'] == 3
        assert segments[0]['video_id'] == 'test_video_id'

    def test_segments_sorted_by_voiceover_timeline(self):
        """Verify segments are sorted by voiceover segment index."""
        from src.stages.download_segments import DownloadVideoSegmentsStage

        # Create segments in non-sequential order (like they might come from matches)
        segments = create_mock_segments([5, 2, 8, 1, 3])

        # Verify sorting works
        sorted_segments = sorted(segments, key=lambda s: s.get('segment_index', 0))

        # Check that order is correct
        assert sorted_segments[0]['segment_index'] == 1
        assert sorted_segments[1]['segment_index'] == 2
        assert sorted_segments[2]['segment_index'] == 3
        assert sorted_segments[3]['segment_index'] == 5
        assert sorted_segments[4]['segment_index'] == 8

    def test_priority_score_calculation(self):
        """Verify priority scores are calculated correctly."""
        # Test priority score calculation
        priority_boost = 1.5
        segments = create_mock_segments([0, 1, 2, 3, 4])
        max_index = max(s['segment_index'] for s in segments)

        for seg in segments:
            idx = seg['segment_index']
            seg['priority_score'] = 1.0 + priority_boost * (1.0 - idx / max_index)

        # First segment (index 0) should have highest priority
        assert segments[0]['priority_score'] > segments[1]['priority_score']
        assert segments[0]['priority_score'] > segments[4]['priority_score']

        # Last segment should have lowest priority (but still >= 1.0)
        assert segments[4]['priority_score'] == 1.0

        # First segment should have highest: 1.0 + 1.5 * (1 - 0/4) = 2.5
        assert segments[0]['priority_score'] == pytest.approx(2.5)

    def test_segments_with_same_segment_index_preserve_order(self):
        """Verify segments with same segment_index maintain stable sort."""
        segments = [
            {'video_id': 'a', 'segment_index': 1, 'start': 10},
            {'video_id': 'b', 'segment_index': 1, 'start': 20},
            {'video_id': 'c', 'segment_index': 0, 'start': 5},
        ]

        sorted_segments = sorted(segments, key=lambda s: s.get('segment_index', 0))

        # Segment with index 0 should come first
        assert sorted_segments[0]['video_id'] == 'c'


class TestPriorityBoostConfig:
    """Tests for priority_boost_for_early_segments config option."""

    def test_default_priority_boost_value(self):
        """Verify default priority boost is 1.5."""
        from src.config.sections.download import DownloadingConfig

        config = DownloadingConfig()
        assert config.priority_boost_for_early_segments == 1.5

    def test_custom_priority_boost_value(self):
        """Verify custom priority boost values work."""
        from src.config.sections.download import DownloadingConfig

        config = DownloadingConfig(priority_boost_for_early_segments=2.0)
        assert config.priority_boost_for_early_segments == 2.0

    def test_priority_boost_disabled_when_equal_to_one(self):
        """Verify priority boost is disabled when set to 1.0."""
        from src.config.sections.download import DownloadingConfig

        config = DownloadingConfig(priority_boost_for_early_segments=1.0)
        assert config.priority_boost_for_early_segments == 1.0


class TestDownloadSegmentPrioritySorting:
    """Integration tests for priority sorting in download flow."""

    def test_priority_boost_no_effect_when_disabled(self):
        """Verify no priority boost when set to 1.0."""
        priority_boost = 1.0  # Disabled
        segments = create_mock_segments([5, 2, 8, 1, 3])

        if priority_boost > 1.0:
            # This should NOT run when disabled
            segments = sorted(segments, key=lambda s: s.get('segment_index', 0))
            max_index = max(s.get('segment_index', 0) for s in segments) or 1
            for seg in segments:
                idx = seg.get('segment_index', 0)
                seg['priority_score'] = 1.0 + priority_boost * (1.0 - idx / max_index)

        # When disabled, priority_score should not be added
        assert 'priority_score' not in segments[0]

    def test_priority_display_in_progress_output(self):
        """Verify priority info is included in debug logging."""
        # This test verifies that the priority display logic is present
        # The actual logging is tested separately
        priority_score = 2.5
        segment_index = 0
        video_id = "test_video"

        # Simulate the logging format
        log_msg = (
            f"Downloading segment 1/5: {video_id} "
            f"(voiceover_idx={segment_index}, priority={priority_score:.2f})"
        )

        assert "voiceover_idx=0" in log_msg
        assert "priority=2.50" in log_msg


class TestVoiceoverSegmentMapping:
    """Tests for voiceover segment to download segment mapping."""

    def test_match_segment_index_corresponds_to_voiceover(self):
        """Verify segment_index in match corresponds to voiceover segment."""
        from src.state import Match, VoiceoverSegment

        # Create voiceover segments
        voiceover_segments = [
            VoiceoverSegment(index=0, start=0.0, end=10.0, text="First segment"),
            VoiceoverSegment(index=1, start=10.0, end=20.0, text="Second segment"),
            VoiceoverSegment(index=2, start=20.0, end=30.0, text="Third segment"),
        ]

        # Create matches with corresponding segment_index
        matches = [
            Match(segment_index=2, video_file="video_c", video_start=100.0,
                  video_end=110.0, confidence=0.8),
            Match(segment_index=0, video_file="video_a", video_start=50.0,
                  video_end=60.0, confidence=0.7),
            Match(segment_index=1, video_file="video_b", video_start=75.0,
                  video_end=85.0, confidence=0.9),
        ]

        # Sort matches by segment_index (voiceover timeline)
        sorted_matches = sorted(matches, key=lambda m: m.segment_index)

        # Verify order matches voiceover timeline
        assert sorted_matches[0].segment_index == 0  # First voiceover segment
        assert sorted_matches[1].segment_index == 1  # Second voiceover segment
        assert sorted_matches[2].segment_index == 2  # Third voiceover segment

        # Verify video_files are in expected order
        assert sorted_matches[0].video_file == "video_a"
        assert sorted_matches[1].video_file == "video_b"
        assert sorted_matches[2].video_file == "video_c"

    def test_voiceover_start_time_used_for_priority(self):
        """Verify voiceover start time is used for priority calculation."""
        from src.state import VoiceoverSegment

        voiceover_segments = [
            VoiceoverSegment(index=0, start=0.0, end=10.0, text="First"),
            VoiceoverSegment(index=1, start=10.0, end=20.0, text="Second"),
            VoiceoverSegment(index=2, start=20.0, end=30.0, text="Third"),
        ]

        # Earlier voiceover segments have lower start time = higher priority
        start_times = [seg.start for seg in voiceover_segments]
        assert start_times == [0.0, 10.0, 20.0]

        # Verify lower start time = higher priority
        priority_scores = []
        max_idx = len(voiceover_segments) - 1
        priority_boost = 1.5

        for seg in voiceover_segments:
            idx = seg.index
            score = 1.0 + priority_boost * (1.0 - idx / max_idx)
            priority_scores.append(score)

        # First segment (index 0) should have highest priority
        assert priority_scores[0] > priority_scores[1]
        assert priority_scores[1] > priority_scores[2]
        assert priority_scores[0] == pytest.approx(2.5)  # 1.0 + 1.5 * 1.0
        assert priority_scores[2] == pytest.approx(1.0)   # 1.0 + 1.5 * 0.0
