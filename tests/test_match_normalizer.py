"""
Tests for MatchNormalizer - converts checkpoint match data to MatchResult format.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.matching.match_normalizer import MatchNormalizer
from src.state import PipelineState, Match
from src.utils import MatchResult, SRTSegment, AlternativeMatch, StrategyMatch


class TestMatchNormalizer:
    """Test MatchNormalizer.normalize() method."""

    def test_normalize_empty_matches(self):
        """Test that empty matches list returns empty list."""
        state = MagicMock(spec=PipelineState)
        state.matches = []
        state.voiceover_segments = []

        normalizer = MatchNormalizer()
        result = normalizer.normalize(state)

        assert result == []

    def test_normalize_already_matchresult_format(self):
        """Test that MatchResult objects pass through unchanged."""
        # Create a mock MatchResult
        mock_vo_seg = MagicMock(spec=SRTSegment)
        mock_video_seg = MagicMock(spec=SRTSegment)
        mock_primary = MagicMock()
        mock_primary.voiceover_segment = mock_vo_seg
        mock_primary.video_segment = mock_video_seg

        match_result = MagicMock(spec=MatchResult)
        match_result.primary_match = mock_primary

        state = MagicMock(spec=PipelineState)
        state.matches = [match_result]

        normalizer = MatchNormalizer()
        result = normalizer.normalize(state)

        assert result == [match_result]

    def test_normalize_simple_match(self):
        """Test normalizing simple Match objects to MatchResult."""
        # Create simple Match objects (like those restored from checkpoint)
        match1 = Match(
            segment_index=0,
            video_file='abc123',
            video_start=5.0,
            video_end=15.0,
            confidence=0.85,
            strategy='semantic',
            reason='Good match'
        )
        match2 = Match(
            segment_index=1,
            video_file='def456',
            video_start=20.0,
            video_end=30.0,
            confidence=0.75,
            strategy='embedding',
            reason='Decent match'
        )

        # Create voiceover segments
        vo_seg1 = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=10.0,
            text="Hello world",
            source_file="voiceover.srt"
        )
        vo_seg2 = SRTSegment(
            index=1,
            start_time=10.0,
            end_time=20.0,
            text="Welcome back",
            source_file="voiceover.srt"
        )

        state = MagicMock(spec=PipelineState)
        state.matches = [match1, match2]
        state.voiceover_segments = [vo_seg1, vo_seg2]

        normalizer = MatchNormalizer()
        result = normalizer.normalize(state)

        assert len(result) == 2

        # Check first result
        assert hasattr(result[0], 'primary_match')
        assert result[0].primary_match.video_segment.source_file == 'abc123'
        assert result[0].primary_match.video_segment.start_time == 5.0
        assert result[0].primary_match.video_segment.end_time == 15.0
        assert result[0].primary_match.confidence == 0.85
        assert result[0].primary_match.reasoning == 'Good match'

        # Check second result
        assert result[1].primary_match.video_segment.source_file == 'def456'
        assert result[1].primary_match.confidence == 0.75

    def test_normalize_with_multi_track_data(self):
        """Test normalizing with alternatives, secondary, and strategy matches (V1-V8)."""
        # Create simple Match object
        match = Match(
            segment_index=0,
            video_file='abc123',
            video_start=5.0,
            video_end=15.0,
            confidence=0.85,
            strategy='semantic',
            reason='Good match'
        )

        # Create voiceover segment
        vo_seg = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=10.0,
            text="Hello world",
            source_file="voiceover.srt"
        )

        # Create raw dict with multi-track data (simulating checkpoint data)
        # Note: AlternativeMatch.from_dict expects nested 'video_segment' dict
        raw_dicts = [{
            'alternatives': [
                {
                    'video_segment': {
                        'index': 0,
                        'start_time': 6.0,
                        'end_time': 16.0,
                        'text': '',
                        'source_file': 'alt1',
                    },
                    'confidence': 0.80,
                    'reasoning': 'Alternative match',
                }
            ],
            'secondary_matches': [
                {
                    'video_segment': {
                        'index': 0,
                        'start_time': 7.0,
                        'end_time': 17.0,
                        'text': '',
                        'source_file': 'sec1',
                    },
                    'confidence': 0.70,
                    'reasoning': 'Secondary match',
                }
            ],
            'strategy_matches': [
                {
                    'video_segment': {
                        'index': 0,
                        'start_time': 8.0,
                        'end_time': 18.0,
                        'text': '',
                        'source_file': 'strat1',
                    },
                    'confidence': 0.60,
                    'reasoning': 'Strategy match',
                }
            ],
            'has_gap': False,
            'gap_reason': '',
        }]

        # Create a real PipelineState-like object instead of MagicMock
        # to properly test _raw_match_dicts attribute access
        class MockState:
            def __init__(self):
                self.matches = [match]
                self.voiceover_segments = [vo_seg]
                self._raw_match_dicts = raw_dicts

        state = MockState()

        normalizer = MatchNormalizer()
        result = normalizer.normalize(state)

        assert len(result) == 1
        match_result = result[0]

        # Check alternatives (V2-V3)
        assert len(match_result.alternatives) == 1
        assert match_result.alternatives[0].video_segment.source_file == 'alt1'

        # Check secondary matches (V4-V6)
        assert len(match_result.secondary_matches) == 1
        assert match_result.secondary_matches[0].video_segment.source_file == 'sec1'

        # Check strategy matches (V7-V8)
        assert len(match_result.strategy_matches) == 1
        assert match_result.strategy_matches[0].video_segment.source_file == 'strat1'

        # Check gap flag
        assert match_result.has_gap is False

    def test_normalize_with_gap(self):
        """Test normalizing with has_gap flag."""
        match = Match(
            segment_index=0,
            video_file='',
            video_start=0.0,
            video_end=0.0,
            confidence=0.0,
            strategy='none',
            reason='No match found'
        )

        vo_seg = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=10.0,
            text="Hello world",
            source_file="voiceover.srt"
        )

        raw_dicts = [{
            'has_gap': True,
            'gap_reason': 'No suitable video found',
        }]

        class MockState:
            def __init__(self):
                self.matches = [match]
                self.voiceover_segments = [vo_seg]
                self._raw_match_dicts = raw_dicts

        state = MockState()

        normalizer = MatchNormalizer()
        result = normalizer.normalize(state)

        assert len(result) == 1
        assert result[0].has_gap is True
        assert result[0].gap_reason == 'No suitable video found'

    def test_normalize_out_of_bounds_segment_index(self):
        """Test normalizing when segment_index exceeds voiceover_segments length."""
        match = Match(
            segment_index=5,  # Out of bounds
            video_file='abc123',
            video_start=5.0,
            video_end=15.0,
            confidence=0.85,
            strategy='semantic',
            reason='Good match'
        )

        # Only 2 voiceover segments
        vo_seg1 = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=10.0,
            text="Hello",
            source_file="voiceover.srt"
        )
        vo_seg2 = SRTSegment(
            index=1,
            start_time=10.0,
            end_time=20.0,
            text="World",
            source_file="voiceover.srt"
        )

        state = MagicMock(spec=PipelineState)
        state.matches = [match]
        state.voiceover_segments = [vo_seg1, vo_seg2]

        normalizer = MatchNormalizer()
        result = normalizer.normalize(state)

        # Should create minimal voiceover segment
        assert len(result) == 1
        assert result[0].primary_match.voiceover_segment.index == 5
        assert result[0].primary_match.voiceover_segment.text == ""

    def test_normalize_cleans_up_raw_dicts(self):
        """Test that _raw_match_dicts is cleaned up after normalization."""
        match = Match(
            segment_index=0,
            video_file='abc123',
            video_start=5.0,
            video_end=15.0,
            confidence=0.85,
            strategy='semantic',
            reason='Good match'
        )

        vo_seg = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=10.0,
            text="Hello world",
            source_file="voiceover.srt"
        )

        class MockState:
            def __init__(self):
                self.matches = [match]
                self.voiceover_segments = [vo_seg]
                self._raw_match_dicts = [{'alternatives': []}]

        state = MockState()

        normalizer = MatchNormalizer()
        result = normalizer.normalize(state)

        # Verify _raw_match_dicts was deleted
        assert not hasattr(state, '_raw_match_dicts') or state._raw_match_dicts is None


class TestMatchNormalizerMultiTrack:
    """Test multi-track (V1-V8) match data extraction."""

    def test_extracts_all_track_types(self):
        """Test that all track types are correctly extracted."""
        match = Match(
            segment_index=0,
            video_file='primary',
            video_start=5.0,
            video_end=15.0,
            confidence=0.90,
            strategy='semantic',
            reason='Primary match'
        )

        vo_seg = SRTSegment(
            index=0,
            start_time=0.0,
            end_time=10.0,
            text="Test content",
            source_file="voiceover.srt"
        )

        # Simulate full multi-track data (V1-V8)
        # Note: AlternativeMatch.from_dict expects nested 'video_segment' dict
        raw_dicts = [{
            'alternatives': [
                {'video_segment': {'index': 0, 'start_time': 6.0, 'end_time': 16.0, 'text': '', 'source_file': 'v2_alt1'}, 'confidence': 0.85, 'reasoning': 'Alt 1'},
                {'video_segment': {'index': 0, 'start_time': 7.0, 'end_time': 17.0, 'text': '', 'source_file': 'v3_alt2'}, 'confidence': 0.80, 'reasoning': 'Alt 2'},
            ],
            'secondary_matches': [
                {'video_segment': {'index': 0, 'start_time': 8.0, 'end_time': 18.0, 'text': '', 'source_file': 'v4_sec1'}, 'confidence': 0.75, 'reasoning': 'Sec 1'},
                {'video_segment': {'index': 0, 'start_time': 9.0, 'end_time': 19.0, 'text': '', 'source_file': 'v5_sec2'}, 'confidence': 0.70, 'reasoning': 'Sec 2'},
                {'video_segment': {'index': 0, 'start_time': 10.0, 'end_time': 20.0, 'text': '', 'source_file': 'v6_sec3'}, 'confidence': 0.65, 'reasoning': 'Sec 3'},
            ],
            'strategy_matches': [
                {'video_segment': {'index': 0, 'start_time': 11.0, 'end_time': 21.0, 'text': '', 'source_file': 'v7_strat1'}, 'confidence': 0.60, 'reasoning': 'Strat 1'},
                {'video_segment': {'index': 0, 'start_time': 12.0, 'end_time': 22.0, 'text': '', 'source_file': 'v8_strat2'}, 'confidence': 0.55, 'reasoning': 'Strat 2'},
            ],
            'has_gap': False,
            'gap_reason': '',
        }]

        # Create a real object instead of MagicMock for proper attribute access
        class MockState:
            def __init__(self):
                self.matches = [match]
                self.voiceover_segments = [vo_seg]
                self._raw_match_dicts = raw_dicts

        state = MockState()

        normalizer = MatchNormalizer()
        result = normalizer.normalize(state)

        assert len(result) == 1
        mr = result[0]

        # V2-V3: Alternatives
        assert len(mr.alternatives) == 2

        # V4-V6: Secondary matches
        assert len(mr.secondary_matches) == 3

        # V7+: Strategy matches
        assert len(mr.strategy_matches) == 2
