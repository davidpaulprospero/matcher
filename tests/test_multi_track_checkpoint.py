"""
Tests for multi-track checkpoint serialization (V2-V8 tracks).

Verifies that alternatives (V2-V3), secondary_matches (V4-V6), strategy_matches
(V7-V8), has_gap, and gap_reason survive checkpoint save → restore → output.
"""

import pytest
from unittest.mock import MagicMock
from dataclasses import dataclass, field
from typing import List

from src.utils import (
    SRTSegment, Match as UtilsMatch, MatchResult,
    AlternativeMatch, StrategyMatch,
)
from src.matching.serialization import (
    _extract_multi_track_data,
    serialize_match_for_match_stage,
    serialize_match_for_iterative_stage,
)
from src.state import Match as StateMatch, PipelineState, restore_matches_from_dicts


# ============================================================================
# Fixtures
# ============================================================================

def _make_vo_seg(index=0):
    return SRTSegment(index=index, start_time=0.0, end_time=5.0, text='voiceover text')


def _make_video_seg(source_file='video1.mp4', start=0.0, end=10.0):
    return SRTSegment(index=0, start_time=start, end_time=end, text='', source_file=source_file)


def _make_alternative(source_file='alt_video.mp4', confidence=0.7, diversity=0.3):
    return AlternativeMatch(
        video_segment=_make_video_seg(source_file),
        video_scene=None,
        confidence=confidence,
        reasoning='alternative reasoning',
        diversity_score=diversity,
    )


def _make_strategy(source_file='strat_video.mp4', confidence=0.6, strategy='embedding_diversity'):
    return StrategyMatch(
        video_segment=_make_video_seg(source_file),
        video_scene=None,
        confidence=confidence,
        reasoning='strategy reasoning',
        strategy=strategy,
    )


def _make_match_result_with_tracks():
    """Create a fully-populated MatchResult with V2-V8 data."""
    primary = UtilsMatch(
        voiceover_segment=_make_vo_seg(),
        video_segment=_make_video_seg('primary.mp4'),
        video_scene=None,
        confidence=0.9,
        reasoning='primary match',
    )
    return MatchResult(
        primary_match=primary,
        alternatives=[
            _make_alternative('alt1.mp4', 0.8, 0.1),
            _make_alternative('alt2.mp4', 0.75, 0.2),
        ],
        secondary_matches=[
            _make_alternative('sec1.mp4', 0.65, 0.8),
            _make_alternative('sec2.mp4', 0.6, 0.9),
            _make_alternative('sec3.mp4', 0.55, 0.95),
        ],
        strategy_matches=[
            _make_strategy('strat1.mp4', 0.5, 'visual_first'),
            _make_strategy('strat2.mp4', 0.45, 'embedding_diversity'),
        ],
        has_gap=True,
        gap_reason='low confidence fallback',
    )


# ============================================================================
# Test: _extract_multi_track_data
# ============================================================================

class TestExtractMultiTrackData:
    """Test the _extract_multi_track_data helper."""

    def test_extracts_alternatives_from_match_result(self):
        """Alternatives are serialized as list of dicts."""
        mr = _make_match_result_with_tracks()
        data = _extract_multi_track_data(mr)
        assert len(data['alternatives']) == 2
        assert data['alternatives'][0]['video_segment']['source_file'] == 'alt1.mp4'
        assert data['alternatives'][1]['confidence'] == 0.75

    def test_extracts_secondary_matches(self):
        """Secondary matches are serialized."""
        mr = _make_match_result_with_tracks()
        data = _extract_multi_track_data(mr)
        assert len(data['secondary_matches']) == 3
        assert data['secondary_matches'][0]['video_segment']['source_file'] == 'sec1.mp4'
        assert data['secondary_matches'][2]['diversity_score'] == 0.95

    def test_extracts_strategy_matches(self):
        """Strategy matches are serialized with strategy field."""
        mr = _make_match_result_with_tracks()
        data = _extract_multi_track_data(mr)
        assert len(data['strategy_matches']) == 2
        assert data['strategy_matches'][0]['strategy'] == 'visual_first'
        assert data['strategy_matches'][1]['strategy'] == 'embedding_diversity'

    def test_extracts_has_gap_and_gap_reason(self):
        """has_gap and gap_reason are preserved."""
        mr = _make_match_result_with_tracks()
        data = _extract_multi_track_data(mr)
        assert data['has_gap'] is True
        assert data['gap_reason'] == 'low confidence fallback'

    def test_returns_empty_for_plain_match(self):
        """Non-MatchResult objects return empty multi-track data."""
        plain = UtilsMatch(
            voiceover_segment=_make_vo_seg(),
            video_segment=_make_video_seg(),
            video_scene=None, confidence=0.8, reasoning='plain',
        )
        data = _extract_multi_track_data(plain)
        assert data['alternatives'] == []
        assert data['secondary_matches'] == []
        assert data['strategy_matches'] == []
        assert data['has_gap'] is False
        assert data['gap_reason'] == ''

    def test_returns_empty_for_state_match(self):
        """state.Match (flat checkpoint match) returns empty multi-track data."""
        flat = StateMatch(segment_index=0, video_file='v.mp4', video_start=0.0, video_end=5.0, confidence=0.5)
        data = _extract_multi_track_data(flat)
        assert data['alternatives'] == []
        assert data['strategy_matches'] == []

    def test_handles_none_alternatives_gracefully(self):
        """MatchResult with None alternatives doesn't crash."""
        mr = MatchResult(
            primary_match=UtilsMatch(
                voiceover_segment=_make_vo_seg(),
                video_segment=_make_video_seg(),
                video_scene=None, confidence=0.8, reasoning='ok',
            ),
            alternatives=None,
            secondary_matches=None,
            strategy_matches=None,
        )
        data = _extract_multi_track_data(mr)
        assert data['alternatives'] == []
        assert data['secondary_matches'] == []
        assert data['strategy_matches'] == []

    def test_has_gap_false_by_default(self):
        """MatchResult without gap set defaults to False."""
        mr = MatchResult(
            primary_match=UtilsMatch(
                voiceover_segment=_make_vo_seg(),
                video_segment=_make_video_seg(),
                video_scene=None, confidence=0.8, reasoning='ok',
            ),
        )
        data = _extract_multi_track_data(mr)
        assert data['has_gap'] is False
        assert data['gap_reason'] == ''


# ============================================================================
# Test: Serialization includes multi-track data
# ============================================================================

class TestSerializationIncludesMultiTrack:
    """Verify serialize_match_for_*_stage includes multi-track keys."""

    def test_match_stage_includes_alternatives(self):
        """Match stage serializer includes alternatives in output."""
        mr = _make_match_result_with_tracks()
        result = serialize_match_for_match_stage(mr, index=0)
        assert len(result['alternatives']) == 2
        assert len(result['secondary_matches']) == 3
        assert len(result['strategy_matches']) == 2
        assert result['has_gap'] is True
        assert result['gap_reason'] == 'low confidence fallback'

    def test_iterative_stage_includes_alternatives(self):
        """Iterative stage serializer includes alternatives in output."""
        mr = _make_match_result_with_tracks()
        result = serialize_match_for_iterative_stage(mr, index=0)
        assert len(result['alternatives']) == 2
        assert len(result['secondary_matches']) == 3
        assert len(result['strategy_matches']) == 2
        assert result['has_gap'] is True

    def test_match_stage_empty_for_plain_match(self):
        """Plain utils.Match produces empty multi-track arrays."""
        match = UtilsMatch(
            voiceover_segment=_make_vo_seg(),
            video_segment=_make_video_seg('plain.mp4'),
            video_scene=None, confidence=0.85, reasoning='test',
        )
        result = serialize_match_for_match_stage(match, index=0)
        assert result['alternatives'] == []
        assert result['secondary_matches'] == []
        assert result['strategy_matches'] == []
        assert result['has_gap'] is False

    def test_primary_match_data_not_affected(self):
        """Multi-track merge doesn't corrupt primary match fields."""
        mr = _make_match_result_with_tracks()
        result = serialize_match_for_match_stage(mr, index=5)
        assert result['segment_index'] == 5
        assert result['source_file'] == 'primary.mp4'
        assert result['confidence'] == 0.9

    def test_iterative_primary_data_not_affected(self):
        """Multi-track merge doesn't corrupt iterative primary fields."""
        mr = _make_match_result_with_tracks()
        result = serialize_match_for_iterative_stage(mr, index=3)
        assert result['video_file'] == 'primary.mp4'
        assert result['confidence'] == 0.9


# ============================================================================
# Test: Round-trip serialization → deserialization
# ============================================================================

class TestMultiTrackRoundTrip:
    """Verify multi-track data survives serialize → deserialize round trip."""

    def test_alternative_round_trip(self):
        """AlternativeMatch survives to_dict → from_dict."""
        alt = _make_alternative('rt_alt.mp4', 0.72, 0.4)
        d = alt.to_dict()
        restored = AlternativeMatch.from_dict(d)
        assert restored.video_segment.source_file == 'rt_alt.mp4'
        assert restored.confidence == 0.72
        assert restored.diversity_score == 0.4
        assert restored.reasoning == 'alternative reasoning'

    def test_strategy_match_round_trip(self):
        """StrategyMatch survives to_dict → from_dict."""
        strat = _make_strategy('rt_strat.mp4', 0.55, 'visual_first')
        d = strat.to_dict()
        restored = StrategyMatch.from_dict(d)
        assert restored.video_segment.source_file == 'rt_strat.mp4'
        assert restored.confidence == 0.55
        assert restored.strategy == 'visual_first'

    def test_full_match_result_round_trip(self):
        """Full MatchResult with all tracks survives to_dict → from_dict."""
        mr = _make_match_result_with_tracks()
        d = mr.to_dict()
        restored = MatchResult.from_dict(d)
        assert len(restored.alternatives) == 2
        assert len(restored.secondary_matches) == 3
        assert len(restored.strategy_matches) == 2
        assert restored.has_gap is True
        assert restored.gap_reason == 'low confidence fallback'
        assert restored.alternatives[0].video_segment.source_file == 'alt1.mp4'
        assert restored.strategy_matches[1].strategy == 'embedding_diversity'

    def test_serialize_then_restore_alternatives(self):
        """Checkpoint-style serialize → raw dict → AlternativeMatch.from_dict."""
        mr = _make_match_result_with_tracks()
        serialized = serialize_match_for_iterative_stage(mr, index=0)

        # Simulate checkpoint restore: read raw dict, reconstruct
        restored_alts = [
            AlternativeMatch.from_dict(a) for a in serialized['alternatives']
        ]
        assert len(restored_alts) == 2
        assert restored_alts[0].video_segment.source_file == 'alt1.mp4'
        assert restored_alts[1].confidence == 0.75

    def test_serialize_then_restore_strategy_matches(self):
        """Checkpoint-style serialize → raw dict → StrategyMatch.from_dict."""
        mr = _make_match_result_with_tracks()
        serialized = serialize_match_for_match_stage(mr, index=0)

        restored_strats = [
            StrategyMatch.from_dict(s) for s in serialized['strategy_matches']
        ]
        assert len(restored_strats) == 2
        assert restored_strats[0].strategy == 'visual_first'


# ============================================================================
# Test: Backward compatibility
# ============================================================================

class TestBackwardCompatibility:
    """Old checkpoints without multi-track keys still work."""

    def test_old_checkpoint_dict_missing_multi_track_keys(self):
        """Old checkpoint dict (no multi-track keys) produces empty lists via .get()."""
        old_dict = {
            'segment_index': 0,
            'video_file': 'old_vid.mp4',
            'video_start': 1.0,
            'video_end': 5.0,
            'confidence': 0.8,
            'strategy': 'semantic',
            'reason': 'good match',
            'face_score': 0.4,
        }
        # Simulating what output.py does with raw dicts
        alternatives = [
            AlternativeMatch.from_dict(a) for a in old_dict.get('alternatives', [])
        ]
        secondary = [
            AlternativeMatch.from_dict(a) for a in old_dict.get('secondary_matches', [])
        ]
        strategy = [
            StrategyMatch.from_dict(s) for s in old_dict.get('strategy_matches', [])
        ]
        has_gap = bool(old_dict.get('has_gap', False))
        gap_reason = old_dict.get('gap_reason', '') or ''

        assert alternatives == []
        assert secondary == []
        assert strategy == []
        assert has_gap is False
        assert gap_reason == ''

    def test_restore_matches_ignores_extra_keys(self):
        """restore_matches_from_dicts ignores multi-track keys (state.Match doesn't use them)."""
        new_dict = {
            'segment_index': 0,
            'video_file': 'new_vid.mp4',
            'video_start': 1.0,
            'video_end': 5.0,
            'confidence': 0.8,
            'strategy': 'semantic',
            'reason': 'good match',
            'face_score': 0.4,
            # Extra keys from new serialization
            'alternatives': [{'video_segment': {'source_file': 'a.mp4'}}],
            'secondary_matches': [],
            'strategy_matches': [],
            'has_gap': True,
            'gap_reason': 'test',
        }
        import logging
        matches = restore_matches_from_dicts([new_dict], logger_instance=logging.getLogger('test'))
        assert matches is not None
        assert len(matches) == 1
        assert matches[0].video_file == 'new_vid.mp4'
        # state.Match doesn't store multi-track data — just ensures no crash
        assert not hasattr(matches[0], 'alternatives') or getattr(matches[0], 'alternatives', None) is None


# ============================================================================
# Test: Output stage _normalize_matches with multi-track restore
# ============================================================================

class TestNormalizeMatchesMultiTrack:
    """Test that _normalize_matches restores multi-track data from raw dicts."""

    def _make_state_with_raw_dicts(self):
        """Create PipelineState with matches and _raw_match_dicts simulating checkpoint restore."""
        state = PipelineState()
        state.voiceover_segments = [
            SRTSegment(index=0, start_time=0.0, end_time=5.0, text='segment 0'),
            SRTSegment(index=1, start_time=5.0, end_time=10.0, text='segment 1'),
        ]
        state.matches = [
            StateMatch(segment_index=0, video_file='v1.mp4', video_start=0.0,
                       video_end=10.0, confidence=0.9),
            StateMatch(segment_index=1, video_file='v2.mp4', video_start=2.0,
                       video_end=8.0, confidence=0.7),
        ]

        # Simulate raw checkpoint dicts with multi-track data
        alt1 = _make_alternative('alt_a.mp4', 0.8, 0.3)
        strat1 = _make_strategy('strat_a.mp4', 0.5, 'visual_first')
        sec1 = _make_alternative('sec_a.mp4', 0.6, 0.9)

        state._raw_match_dicts = [
            {
                'segment_index': 0, 'video_file': 'v1.mp4',
                'alternatives': [alt1.to_dict()],
                'secondary_matches': [sec1.to_dict()],
                'strategy_matches': [strat1.to_dict()],
                'has_gap': False, 'gap_reason': '',
            },
            {
                'segment_index': 1, 'video_file': 'v2.mp4',
                'alternatives': [],
                'secondary_matches': [],
                'strategy_matches': [],
                'has_gap': True, 'gap_reason': 'low conf',
            },
        ]
        return state

    def test_normalize_restores_alternatives(self):
        """_normalize_matches reconstructs AlternativeMatch objects from raw dicts."""
        from src.stages.output import OutputStage
        stage = OutputStage()
        state = self._make_state_with_raw_dicts()

        normalized = stage._normalize_matches(state)

        assert len(normalized) == 2
        assert len(normalized[0].alternatives) == 1
        assert normalized[0].alternatives[0].video_segment.source_file == 'alt_a.mp4'
        assert normalized[0].alternatives[0].confidence == 0.8

    def test_normalize_restores_secondary_matches(self):
        """_normalize_matches reconstructs secondary AlternativeMatch objects."""
        from src.stages.output import OutputStage
        stage = OutputStage()
        state = self._make_state_with_raw_dicts()

        normalized = stage._normalize_matches(state)

        assert len(normalized[0].secondary_matches) == 1
        assert normalized[0].secondary_matches[0].video_segment.source_file == 'sec_a.mp4'
        assert normalized[0].secondary_matches[0].diversity_score == 0.9

    def test_normalize_restores_strategy_matches(self):
        """_normalize_matches reconstructs StrategyMatch objects."""
        from src.stages.output import OutputStage
        stage = OutputStage()
        state = self._make_state_with_raw_dicts()

        normalized = stage._normalize_matches(state)

        assert len(normalized[0].strategy_matches) == 1
        assert normalized[0].strategy_matches[0].strategy == 'visual_first'
        assert normalized[0].strategy_matches[0].video_segment.source_file == 'strat_a.mp4'

    def test_normalize_restores_has_gap(self):
        """_normalize_matches restores has_gap and gap_reason."""
        from src.stages.output import OutputStage
        stage = OutputStage()
        state = self._make_state_with_raw_dicts()

        normalized = stage._normalize_matches(state)

        assert normalized[0].has_gap is False
        assert normalized[0].gap_reason == ''
        assert normalized[1].has_gap is True
        assert normalized[1].gap_reason == 'low conf'

    def test_normalize_cleans_up_raw_dicts(self):
        """_normalize_matches removes _raw_match_dicts from state after conversion."""
        from src.stages.output import OutputStage
        stage = OutputStage()
        state = self._make_state_with_raw_dicts()

        stage._normalize_matches(state)

        assert not hasattr(state, '_raw_match_dicts')

    def test_normalize_without_raw_dicts(self):
        """Without _raw_match_dicts, normalize produces empty multi-track lists."""
        from src.stages.output import OutputStage
        stage = OutputStage()
        state = PipelineState()
        state.voiceover_segments = [
            SRTSegment(index=0, start_time=0.0, end_time=5.0, text='seg'),
        ]
        state.matches = [
            StateMatch(segment_index=0, video_file='v.mp4', video_start=0.0, video_end=5.0, confidence=0.8),
        ]

        normalized = stage._normalize_matches(state)

        assert len(normalized) == 1
        assert normalized[0].alternatives == []
        assert normalized[0].secondary_matches == []
        assert normalized[0].strategy_matches == []

    def test_normalize_preserves_primary_match(self):
        """Primary match data is correct regardless of multi-track presence."""
        from src.stages.output import OutputStage
        stage = OutputStage()
        state = self._make_state_with_raw_dicts()

        normalized = stage._normalize_matches(state)

        assert normalized[0].primary_match.video_segment.source_file == 'v1.mp4'
        assert normalized[0].primary_match.confidence == 0.9
        assert normalized[1].primary_match.video_segment.source_file == 'v2.mp4'
        assert normalized[1].primary_match.confidence == 0.7


# ============================================================================
# Test: Match stage restore stashes raw dicts
# ============================================================================

class TestMatchStageRestoreStashesRawDicts:
    """Verify MATCH stage restore() sets _raw_match_dicts on state."""

    def test_restore_sets_raw_match_dicts(self):
        """MATCH restore() stashes raw checkpoint dicts on state."""
        from src.stages.match import MatchStage
        stage = MatchStage()
        state = PipelineState()

        mock_checkpoint = MagicMock()
        mock_checkpoint.get_stage_data.return_value = {
            'matches': [
                {'segment_index': 0, 'video_file': 'v.mp4', 'confidence': 0.8,
                 'alternatives': [{'video_segment': {'source_file': 'a.mp4',
                 'start_time': 0, 'end_time': 5, 'text': '', 'index': 0},
                 'video_scene': None, 'confidence': 0.7, 'reasoning': 'alt',
                 'diversity_score': 0.3}]},
            ]
        }

        result = stage.restore(state, mock_checkpoint)
        assert result is True
        assert hasattr(state, '_raw_match_dicts')
        assert len(state._raw_match_dicts) == 1
        assert 'alternatives' in state._raw_match_dicts[0]

    def test_restore_without_matches_no_raw_dicts(self):
        """MATCH restore() without matches data doesn't set _raw_match_dicts."""
        from src.stages.match import MatchStage
        stage = MatchStage()
        state = PipelineState()

        mock_checkpoint = MagicMock()
        mock_checkpoint.get_stage_data.return_value = {
            'match_count': 0,
        }

        result = stage.restore(state, mock_checkpoint)
        assert result is True
        assert not hasattr(state, '_raw_match_dicts')


# ============================================================================
# Test: Iterative match stage restore stashes raw dicts
# ============================================================================

class TestIterativeMatchRestoreStashesRawDicts:
    """Verify ITERATIVE_MATCH restore() sets _raw_match_dicts on state."""

    def test_restore_sets_raw_match_dicts(self):
        """ITERATIVE_MATCH restore() stashes raw checkpoint dicts on state."""
        from src.stages.iterative_match import IterativeMatchStage
        stage = IterativeMatchStage()
        state = PipelineState()

        mock_checkpoint = MagicMock()
        mock_checkpoint.get_stage_data.return_value = {
            'matches': [
                {'segment_index': 0, 'video_file': 'v.mp4', 'confidence': 0.8,
                 'strategy_matches': [{'video_segment': {'source_file': 's.mp4',
                 'start_time': 0, 'end_time': 5, 'text': '', 'index': 0},
                 'video_scene': None, 'confidence': 0.5, 'reasoning': 'strat',
                 'strategy': 'visual_first'}]},
            ]
        }

        result = stage.restore(state, mock_checkpoint)
        assert result is True
        assert hasattr(state, '_raw_match_dicts')
        assert len(state._raw_match_dicts) == 1
        assert 'strategy_matches' in state._raw_match_dicts[0]
