"""
Tests for harmonized match serialization/deserialization between MATCH and ITERATIVE_MATCH.

Verifies that both stages produce identical Match objects from the same checkpoint
dict input, including confidence clamping, video_file validation, and type coercion.
"""

import pytest
from unittest.mock import MagicMock

from src.state import Match, restore_matches_from_dicts
from src.stages.match import MatchStage
from src.stages.iterative_match import IterativeMatchStage
from src.state import PipelineState


# ============================================================================
# Test Data
# ============================================================================

VALID_MATCH_DICT = {
    'segment_index': 0,
    'video_file': 'abc123',
    'video_start': 5.0,
    'video_end': 15.0,
    'confidence': 0.85,
    'strategy': 'semantic',
    'reason': 'High similarity',
    'face_score': 0.3,
}

LEGACY_MATCH_DICT = {
    'segment_index': 1,
    'source_file': 'def456',  # old format
    'start_time': 2.0,  # old format
    'confidence': 0.7,
    'strategy': 'embedding',
    'reason': 'Embedding match',
}


# ============================================================================
# Match.from_dict unit tests
# ============================================================================

class TestMatchFromDict:
    """Test Match.from_dict classmethod."""

    def test_valid_dict(self):
        match = Match.from_dict(VALID_MATCH_DICT)
        assert match.segment_index == 0
        assert match.video_file == 'abc123'
        assert match.video_start == 5.0
        assert match.video_end == 15.0
        assert match.confidence == 0.85
        assert match.strategy == 'semantic'
        assert match.reason == 'High similarity'
        assert match.face_score == 0.3

    def test_legacy_source_file_format(self):
        match = Match.from_dict(LEGACY_MATCH_DICT)
        assert match.video_file == 'def456'
        assert match.video_start == 2.0
        # video_end defaults to video_start + 10.0 when missing
        assert match.video_end == 12.0

    def test_confidence_clamp_negative(self):
        data = {**VALID_MATCH_DICT, 'confidence': -0.5}
        match = Match.from_dict(data)
        assert match.confidence == 0.0

    def test_confidence_clamp_above_one(self):
        data = {**VALID_MATCH_DICT, 'confidence': 1.5}
        match = Match.from_dict(data)
        assert match.confidence == 1.0

    def test_confidence_clamp_large_negative(self):
        data = {**VALID_MATCH_DICT, 'confidence': -100.0}
        match = Match.from_dict(data)
        assert match.confidence == 0.0

    def test_confidence_clamp_large_positive(self):
        data = {**VALID_MATCH_DICT, 'confidence': 999.9}
        match = Match.from_dict(data)
        assert match.confidence == 1.0

    def test_confidence_boundary_zero(self):
        data = {**VALID_MATCH_DICT, 'confidence': 0.0}
        match = Match.from_dict(data)
        assert match.confidence == 0.0

    def test_confidence_boundary_one(self):
        data = {**VALID_MATCH_DICT, 'confidence': 1.0}
        match = Match.from_dict(data)
        assert match.confidence == 1.0

    def test_rejects_not_dict(self):
        with pytest.raises(ValueError, match="not a dict"):
            Match.from_dict("not a dict")

    def test_rejects_empty_video_file(self):
        data = {**VALID_MATCH_DICT, 'video_file': ''}
        with pytest.raises(ValueError, match="invalid video_file"):
            Match.from_dict(data)

    def test_rejects_none_video_file(self):
        data = {**VALID_MATCH_DICT, 'video_file': None}
        # source_file also absent, so falls back to ''
        data.pop('source_file', None)
        with pytest.raises(ValueError, match="invalid video_file"):
            Match.from_dict(data)

    def test_rejects_invalid_segment_index_type(self):
        data = {**VALID_MATCH_DICT, 'segment_index': 'bad'}
        with pytest.raises(ValueError, match="invalid segment_index"):
            Match.from_dict(data)

    def test_rejects_invalid_confidence_type(self):
        data = {**VALID_MATCH_DICT, 'confidence': 'high'}
        with pytest.raises(ValueError, match="invalid confidence"):
            Match.from_dict(data)

    def test_default_strategy(self):
        data = {**VALID_MATCH_DICT}
        del data['strategy']
        match = Match.from_dict(data, default_strategy='test_default')
        assert match.strategy == 'test_default'

    def test_index_fallback_for_segment_index(self):
        data = {**VALID_MATCH_DICT}
        del data['segment_index']
        match = Match.from_dict(data, index=7)
        assert match.segment_index == 7

    def test_float_segment_index_coerced(self):
        data = {**VALID_MATCH_DICT, 'segment_index': 3.0}
        match = Match.from_dict(data)
        assert match.segment_index == 3
        assert isinstance(match.segment_index, int)

    def test_int_confidence_coerced(self):
        data = {**VALID_MATCH_DICT, 'confidence': 1}
        match = Match.from_dict(data)
        assert match.confidence == 1.0
        assert isinstance(match.confidence, float)


# ============================================================================
# restore_matches_from_dicts unit tests
# ============================================================================

class TestRestoreMatchesFromDicts:
    """Test the shared restore_matches_from_dicts helper."""

    def test_valid_list(self):
        matches = restore_matches_from_dicts([VALID_MATCH_DICT])
        assert len(matches) == 1
        assert matches[0].video_file == 'abc123'

    def test_not_a_list_returns_none(self):
        result = restore_matches_from_dicts("not a list")
        assert result is None

    def test_empty_list(self):
        result = restore_matches_from_dicts([])
        assert result == []

    def test_all_invalid_returns_none(self):
        invalid = [{'video_file': ''}, {'not_a_match': True}]
        result = restore_matches_from_dicts(invalid)
        assert result is None

    def test_mixed_valid_invalid(self):
        data = [VALID_MATCH_DICT, {'video_file': ''}, LEGACY_MATCH_DICT]
        result = restore_matches_from_dicts(data)
        assert len(result) == 2

    def test_default_strategy_propagated(self):
        data = [{**VALID_MATCH_DICT}]
        del data[0]['strategy']
        result = restore_matches_from_dicts(data, default_strategy='iterative_restored')
        assert result[0].strategy == 'iterative_restored'


# ============================================================================
# Stage restore harmonization tests
# ============================================================================

class TestStageRestoreHarmonization:
    """Verify MATCH and ITERATIVE_MATCH produce identical Match objects from same data."""

    @pytest.fixture
    def match_stage(self):
        return MatchStage()

    @pytest.fixture
    def iterative_stage(self):
        return IterativeMatchStage()

    @pytest.fixture
    def state(self):
        return PipelineState()

    def _make_checkpoint(self, matches_data, stage_name='MATCH'):
        checkpoint = MagicMock()
        checkpoint.should_skip_stage.return_value = False
        checkpoint.get_stage_data.return_value = {
            'matches': matches_data,
            'passes_completed': 1,
            'total_gaps_filled': 0,
        }
        return checkpoint

    def test_identical_objects_from_same_input(self, match_stage, iterative_stage):
        """Both stages produce equivalent Match objects from same checkpoint dict."""
        test_data = [
            {
                'segment_index': 0,
                'video_file': 'vid_001',
                'video_start': 0.0,
                'video_end': 10.0,
                'confidence': 0.9,
                'strategy': 'semantic',
                'reason': 'test',
                'face_score': 0.4,
            },
            {
                'segment_index': 1,
                'video_file': 'vid_002',
                'video_start': 5.0,
                'video_end': 15.0,
                'confidence': 0.6,
                'strategy': 'embedding',
                'reason': 'fallback',
                'face_score': 0.7,
            },
        ]

        state_match = PipelineState()
        state_iter = PipelineState()

        cp_match = self._make_checkpoint(test_data, 'MATCH')
        cp_iter = self._make_checkpoint(test_data, 'ITERATIVE_MATCH')

        assert match_stage.restore(state_match, cp_match) is True
        assert iterative_stage.restore(state_iter, cp_iter) is True

        assert len(state_match.matches) == len(state_iter.matches)
        for m1, m2 in zip(state_match.matches, state_iter.matches):
            assert m1.segment_index == m2.segment_index
            assert m1.video_file == m2.video_file
            assert m1.video_start == m2.video_start
            assert m1.video_end == m2.video_end
            assert m1.confidence == m2.confidence
            # strategy differs by design (restored vs iterative_restored) only when
            # the source data has no strategy key; when present, both use source value
            assert m1.strategy == m2.strategy
            assert m1.reason == m2.reason
            assert m1.face_score == m2.face_score

    def test_both_clamp_negative_confidence(self, match_stage, iterative_stage):
        """Both stages clamp negative confidence to 0."""
        test_data = [{
            'segment_index': 0,
            'video_file': 'vid_001',
            'video_start': 0.0,
            'video_end': 10.0,
            'confidence': -0.5,
        }]

        state_match = PipelineState()
        state_iter = PipelineState()

        assert match_stage.restore(state_match, self._make_checkpoint(test_data)) is True
        assert iterative_stage.restore(state_iter, self._make_checkpoint(test_data)) is True

        assert state_match.matches[0].confidence == 0.0
        assert state_iter.matches[0].confidence == 0.0

    def test_both_clamp_high_confidence(self, match_stage, iterative_stage):
        """Both stages clamp confidence > 1 to 1."""
        test_data = [{
            'segment_index': 0,
            'video_file': 'vid_001',
            'video_start': 0.0,
            'video_end': 10.0,
            'confidence': 2.5,
        }]

        state_match = PipelineState()
        state_iter = PipelineState()

        assert match_stage.restore(state_match, self._make_checkpoint(test_data)) is True
        assert iterative_stage.restore(state_iter, self._make_checkpoint(test_data)) is True

        assert state_match.matches[0].confidence == 1.0
        assert state_iter.matches[0].confidence == 1.0

    def test_both_reject_empty_video_file(self, match_stage, iterative_stage):
        """Both stages reject matches with empty video_file."""
        test_data = [{
            'segment_index': 0,
            'video_file': '',
            'video_start': 0.0,
            'video_end': 10.0,
            'confidence': 0.9,
        }]

        state_match = PipelineState()
        state_iter = PipelineState()

        # Both should return False since the only match is invalid
        assert match_stage.restore(state_match, self._make_checkpoint(test_data)) is False
        assert iterative_stage.restore(state_iter, self._make_checkpoint(test_data)) is False

    def test_both_handle_legacy_source_file(self, match_stage, iterative_stage):
        """Both stages handle legacy source_file field."""
        test_data = [{
            'segment_index': 0,
            'source_file': 'legacy_vid',
            'start_time': 3.0,
            'confidence': 0.8,
        }]

        state_match = PipelineState()
        state_iter = PipelineState()

        assert match_stage.restore(state_match, self._make_checkpoint(test_data)) is True
        assert iterative_stage.restore(state_iter, self._make_checkpoint(test_data)) is True

        assert state_match.matches[0].video_file == 'legacy_vid'
        assert state_iter.matches[0].video_file == 'legacy_vid'
        assert state_match.matches[0].video_start == state_iter.matches[0].video_start

    def test_default_strategy_differs_when_absent(self, match_stage, iterative_stage):
        """When strategy is missing, each stage uses its own default label."""
        test_data = [{
            'segment_index': 0,
            'video_file': 'vid_001',
            'video_start': 0.0,
            'video_end': 10.0,
            'confidence': 0.8,
            # no 'strategy' key
        }]

        state_match = PipelineState()
        state_iter = PipelineState()

        assert match_stage.restore(state_match, self._make_checkpoint(test_data)) is True
        assert iterative_stage.restore(state_iter, self._make_checkpoint(test_data)) is True

        assert state_match.matches[0].strategy == 'restored'
        assert state_iter.matches[0].strategy == 'iterative_restored'
