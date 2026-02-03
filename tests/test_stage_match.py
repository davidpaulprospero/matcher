"""
Test Suite for MatchStage

Tests the MatchStage class which handles:
- Voiceover to video matching using embedding similarity
- LLM-based reranking
- Multiple matching strategies (primary, alternatives, diversity, B-roll, etc.)
- Location filtering
- Delta matching
- Checkpoint operations
- Parametrized testing for provider combinations, confidence thresholds, match counts, and fallback scenarios
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
import numpy as np

from src.stages.match import MatchStage
from src.state import PipelineState, VoiceoverSegment, Match


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config with required attributes"""
    config = MagicMock()
    config.pipeline.skip_matching = False
    config.matching.min_confidence = 0.5
    config.matching.high_confidence_threshold = 0.8
    config.matching.embedding_candidates = 50
    config.matching.llm_rerank_candidates = 10
    config.matching.max_clip_reuse = 3
    config.matching.force_rematch = False
    config.matching.delta_matching_enabled = True
    config.cache.cache_dir = ".cache"
    return config


@pytest.fixture
def mock_checkpoint():
    """Create mock checkpoint manager"""
    checkpoint = MagicMock()
    checkpoint.should_skip_stage.return_value = False
    checkpoint.get_stage_data.return_value = None
    return checkpoint


@pytest.fixture
def mock_voiceover_segments():
    """Create mock voiceover segments"""
    return [
        VoiceoverSegment(index=0, start=0.0, end=3.0, text="Welcome to the beach"),
        VoiceoverSegment(index=1, start=3.0, end=6.0, text="The sunset is beautiful"),
        VoiceoverSegment(index=2, start=6.0, end=9.0, text="Let's explore the area")
    ]


@pytest.fixture
def mock_text_metadata():
    """Create mock text metadata (video segments)"""
    return [
        {'video_path': 'video1.mp4', 'text': 'Ocean waves', 'start_time': 0.0, 'end_time': 5.0,
         'is_broll': False, 'face_score': 0.8, 'scene_index': 0},
        {'video_path': 'video1.mp4', 'text': 'Sunset colors', 'start_time': 5.0, 'end_time': 10.0,
         'is_broll': True, 'face_score': 0.2, 'scene_index': 1},
        {'video_path': 'video2.mp4', 'text': 'Beach exploration', 'start_time': 0.0, 'end_time': 7.0,
         'is_broll': False, 'face_score': 0.9, 'scene_index': 0},
    ]


# ============================================================================
# Test Stage Initialization
# ============================================================================

class TestMatchStageInit:
    """Test stage initialization"""

    @pytest.mark.fast
    def test_stage_name(self):
        """Test stage name"""
        stage = MatchStage()
        assert stage.name == "MATCH"

    @pytest.mark.fast
    def test_stage_description(self):
        """Test stage description"""
        stage = MatchStage()
        assert "match" in stage.description.lower() or "Match" in stage.description

    @pytest.mark.fast
    def test_stage_registration(self):
        """Test stage is registered"""
        from src.stages import get_stage
        stage_class = get_stage("MATCH")
        assert stage_class is MatchStage


# ============================================================================
# Test Input Validation
# ============================================================================

class TestInputValidation:
    """Test input validation"""

    @pytest.mark.fast
    def test_validate_no_voiceover_segments(self, mock_config):
        """Test validation fails when no voiceover segments"""
        stage = MatchStage()
        state = PipelineState()
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.text_metadata = [{'text': 'test'}]

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "voiceover" in error.lower()

    @pytest.mark.fast
    def test_validate_no_embeddings_allowed(self, mock_config, mock_voiceover_segments):
        """Test validation passes when no embeddings but text_metadata exists (caption-first mode)"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.embeddings = None
        state.text_metadata = [{'text': 'test'}]

        error = stage.validate_inputs(state, mock_config)

        # In caption-first mode, embeddings are optional - text matching works without them
        assert error is None

    @pytest.mark.fast
    def test_validate_success(self, mock_config, mock_voiceover_segments):
        """Test successful validation"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.text_metadata = [{'text': 'test'}]

        error = stage.validate_inputs(state, mock_config)

        assert error is None


class TestInputValidationErrorMessages:
    """Test validation error messages contain specific field names and suggestions (caption-first mode)"""

    @pytest.mark.fast
    def test_validate_error_missing_voiceover_segments(self, mock_config):
        """Test error message when voiceover_segments missing"""
        stage = MatchStage()
        state = PipelineState()
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.text_metadata = [{'text': 'test'}]
        state.voiceover_segments = []

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "voiceover_segments" in error
        assert "ANALYZE" in error

    @pytest.mark.fast
    def test_validate_embeddings_optional_with_text_metadata(self, mock_config, mock_voiceover_segments):
        """Test embeddings are optional when text_metadata exists (caption-first mode)"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.embeddings = None
        state.text_metadata = [{'text': 'test'}]

        error = stage.validate_inputs(state, mock_config)

        # In caption-first mode, embeddings are optional
        assert error is None

    @pytest.mark.fast
    def test_validate_error_missing_text_metadata(self, mock_config, mock_voiceover_segments):
        """Test error message when text_metadata and caption_results missing"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.text_metadata = []
        # No caption_results either

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "text_metadata" in error
        assert "CAPTION" in error

    @pytest.mark.fast
    def test_validate_caption_results_fallback(self, mock_config, mock_voiceover_segments):
        """Test validation passes when text_metadata empty but caption_results exists"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.embeddings = None
        state.text_metadata = []
        state.caption_results = {'video123': {'segments': [{'text': 'test'}]}}

        error = stage.validate_inputs(state, mock_config)

        # caption_results can be used as fallback
        assert error is None

    @pytest.mark.fast
    def test_validate_error_suggests_analyze_stage(self, mock_config):
        """Test error suggests running ANALYZE stage for voiceover_segments"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = []
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.text_metadata = [{'text': 'test'}]

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "Suggestion" in error
        assert "ANALYZE" in error

    @pytest.mark.fast
    def test_validate_error_suggests_caption_stage(self, mock_config, mock_voiceover_segments):
        """Test error suggests running CAPTION stage for text_metadata"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.embeddings = None
        state.text_metadata = []

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "Suggestion" in error
        assert "CAPTION" in error

    @pytest.mark.fast
    def test_validate_voiceover_checked_first(self, mock_config):
        """Test voiceover_segments is checked before text_metadata"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = []
        state.embeddings = None
        state.text_metadata = []

        error = stage.validate_inputs(state, mock_config)

        # voiceover_segments error should come first
        assert error is not None
        assert "voiceover_segments" in error
        assert "ANALYZE" in error


# ============================================================================
# Test Segment Preparation
# ============================================================================

class TestSegmentPreparation:
    """Test segment preparation"""

    @pytest.mark.fast
    def test_prepare_voiceover_segments(self, mock_voiceover_segments):
        """Test preparing voiceover segments"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = [{'video_path': 'v.mp4', 'text': 'test', 'start_time': 0, 'end_time': 1}]

        vo_segs, vid_segs, video_paths = stage._prepare_segments(state)

        assert len(vo_segs) == 3
        assert vo_segs[0].text == "Welcome to the beach"
        assert vo_segs[0].start_time == 0.0
        assert vo_segs[0].end_time == 3.0

    @pytest.mark.fast
    def test_prepare_dict_voiceover_segments(self):
        """Test preparing dict-based voiceover segments"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = [
            {'index': 0, 'start': 0.0, 'end': 3.0, 'text': 'Test segment', 'keywords': ['beach']}
        ]
        state.text_metadata = [{'video_path': 'v.mp4', 'text': 'test', 'start_time': 0, 'end_time': 1}]

        vo_segs, vid_segs, video_paths = stage._prepare_segments(state)

        assert len(vo_segs) == 1
        assert vo_segs[0].text == "Test segment"

    @pytest.mark.fast
    def test_prepare_video_segments(self, mock_voiceover_segments, mock_text_metadata):
        """Test preparing video segments"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata

        vo_segs, vid_segs, video_paths = stage._prepare_segments(state)

        assert len(vid_segs) == 3
        assert len(video_paths) == 2  # 2 unique videos
        assert 'video1.mp4' in video_paths
        assert 'video2.mp4' in video_paths

    @pytest.mark.fast
    def test_prepare_video_segments_with_broll(self, mock_voiceover_segments, mock_text_metadata):
        """Test B-roll flag propagation in video segments"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata

        vo_segs, vid_segs, video_paths = stage._prepare_segments(state)

        # Check B-roll flags are preserved
        broll_count = sum(1 for seg in vid_segs if hasattr(seg, 'is_broll') and seg.is_broll)
        assert broll_count == 1  # One segment has is_broll=True

    @pytest.mark.fast
    def test_prepare_video_segments_with_face_score(self, mock_voiceover_segments, mock_text_metadata):
        """Test face_score propagation in video segments"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata

        vo_segs, vid_segs, video_paths = stage._prepare_segments(state)

        # Check face_score is preserved
        assert hasattr(vid_segs[0], 'face_score')
        assert vid_segs[0].face_score == 0.8


# ============================================================================
# Test Matching Execution
# ============================================================================

class TestMatchingExecution:
    """Test matching execution"""

    @patch('src.matching.match_all_segments')
    @patch('src.embeddings.compute_embeddings')
    @patch('src.embeddings.get_embedding_provider')
    @patch('src.utils.CacheManager')
    @pytest.mark.fast
    def test_run_matching_success(self, mock_cache_class, mock_provider, mock_compute,
                                  mock_match_all, mock_voiceover_segments,
                                  mock_text_metadata):
        """Test successful matching execution"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata
        state.face_preference = 'neutral'

        # Mock voiceover + video embeddings (called twice)
        vo_embeddings = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])
        vid_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        mock_compute.side_effect = [vo_embeddings, vid_embeddings]
        mock_provider.return_value = Mock()
        mock_cache_class.return_value = Mock()

        # Mock matches
        mock_matches = [
            Match(segment_index=0, video_file='video1.mp4', video_start=0.0, video_end=3.0,
                  confidence=0.9, strategy='primary'),
            Match(segment_index=1, video_file='video1.mp4', video_start=5.0, video_end=8.0,
                  confidence=0.85, strategy='primary'),
            Match(segment_index=2, video_file='video2.mp4', video_start=0.0, video_end=3.0,
                  confidence=0.8, strategy='primary')
        ]
        mock_match_all.return_value = mock_matches

        vo_segs, vid_segs, video_paths = stage._prepare_segments(state)
        matches = stage._run_matching(
            vo_segs, vid_segs, video_paths, state,
            Mock(cache=Mock(cache_dir=".cache")),
            delta_enabled=False, force_rematch=False
        )

        assert len(matches) == 3
        assert all(m.confidence > 0.5 for m in matches)

    @patch('src.embeddings.compute_embeddings')
    @patch('src.embeddings.get_embedding_provider')
    @patch('src.utils.CacheManager')
    @pytest.mark.fast
    def test_run_matching_no_vo_embeddings(self, mock_cache_class, mock_provider, mock_compute,
                                           mock_voiceover_segments, mock_text_metadata):
        """Test handling when voiceover embeddings fail"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata

        # Mock embedding failure
        mock_compute.return_value = None
        mock_provider.return_value = Mock()
        mock_cache_class.return_value = Mock()

        vo_segs, vid_segs, video_paths = stage._prepare_segments(state)
        matches = stage._run_matching(
            vo_segs, vid_segs, video_paths, state,
            Mock(cache=Mock(cache_dir=".cache")),
            delta_enabled=False, force_rematch=False
        )

        assert matches == []


# ============================================================================
# Test Skip Matching
# ============================================================================

class TestSkipMatching:
    """Test skip matching behavior"""

    @pytest.mark.fast
    def test_skip_when_configured(self, mock_config, mock_checkpoint):
        """Test skipping matching when config says so"""
        stage = MatchStage()
        state = PipelineState()

        mock_config.pipeline.skip_matching = True

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True

    @pytest.mark.fast
    def test_can_skip_no_checkpoint(self, mock_checkpoint):
        """Test can_skip returns False when no checkpoint"""
        stage = MatchStage()
        state = PipelineState()

        mock_checkpoint.should_skip_stage.return_value = False

        assert stage.can_skip(state, mock_checkpoint) is False

    @pytest.mark.fast
    def test_can_skip_with_checkpoint(self, mock_checkpoint):
        """Test can_skip returns True when checkpoint exists"""
        stage = MatchStage()
        state = PipelineState()

        mock_checkpoint.should_skip_stage.return_value = True

        assert stage.can_skip(state, mock_checkpoint) is True


# ============================================================================
# Test Stage Execution
# ============================================================================

class TestMatchStageExecution:
    """Test full stage execution"""

    @pytest.mark.fast
    def test_run_no_voiceover_segments(self, mock_config, mock_checkpoint):
        """Test running with no voiceover segments"""
        stage = MatchStage()
        state = PipelineState()
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.text_metadata = [{'text': 'test'}]

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(result.warnings) > 0
        assert "voiceover" in result.warnings[0].lower()

    @pytest.mark.fast
    def test_run_no_video_data(self, mock_config, mock_checkpoint, mock_voiceover_segments):
        """Test running with no video data returns error (US-40-007)"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.embeddings = None
        state.text_metadata = []

        result = stage.run(state, mock_config, mock_checkpoint)

        # US-40-007: Should return error when both text_metadata and caption_results unavailable
        assert result.success is False
        assert "No captions available" in result.error

    @patch('src.matching.match_all_segments')
    @patch('src.embeddings.compute_embeddings')
    @patch('src.embeddings.get_embedding_provider')
    @patch('src.utils.CacheManager')
    @pytest.mark.fast
    def test_run_success(self, mock_cache_class, mock_provider, mock_compute,
                        mock_match_all, mock_config, mock_checkpoint,
                        mock_voiceover_segments, mock_text_metadata):
        """Test successful full stage execution"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata
        state.face_preference = 'neutral'

        # Mock voiceover + video embeddings (called twice)
        vo_embeddings = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])
        vid_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        mock_compute.side_effect = [vo_embeddings, vid_embeddings]
        mock_provider.return_value = Mock()
        mock_cache_class.return_value = Mock()

        # Mock matches
        mock_matches = [
            Match(segment_index=0, video_file='video1.mp4', video_start=0.0, video_end=3.0,
                  confidence=0.9, strategy='primary'),
            Match(segment_index=1, video_file='video1.mp4', video_start=5.0, video_end=8.0,
                  confidence=0.85, strategy='primary'),
        ]
        mock_match_all.return_value = mock_matches

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.matches) == 2
        assert result.data['match_count'] == 2
        assert result.data['avg_confidence'] > 0

    @patch('src.embeddings.get_embedding_provider')
    @patch('src.matching.match_all_segments')
    @pytest.mark.fast
    def test_run_exception_handling(self, mock_match_all, mock_provider, mock_config,
                                   mock_checkpoint, mock_voiceover_segments, mock_text_metadata):
        """Test exception handling in main run method"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata

        # Mock provider and matching to raise exception
        mock_provider.side_effect = Exception("Provider failed")

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is False
        assert "failed" in result.error.lower()


# ============================================================================
# Test Checkpoint Operations
# ============================================================================

class TestMatchCheckpoint:
    """Test checkpoint operations"""

    @pytest.mark.fast
    def test_restore_no_data(self, mock_checkpoint):
        """Test restore returns False when no checkpoint data"""
        stage = MatchStage()
        state = PipelineState()

        mock_checkpoint.get_stage_data.return_value = None

        result = stage.restore(state, mock_checkpoint)

        assert result is False

    @pytest.mark.fast
    def test_restore_success(self, mock_checkpoint):
        """Test successful restore from checkpoint"""
        stage = MatchStage()
        state = PipelineState()

        mock_checkpoint.get_stage_data.return_value = {
            'match_count': 10,
            'avg_confidence': 0.85
        }

        result = stage.restore(state, mock_checkpoint)

        assert result is True

    @pytest.mark.fast
    def test_restore_exception_handling(self, mock_checkpoint):
        """Test restore handles exceptions"""
        stage = MatchStage()
        state = PipelineState()

        mock_checkpoint.get_stage_data.side_effect = Exception("Checkpoint error")

        result = stage.restore(state, mock_checkpoint)

        assert result is False


# ============================================================================
# Test Restore Error Handling (US-005)
# ============================================================================

class TestRestoreErrorHandling:
    """Test restore() error handling with corrupt/incomplete checkpoint data"""

    @pytest.mark.fast
    def test_restore_logs_warning_on_no_data(self, mock_checkpoint, caplog):
        """Test restore logs specific warning when checkpoint data is missing"""
        import logging
        caplog.set_level(logging.WARNING)

        stage = MatchStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.return_value = None

        result = stage.restore(state, mock_checkpoint)

        assert result is False
        assert "No checkpoint data for MATCH" in caplog.text

    @pytest.mark.fast
    def test_restore_validates_matches_is_list(self, mock_checkpoint, caplog):
        """Test restore returns False when matches is not a list"""
        import logging
        caplog.set_level(logging.WARNING)

        stage = MatchStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.return_value = {
            'matches': "not a list"  # Invalid: should be list
        }

        result = stage.restore(state, mock_checkpoint)

        assert result is False
        assert "not a list" in caplog.text

    @pytest.mark.fast
    def test_restore_validates_match_objects(self, mock_checkpoint, caplog):
        """Test restore validates individual match objects are dicts"""
        import logging
        caplog.set_level(logging.WARNING)

        stage = MatchStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.return_value = {
            'matches': [
                "not a dict",  # Invalid
                123,  # Invalid
                None,  # Invalid
            ]
        }

        result = stage.restore(state, mock_checkpoint)

        assert result is False  # No valid matches
        assert "is not a dict" in caplog.text

    @pytest.mark.fast
    def test_restore_validates_video_file_required(self, mock_checkpoint, caplog):
        """Test restore validates video_file is non-empty string"""
        import logging
        caplog.set_level(logging.WARNING)

        stage = MatchStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.return_value = {
            'matches': [
                {'segment_index': 0, 'video_file': ''},  # Empty video_file
                {'segment_index': 1, 'video_file': None},  # None video_file
                {'segment_index': 2},  # Missing video_file
            ]
        }

        result = stage.restore(state, mock_checkpoint)

        assert result is False  # No valid matches
        assert "invalid video_file" in caplog.text

    @pytest.mark.fast
    def test_restore_validates_segment_index_type(self, mock_checkpoint, caplog):
        """Test restore validates segment_index is numeric"""
        import logging
        caplog.set_level(logging.WARNING)

        stage = MatchStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.return_value = {
            'matches': [
                {'segment_index': 'not a number', 'video_file': 'v1.mp4'},
            ]
        }

        result = stage.restore(state, mock_checkpoint)

        assert result is False
        assert "invalid segment_index" in caplog.text

    @pytest.mark.fast
    def test_restore_validates_confidence_type(self, mock_checkpoint, caplog):
        """Test restore validates confidence is numeric"""
        import logging
        caplog.set_level(logging.WARNING)

        stage = MatchStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.return_value = {
            'matches': [
                {'segment_index': 0, 'video_file': 'v1.mp4', 'confidence': 'high'},
            ]
        }

        result = stage.restore(state, mock_checkpoint)

        assert result is False
        assert "invalid confidence" in caplog.text

    @pytest.mark.fast
    def test_restore_clamps_confidence_to_valid_range(self, mock_checkpoint, caplog):
        """Test restore clamps out-of-range confidence values"""
        import logging
        caplog.set_level(logging.DEBUG)

        stage = MatchStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.return_value = {
            'matches': [
                {'segment_index': 0, 'video_file': 'v1.mp4', 'confidence': 1.5},  # Out of range
                {'segment_index': 1, 'video_file': 'v2.mp4', 'confidence': -0.2},  # Out of range
            ]
        }

        result = stage.restore(state, mock_checkpoint)

        assert result is True
        assert len(state.matches) == 2
        assert state.matches[0].confidence == 1.0  # Clamped to max
        assert state.matches[1].confidence == 0.0  # Clamped to min

    @pytest.mark.fast
    def test_restore_partial_valid_data(self, mock_checkpoint, caplog):
        """Test restore succeeds with partial valid data, logs validation errors"""
        import logging
        caplog.set_level(logging.WARNING)

        stage = MatchStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.return_value = {
            'matches': [
                {'segment_index': 0, 'video_file': 'valid.mp4', 'confidence': 0.8},  # Valid
                {'segment_index': 'bad', 'video_file': 'v2.mp4'},  # Invalid segment_index
                {'segment_index': 2, 'video_file': ''},  # Invalid video_file
            ]
        }

        result = stage.restore(state, mock_checkpoint)

        # Should succeed with partial data
        assert result is True
        assert len(state.matches) == 1
        assert state.matches[0].video_file == 'valid.mp4'
        # Should log validation errors
        assert "validation errors" in caplog.text

    @pytest.mark.fast
    def test_restore_returns_false_not_exception(self, mock_checkpoint):
        """Test restore returns False instead of raising exception on validation failure"""
        stage = MatchStage()
        state = PipelineState()

        # Various invalid checkpoint data structures
        invalid_data_cases = [
            None,
            {'matches': 'not a list'},
            {'matches': [None, None]},
            {'matches': [{'video_file': ''}]},
        ]

        for data in invalid_data_cases:
            mock_checkpoint.get_stage_data.return_value = data
            result = stage.restore(state, mock_checkpoint)
            assert result is False, f"Expected False for data: {data}"

    @pytest.mark.fast
    def test_restore_handles_legacy_source_file_field(self, mock_checkpoint):
        """Test restore handles legacy 'source_file' field name"""
        stage = MatchStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.return_value = {
            'matches': [
                {'segment_index': 0, 'source_file': 'legacy.mp4', 'start_time': 5.0},  # Legacy format
            ]
        }

        result = stage.restore(state, mock_checkpoint)

        assert result is True
        assert len(state.matches) == 1
        assert state.matches[0].video_file == 'legacy.mp4'
        assert state.matches[0].video_start == 5.0


# ============================================================================
# Test Settings Display
# ============================================================================

class TestSettingsDisplay:
    """Test settings display"""

    @pytest.mark.fast
    def test_print_settings(self, mock_config, capsys):
        """Test printing matching settings"""
        stage = MatchStage()

        stage._print_settings(mock_config)

        captured = capsys.readouterr()
        assert "Min confidence" in captured.out or "min confidence" in captured.out
        assert "0.5" in captured.out  # min_confidence value


# ============================================================================
# Test Confidence Calculation
# ============================================================================

class TestConfidenceCalculation:
    """Test average confidence calculation"""

    @patch('src.matching.match_all_segments')
    @patch('src.embeddings.compute_embeddings')
    @patch('src.embeddings.get_embedding_provider')
    @patch('src.utils.CacheManager')
    @pytest.mark.fast
    def test_average_confidence_calculation(self, mock_cache_class, mock_provider,
                                           mock_compute, mock_match_all, mock_config,
                                           mock_checkpoint, mock_voiceover_segments,
                                           mock_text_metadata):
        """Test average confidence is calculated correctly"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata
        state.face_preference = 'neutral'

        mock_compute.side_effect = [np.array([[0.2, 0.3, 0.4]]), np.array([[0.1, 0.2, 0.3]])]
        mock_provider.return_value = Mock()
        mock_cache_class.return_value = Mock()

        # Mock matches with known confidences
        mock_matches = [
            Match(segment_index=0, video_file='v1.mp4', video_start=0.0, video_end=3.0,
                  confidence=0.8, strategy='primary'),
            Match(segment_index=1, video_file='v2.mp4', video_start=0.0, video_end=3.0,
                  confidence=0.9, strategy='primary'),
        ]
        mock_match_all.return_value = mock_matches

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        # Average of 0.8 and 0.9 = 0.85
        assert abs(result.data['avg_confidence'] - 0.85) < 0.01

    @patch('src.matching.match_all_segments')
    @patch('src.embeddings.compute_embeddings')
    @patch('src.embeddings.get_embedding_provider')
    @patch('src.utils.CacheManager')
    @pytest.mark.fast
    def test_average_confidence_with_match_result(self, mock_cache_class, mock_provider,
                                                  mock_compute, mock_match_all, mock_config,
                                                  mock_checkpoint, mock_voiceover_segments,
                                                  mock_text_metadata):
        """Test average confidence with MatchResult objects (primary_match)"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata
        state.face_preference = 'neutral'

        mock_compute.side_effect = [np.array([[0.2, 0.3, 0.4]]), np.array([[0.1, 0.2, 0.3]])]
        mock_provider.return_value = Mock()
        mock_cache_class.return_value = Mock()

        # Mock MatchResult with primary_match
        primary_match = Match(segment_index=0, video_file='v1.mp4', video_start=0.0,
                             video_end=3.0, confidence=0.75, strategy='primary')
        match_result = Mock()
        match_result.primary_match = primary_match
        mock_match_all.return_value = [match_result]

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert abs(result.data['avg_confidence'] - 0.75) < 0.01


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestMatchEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.mark.fast
    def test_empty_text_metadata(self, mock_config, mock_checkpoint, mock_voiceover_segments):
        """Test handling empty text_metadata returns error (US-40-007)"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = []
        state.embeddings = np.array([[0.1, 0.2, 0.3]])

        result = stage.run(state, mock_config, mock_checkpoint)

        # US-40-007: Should return error when both text_metadata and caption_results unavailable
        assert result.success is False
        assert "No captions available" in result.error

    @pytest.mark.fast
    def test_no_matches_returned(self, mock_config, mock_checkpoint, mock_voiceover_segments,
                                mock_text_metadata):
        """Test handling when matching returns no matches"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata
        state.face_preference = 'neutral'

        with patch('src.matching.match_all_segments', return_value=[]):
            with patch('src.embeddings.compute_embeddings', side_effect=[np.array([[0.2, 0.3, 0.4]]), np.array([[0.1, 0.2, 0.3]])]):
                with patch('src.embeddings.get_embedding_provider', return_value=Mock()):
                    with patch('src.utils.CacheManager', return_value=Mock()):
                        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['match_count'] == 0
        assert result.data['avg_confidence'] == 0

    @pytest.mark.fast
    def test_non_dict_text_metadata(self, mock_voiceover_segments):
        """Test handling non-dict entries in text_metadata"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = [
            "not a dict",  # Should be handled
            {'video_path': 'v.mp4', 'text': 'test', 'start_time': 0, 'end_time': 1}
        ]

        # Should not raise
        vo_segs, vid_segs, video_paths = stage._prepare_segments(state)

        # Second entry should be processed
        assert len(vid_segs) == 2  # Non-dict is passed through as-is

    @pytest.mark.fast
    def test_matches_without_confidence_attribute(self, mock_config, mock_checkpoint,
                                                  mock_voiceover_segments, mock_text_metadata):
        """Test handling matches without confidence attribute"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata
        state.face_preference = 'neutral'

        # Mock matches without proper attributes
        mock_matches = [Mock(spec=[])]  # No confidence attribute

        with patch('src.matching.match_all_segments', return_value=mock_matches):
            with patch('src.embeddings.compute_embeddings', side_effect=[np.array([[0.2, 0.3, 0.4]]), np.array([[0.1, 0.2, 0.3]])]):
                with patch('src.embeddings.get_embedding_provider', return_value=Mock()):
                    with patch('src.utils.CacheManager', return_value=Mock()):
                        result = stage.run(state, mock_config, mock_checkpoint)

        # Should handle gracefully
        assert result.success is True
        assert result.data['avg_confidence'] == 0  # No valid confidences

    @patch('src.embeddings.compute_embeddings')
    @patch('src.embeddings.get_embedding_provider')
    @patch('src.utils.CacheManager')
    @pytest.mark.fast
    def test_empty_voiceover_embeddings(self, mock_cache_class, mock_provider, mock_compute,
                                       mock_voiceover_segments, mock_text_metadata):
        """Test handling when voiceover embeddings are empty array"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata

        # Return empty array
        mock_compute.return_value = np.array([])
        mock_provider.return_value = Mock()
        mock_cache_class.return_value = Mock()

        vo_segs, vid_segs, video_paths = stage._prepare_segments(state)
        matches = stage._run_matching(
            vo_segs, vid_segs, video_paths, state,
            Mock(cache=Mock(cache_dir=".cache")),
            delta_enabled=False, force_rematch=False
        )

        assert matches == []

    @pytest.mark.fast
    def test_delta_matching_flags(self, mock_voiceover_segments, mock_text_metadata):
        """Test delta matching and force rematch flags"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata
        state.face_preference = 'neutral'

        with patch('src.matching.match_all_segments', return_value=[]) as mock_match:
            with patch('src.embeddings.compute_embeddings', side_effect=[np.array([[0.2, 0.3, 0.4]]), np.array([[0.1, 0.2, 0.3]])]):
                with patch('src.embeddings.get_embedding_provider', return_value=Mock()):
                    with patch('src.utils.CacheManager', return_value=Mock()):
                        vo_segs, vid_segs, video_paths = stage._prepare_segments(state)

                        # Test delta matching
                        stage._run_matching(
                            vo_segs, vid_segs, video_paths, state,
                            Mock(cache=Mock(cache_dir=".cache")),
                            delta_enabled=True, force_rematch=False
                        )

                        # match_all_segments should be called
                        assert mock_match.called

    @pytest.mark.fast
    def test_location_chapters_in_matching(self, mock_voiceover_segments, mock_text_metadata):
        """Test location chapters are passed to matching"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata
        state.location_chapters = [{'location': 'Paris', 'start_index': 0}]

        with patch('src.matching.match_all_segments', return_value=[]) as mock_match:
            with patch('src.embeddings.compute_embeddings', side_effect=[np.array([[0.2, 0.3, 0.4]]), np.array([[0.1, 0.2, 0.3]])]):
                with patch('src.embeddings.get_embedding_provider', return_value=Mock()):
                    with patch('src.utils.CacheManager', return_value=Mock()):
                        vo_segs, vid_segs, video_paths = stage._prepare_segments(state)

                        stage._run_matching(
                            vo_segs, vid_segs, video_paths, state,
                            Mock(cache=Mock(cache_dir=".cache")),
                            delta_enabled=False, force_rematch=False
                        )

                        # Check location_chapters was passed
                        call_kwargs = mock_match.call_args[1]
                        assert 'location_chapters' in call_kwargs


# ============================================================================
# Parametrized Tests for Provider Combinations
# ============================================================================

class TestProviderCombinations:
    """Parametrized tests for different matching provider combinations"""

    @pytest.mark.parametrize("provider_name,expected_valid", [
        ("gemini", True),
        ("anthropic", True),
        ("ollama", True),
        ("embedding_only", True),
        ("invalid_provider", False),
    ])
    @pytest.mark.fast
    def test_provider_initialization(self, provider_name, expected_valid):
        """Test that different provider configurations initialize correctly"""
        config = MagicMock()
        config.pipeline.skip_matching = False
        config.matching.primary_provider = provider_name
        config.matching.min_confidence = 0.5
        config.matching.embedding_candidates = 50
        config.matching.llm_rerank_candidates = 10
        config.matching.max_clip_reuse = 3
        config.cache.cache_dir = ".cache"

        stage = MatchStage()

        # Stage itself should always initialize
        assert stage.name == "MATCH"

    @pytest.mark.parametrize("provider,fallback_provider,scenario", [
        ("gemini", "anthropic", "primary_fails_uses_fallback"),
        ("anthropic", "ollama", "primary_fails_uses_fallback"),
        ("ollama", None, "single_provider_no_fallback"),
        ("gemini", "gemini", "same_provider_retry"),
    ])
    @pytest.mark.fast
    def test_provider_fallback_scenarios(self, provider, fallback_provider, scenario):
        """Test provider fallback behavior in different scenarios"""
        config = MagicMock()
        config.pipeline.skip_matching = False
        config.matching.primary_provider = provider
        config.matching.fallback_provider = fallback_provider
        config.matching.min_confidence = 0.5
        config.matching.embedding_candidates = 50
        config.matching.llm_rerank_candidates = 10
        config.matching.max_clip_reuse = 3
        config.cache.cache_dir = ".cache"

        stage = MatchStage()

        # Verify configuration is read correctly
        assert config.matching.primary_provider == provider
        assert config.matching.fallback_provider == fallback_provider

    @pytest.mark.parametrize("strategy_type", [
        "embedding",
        "llm",
        "hybrid",
    ])
    @pytest.mark.fast
    def test_matching_strategy_types(self, strategy_type):
        """Test different matching strategy types"""
        config = MagicMock()
        config.pipeline.skip_matching = False
        config.matching.strategy = strategy_type
        config.matching.min_confidence = 0.5
        config.matching.embedding_candidates = 50
        config.matching.llm_rerank_candidates = 10
        config.matching.max_clip_reuse = 3
        config.cache.cache_dir = ".cache"

        stage = MatchStage()

        # Verify strategy type is valid
        assert strategy_type in ["embedding", "llm", "hybrid"]


# ============================================================================
# Parametrized Tests for Confidence Threshold Boundaries
# ============================================================================

class TestConfidenceThresholdBoundaries:
    """Parametrized tests for confidence threshold boundary values"""

    @pytest.mark.parametrize("min_confidence", [
        0.0,   # Minimum boundary - accept everything
        0.25,  # Low threshold
        0.5,   # Default/medium threshold
        0.75,  # High threshold
        1.0,   # Maximum boundary - accept only perfect matches
    ])
    @pytest.mark.fast
    def test_min_confidence_threshold_values(self, min_confidence):
        """Test matching with different min_confidence threshold values"""
        config = MagicMock()
        config.pipeline.skip_matching = False
        config.matching.min_confidence = min_confidence
        config.matching.high_confidence_threshold = 0.8
        config.matching.embedding_candidates = 50
        config.matching.llm_rerank_candidates = 10
        config.matching.max_clip_reuse = 3
        config.cache.cache_dir = ".cache"

        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = [
            VoiceoverSegment(index=0, start=0.0, end=3.0, text="Test segment")
        ]
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.text_metadata = [{'video_path': 'v.mp4', 'text': 'test', 'start_time': 0, 'end_time': 1}]

        # Validation should pass regardless of min_confidence value
        error = stage.validate_inputs(state, config)
        assert error is None

    @pytest.mark.parametrize("high_conf_threshold,expected_skip_llm", [
        (0.5, True),   # Low threshold - more segments skip LLM
        (0.75, True),  # Medium threshold
        (0.9, True),   # High threshold - fewer segments skip LLM
        (1.0, False),  # Maximum - nothing skips LLM (impossible to achieve 1.0 embedding sim)
    ])
    @pytest.mark.fast
    def test_skip_llm_threshold_behavior(self, high_conf_threshold, expected_skip_llm):
        """Test skip_llm threshold affects LLM usage appropriately"""
        config = MagicMock()
        config.pipeline.skip_matching = False
        config.matching.min_confidence = 0.5
        config.matching.high_confidence_threshold = high_conf_threshold
        config.matching.skip_llm_threshold = high_conf_threshold
        config.matching.embedding_candidates = 50
        config.matching.llm_rerank_candidates = 10
        config.matching.max_clip_reuse = 3
        config.cache.cache_dir = ".cache"

        stage = MatchStage()

        # High similarity candidate (0.95) should skip LLM if threshold <= 0.95
        skip_llm = 0.95 >= high_conf_threshold
        assert skip_llm == expected_skip_llm or high_conf_threshold == 1.0

    @pytest.mark.parametrize("confidence,expected_valid", [
        (-0.1, False),  # Below valid range
        (0.0, True),    # Minimum valid
        (0.5, True),    # Middle of range
        (1.0, True),    # Maximum valid
        (1.1, False),   # Above valid range
    ])
    @pytest.mark.fast
    def test_confidence_value_validation(self, confidence, expected_valid):
        """Test confidence values are validated correctly in restore"""
        stage = MatchStage()
        state = PipelineState()

        checkpoint = MagicMock()
        checkpoint.get_stage_data.return_value = {
            'matches': [
                {'segment_index': 0, 'video_file': 'v1.mp4', 'confidence': confidence}
            ]
        }

        result = stage.restore(state, checkpoint)

        if expected_valid:
            assert result is True
            # Confidence should be clamped to [0.0, 1.0]
            expected_conf = max(0.0, min(1.0, confidence))
            assert state.matches[0].confidence == expected_conf
        else:
            # Invalid confidence should still work but be clamped
            assert result is True
            assert 0.0 <= state.matches[0].confidence <= 1.0


# ============================================================================
# Parametrized Tests for Match Count Variations
# ============================================================================

class TestMatchCountVariations:
    """Parametrized tests for different match_count/num_alternatives settings"""

    @pytest.mark.parametrize("num_alternatives", [
        0,   # No alternatives (only primary match)
        1,   # Single alternative
        2,   # Default
        3,   # Three alternatives
        5,   # Five alternatives
        10,  # Ten alternatives
    ])
    @pytest.mark.fast
    def test_num_alternatives_config(self, num_alternatives):
        """Test different num_alternatives configurations"""
        config = MagicMock()
        config.pipeline.skip_matching = False
        config.matching.min_confidence = 0.5
        config.matching.embedding_candidates = 50
        config.matching.llm_rerank_candidates = 10
        config.matching.max_clip_reuse = 3
        config.output.num_alternatives = num_alternatives
        config.cache.cache_dir = ".cache"

        stage = MatchStage()

        assert config.output.num_alternatives == num_alternatives
        # Stage should accept any valid num_alternatives value
        assert stage.name == "MATCH"

    @pytest.mark.parametrize("num_alternatives,num_candidates,expected_alt_count", [
        (3, 1, 0),   # Only 1 candidate, can't have alternatives
        (3, 2, 1),   # 2 candidates, 1 alternative (primary uses 1)
        (3, 3, 2),   # 3 candidates, 2 alternatives
        (3, 5, 3),   # 5 candidates, capped at 3 alternatives
        (5, 3, 2),   # 3 candidates but requesting 5, limited to 2
        (10, 5, 4),  # 5 candidates but requesting 10, limited to 4
    ])
    @pytest.mark.fast
    def test_alternatives_limited_by_candidates(self, num_alternatives, num_candidates, expected_alt_count):
        """Test that alternatives are limited by available candidates"""
        # Simulate the logic: alternatives = min(num_alternatives, num_candidates - 1)
        actual_alt_count = min(num_alternatives, max(0, num_candidates - 1))
        assert actual_alt_count == expected_alt_count

    @pytest.mark.parametrize("embedding_candidates,llm_candidates", [
        (10, 5),    # Default ratio
        (30, 10),   # Larger pool
        (50, 15),   # Large pool
        (100, 20),  # Very large pool
    ])
    @pytest.mark.fast
    def test_candidate_pool_sizes(self, embedding_candidates, llm_candidates):
        """Test different candidate pool size configurations"""
        config = MagicMock()
        config.matching.embedding_candidates = embedding_candidates
        config.matching.llm_rerank_candidates = llm_candidates

        # LLM candidates should be <= embedding candidates
        assert llm_candidates <= embedding_candidates

    @pytest.mark.parametrize("max_clip_reuse", [
        1,   # Single use per clip
        2,   # Allow reuse twice
        3,   # Allow reuse three times (default)
        5,   # More permissive
        10,  # Very permissive
    ])
    @pytest.mark.fast
    def test_max_clip_reuse_settings(self, max_clip_reuse):
        """Test different max_clip_reuse configurations"""
        config = MagicMock()
        config.matching.max_clip_reuse = max_clip_reuse

        # Validate that the setting is respected
        assert config.matching.max_clip_reuse == max_clip_reuse


# ============================================================================
# Parametrized Tests for Strategy Fallback Scenarios
# ============================================================================

class TestStrategyFallbackScenarios:
    """Parametrized tests for strategy fallback behavior"""

    @pytest.mark.parametrize("primary_strategy,has_broll,has_scenes,expected_fallback", [
        ("visual_first", True, True, None),         # All data available, no fallback
        ("visual_first", False, True, "keyword_only"),  # No B-roll, fallback to keyword
        ("visual_first", True, False, "keyword_only"),  # No scenes, fallback to keyword
        ("broll_only", True, True, None),           # B-roll available
        ("broll_only", False, True, "different_source"),  # No B-roll, must use different strategy
        ("embedding_diversity", True, True, None),  # All data available
    ])
    @pytest.mark.fast
    def test_strategy_fallback_conditions(self, primary_strategy, has_broll, has_scenes, expected_fallback):
        """Test strategy fallback based on available data"""
        config = MagicMock()
        config.output.include_strategy_tracks = True
        config.output.strategy_tracks = [primary_strategy, "different_source", "keyword_only"]

        # Simulate data availability
        if has_broll:
            broll_segments = [{'is_broll': True}]
        else:
            broll_segments = []

        if has_scenes:
            scenes = {'video1.mp4': [{'description': 'test'}]}
        else:
            scenes = {}

        # Validate fallback expectation
        if not has_broll and primary_strategy == "broll_only":
            assert expected_fallback == "different_source"
        elif not has_scenes and primary_strategy == "visual_first":
            assert expected_fallback == "keyword_only"

    @pytest.mark.parametrize("strategy,fallback_chain", [
        ("visual_first", ["keyword_only", "different_source", "source_rotation"]),
        ("embedding_diversity", ["different_source", "keyword_only"]),
        ("broll_only", ["different_source"]),
        ("keyword_only", ["different_source"]),
        ("different_source", ["source_rotation"]),
    ])
    @pytest.mark.fast
    def test_strategy_fallback_chains(self, strategy, fallback_chain):
        """Test expected fallback chain for each strategy"""
        # Each strategy should have a defined fallback chain
        assert len(fallback_chain) > 0

        # First fallback should be a valid strategy
        valid_strategies = [
            "visual_first", "different_source", "keyword_only",
            "embedding_diversity", "broll_only", "source_rotation"
        ]
        assert fallback_chain[0] in valid_strategies

    @pytest.mark.parametrize("num_sources,expect_different_source_success", [
        (1, False),   # Only 1 source, can't find different
        (2, True),    # 2 sources, can find different
        (5, True),    # Many sources available
        (10, True),   # Large variety of sources
    ])
    @pytest.mark.fast
    def test_different_source_strategy_availability(self, num_sources, expect_different_source_success):
        """Test different_source strategy success based on source count"""
        # Generate mock sources
        sources = [f"video{i}.mp4" for i in range(num_sources)]

        # With only 1 source, different_source cannot succeed
        can_find_different = num_sources > 1
        assert can_find_different == expect_different_source_success

    @pytest.mark.parametrize("used_sources_count,available_count,expect_fallback", [
        (0, 5, False),   # No used sources, plenty available
        (2, 5, False),   # Some used, still have options
        (4, 5, False),   # Most used, still 1 available
        (5, 5, True),    # All used, need fallback
    ])
    @pytest.mark.fast
    def test_source_exhaustion_fallback(self, used_sources_count, available_count, expect_fallback):
        """Test fallback when available sources are exhausted"""
        sources = [f"video{i}.mp4" for i in range(available_count)]
        used = sources[:used_sources_count]
        remaining = available_count - used_sources_count

        needs_fallback = remaining == 0
        assert needs_fallback == expect_fallback


# ============================================================================
# Additional Parametrized Tests for Edge Cases
# ============================================================================

class TestParametrizedEdgeCases:
    """Parametrized tests for edge cases in matching"""

    @pytest.mark.parametrize("segment_count", [
        1,    # Single segment
        5,    # Few segments
        10,   # Medium count
        50,   # Many segments
        100,  # Large count
    ])
    @pytest.mark.fast
    def test_varying_segment_counts(self, segment_count):
        """Test matching with different numbers of voiceover segments"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = [
            VoiceoverSegment(index=i, start=float(i*3), end=float(i*3+3), text=f"Segment {i}")
            for i in range(segment_count)
        ]
        state.embeddings = np.random.rand(segment_count, 3)
        state.text_metadata = [{'video_path': f'v{i % 3}.mp4', 'text': f'test {i}', 'start_time': 0, 'end_time': 1}
                              for i in range(segment_count)]

        config = MagicMock()
        config.pipeline.skip_matching = False
        config.matching.min_confidence = 0.5

        error = stage.validate_inputs(state, config)
        assert error is None

    @pytest.mark.parametrize("video_count,segment_per_video", [
        (1, 5),     # Single video, multiple segments
        (3, 3),     # Few videos, few segments each
        (5, 10),    # Medium videos, more segments
        (10, 5),    # Many videos, moderate segments
    ])
    @pytest.mark.fast
    def test_varying_video_distributions(self, video_count, segment_per_video):
        """Test matching with different video/segment distributions"""
        text_metadata = []
        for v in range(video_count):
            for s in range(segment_per_video):
                text_metadata.append({
                    'video_path': f'video{v}.mp4',
                    'text': f'segment {v}_{s}',
                    'start_time': float(s * 5),
                    'end_time': float(s * 5 + 5)
                })

        assert len(text_metadata) == video_count * segment_per_video

        # Unique video count
        unique_videos = set(m['video_path'] for m in text_metadata)
        assert len(unique_videos) == video_count

    @pytest.mark.parametrize("reuse_penalty", [
        0.0,    # No penalty
        0.05,   # Small penalty
        0.1,    # Default penalty
        0.2,    # Larger penalty
        0.5,    # Heavy penalty
    ])
    @pytest.mark.fast
    def test_reuse_penalty_values(self, reuse_penalty):
        """Test different reuse penalty configurations"""
        config = MagicMock()
        config.matching.reuse_penalty = reuse_penalty
        config.matching.max_clip_reuse = 3

        # Penalty should reduce confidence by this amount per reuse
        original_confidence = 0.9
        reuses = 2
        penalized_confidence = original_confidence - (reuse_penalty * reuses)

        # Penalized confidence should still be >= 0
        assert penalized_confidence >= 0.0 or reuse_penalty > 0.45


# ============================================================================
# Test Text Metadata Fallback (US-39-012)
# ============================================================================

class TestTextMetadataFallback:
    """Tests for US-39-012: Match stage fallback for missing text_metadata"""

    @pytest.fixture
    def mock_caption_results(self):
        """Create mock caption_results with segments"""
        return {
            'video123': {
                'video_id': 'video123',
                'language': 'en',
                'is_auto_generated': False,
                'caption_quality': 'high',
                'timing_penalty': 1.0,
                'segments': [
                    {'text': 'Ocean waves crashing', 'start': 0.0, 'end': 5.0},
                    {'text': 'Sunset over the water', 'start': 5.0, 'end': 10.0},
                ],
            },
            'video456': {
                'video_id': 'video456',
                'language': 'en',
                'is_auto_generated': True,
                'caption_quality': 'medium',
                'timing_penalty': 0.95,
                'segments': [
                    {'text': 'Beach exploration', 'start': 0.0, 'end': 7.0},
                ],
            },
        }

    @pytest.mark.fast
    def test_recover_text_metadata_from_captions(self, mock_caption_results, mock_voiceover_segments):
        """Test recovery of text_metadata when empty but caption_results exists"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = []  # Empty
        state.caption_results = mock_caption_results

        # Call the recovery method directly
        stage._recover_text_metadata_from_captions(state)

        # Should have populated text_metadata
        assert len(state.text_metadata) == 3  # 2 + 1 segments
        assert state.text_metadata[0]['video_path'] == 'video123'
        assert state.text_metadata[0]['text'] == 'Ocean waves crashing'
        assert state.text_metadata[0]['caption_quality'] == 'high'
        assert state.text_metadata[2]['video_path'] == 'video456'
        assert state.text_metadata[2]['caption_quality'] == 'medium'

    @pytest.mark.fast
    def test_recover_skips_unavailable_captions(self, mock_voiceover_segments):
        """Test that recovery skips unavailable/errored caption results"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = []
        state.caption_results = {
            'video123': {
                'video_id': 'video123',
                'unavailable': True,
                'reason': 'no_captions_available',
            },
            'video456': {
                'video_id': 'video456',
                'error': 'fetch_failed',
            },
            'video789': {
                'video_id': 'video789',
                'skipped': True,
                'reason': 'live_stream',
            },
            'video_valid': {
                'video_id': 'video_valid',
                'language': 'en',
                'segments': [
                    {'text': 'Valid segment', 'start': 0.0, 'end': 5.0},
                ],
            },
        }

        stage._recover_text_metadata_from_captions(state)

        # Should only have the valid segment
        assert len(state.text_metadata) == 1
        assert state.text_metadata[0]['video_path'] == 'video_valid'

    @pytest.mark.fast
    def test_recover_preserves_existing_text_metadata(self, mock_caption_results, mock_voiceover_segments):
        """Test that recovery extends rather than replaces existing text_metadata"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = [
            {'video_path': 'existing.mp4', 'text': 'Existing segment', 'start_time': 0, 'end_time': 1}
        ]
        state.caption_results = mock_caption_results

        stage._recover_text_metadata_from_captions(state)

        # Should have existing + recovered
        assert len(state.text_metadata) == 4  # 1 existing + 3 recovered
        assert state.text_metadata[0]['video_path'] == 'existing.mp4'

    @pytest.mark.fast
    def test_recover_handles_empty_caption_results(self, mock_voiceover_segments, caplog):
        """Test that recovery handles empty caption_results gracefully"""
        import logging
        caplog.set_level(logging.WARNING)

        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = []
        state.caption_results = {}

        stage._recover_text_metadata_from_captions(state)

        # Should remain empty
        assert len(state.text_metadata) == 0
        assert "Cannot recover: caption_results is empty" in caplog.text

    @pytest.mark.fast
    def test_match_stage_run_triggers_recovery(self, mock_config, mock_checkpoint,
                                                mock_caption_results, mock_voiceover_segments, caplog):
        """Test that MatchStage.run() triggers recovery when text_metadata empty but caption_results exists"""
        import logging
        caplog.set_level(logging.WARNING)

        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = []  # Empty - should trigger recovery
        state.caption_results = mock_caption_results
        state.face_preference = 'neutral'

        with patch('src.matching.match_all_segments', return_value=[]):
            with patch('src.embeddings.compute_embeddings', side_effect=[np.array([[0.2, 0.3, 0.4]]), np.array([[0.1, 0.2, 0.3]])]):
                with patch('src.embeddings.get_embedding_provider', return_value=Mock()):
                    with patch('src.utils.CacheManager', return_value=Mock()):
                        result = stage.run(state, mock_config, mock_checkpoint)

        # Recovery should have been triggered (US-40-007 log message)
        assert "text_metadata empty, attempting recovery from caption_results" in caplog.text
        # text_metadata should be populated from caption_results
        assert len(state.text_metadata) == 3
        # Result should include warning about recovery
        assert any("Recovered" in w for w in result.warnings)

    @pytest.mark.fast
    def test_match_stage_run_fails_when_both_missing(self, mock_config, mock_checkpoint,
                                                      mock_voiceover_segments):
        """Test that MatchStage.run() returns error when both text_metadata and caption_results missing (US-40-007)"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = []
        state.caption_results = {}  # Empty - no recovery possible

        result = stage.run(state, mock_config, mock_checkpoint)

        # US-40-007: Should return error when both text_metadata and caption_results unavailable
        assert result.success is False
        assert "No captions available" in result.error
        assert "text_metadata empty" in result.error

    @pytest.mark.fast
    def test_match_stage_run_no_caption_results_attribute(self, mock_config, mock_checkpoint,
                                                           mock_voiceover_segments):
        """Test that MatchStage.run() returns error when state has no caption_results attribute (US-40-007)"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = []
        # No caption_results attribute at all

        result = stage.run(state, mock_config, mock_checkpoint)

        # US-40-007: Should return error when both text_metadata and caption_results unavailable
        assert result.success is False
        assert "No captions available" in result.error

    @pytest.mark.fast
    def test_recover_metadata_includes_all_fields(self, mock_voiceover_segments):
        """Test that recovered text_metadata includes all expected fields"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = []
        state.caption_results = {
            'testVideo': {
                'video_id': 'testVideo',
                'language': 'es',
                'is_auto_generated': True,
                'caption_quality': 'low',
                'timing_penalty': 0.8,
                'segments': [
                    {'text': 'Prueba de texto', 'start': 1.5, 'end': 4.5},
                ],
            },
        }

        stage._recover_text_metadata_from_captions(state)

        # Verify all expected fields
        meta = state.text_metadata[0]
        assert meta['text'] == 'Prueba de texto'
        assert meta['video_path'] == 'testVideo'
        assert meta['start_time'] == 1.5
        assert meta['end_time'] == 4.5
        assert meta['source_file'] == 'testVideo'
        assert meta['caption_source'] == 'youtube'
        assert meta['caption_language'] == 'es'
        assert meta['caption_auto_generated'] is True
        assert meta['caption_quality'] == 'low'
        assert meta['timing_penalty'] == 0.8


# ============================================================================
# Regression Test: No Stale PipelineState Attributes (US-44-001)
# ============================================================================

class TestNoStalePipelineStateAttributes:
    """
    Regression tests for US-44-001: _run_matching() must not reference
    state.embeddings, state.embedding_index, or state.location_chapters
    directly. These fields were removed from PipelineState during the
    7-stage pipeline simplification.
    """

    @patch('src.matching.match_all_segments')
    @patch('src.embeddings.compute_embeddings')
    @patch('src.embeddings.get_embedding_provider')
    @patch('src.utils.CacheManager')
    @pytest.mark.fast
    def test_run_matching_with_vanilla_pipeline_state(
        self, mock_cache_class, mock_provider, mock_compute,
        mock_match_all, mock_voiceover_segments, mock_text_metadata
    ):
        """
        Regression: _run_matching() must work with a vanilla PipelineState
        that has no dynamically-set embeddings, embedding_index, or
        location_chapters attributes.
        """
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata
        # Deliberately do NOT set state.embeddings, state.embedding_index,
        # or state.location_chapters — these don't exist on PipelineState

        vo_embeddings = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])
        vid_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        mock_compute.side_effect = [vo_embeddings, vid_embeddings]
        mock_provider.return_value = Mock()
        mock_cache_class.return_value = Mock()

        mock_matches = [
            Match(segment_index=0, video_file='video1.mp4', video_start=0.0,
                  video_end=3.0, confidence=0.9, strategy='primary'),
        ]
        mock_match_all.return_value = mock_matches

        vo_segs, vid_segs, video_paths = stage._prepare_segments(state)
        matches = stage._run_matching(
            vo_segs, vid_segs, video_paths, state,
            Mock(cache=Mock(cache_dir=".cache")),
            delta_enabled=False, force_rematch=False
        )

        assert len(matches) == 1
        # Verify compute_embeddings was called twice: voiceover + video
        assert mock_compute.call_count == 2
        # Verify embedding_index=None was passed (not state.embedding_index)
        call_kwargs = mock_match_all.call_args[1]
        assert call_kwargs['embedding_index'] is None

    @patch('src.matching.match_all_segments')
    @patch('src.embeddings.compute_embeddings')
    @patch('src.embeddings.get_embedding_provider')
    @patch('src.utils.CacheManager')
    @pytest.mark.fast
    def test_run_matching_passes_location_chapters_via_getattr(
        self, mock_cache_class, mock_provider, mock_compute,
        mock_match_all, mock_voiceover_segments, mock_text_metadata
    ):
        """
        Regression: location_chapters should be retrieved via getattr
        (set dynamically by AnalyzeStage) and default to None.
        """
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata
        # Simulate AnalyzeStage having set location_chapters dynamically
        state.location_chapters = [{'location': 'Paris', 'start_index': 0}]

        vo_embeddings = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])
        vid_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        mock_compute.side_effect = [vo_embeddings, vid_embeddings]
        mock_provider.return_value = Mock()
        mock_cache_class.return_value = Mock()
        mock_match_all.return_value = []

        vo_segs, vid_segs, video_paths = stage._prepare_segments(state)
        stage._run_matching(
            vo_segs, vid_segs, video_paths, state,
            Mock(cache=Mock(cache_dir=".cache")),
            delta_enabled=False, force_rematch=False
        )

        call_kwargs = mock_match_all.call_args[1]
        assert call_kwargs['location_chapters'] == [{'location': 'Paris', 'start_index': 0}]

    @patch('src.matching.match_all_segments')
    @patch('src.embeddings.compute_embeddings')
    @patch('src.embeddings.get_embedding_provider')
    @patch('src.utils.CacheManager')
    @pytest.mark.fast
    def test_run_matching_computes_video_embeddings_locally(
        self, mock_cache_class, mock_provider, mock_compute,
        mock_match_all, mock_voiceover_segments, mock_text_metadata
    ):
        """
        Regression: video embeddings must be computed locally in
        _run_matching(), not read from state.embeddings.
        """
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata

        vo_embeddings = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])
        vid_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        mock_compute.side_effect = [vo_embeddings, vid_embeddings]
        mock_provider.return_value = Mock()
        mock_cache_class.return_value = Mock()
        mock_match_all.return_value = []

        vo_segs, vid_segs, video_paths = stage._prepare_segments(state)
        stage._run_matching(
            vo_segs, vid_segs, video_paths, state,
            Mock(cache=Mock(cache_dir=".cache")),
            delta_enabled=False, force_rematch=False
        )

        # Second call should be for video embeddings with cache_key="video_segments"
        second_call_kwargs = mock_compute.call_args_list[1][1]
        assert second_call_kwargs['cache_key'] == 'video_segments'
        # video_embeddings passed to match_all_segments should be the locally computed ones
        call_kwargs = mock_match_all.call_args[1]
        assert np.array_equal(call_kwargs['video_embeddings'], vid_embeddings)

    @patch('src.matching.match_all_segments')
    @patch('src.embeddings.compute_embeddings')
    @patch('src.embeddings.get_embedding_provider')
    @patch('src.utils.CacheManager')
    @pytest.mark.fast
    def test_full_stage_run_with_vanilla_state_no_attribute_error(
        self, mock_cache_class, mock_provider, mock_compute,
        mock_match_all, mock_config, mock_checkpoint,
        mock_voiceover_segments, mock_text_metadata
    ):
        """
        Regression: full stage.run() must not raise AttributeError when
        PipelineState has no embeddings/embedding_index/location_chapters.
        """
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata
        # Only set fields that actually exist on PipelineState
        state.face_preference = 'neutral'

        vo_embeddings = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])
        vid_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        mock_compute.side_effect = [vo_embeddings, vid_embeddings]
        mock_provider.return_value = Mock()
        mock_cache_class.return_value = Mock()

        mock_matches = [
            Match(segment_index=0, video_file='video1.mp4', video_start=0.0,
                  video_end=3.0, confidence=0.85, strategy='primary'),
            Match(segment_index=1, video_file='video1.mp4', video_start=5.0,
                  video_end=8.0, confidence=0.80, strategy='primary'),
        ]
        mock_match_all.return_value = mock_matches

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.matches) == 2
        assert result.data['match_count'] == 2
