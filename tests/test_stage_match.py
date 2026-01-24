"""
Test Suite for MatchStage

Tests the MatchStage class which handles:
- Voiceover to video matching using embedding similarity
- LLM-based reranking
- Multiple matching strategies (primary, alternatives, diversity, B-roll, etc.)
- Location filtering
- Delta matching
- Checkpoint operations
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

    def test_stage_name(self):
        """Test stage name"""
        stage = MatchStage()
        assert stage.name == "MATCH"

    def test_stage_description(self):
        """Test stage description"""
        stage = MatchStage()
        assert "match" in stage.description.lower() or "Match" in stage.description

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

    def test_validate_no_voiceover_segments(self, mock_config):
        """Test validation fails when no voiceover segments"""
        stage = MatchStage()
        state = PipelineState()
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.text_metadata = [{'text': 'test'}]

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "voiceover" in error.lower()

    def test_validate_no_embeddings(self, mock_config, mock_voiceover_segments):
        """Test validation fails when no embeddings"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.embeddings = None
        state.text_metadata = [{'text': 'test'}]

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "embedding" in error.lower()

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
    """Test validation error messages contain specific field names and suggestions"""

    def test_validate_error_contains_field_name_voiceover_segments(self, mock_config):
        """Test error message contains 'voiceover_segments' field name"""
        stage = MatchStage()
        state = PipelineState()
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.text_metadata = [{'text': 'test'}]
        state.voiceover_segments = []

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "voiceover_segments" in error
        assert "Missing required fields" in error

    def test_validate_error_contains_field_name_embeddings(self, mock_config, mock_voiceover_segments):
        """Test error message contains 'embeddings' field name"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.embeddings = None
        state.text_metadata = [{'text': 'test'}]

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "embeddings" in error
        assert "Missing required fields" in error

    def test_validate_error_contains_field_name_text_metadata(self, mock_config, mock_voiceover_segments):
        """Test error message contains 'text_metadata' field name"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.text_metadata = []

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "text_metadata" in error
        assert "Missing required fields" in error

    def test_validate_error_contains_multiple_field_names(self, mock_config):
        """Test error message contains multiple missing field names"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = []
        state.embeddings = None
        state.text_metadata = []

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "voiceover_segments" in error
        assert "embeddings" in error
        assert "text_metadata" in error
        assert "Missing required fields" in error

    def test_validate_error_suggests_analyze_stage(self, mock_config):
        """Test error suggests running ANALYZE stage for voiceover_segments"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = []
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.text_metadata = [{'text': 'test'}]

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "Suggestion:" in error
        assert "ANALYZE" in error

    def test_validate_error_suggests_transcribe_stage(self, mock_config, mock_voiceover_segments):
        """Test error suggests running TRANSCRIBE stage for embeddings"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.embeddings = None
        state.text_metadata = []

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "Suggestion:" in error
        assert "TRANSCRIBE" in error

    def test_validate_error_suggests_multiple_stages(self, mock_config):
        """Test error suggests multiple stages when multiple fields missing"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = []
        state.embeddings = None
        state.text_metadata = []

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "Suggestion:" in error
        assert "ANALYZE" in error
        assert "TRANSCRIBE" in error


# ============================================================================
# Test Segment Preparation
# ============================================================================

class TestSegmentPreparation:
    """Test segment preparation"""

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
    def test_run_matching_success(self, mock_cache_class, mock_provider, mock_compute,
                                  mock_match_all, mock_voiceover_segments,
                                  mock_text_metadata):
        """Test successful matching execution"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata
        state.embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        state.embedding_index = Mock()
        state.face_preference = 'neutral'
        state.location_chapters = []

        # Mock voiceover embeddings
        vo_embeddings = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])
        mock_compute.return_value = vo_embeddings
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
    def test_run_matching_no_vo_embeddings(self, mock_cache_class, mock_provider, mock_compute,
                                           mock_voiceover_segments, mock_text_metadata):
        """Test handling when voiceover embeddings fail"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.embedding_index = Mock()

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

    def test_skip_when_configured(self, mock_config, mock_checkpoint):
        """Test skipping matching when config says so"""
        stage = MatchStage()
        state = PipelineState()

        mock_config.pipeline.skip_matching = True

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True

    def test_can_skip_no_checkpoint(self, mock_checkpoint):
        """Test can_skip returns False when no checkpoint"""
        stage = MatchStage()
        state = PipelineState()

        mock_checkpoint.should_skip_stage.return_value = False

        assert stage.can_skip(state, mock_checkpoint) is False

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

    def test_run_no_video_data(self, mock_config, mock_checkpoint, mock_voiceover_segments):
        """Test running with no video data"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.embeddings = None
        state.text_metadata = []

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(result.warnings) > 0
        assert "video" in result.warnings[0].lower() or "embedding" in result.warnings[0].lower()

    @patch('src.matching.match_all_segments')
    @patch('src.embeddings.compute_embeddings')
    @patch('src.embeddings.get_embedding_provider')
    @patch('src.utils.CacheManager')
    def test_run_success(self, mock_cache_class, mock_provider, mock_compute,
                        mock_match_all, mock_config, mock_checkpoint,
                        mock_voiceover_segments, mock_text_metadata):
        """Test successful full stage execution"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata
        state.embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        state.embedding_index = Mock()
        state.face_preference = 'neutral'
        state.location_chapters = []

        # Mock embeddings
        mock_compute.return_value = np.array([[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])
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
    def test_run_exception_handling(self, mock_match_all, mock_provider, mock_config,
                                   mock_checkpoint, mock_voiceover_segments, mock_text_metadata):
        """Test exception handling in main run method"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.embedding_index = Mock()

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

    def test_restore_no_data(self, mock_checkpoint):
        """Test restore returns False when no checkpoint data"""
        stage = MatchStage()
        state = PipelineState()

        mock_checkpoint.get_stage_data.return_value = None

        result = stage.restore(state, mock_checkpoint)

        assert result is False

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

    def test_restore_exception_handling(self, mock_checkpoint):
        """Test restore handles exceptions"""
        stage = MatchStage()
        state = PipelineState()

        mock_checkpoint.get_stage_data.side_effect = Exception("Checkpoint error")

        result = stage.restore(state, mock_checkpoint)

        assert result is False


# ============================================================================
# Test Settings Display
# ============================================================================

class TestSettingsDisplay:
    """Test settings display"""

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
    def test_average_confidence_calculation(self, mock_cache_class, mock_provider,
                                           mock_compute, mock_match_all, mock_config,
                                           mock_checkpoint, mock_voiceover_segments,
                                           mock_text_metadata):
        """Test average confidence is calculated correctly"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.embedding_index = Mock()
        state.face_preference = 'neutral'

        mock_compute.return_value = np.array([[0.2, 0.3, 0.4]])
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
    def test_average_confidence_with_match_result(self, mock_cache_class, mock_provider,
                                                  mock_compute, mock_match_all, mock_config,
                                                  mock_checkpoint, mock_voiceover_segments,
                                                  mock_text_metadata):
        """Test average confidence with MatchResult objects (primary_match)"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.embedding_index = Mock()
        state.face_preference = 'neutral'

        mock_compute.return_value = np.array([[0.2, 0.3, 0.4]])
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

    def test_empty_text_metadata(self, mock_config, mock_checkpoint, mock_voiceover_segments):
        """Test handling empty text_metadata"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = []
        state.embeddings = np.array([[0.1, 0.2, 0.3]])

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(result.warnings) > 0

    def test_no_matches_returned(self, mock_config, mock_checkpoint, mock_voiceover_segments,
                                mock_text_metadata):
        """Test handling when matching returns no matches"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.embedding_index = Mock()

        with patch('src.matching.match_all_segments', return_value=[]):
            with patch('src.embeddings.compute_embeddings', return_value=np.array([[0.2, 0.3, 0.4]])):
                with patch('src.embeddings.get_embedding_provider', return_value=Mock()):
                    with patch('src.utils.CacheManager', return_value=Mock()):
                        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['match_count'] == 0
        assert result.data['avg_confidence'] == 0

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

    def test_matches_without_confidence_attribute(self, mock_config, mock_checkpoint,
                                                  mock_voiceover_segments, mock_text_metadata):
        """Test handling matches without confidence attribute"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.embedding_index = Mock()

        # Mock matches without proper attributes
        mock_matches = [Mock(spec=[])]  # No confidence attribute

        with patch('src.matching.match_all_segments', return_value=mock_matches):
            with patch('src.embeddings.compute_embeddings', return_value=np.array([[0.2, 0.3, 0.4]])):
                with patch('src.embeddings.get_embedding_provider', return_value=Mock()):
                    with patch('src.utils.CacheManager', return_value=Mock()):
                        result = stage.run(state, mock_config, mock_checkpoint)

        # Should handle gracefully
        assert result.success is True
        assert result.data['avg_confidence'] == 0  # No valid confidences

    @patch('src.embeddings.compute_embeddings')
    @patch('src.embeddings.get_embedding_provider')
    @patch('src.utils.CacheManager')
    def test_empty_voiceover_embeddings(self, mock_cache_class, mock_provider, mock_compute,
                                       mock_voiceover_segments, mock_text_metadata):
        """Test handling when voiceover embeddings are empty array"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.embedding_index = Mock()

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

    def test_delta_matching_flags(self, mock_voiceover_segments, mock_text_metadata):
        """Test delta matching and force rematch flags"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.embedding_index = Mock()

        with patch('src.matching.match_all_segments', return_value=[]) as mock_match:
            with patch('src.embeddings.compute_embeddings', return_value=np.array([[0.2, 0.3, 0.4]])):
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

    def test_location_chapters_in_matching(self, mock_voiceover_segments, mock_text_metadata):
        """Test location chapters are passed to matching"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = mock_voiceover_segments
        state.text_metadata = mock_text_metadata
        state.embeddings = np.array([[0.1, 0.2, 0.3]])
        state.embedding_index = Mock()
        state.location_chapters = [{'location': 'Paris', 'start_index': 0}]

        with patch('src.matching.match_all_segments', return_value=[]) as mock_match:
            with patch('src.embeddings.compute_embeddings', return_value=np.array([[0.2, 0.3, 0.4]])):
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
