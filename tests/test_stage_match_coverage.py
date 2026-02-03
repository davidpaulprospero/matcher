"""
Additional coverage tests for src/stages/match.py

Tests cover edge cases:
- Skip matching config
- Empty voiceover segments
- Empty text metadata/embeddings
- Delta matching flags
- Run logger stats
- Prepare segments with dict/object conversion
- B-roll segment creation from text_metadata
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.stages.match import MatchStage
from src.state import PipelineState


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config for matching"""
    config = MagicMock()
    config.pipeline.skip_matching = False
    config.matching.min_confidence = 0.5
    config.matching.high_confidence_threshold = 0.8
    config.matching.embedding_candidates = 10
    config.matching.llm_rerank_candidates = 5
    config.matching.max_clip_reuse = 3
    config.matching.force_rematch = False
    config.matching.delta_matching_enabled = True
    config.cache.cache_dir = ".cache"
    return config


@pytest.fixture
def mock_checkpoint():
    """Create mock checkpoint"""
    checkpoint = MagicMock()
    checkpoint.should_skip_stage.return_value = False
    checkpoint.get_stage_data.return_value = None
    return checkpoint


# ============================================================================
# Test Skip Matching
# ============================================================================

class TestMatchStageSkip:
    """Test skip matching behavior"""

    @pytest.mark.fast
    def test_skip_matching_config_true(self, mock_config, mock_checkpoint):
        """Test skipping when config.pipeline.skip_matching=true"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = [Mock()]
        state.text_metadata = [{}]
        state.embeddings = np.array([[0.1, 0.2]])

        mock_config.pipeline.skip_matching = True

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True


# ============================================================================
# Test Empty Inputs
# ============================================================================

class TestMatchStageEmptyInputs:
    """Test handling of empty inputs"""

    @pytest.mark.fast
    def test_no_voiceover_segments(self, mock_config, mock_checkpoint):
        """Test when no voiceover segments"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = []
        state.text_metadata = [{}]
        state.embeddings = np.array([[0.1, 0.2]])

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert "No voiceover segments" in result.warnings or result.data.get('matches') == []

    @pytest.mark.fast
    def test_no_text_metadata(self, mock_config, mock_checkpoint):
        """Test when no text metadata returns error (US-40-007)"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = [Mock(text="Test", index=0, start=0, end=5)]
        state.text_metadata = []

        result = stage.run(state, mock_config, mock_checkpoint)

        # US-40-007: Empty text_metadata with no caption_results should fail
        assert result.success is False
        assert "No captions available" in result.error

    @pytest.mark.fast
    def test_empty_embeddings(self, mock_config, mock_checkpoint):
        """Test when voiceover embeddings computation returns empty"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = [Mock(text="Test", index=0, start=0, end=5)]
        state.text_metadata = [{}]

        # Mock compute_embeddings to return empty for voiceover (triggers early return)
        with patch('src.embeddings.compute_embeddings', return_value=np.array([])):
            with patch('src.embeddings.get_embedding_provider', return_value=Mock()):
                with patch('src.utils.CacheManager', return_value=Mock()):
                    result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data.get('matches') == [] or result.data.get('match_count') == 0


# ============================================================================
# Test Prepare Segments
# ============================================================================

class TestPrepareSegments:
    """Test _prepare_segments method"""

    @pytest.mark.fast
    def test_prepare_voiceover_segment_objects(self, mock_config):
        """Test preparing VoiceoverSegment objects"""
        stage = MatchStage()
        state = PipelineState()

        # Create VoiceoverSegment-like objects
        vo_seg = Mock()
        vo_seg.text = "Hello world"
        vo_seg.index = 0
        vo_seg.start = 0.0
        vo_seg.end = 5.0
        state.voiceover_segments = [vo_seg]
        state.text_metadata = []

        vo_segments, video_segments, paths = stage._prepare_segments(state)

        assert len(vo_segments) == 1
        assert vo_segments[0].text == "Hello world"
        assert vo_segments[0].start_time == 0.0
        assert vo_segments[0].end_time == 5.0

    @pytest.mark.fast
    def test_prepare_voiceover_segment_dicts(self, mock_config):
        """Test preparing voiceover segments from dicts"""
        stage = MatchStage()
        state = PipelineState()

        # Dict-based segment
        vo_dict = {
            'index': 1,
            'text': 'Dict segment',
            'start_time': 10.0,
            'end_time': 15.0,
            'keywords': ['test'],
            'entities': ['Person'],
            'chapter_topics': ['Topic1']
        }
        state.voiceover_segments = [vo_dict]
        state.text_metadata = []

        vo_segments, video_segments, paths = stage._prepare_segments(state)

        assert len(vo_segments) == 1
        assert vo_segments[0].text == 'Dict segment'
        assert vo_segments[0].keywords == ['test']

    @pytest.mark.fast
    def test_prepare_voiceover_segment_dict_alt_keys(self, mock_config):
        """Test preparing dicts with alternate key names"""
        stage = MatchStage()
        state = PipelineState()

        # Dict with 'start' and 'end' instead of 'start_time', 'end_time'
        vo_dict = {
            'text': 'Alt keys',
            'start': 5.0,
            'end': 10.0
        }
        state.voiceover_segments = [vo_dict]
        state.text_metadata = []

        vo_segments, video_segments, paths = stage._prepare_segments(state)

        assert len(vo_segments) == 1
        assert vo_segments[0].start_time == 5.0
        assert vo_segments[0].end_time == 10.0

    @pytest.mark.fast
    def test_prepare_video_segments_from_dict_metadata(self, mock_config):
        """Test preparing video segments from dict metadata"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = []

        meta = {
            'start_time': 0.0,
            'end_time': 10.0,
            'text': 'Video segment text',
            'video_path': '/path/to/video.mp4',
            'source': 'youtube',
            'face_score': 0.8,
            'is_broll': True,
            'scene_index': 5
        }
        state.text_metadata = [meta]

        vo_segments, video_segments, paths = stage._prepare_segments(state)

        assert len(video_segments) == 1
        assert video_segments[0].text == 'Video segment text'
        assert video_segments[0].source == 'youtube'
        assert video_segments[0].face_score == 0.8
        assert video_segments[0].is_broll is True
        assert video_segments[0].scene_index == 5
        assert '/path/to/video.mp4' in paths

    @pytest.mark.fast
    def test_prepare_voiceover_segment_without_text_attr(self, mock_config):
        """Test line 211: Segment without .text attribute is passed through as-is"""
        stage = MatchStage()
        state = PipelineState()

        # Create an object WITHOUT a .text attribute to trigger else branch (line 211)
        # This simulates an edge case where segment type is unknown
        class ProcessedSegment:
            """Marker class for already-processed segments without .text"""
            def __init__(self):
                self.index = 99
                self.start_time = 1.0
                self.end_time = 2.0
                # Deliberately no .text attribute

        processed = ProcessedSegment()
        state.voiceover_segments = [processed]
        state.text_metadata = []

        vo_segments, video_segments, paths = stage._prepare_segments(state)

        assert len(vo_segments) == 1
        # The processed segment should be passed through as-is (line 211)
        assert vo_segments[0] is processed
        assert vo_segments[0].index == 99

    @pytest.mark.fast
    def test_prepare_video_segments_from_objects(self, mock_config):
        """Test preparing video segments from objects"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = []

        meta_obj = Mock()
        meta_obj.source_file = '/path/to/video.mp4'
        state.text_metadata = [meta_obj]

        vo_segments, video_segments, paths = stage._prepare_segments(state)

        assert len(video_segments) == 1
        assert video_segments[0] is meta_obj

    @pytest.mark.fast
    def test_prepare_counts_broll_entries(self, mock_config, caplog):
        """Test that B-roll entries are counted and logged"""
        import logging

        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = []

        # Mix of B-roll and non-B-roll
        state.text_metadata = [
            {'is_broll': True, 'text': 'broll 1'},
            {'is_broll': True, 'text': 'broll 2'},
            {'is_broll': False, 'text': 'not broll'},
            {'text': 'no is_broll key'}
        ]

        with caplog.at_level(logging.INFO):
            vo_segments, video_segments, paths = stage._prepare_segments(state)

        # Should have logged B-roll counts
        assert any("is_broll=True" in r.message for r in caplog.records) or len(video_segments) == 4


# ============================================================================
# Test Validate Inputs
# ============================================================================

class TestMatchValidateInputs:
    """Test validate_inputs method"""

    @pytest.mark.fast
    def test_validate_no_voiceover(self, mock_config):
        """Test validation fails with no voiceover"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = []
        state.embeddings = np.array([[0.1]])

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "voiceover" in error.lower()

    @pytest.mark.fast
    def test_validate_no_text_data(self, mock_config):
        """Test validation fails with no text_metadata or caption_results"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = [Mock()]
        state.text_metadata = []  # Empty

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "text_metadata" in error.lower()

    @pytest.mark.fast
    def test_validate_success(self, mock_config):
        """Test validation succeeds with valid inputs"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = [Mock()]
        state.embeddings = np.array([[0.1, 0.2]])
        state.text_metadata = [{'video_path': 'v.mp4', 'text': 'test'}]  # Required field

        error = stage.validate_inputs(state, mock_config)

        assert error is None


# ============================================================================
# Test Checkpoint Operations
# ============================================================================

class TestMatchCheckpoint:
    """Test checkpoint operations"""

    @pytest.mark.fast
    def test_can_skip_false(self, mock_checkpoint):
        """Test can_skip returns False when not in checkpoint"""
        stage = MatchStage()
        state = PipelineState()
        mock_checkpoint.should_skip_stage.return_value = False

        result = stage.can_skip(state, mock_checkpoint)

        assert result is False

    @pytest.mark.fast
    def test_can_skip_true(self, mock_checkpoint):
        """Test can_skip returns True when in checkpoint"""
        stage = MatchStage()
        state = PipelineState()
        mock_checkpoint.should_skip_stage.return_value = True

        result = stage.can_skip(state, mock_checkpoint)

        assert result is True

    @pytest.mark.fast
    def test_restore_no_data(self, mock_checkpoint):
        """Test restore returns False when no data"""
        stage = MatchStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.return_value = None

        result = stage.restore(state, mock_checkpoint)

        assert result is False

    @pytest.mark.fast
    def test_restore_with_data(self, mock_checkpoint):
        """Test restore returns True when data exists"""
        stage = MatchStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.return_value = {'match_count': 5}

        result = stage.restore(state, mock_checkpoint)

        assert result is True

    @pytest.mark.fast
    def test_restore_exception(self, mock_checkpoint, caplog):
        """Test restore handles exceptions"""
        import logging

        stage = MatchStage()
        state = PipelineState()
        mock_checkpoint.get_stage_data.side_effect = Exception("Restore error")

        with caplog.at_level(logging.WARNING):
            result = stage.restore(state, mock_checkpoint)

        assert result is False


# ============================================================================
# Test Run Matching
# ============================================================================

class TestRunMatching:
    """Test _run_matching method"""

    @pytest.mark.fast
    def test_run_matching_success(self, mock_config):
        """Test successful matching run"""
        stage = MatchStage()
        state = PipelineState()
        state.face_preference = 0.5

        # Mock voiceover segment
        vo_seg = Mock()
        vo_seg.text = "Test segment"

        # Patch imports inside _run_matching
        # compute_embeddings is called twice: voiceover + video
        with patch('src.matching.match_all_segments', return_value=[Mock(confidence=0.9)]) as mock_match:
            with patch('src.embeddings.compute_embeddings', side_effect=[np.array([[0.3, 0.4]]), np.array([[0.1, 0.2]])]) as mock_compute:
                with patch('src.embeddings.get_embedding_provider', return_value=Mock()):
                    with patch('src.utils.CacheManager', return_value=Mock()):
                        matches = stage._run_matching(
                            vo_segments=[vo_seg],
                            video_segments=[],
                            all_video_paths=[],
                            state=state,
                            config=mock_config,
                            delta_enabled=True,
                            force_rematch=False
                        )

        assert len(matches) == 1
        assert mock_compute.call_count == 2  # voiceover + video
        mock_match.assert_called_once()

    @pytest.mark.fast
    def test_run_matching_empty_embeddings(self, mock_config):
        """Test matching with failed embedding computation"""
        stage = MatchStage()
        state = PipelineState()

        vo_seg = Mock()
        vo_seg.text = "Test segment"

        # Return None for voiceover embeddings (early exit before video embeddings)
        with patch('src.embeddings.compute_embeddings', return_value=None):
            with patch('src.embeddings.get_embedding_provider', return_value=Mock()):
                with patch('src.utils.CacheManager', return_value=Mock()):
                    matches = stage._run_matching(
                        vo_segments=[vo_seg],
                        video_segments=[],
                        all_video_paths=[],
                        state=state,
                        config=mock_config,
                        delta_enabled=True,
                        force_rematch=False
                    )

        assert matches == []


# ============================================================================
# Test Exception Handling
# ============================================================================

class TestMatchExceptionHandling:
    """Test exception handling in run"""

    @pytest.mark.fast
    def test_run_exception(self, mock_config, mock_checkpoint):
        """Test that exceptions are caught and returned as failure"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = [Mock(text="Test", index=0, start=0, end=5)]
        state.text_metadata = [{'text': 'Test'}]
        state.embeddings = np.array([[0.1, 0.2]])

        with patch.object(stage, '_prepare_segments', side_effect=Exception("Prepare failed")):
            result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is False
        assert "Prepare failed" in result.error


# ============================================================================
# Test Confidence Stats
# ============================================================================

class TestConfidenceStats:
    """Test confidence statistics calculation"""

    @pytest.mark.fast
    def test_confidence_with_match_results(self, mock_config, mock_checkpoint):
        """Test confidence calculation with MatchResult objects"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = [Mock(text="Test", index=0, start=0, end=5)]
        state.text_metadata = [{'text': 'Video'}]
        state.face_preference = 0.5

        # Create MatchResult-like objects with primary_match
        match_result = Mock()
        match_result.primary_match = Mock(confidence=0.85)

        # Mock run logger
        mock_run_logger = Mock()

        with patch('src.matching.match_all_segments', return_value=[match_result]):
            with patch('src.embeddings.compute_embeddings', side_effect=[np.array([[0.3, 0.4]]), np.array([[0.1, 0.2]])]):
                with patch('src.embeddings.get_embedding_provider', return_value=Mock()):
                    with patch('src.utils.CacheManager', return_value=Mock()):
                        with patch('src.stages.match.get_global_logger', return_value=mock_run_logger):
                            result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['avg_confidence'] == pytest.approx(0.85, rel=0.01)
        mock_run_logger.set_stats.assert_called()

    @pytest.mark.fast
    def test_confidence_with_direct_match(self, mock_config, mock_checkpoint):
        """Test confidence calculation with direct Match objects"""
        stage = MatchStage()
        state = PipelineState()
        state.voiceover_segments = [Mock(text="Test", index=0, start=0, end=5)]
        state.text_metadata = [{'text': 'Video'}]
        state.face_preference = 0.5

        # Create direct Match object (no primary_match attribute)
        match = Mock(spec=['confidence'])  # Use spec to control attributes
        match.confidence = 0.75

        with patch('src.matching.match_all_segments', return_value=[match]):
            with patch('src.embeddings.compute_embeddings', side_effect=[np.array([[0.3, 0.4]]), np.array([[0.1, 0.2]])]):
                with patch('src.embeddings.get_embedding_provider', return_value=Mock()):
                    with patch('src.utils.CacheManager', return_value=Mock()):
                        with patch('src.stages.match.get_global_logger', return_value=Mock()):
                            result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data['avg_confidence'] == pytest.approx(0.75, rel=0.01)
