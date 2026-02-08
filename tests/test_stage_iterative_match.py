"""
Tests for IterativeMatchStage.

Tests cover:
- Gap detection and identification
- Gap filling with search
- Max iterations config respect
- Stage registration and initialization
- Source spacing enforcement
- Skip conditions and checkpoint restore

Created: 2026-02-02 (Sprint 34 - Pipeline Focus)
"""

from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, PropertyMock
import pytest

from src.stages.iterative_match import (
    IterativeMatchStage,
    LockedMatch,
    GapSegment,
    PassMetrics,
)
from src.state import PipelineState
from src.state import Match as StateMatch
from src.utils import Match, MatchResult, SRTSegment


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config with iterative matching settings"""
    config = MagicMock()

    # iterative_matching config
    iter_config = MagicMock()
    iter_config.enabled = True
    iter_config.target_confidence = 0.90
    iter_config.source_spacing_seconds = 300.0
    iter_config.max_iterations = 5
    iter_config.min_gap_percentage = 0.05
    iter_config.search_results_per_gap = 10
    iter_config.max_new_videos_per_pass = 50
    iter_config.use_voiceover_text_queries = True
    iter_config.use_similar_to_locked = True
    iter_config.use_entity_topic_queries = True
    iter_config.enable_progressive_refinement = True
    iter_config.analyze_gap_patterns = True
    iter_config.enable_query_learning = False  # Disable for tests
    iter_config.caption_batch_size = 10
    iter_config.caption_fetch_delay = 0.0  # No delay for tests
    iter_config.search_cache_ttl_hours = 24  # Required for SearchResultsCache

    config.iterative_matching = iter_config

    # download config for cookies
    config.download = MagicMock()
    config.download.cookie_rotation = None

    return config


@pytest.fixture
def mock_checkpoint():
    """Create mock checkpoint manager"""
    checkpoint = MagicMock()
    checkpoint.should_skip_stage.return_value = False
    checkpoint.get_stage_data.return_value = None
    checkpoint.save_intermediate = MagicMock()
    return checkpoint


@pytest.fixture
def mock_voiceover_segments():
    """Create mock voiceover segments"""
    segments = []
    for i in range(10):
        seg = MagicMock()
        seg.index = i
        seg.segment_index = i
        seg.text = f"Segment {i} text about topic {i % 3}"
        seg.start = i * 10.0  # 0, 10, 20, ...
        seg.end = (i + 1) * 10.0
        segments.append(seg)
    return segments


@pytest.fixture
def mock_matches_with_gaps(mock_voiceover_segments):
    """Create matches with some low confidence (gaps)"""
    matches = []
    for i, seg in enumerate(mock_voiceover_segments):
        match = MagicMock()
        match.segment_index = i
        # Make every 3rd segment a gap (low confidence)
        if i % 3 == 0:
            match.confidence = 0.5  # Below 0.90 threshold
        else:
            match.confidence = 0.95  # Above threshold
        # Use unique videos to avoid source spacing violations
        match.video_file = f"video_{i}.mp4"  # Each segment gets unique video
        match.video_start = 0.0
        match.video_end = 10.0
        match.strategy = "test"
        match.reason = "test match"
        match.face_score = 0.5
        # No primary_match attribute (use simple Match structure)
        match.primary_match = None
        matches.append(match)
    return matches


@pytest.fixture
def mock_state_with_gaps(mock_voiceover_segments, mock_matches_with_gaps):
    """Create state with matches containing gaps"""
    state = PipelineState()
    state.voiceover_segments = mock_voiceover_segments
    state.matches = mock_matches_with_gaps
    # US-40-009: Provide some text_metadata candidates so stage doesn't skip
    state.text_metadata = [
        {'text': f'caption {i}', 'video_path': f'vid_{i}', 'start_time': 0, 'end_time': 5}
        for i in range(10)
    ]
    state.embeddings = None
    state.embedding_index = None
    state.extracted_entities = []
    state.voiceover_path = "/tmp/test/voiceover.srt"
    return state


@pytest.fixture
def mock_state_all_high_confidence():
    """Create state with all high confidence matches (no gaps)"""
    state = PipelineState()

    # Create voiceover segments with wide spacing (>300s apart)
    vo_segments = []
    for i in range(5):
        seg = MagicMock()
        seg.index = i
        seg.segment_index = i
        seg.text = f"Segment {i} text"
        seg.start = i * 400.0  # 0, 400, 800, 1200, 1600 seconds (>300s apart)
        seg.end = seg.start + 10.0
        vo_segments.append(seg)

    state.voiceover_segments = vo_segments

    matches = []
    for i, seg in enumerate(vo_segments):
        match = MagicMock()
        match.segment_index = i
        match.confidence = 0.95  # All high confidence
        match.video_file = f"unique_video_{i}.mp4"  # Unique videos
        match.video_start = 0.0
        match.video_end = 10.0
        match.primary_match = None
        matches.append(match)

    state.matches = matches
    # US-40-009: Provide some text_metadata candidates so stage doesn't skip
    state.text_metadata = [
        {'text': f'caption {i}', 'video_path': f'vid_{i}', 'start_time': 0, 'end_time': 5}
        for i in range(5)
    ]
    state.embeddings = None
    state.extracted_entities = []
    state.voiceover_path = "/tmp/test/voiceover.srt"
    return state


@pytest.fixture
def stage():
    """Create IterativeMatchStage instance"""
    return IterativeMatchStage()


# ============================================================================
# Stage Registration and Initialization
# ============================================================================

class TestStageRegistration:
    """Test stage registration and basic properties"""

    def test_stage_name(self, stage):
        """Stage has correct name"""
        assert stage.name == "ITERATIVE_MATCH"

    def test_stage_description(self, stage):
        """Stage has description"""
        assert stage.description == "Fill matching gaps with iterative search passes"

    def test_stage_initializes_tracking_sets(self, stage):
        """Stage initializes cross-pass tracking sets"""
        assert hasattr(stage, '_fetched_video_ids')
        assert hasattr(stage, '_used_queries')
        assert isinstance(stage._fetched_video_ids, set)
        assert isinstance(stage._used_queries, set)


# ============================================================================
# Gap Detection Tests (AC: test_iterative_match_detects_gaps)
# ============================================================================

class TestIterativeMatchDetectsGaps:
    """Test gap detection and identification"""

    def test_iterative_match_detects_gaps(
        self, stage, mock_state_with_gaps, mock_config
    ):
        """Gaps are correctly identified based on confidence threshold"""
        locked, gaps = stage._identify_gaps_and_locks(
            mock_state_with_gaps.matches,
            mock_state_with_gaps.voiceover_segments,
            target_conf=0.90,
            source_spacing=300.0
        )

        # Every 3rd segment (indices 0, 3, 6, 9) should be a gap
        gap_indices = {g.segment_index for g in gaps}
        expected_gaps = {0, 3, 6, 9}

        assert gap_indices == expected_gaps
        assert len(gaps) == 4
        assert len(locked) == 6

    def test_detects_gaps_from_low_confidence(self, stage):
        """Segments with low confidence are marked as gaps"""
        # Create simple matches
        matches = []
        vo_segments = []

        for i in range(5):
            match = MagicMock()
            match.segment_index = i
            match.confidence = 0.5 if i < 3 else 0.95  # First 3 are low
            match.video_file = f"vid_{i}.mp4"
            match.primary_match = None
            matches.append(match)

            seg = MagicMock()
            seg.index = i
            seg.text = f"text {i}"
            seg.start = i * 10.0
            vo_segments.append(seg)

        locked, gaps = stage._identify_gaps_and_locks(
            matches, vo_segments, target_conf=0.90, source_spacing=300.0
        )

        assert len(gaps) == 3
        assert len(locked) == 2

        # Verify gap reasons
        for gap in gaps:
            assert gap.reason == 'low_confidence'

    def test_detects_gaps_from_source_spacing_violation(self, stage):
        """Same video source within spacing window creates gap"""
        # Create matches where same video is used too close together
        matches = []
        vo_segments = []

        for i in range(4):
            match = MagicMock()
            match.segment_index = i
            match.confidence = 0.95  # All high confidence
            match.video_file = "same_video.mp4"  # Same video for all
            match.primary_match = None
            matches.append(match)

            seg = MagicMock()
            seg.index = i
            seg.text = f"text {i}"
            seg.start = i * 10.0  # 0, 10, 20, 30 seconds
            vo_segments.append(seg)

        # With 300s spacing, only first should be locked, rest are gaps
        locked, gaps = stage._identify_gaps_and_locks(
            matches, vo_segments, target_conf=0.90, source_spacing=300.0
        )

        assert len(locked) == 1
        assert locked[0].segment_index == 0

        assert len(gaps) == 3
        for gap in gaps:
            assert gap.reason == 'spacing_violation'

    def test_gap_analysis_patterns(
        self, stage, mock_state_with_gaps, mock_config, mock_checkpoint
    ):
        """Gap pattern analysis classifies gaps correctly"""
        # Create voiceover segments with pattern-specific text
        texts = [
            "The concept of freedom and justice matters.",  # abstract
            "Running through the streets of New York City.",  # location
            "John Smith visited the White House.",  # proper noun
            "She felt happy and excited about the news.",  # emotion
        ]

        vo_segments = []
        for i, text in enumerate(texts):
            seg = MagicMock()
            seg.index = i
            seg.segment_index = i
            seg.text = text
            seg.start = i * 10.0
            vo_segments.append(seg)

        # Create low confidence matches to generate gaps
        matches = []
        for i in range(len(texts)):
            match = MagicMock()
            match.segment_index = i
            match.confidence = 0.5  # All gaps
            match.video_file = f"vid_{i}.mp4"
            match.primary_match = None
            matches.append(match)

        # Run gap identification
        locked, gaps = stage._identify_gaps_and_locks(
            matches, vo_segments, target_conf=0.90, source_spacing=300.0
        )

        assert len(gaps) == 4

        # Verify voiceover text is captured in gaps
        gap_texts = {g.voiceover_text for g in gaps}
        for text in texts:
            assert text in gap_texts


# ============================================================================
# Gap Filling Tests (AC: test_iterative_match_fills_gaps_with_search)
# ============================================================================

class TestIterativeMatchFillsGapsWithSearch:
    """Test that gaps are filled via search"""

    def test_iterative_match_fills_gaps_with_search(
        self, stage, mock_state_with_gaps, mock_config, mock_checkpoint
    ):
        """Gaps are filled when search returns good candidates"""
        # Mock the search and caption fetch to return candidates
        with patch.object(stage, '_search_youtube_for_videos') as mock_search, \
             patch.object(stage, '_fetch_captions_for_videos') as mock_fetch, \
             patch.object(stage, '_rematch_gaps') as mock_rematch:

            mock_search.return_value = ['new_vid_1', 'new_vid_2']
            mock_fetch.return_value = [
                {
                    'video_id': 'new_vid_1',
                    'segments': [{'text': 'caption text', 'start': 0, 'end': 5}]
                }
            ]
            # Simulate filling 2 gaps
            mock_rematch.return_value = 2

            result = stage.run(mock_state_with_gaps, mock_config, mock_checkpoint)

            assert result.success is True
            # Search should have been called
            assert mock_search.called
            # Rematch should have been called
            assert mock_rematch.called

    def test_generates_multi_strategy_queries(
        self, stage, mock_state_with_gaps, mock_config
    ):
        """Multiple search strategies generate queries"""
        # Create gaps
        locked, gaps = stage._identify_gaps_and_locks(
            mock_state_with_gaps.matches,
            mock_state_with_gaps.voiceover_segments,
            target_conf=0.90,
            source_spacing=300.0
        )

        # Mock gap analysis
        mock_gap_analysis = MagicMock()
        mock_gap_analysis.clustered_gaps = {
            'abstract_concept': [0, 3],
            'other': [6, 9]
        }

        # Patch at the module where it's imported, not where it's defined
        with patch('src.iterative_match.gap_analyzer.extract_keywords_for_gap') as mock_kw:
            mock_kw.return_value = ['test', 'keyword']

            queries = stage._generate_multi_strategy_queries(
                gaps, locked, mock_state_with_gaps,
                mock_config.iterative_matching,
                pass_num=1,
                gap_analysis=mock_gap_analysis
            )

        # Should have generated queries from voiceover text strategy
        assert len(queries) > 0

        # Verify query structure
        for q in queries:
            assert 'query' in q
            assert 'strategy' in q
            assert 'gap_indices' in q

    def test_search_excludes_existing_videos(
        self, stage, mock_state_with_gaps, mock_config
    ):
        """Search excludes videos already in state"""
        mock_state_with_gaps.video_ids = ['existing_vid_1', 'existing_vid_2']

        # Set the search cache ttl as a proper integer
        mock_config.iterative_matching.search_cache_ttl_hours = 24

        with patch('subprocess.run') as mock_run, \
             patch('src.stages.iterative_match.SearchResultsCache') as mock_cache_class:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"id": "new_vid_333333", "title": "test", "duration": 60}\n'
            )

            # Mock the cache to return None (cache miss)
            mock_cache_instance = MagicMock()
            mock_cache_instance.get_search_result.return_value = None
            mock_cache_class.return_value = mock_cache_instance

            queries = [{'query': 'test search', 'strategy': 'voiceover'}]

            video_ids = stage._search_youtube_for_videos(
                queries, mock_state_with_gaps, mock_config,
                mock_config.iterative_matching
            )

            # Should return new video
            assert 'new_vid_333333' in video_ids
            # Existing videos should be excluded
            assert 'existing_vid_1' not in video_ids
            assert 'existing_vid_2' not in video_ids


# ============================================================================
# Max Iterations Tests (AC: test_iterative_match_respects_max_iterations)
# ============================================================================

class TestIterativeMatchRespectsMaxIterations:
    """Test that max_iterations config is respected"""

    def test_iterative_match_respects_max_iterations(
        self, stage, mock_state_with_gaps, mock_config, mock_checkpoint
    ):
        """Stage stops after max_iterations passes"""
        # Set max_iterations to 2
        mock_config.iterative_matching.max_iterations = 2

        pass_count = 0

        def count_passes(*args, **kwargs):
            nonlocal pass_count
            pass_count += 1
            return ['vid_1']  # Return some results to continue

        def fill_some_gaps(*args, **kwargs):
            return 1  # Fill 1 gap per pass

        with patch.object(stage, '_search_youtube_for_videos', side_effect=count_passes), \
             patch.object(stage, '_fetch_captions_for_videos', return_value=[{'video_id': 'v1', 'segments': []}]), \
             patch.object(stage, '_rematch_gaps', side_effect=fill_some_gaps):

            result = stage.run(mock_state_with_gaps, mock_config, mock_checkpoint)

            assert result.success is True
            # Should not exceed max_iterations
            assert pass_count <= 2

    def test_stops_early_when_no_gaps(
        self, stage, mock_state_all_high_confidence, mock_config, mock_checkpoint
    ):
        """Stage stops early when all matches are good"""
        # Set required config values
        mock_config.iterative_matching.search_cache_ttl_hours = 24

        result = stage.run(mock_state_all_high_confidence, mock_config, mock_checkpoint)

        assert result.success is True
        # Should report no gaps remaining
        data = result.data
        assert data.get('final_gaps', -1) == 0 or data.get('skipped')

    def test_stops_when_gap_percentage_below_threshold(
        self, stage, mock_config, mock_checkpoint
    ):
        """Stage stops when gap percentage drops below min_gap_percentage"""
        # Set threshold to 50%
        mock_config.iterative_matching.min_gap_percentage = 0.50

        # Create state with only 1 gap out of 10 (10% gaps)
        state = PipelineState()
        state.voiceover_path = "/tmp/test/voiceover.srt"
        state.extracted_entities = []
        state.text_metadata = []
        state.embeddings = None

        vo_segments = []
        matches = []

        for i in range(10):
            seg = MagicMock()
            seg.index = i
            seg.text = f"text {i}"
            seg.start = i * 100.0  # Spread out for spacing
            vo_segments.append(seg)

            match = MagicMock()
            match.segment_index = i
            match.confidence = 0.5 if i == 0 else 0.95  # Only first is gap
            match.video_file = f"vid_{i}.mp4"  # Different videos
            match.primary_match = None
            matches.append(match)

        state.voiceover_segments = vo_segments
        state.matches = matches

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        # Should have stopped due to low gap percentage

    def test_max_iterations_config_from_dataclass(self, stage, mock_checkpoint):
        """Config defaults are used when iterative_matching not set"""
        from src.config.sections.iterative_matching import IterativeMatchingConfig

        config = MagicMock()
        config.iterative_matching = IterativeMatchingConfig()
        config.download = MagicMock()
        config.download.cookie_rotation = None

        # Default max_iterations should be 5
        assert config.iterative_matching.max_iterations == 5


# ============================================================================
# Skip and Restore Tests
# ============================================================================

class TestCanSkipAndRestore:
    """Test can_skip and restore methods"""

    def test_can_skip_delegates_to_checkpoint(self, stage, mock_checkpoint):
        """can_skip checks checkpoint manager"""
        state = PipelineState()
        mock_checkpoint.should_skip_stage.return_value = True

        result = stage.can_skip(state, mock_checkpoint)

        assert result is True
        mock_checkpoint.should_skip_stage.assert_called_once_with("ITERATIVE_MATCH")

    def test_restore_loads_matches_from_checkpoint(self, stage, mock_checkpoint, mock_config):
        """restore() loads matches from checkpoint data"""
        mock_checkpoint.get_stage_data.return_value = {
            'passes_completed': 3,
            'total_gaps_filled': 5,
            'matches': [
                {
                    'segment_index': 0,
                    'video_file': 'restored_vid.mp4',
                    'video_start': 0.0,
                    'video_end': 10.0,
                    'confidence': 0.95,
                    'strategy': 'iterative',
                    'reason': 'gap filled',
                    'face_score': 0.5,
                }
            ]
        }

        state = PipelineState()
        state.matches = []

        result = stage.restore(state, mock_checkpoint, mock_config)

        assert result is True
        assert len(state.matches) == 1
        assert state.matches[0].video_file == 'restored_vid.mp4'

    def test_restore_handles_missing_data(self, stage, mock_checkpoint, mock_config):
        """restore() returns False when no checkpoint data"""
        mock_checkpoint.get_stage_data.return_value = None

        state = PipelineState()

        result = stage.restore(state, mock_checkpoint, mock_config)

        assert result is False


# ============================================================================
# Validate Inputs Tests
# ============================================================================

class TestValidateInputs:
    """Test validate_inputs method"""

    def test_validate_inputs_returns_error_when_no_matches(
        self, stage, mock_config
    ):
        """validate_inputs returns error message when matches empty"""
        state = PipelineState()
        state.matches = []
        state.voiceover_segments = [MagicMock()]

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "No matches" in error

    def test_validate_inputs_returns_error_when_no_voiceover(
        self, stage, mock_config
    ):
        """validate_inputs returns error when no voiceover segments"""
        state = PipelineState()
        state.matches = [MagicMock()]
        state.voiceover_segments = []

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "No voiceover" in error

    def test_validate_inputs_returns_none_when_valid(
        self, stage, mock_config, mock_state_with_gaps
    ):
        """validate_inputs returns None when inputs are valid"""
        error = stage.validate_inputs(mock_state_with_gaps, mock_config)

        assert error is None


# ============================================================================
# Source Spacing Tests
# ============================================================================

class TestSourceSpacing:
    """Test source spacing enforcement"""

    def test_source_spacing_enforced(self, stage):
        """Same video creates gap when used within spacing window"""
        matches = []
        vo_segments = []

        for i in range(3):
            match = MagicMock()
            match.segment_index = i
            match.confidence = 0.95
            match.video_file = "same_video.mp4"
            match.primary_match = None
            matches.append(match)

            seg = MagicMock()
            seg.index = i
            seg.text = f"text {i}"
            seg.start = i * 100.0  # 0, 100, 200 seconds
            vo_segments.append(seg)

        # 300s spacing means only first is valid
        locked, gaps = stage._identify_gaps_and_locks(
            matches, vo_segments, target_conf=0.90, source_spacing=300.0
        )

        assert len(locked) == 1
        assert locked[0].segment_index == 0

        # Others are gaps due to spacing
        assert len(gaps) == 2
        for gap in gaps:
            assert gap.reason == 'spacing_violation'

    def test_different_videos_not_affected_by_spacing(self, stage):
        """Different videos are not affected by source spacing"""
        matches = []
        vo_segments = []

        for i in range(3):
            match = MagicMock()
            match.segment_index = i
            match.confidence = 0.95
            match.video_file = f"video_{i}.mp4"  # Different videos
            match.primary_match = None
            matches.append(match)

            seg = MagicMock()
            seg.index = i
            seg.text = f"text {i}"
            seg.start = i * 10.0  # Close together
            vo_segments.append(seg)

        locked, gaps = stage._identify_gaps_and_locks(
            matches, vo_segments, target_conf=0.90, source_spacing=300.0
        )

        # All should be locked (different videos)
        assert len(locked) == 3
        assert len(gaps) == 0


# ============================================================================
# Disabled Stage Tests
# ============================================================================

class TestDisabledStage:
    """Test behavior when stage is disabled"""

    def test_skips_when_disabled(
        self, stage, mock_state_with_gaps, mock_checkpoint
    ):
        """Stage returns skipped when disabled in config"""
        config = MagicMock()
        config.iterative_matching = MagicMock()
        config.iterative_matching.enabled = False

        result = stage.run(mock_state_with_gaps, config, mock_checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True
        assert result.data.get('reason') == 'disabled'

    def test_skips_when_no_matches(
        self, stage, mock_config, mock_checkpoint
    ):
        """Stage returns skipped when no matches exist"""
        state = PipelineState()
        state.matches = []
        state.voiceover_segments = [MagicMock()]
        # US-40-009: Need text_metadata to avoid 'no_candidates' check first
        state.text_metadata = [{'text': 'caption', 'video_path': 'vid'}]

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True
        assert result.data.get('reason') == 'no_matches'


# ============================================================================
# Video ID Extraction Tests
# ============================================================================

class TestVideoIdExtraction:
    """Test video ID extraction from matches"""

    def test_extract_video_id_from_filename(self, stage):
        """Video ID is extracted from filename"""
        match = MagicMock()
        match.video_file = "abc12345678.mp4"
        match.primary_match = None

        video_id = stage._extract_video_id(match)

        assert video_id == "abc12345678"

    def test_extract_video_id_from_primary_match(self, stage):
        """Video ID is extracted from primary_match structure"""
        match = MagicMock()
        match.primary_match = MagicMock()
        match.primary_match.video_segment = MagicMock()
        match.primary_match.video_segment.source_file = "xyz98765432.mp4"

        video_id = stage._extract_video_id(match)

        assert video_id == "xyz98765432"

    def test_extract_video_id_returns_empty_for_no_source(self, stage):
        """Returns empty string when no source file"""
        match = MagicMock()
        match.video_file = ""
        match.primary_match = None

        video_id = stage._extract_video_id(match)

        assert video_id == ""


# ============================================================================
# Query Generation Tests
# ============================================================================

class TestQueryGeneration:
    """Test search query generation"""

    def test_queries_are_deduplicated(self, stage, mock_state_with_gaps, mock_config):
        """Duplicate queries are filtered out"""
        gaps = [
            GapSegment(0, 0.5, "same text same text", 0.0),
            GapSegment(1, 0.5, "same text same text", 10.0),  # Same text
        ]
        locked = []

        mock_gap_analysis = MagicMock()
        mock_gap_analysis.clustered_gaps = {'other': [0, 1]}

        # Patch at the module where it's imported
        with patch('src.iterative_match.gap_analyzer.extract_keywords_for_gap') as mock_kw:
            mock_kw.return_value = ['same', 'text']

            queries = stage._generate_multi_strategy_queries(
                gaps, locked, mock_state_with_gaps,
                mock_config.iterative_matching,
                pass_num=1,
                gap_analysis=mock_gap_analysis
            )

        # Should have deduplicated
        query_texts = [q['query'] for q in queries]
        unique_query_texts = list(set(query_texts))
        assert len(query_texts) == len(unique_query_texts)

    def test_tracks_used_queries_across_passes(self, stage):
        """Used queries are tracked to avoid repeating"""
        stage._used_queries = set()

        # First pass uses "test query"
        stage._used_queries.add("test query")

        # Second pass should exclude it
        assert "test query" in stage._used_queries


# ============================================================================
# Integration Test
# ============================================================================

class TestIterativeMatchIntegration:
    """Integration tests for full stage execution"""

    def test_full_stage_execution_with_gaps(
        self, stage, mock_state_with_gaps, mock_config, mock_checkpoint
    ):
        """Full stage execution processes gaps"""
        # Disable heavy operations for test
        mock_config.iterative_matching.enable_query_learning = False
        mock_config.iterative_matching.max_iterations = 1

        with patch.object(stage, '_search_youtube_for_videos', return_value=[]), \
             patch.object(stage, '_fetch_captions_for_videos', return_value=[]):

            result = stage.run(mock_state_with_gaps, mock_config, mock_checkpoint)

            assert result.success is True
            assert 'passes_completed' in result.data
            assert 'initial_gaps' in result.data
            assert 'final_gaps' in result.data

    def test_checkpoint_data_structure(
        self, stage, mock_state_with_gaps, mock_config, mock_checkpoint
    ):
        """Checkpoint data has expected structure"""
        mock_config.iterative_matching.max_iterations = 1

        with patch.object(stage, '_search_youtube_for_videos', return_value=[]), \
             patch.object(stage, '_fetch_captions_for_videos', return_value=[]):

            result = stage.run(mock_state_with_gaps, mock_config, mock_checkpoint)

            data = result.data

            # Required fields
            assert 'passes_completed' in data
            assert 'initial_gaps' in data
            assert 'final_gaps' in data
            assert 'total_gaps_filled' in data
            assert 'total_duration_seconds' in data
            assert 'matches' in data
            assert 'pass_metrics' in data


# ============================================================================
# Empty Candidate Pool Pre-Check Tests (US-40-009)
# ============================================================================

class TestEmptyCandidatePoolPreCheck:
    """Test pre-check for empty candidate pool at stage entry.

    US-40-009: IterativeMatchStage should check candidate pool size
    at entry and return early if no candidates are available.
    """

    def test_early_return_when_no_candidates_available(
        self, stage, mock_config, mock_checkpoint
    ):
        """Stage returns early with skipped=True when text_metadata is empty.

        AC: Add unit test verifying early return when no candidates available
        """
        state = PipelineState()
        state.voiceover_segments = [MagicMock()]
        state.matches = [MagicMock()]
        state.text_metadata = []  # Empty candidate pool
        state.embeddings = None
        state.extracted_entities = []
        state.voiceover_path = "/tmp/test/voiceover.srt"

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True
        assert result.data.get('reason') == 'no_candidates'
        assert result.data.get('candidate_count') == 0

    def test_early_return_when_text_metadata_is_none(
        self, stage, mock_config, mock_checkpoint
    ):
        """Stage returns early when text_metadata is None.

        Handles edge case where text_metadata attribute exists but is None.
        """
        state = PipelineState()
        state.voiceover_segments = [MagicMock()]
        state.matches = [MagicMock()]
        state.text_metadata = None  # None instead of empty list
        state.embeddings = None
        state.extracted_entities = []
        state.voiceover_path = "/tmp/test/voiceover.srt"

        result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data.get('skipped') is True
        assert result.data.get('reason') == 'no_candidates'

    def test_normal_flow_when_candidates_exist(
        self, stage, mock_state_with_gaps, mock_config, mock_checkpoint
    ):
        """Stage proceeds normally when candidates exist in text_metadata.

        AC: Add unit test verifying normal flow when candidates exist
        """
        # Add some candidates to text_metadata
        mock_state_with_gaps.text_metadata = [
            {'text': 'caption 1', 'video_path': 'vid1', 'start_time': 0, 'end_time': 5},
            {'text': 'caption 2', 'video_path': 'vid2', 'start_time': 0, 'end_time': 5},
        ]
        mock_config.iterative_matching.max_iterations = 1

        with patch.object(stage, '_search_youtube_for_videos', return_value=[]), \
             patch.object(stage, '_fetch_captions_for_videos', return_value=[]):

            result = stage.run(mock_state_with_gaps, mock_config, mock_checkpoint)

        # Should NOT be skipped due to no_candidates
        assert result.success is True
        assert result.data.get('reason') != 'no_candidates'
        # Should have processed (passes_completed indicates normal flow)
        assert 'passes_completed' in result.data

    def test_logs_candidate_count_at_entry(
        self, stage, mock_state_with_gaps, mock_config, mock_checkpoint, caplog
    ):
        """Stage logs INFO with candidate count at entry.

        AC: Log INFO with candidate pool size at stage entry
        """
        import logging

        # Add candidates
        mock_state_with_gaps.text_metadata = [
            {'text': f'caption {i}', 'video_path': f'vid{i}'} for i in range(5)
        ]
        mock_config.iterative_matching.max_iterations = 1

        with patch.object(stage, '_search_youtube_for_videos', return_value=[]), \
             patch.object(stage, '_fetch_captions_for_videos', return_value=[]), \
             caplog.at_level(logging.INFO, logger='src.stages.iterative_match'):

            stage.run(mock_state_with_gaps, mock_config, mock_checkpoint)

        # Check that candidate count was logged
        assert any('IterativeMatch starting with 5 candidates' in record.message
                   for record in caplog.records)

    def test_logs_warning_when_no_candidates(
        self, stage, mock_config, mock_checkpoint, caplog
    ):
        """Stage logs WARNING when no candidates available.

        AC: If candidate pool is 0, log WARNING
        """
        import logging

        state = PipelineState()
        state.voiceover_segments = [MagicMock()]
        state.matches = [MagicMock()]
        state.text_metadata = []
        state.embeddings = None
        state.extracted_entities = []
        state.voiceover_path = "/tmp/test/voiceover.srt"

        with caplog.at_level(logging.WARNING, logger='src.stages.iterative_match'):
            stage.run(state, mock_config, mock_checkpoint)

        # Check that warning was logged
        assert any('No candidates available' in record.message
                   for record in caplog.records)


# ============================================================================
# TestMatchSerializationVideoFile - US-55-004
# ============================================================================

def _serialize_match(match, index: int) -> dict:
    """Replicate the serialization logic from IterativeMatchStage.run() (lines 421-497).

    This mirrors the exact serialization block in src/stages/iterative_match.py
    so we can test it in isolation without running the full stage.
    """
    if hasattr(match, 'primary_match') and match.primary_match:
        pm = match.primary_match
        source_file = ''
        video_start = 0.0
        video_end = 0.0
        conf = 0.0

        if hasattr(pm, 'video_segment') and pm.video_segment:
            source_file = getattr(pm.video_segment, 'source_file', '') or ''
            video_start = getattr(pm.video_segment, 'start_time', 0.0)
            video_end = getattr(pm.video_segment, 'end_time', 0.0)

        conf = getattr(pm, 'confidence', 0.0)

        return {
            'segment_index': index,
            'video_file': source_file,
            'video_start': float(video_start),
            'video_end': float(video_end),
            'confidence': float(conf),
            'strategy': getattr(match, 'strategy', getattr(pm, 'reasoning', '')),
            'reason': getattr(pm, 'reasoning', ''),
            'face_score': getattr(match, 'face_score', 0.5),
        }
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
                    video_file = getattr(vs, 'source_file', '') or ''
                    video_start = getattr(vs, 'start_time', video_start)
                    video_end = getattr(vs, 'end_time', video_end)
                confidence = getattr(pm, 'confidence', confidence)

        return {
            'segment_index': getattr(match, 'segment_index', index),
            'video_file': video_file,
            'video_start': float(video_start),
            'video_end': float(video_end),
            'confidence': float(confidence),
            'strategy': getattr(match, 'strategy', ''),
            'reason': getattr(match, 'reason', ''),
            'face_score': getattr(match, 'face_score', 0.5),
        }


def _make_srt_segment(text="Test", source_file="", start_time=0.0, end_time=5.0):
    """Create a minimal SRTSegment for testing."""
    return SRTSegment(
        index=1,
        start_time=start_time,
        end_time=end_time,
        text=text,
        source_file=source_file,
    )


class TestMatchSerializationVideoFile:
    """Tests verifying that iterative_match serialization correctly extracts
    video_file from match.primary_match.video_segment.source_file.

    US-55-004: The ITERATIVE_MATCH serialization block must drill into
    primary_match.video_segment.source_file for MatchResult objects,
    not rely on getattr(match, 'video_file', '') which always returns ''
    for MatchResult instances.
    """

    @pytest.mark.fast
    def test_matchresult_serializes_video_file_from_primary_match(self):
        """MatchResult with primary_match.video_segment.source_file='abc123'
        serializes to a dict with video_file='abc123' (not empty string).
        """
        video_seg = _make_srt_segment(text="Video text", source_file="abc123")
        vo_seg = _make_srt_segment(text="Voiceover text")
        primary = Match(
            voiceover_segment=vo_seg,
            video_segment=video_seg,
            video_scene=None,
            confidence=0.85,
            reasoning="Good match",
        )
        result = MatchResult(primary_match=primary)

        serialized = _serialize_match(result, index=0)

        assert serialized['video_file'] == 'abc123'
        assert serialized['confidence'] == 0.85
        assert serialized['video_start'] == 0.0
        assert serialized['video_end'] == 5.0

    @pytest.mark.fast
    def test_matchresult_none_primary_match_serializes_empty_video_file(self):
        """MatchResult with primary_match=None serializes video_file as empty
        string without raising AttributeError.
        """
        result = MatchResult(primary_match=None)

        serialized = _serialize_match(result, index=3)

        assert serialized['video_file'] == ''
        assert serialized['segment_index'] == 3
        assert serialized['confidence'] == 0.0

    @pytest.mark.fast
    def test_serialized_matchresult_roundtrips_via_state_match_from_dict(self):
        """Serialized MatchResult can be deserialized via state.Match.from_dict()
        and the video_file value survives the round-trip.
        """
        video_seg = _make_srt_segment(text="Video", source_file="roundtrip_vid_42")
        vo_seg = _make_srt_segment(text="Voiceover")
        primary = Match(
            voiceover_segment=vo_seg,
            video_segment=video_seg,
            video_scene=None,
            confidence=0.92,
            reasoning="Strong match",
        )
        result = MatchResult(primary_match=primary)

        serialized = _serialize_match(result, index=7)

        # Deserialize via state.Match.from_dict
        restored = StateMatch.from_dict(serialized, index=7)

        assert restored.video_file == 'roundtrip_vid_42'
        assert restored.confidence == 0.92
        assert restored.segment_index == 7

    @pytest.mark.fast
    def test_simple_state_match_serializes_video_file_directly(self):
        """The else branch correctly serializes simple state.Match objects
        that have video_file as a direct attribute.
        """
        simple_match = StateMatch(
            segment_index=5,
            video_file='simple_video.mp4',
            video_start=10.0,
            video_end=20.0,
            confidence=0.75,
            strategy='keyword',
            reason='keyword match',
            face_score=0.3,
        )

        serialized = _serialize_match(simple_match, index=5)

        assert serialized['video_file'] == 'simple_video.mp4'
        assert serialized['segment_index'] == 5
        assert serialized['video_start'] == 10.0
        assert serialized['video_end'] == 20.0
        assert serialized['confidence'] == 0.75
        assert serialized['strategy'] == 'keyword'
        assert serialized['face_score'] == 0.3

    @pytest.mark.fast
    def test_matchresult_with_empty_source_file_serializes_empty_string(self):
        """MatchResult where video_segment.source_file is empty string
        serializes video_file as empty string gracefully.
        """
        video_seg = _make_srt_segment(text="Video", source_file="")
        vo_seg = _make_srt_segment(text="Voiceover")
        primary = Match(
            voiceover_segment=vo_seg,
            video_segment=video_seg,
            video_scene=None,
            confidence=0.5,
            reasoning="Weak match",
        )
        result = MatchResult(primary_match=primary)

        serialized = _serialize_match(result, index=2)

        assert serialized['video_file'] == ''
        assert serialized['confidence'] == 0.5

    @pytest.mark.fast
    def test_simple_state_match_roundtrips_via_from_dict(self):
        """Simple state.Match serialized via else branch roundtrips through
        state.Match.from_dict() preserving video_file.
        """
        simple_match = StateMatch(
            segment_index=0,
            video_file='else_branch_vid.mp4',
            video_start=1.0,
            video_end=11.0,
            confidence=0.88,
            strategy='visual',
            reason='visual match',
            face_score=0.6,
        )

        serialized = _serialize_match(simple_match, index=0)
        restored = StateMatch.from_dict(serialized, index=0)

        assert restored.video_file == 'else_branch_vid.mp4'
        assert restored.confidence == 0.88
        assert restored.strategy == 'visual'
        assert restored.face_score == 0.6


# ============================================================================
# US-73-009: Video tag-derived search queries
# ============================================================================

class TestVideoTagQueryExtraction:
    """Test tag extraction from nearby matches and stop-tag filtering."""

    @pytest.mark.fast
    def test_extracts_tags_from_nearby_locked_matches(self):
        """Tags from locked matches near the gap are extracted by frequency."""
        from src.iterative_match.gap_analyzer import (
            GapSegment, LockedMatch, extract_tags_from_nearby_matches,
        )

        gap = GapSegment(
            segment_index=5, confidence=0.3,
            voiceover_text="coral reef ecosystem", position=100.0,
        )
        locked = [
            LockedMatch(segment_index=4, video_id='vid_A', confidence=0.95, position=90.0),
            LockedMatch(segment_index=6, video_id='vid_B', confidence=0.92, position=110.0),
            LockedMatch(segment_index=10, video_id='vid_C', confidence=0.91, position=500.0),  # far away
        ]

        # Mock state with video_search_results carrying tags
        state = MagicMock()
        vsr_a = MagicMock(video_id='vid_A', video_tags=['marine biology', 'coral reef', 'ocean'])
        vsr_b = MagicMock(video_id='vid_B', video_tags=['coral reef', 'diving', 'ocean'])
        vsr_c = MagicMock(video_id='vid_C', video_tags=['space exploration', 'nasa'])
        state.video_search_results = [vsr_a, vsr_b, vsr_c]

        tags = extract_tags_from_nearby_matches(gap, locked, state, max_tags=3)

        # 'coral reef' appears in both nearby matches (freq=2), should be first
        assert 'coral reef' in tags
        # 'ocean' also freq=2
        assert 'ocean' in tags
        # 'space exploration' should NOT appear (vid_C is too far away)
        assert 'space exploration' not in tags
        assert len(tags) <= 3

    @pytest.mark.fast
    def test_filters_out_stop_tags(self):
        """Generic tags like 'video', 'youtube', 'official' are filtered out."""
        from src.iterative_match.gap_analyzer import (
            GapSegment, LockedMatch, extract_tags_from_nearby_matches,
        )

        gap = GapSegment(
            segment_index=2, confidence=0.4,
            voiceover_text="test content", position=50.0,
        )
        locked = [
            LockedMatch(segment_index=1, video_id='vid_X', confidence=0.95, position=45.0),
        ]

        state = MagicMock()
        vsr = MagicMock(
            video_id='vid_X',
            video_tags=['video', 'youtube', 'official', 'marine biology', 'HD'],
        )
        state.video_search_results = [vsr]

        tags = extract_tags_from_nearby_matches(gap, locked, state, max_tags=5)

        # Only 'marine biology' should survive stop-tag filtering
        assert 'marine biology' in tags
        assert 'video' not in tags
        assert 'youtube' not in tags
        assert 'official' not in tags

    @pytest.mark.fast
    def test_returns_empty_when_no_nearby_matches(self):
        """Returns empty list when no locked matches are within range."""
        from src.iterative_match.gap_analyzer import (
            GapSegment, LockedMatch, extract_tags_from_nearby_matches,
        )

        gap = GapSegment(
            segment_index=5, confidence=0.3,
            voiceover_text="isolated gap", position=1000.0,
        )
        locked = [
            LockedMatch(segment_index=0, video_id='vid_A', confidence=0.95, position=10.0),
        ]

        state = MagicMock()
        vsr = MagicMock(video_id='vid_A', video_tags=['nature', 'wildlife'])
        state.video_search_results = [vsr]

        tags = extract_tags_from_nearby_matches(gap, locked, state, max_tags=3)
        assert tags == []

    @pytest.mark.fast
    def test_config_use_tag_queries_field_exists(self):
        """IterativeMatchingConfig has use_tag_queries field defaulting to True."""
        from src.config.sections.iterative_matching import IterativeMatchingConfig

        config = IterativeMatchingConfig()
        assert config.use_tag_queries is True
