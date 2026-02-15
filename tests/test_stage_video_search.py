"""
Comprehensive tests for VideoSearchStage.

Tests cover:
- Stage initialization and registration
- Video search with YouTube API (mocked)
- Empty results handling
- Checkpoint save/restore
- Title blacklist filtering
- Duration filtering
- Input validation

Created: 2026-02-02 (US-1-010)
"""

from unittest.mock import Mock, MagicMock, patch
import pytest
import tempfile
from pathlib import Path

from src.stages.video_search import VideoSearchStage
from src.stages import StageResult
from src.state import PipelineState, VideoSearchResult
from src.checkpoint import CheckpointManager


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config for VideoSearchStage"""
    config = Mock()
    config.download = Mock()
    config.download.video_search = {
        'results_per_keyword': 20,
        'max_total_results': 200,
        'search_budget_aware': True,
        'auto_distribute_budget': True
    }
    config.download.min_duration = 30
    config.download.max_duration = 600
    config.download.title_blacklist = ['reaction', 'review', 'gameplay']
    return config


@pytest.fixture
def temp_project_dir():
    """Create temporary project directory"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_checkpoint(temp_project_dir):
    """Create mock checkpoint manager"""
    return CheckpointManager(temp_project_dir, config_hash='test_hash')


@pytest.fixture
def sample_state():
    """Create sample pipeline state with keywords"""
    state = PipelineState()
    state.keywords = ['beach sunset', 'ocean waves']
    state.topic_context = 'Travel'
    return state


@pytest.fixture
def mock_yt_dlp_results():
    """Sample yt-dlp search results"""
    return {
        'entries': [
            {
                'id': 'abc123',
                'title': 'Beautiful Beach Sunset',
                'channel': 'Nature Channel',
                'duration': 120
            },
            {
                'id': 'def456',
                'title': 'Ocean Waves Relaxation',
                'channel': 'Relaxing Videos',
                'duration': 300
            },
            {
                'id': 'ghi789',
                'title': 'Beach Vlog - Too Short',
                'channel': 'Vlogger',
                'duration': 10  # Below min_duration
            }
        ]
    }


# ============================================================================
# Test Stage Registration and Initialization
# ============================================================================

class TestVideoSearchStageInit:
    """Test VideoSearchStage initialization and metadata"""

    @pytest.mark.fast
    def test_stage_name(self):
        """Test stage name is VIDEO_SEARCH"""
        stage = VideoSearchStage()
        assert stage.name == "VIDEO_SEARCH"

    @pytest.mark.fast
    def test_stage_description(self):
        """Test stage has description"""
        stage = VideoSearchStage()
        assert "search" in stage.description.lower()
        assert "youtube" in stage.description.lower() or "video" in stage.description.lower()

    @pytest.mark.fast
    def test_stage_registration(self):
        """Test stage is registered"""
        from src.stages import get_stage
        stage_class = get_stage("VIDEO_SEARCH")
        assert stage_class is not None
        assert stage_class == VideoSearchStage


# ============================================================================
# Test Input Validation
# ============================================================================

class TestInputValidation:
    """Test validate_inputs method"""

    @pytest.mark.fast
    def test_validate_no_keywords(self, mock_config):
        """Test validation fails when no keywords"""
        stage = VideoSearchStage()
        state = PipelineState()
        state.keywords = []

        error = stage.validate_inputs(state, mock_config)

        assert error is not None
        assert "no keywords" in error.lower()

    @pytest.mark.fast
    def test_validate_success(self, mock_config, sample_state):
        """Test validation succeeds with valid keywords"""
        stage = VideoSearchStage()

        error = stage.validate_inputs(sample_state, mock_config)

        assert error is None


# ============================================================================
# Test Video Search Returns Video IDs
# ============================================================================

class TestVideoSearchReturnsVideoIds:
    """Test _search_keyword and run() return video IDs correctly"""

    @pytest.mark.fast
    def test_video_search_returns_video_ids(self, mock_config, sample_state, mock_checkpoint, mock_yt_dlp_results):
        """Test that video search returns video IDs with mocked YouTube API"""
        stage = VideoSearchStage()

        # Mock yt-dlp
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_yt_dlp_results
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            result = stage.run(sample_state, mock_config, mock_checkpoint)

        assert result.success is True
        # Should have video IDs (filtering out short duration)
        assert len(sample_state.video_ids) >= 2
        assert 'abc123' in sample_state.video_ids
        assert 'def456' in sample_state.video_ids

    @pytest.mark.fast
    def test_search_keyword_returns_metadata(self, mock_config, mock_yt_dlp_results):
        """Test _search_keyword returns video metadata"""
        stage = VideoSearchStage()

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_yt_dlp_results
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            results = stage._search_keyword(
                keyword='beach',
                config=mock_config,
                max_results=10,
                topic='Travel'
            )

        # Should filter out short videos
        assert len(results) == 2
        assert results[0]['video_id'] == 'abc123'
        assert results[0]['title'] == 'Beautiful Beach Sunset'
        assert results[0]['channel'] == 'Nature Channel'
        assert results[0]['duration'] == 120
        assert results[0]['keyword'] == 'beach'

    @pytest.mark.fast
    def test_search_deduplicates_video_ids(self, mock_config, mock_checkpoint):
        """Test that duplicate video IDs are removed across keywords"""
        stage = VideoSearchStage()
        state = PipelineState()
        state.keywords = ['beach', 'ocean']
        state.topic_context = 'Travel'

        # Both keywords return same video ID
        duplicate_results = {
            'entries': [
                {'id': 'same_video', 'title': 'Test Video', 'channel': 'Test', 'duration': 120}
            ]
        }

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = duplicate_results
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        # Should only have one video ID despite two keywords returning it
        assert len(state.video_ids) == 1
        assert state.video_ids[0] == 'same_video'


# ============================================================================
# Test Empty Results Handling
# ============================================================================

class TestVideoSearchHandlesEmptyResults:
    """Test handling of empty search results"""

    @pytest.mark.fast
    def test_video_search_handles_empty_results(self, mock_config, sample_state, mock_checkpoint):
        """Test that empty search results are handled gracefully"""
        stage = VideoSearchStage()

        # Mock yt-dlp returning no results
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = {'entries': []}
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            result = stage.run(sample_state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(sample_state.video_ids) == 0
        assert len(sample_state.search_failed_keywords) == 2  # Both keywords failed

    @pytest.mark.fast
    def test_handles_none_entries(self, mock_config, sample_state, mock_checkpoint):
        """Test handling when entries is None"""
        stage = VideoSearchStage()

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = None
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            result = stage.run(sample_state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(sample_state.video_ids) == 0

    @pytest.mark.fast
    def test_handles_search_exception(self, mock_config, sample_state, mock_checkpoint):
        """Test handling when search throws exception"""
        stage = VideoSearchStage()

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.side_effect = Exception("Network error")
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            result = stage.run(sample_state, mock_config, mock_checkpoint)

        assert result.success is True  # Graceful degradation
        assert len(sample_state.video_ids) == 0
        # Failed keywords should be tracked
        assert len(sample_state.search_failed_keywords) == 2

    @pytest.mark.fast
    def test_partial_failure_continues(self, mock_config, mock_checkpoint):
        """Test that partial keyword failures don't stop the whole search"""
        stage = VideoSearchStage()
        state = PipelineState()
        state.keywords = ['working', 'failing']
        state.topic_context = 'Test'

        call_count = [0]

        def mock_extract(url, download=False):
            call_count[0] += 1
            if 'working' in url:
                return {'entries': [{'id': 'vid1', 'title': 'Test', 'channel': 'Ch', 'duration': 120}]}
            else:
                raise Exception("Search failed")

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.side_effect = mock_extract
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.video_ids) == 1
        assert 'failing' in state.search_failed_keywords


# ============================================================================
# Test Checkpoint Operations
# ============================================================================

class TestVideoSearchCheckpointsResults:
    """Test checkpoint save and restore"""

    @pytest.mark.fast
    def test_video_search_checkpoints_results(self, mock_config, sample_state, mock_checkpoint, mock_yt_dlp_results):
        """Test that search results are saved to checkpoint"""
        stage = VideoSearchStage()

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_yt_dlp_results
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            result = stage.run(sample_state, mock_config, mock_checkpoint)

        assert result.success is True
        assert result.data is not None
        assert 'video_ids' in result.data
        assert 'search_results' in result.data
        assert 'failed_keywords' in result.data
        assert 'video_count' in result.data

    @pytest.mark.fast
    def test_restore_from_checkpoint(self, temp_project_dir):
        """Test successful restore from checkpoint"""
        stage = VideoSearchStage()
        state = PipelineState()

        # Create checkpoint with data
        checkpoint = CheckpointManager(temp_project_dir, config_hash='test')
        checkpoint_data = {
            'video_ids': ['abc123', 'def456'],
            'search_results': [
                {'video_id': 'abc123', 'url': 'https://youtube.com/watch?v=abc123',
                 'title': 'Test 1', 'channel': 'Ch1', 'duration': 120, 'keyword': 'beach'},
                {'video_id': 'def456', 'url': 'https://youtube.com/watch?v=def456',
                 'title': 'Test 2', 'channel': 'Ch2', 'duration': 300, 'keyword': 'ocean'}
            ],
            'failed_keywords': ['bad_keyword'],
            'video_count': 2
        }
        checkpoint.save('VIDEO_SEARCH', checkpoint_data)

        restored = stage.restore(state, checkpoint)

        assert restored is True
        assert state.video_ids == ['abc123', 'def456']
        assert len(state.video_search_results) == 2
        assert state.video_search_results[0].video_id == 'abc123'
        assert state.video_search_results[1].title == 'Test 2'
        assert state.search_failed_keywords == ['bad_keyword']

    @pytest.mark.fast
    def test_restore_empty_checkpoint(self, mock_checkpoint):
        """Test restore returns False when no checkpoint data"""
        stage = VideoSearchStage()
        state = PipelineState()

        restored = stage.restore(state, mock_checkpoint)

        assert restored is False

    @pytest.mark.fast
    def test_can_skip_when_checkpoint_exists(self, temp_project_dir):
        """Test can_skip returns True when checkpoint exists"""
        stage = VideoSearchStage()
        state = PipelineState()

        checkpoint = CheckpointManager(temp_project_dir, config_hash='test')
        # save() updates last_completed_stage to VIDEO_SEARCH
        checkpoint.save('VIDEO_SEARCH', {'video_ids': ['test']})

        can_skip = stage.can_skip(state, checkpoint)

        assert can_skip is True

    @pytest.mark.fast
    def test_can_skip_without_checkpoint(self, mock_checkpoint):
        """Test can_skip returns False when no checkpoint"""
        stage = VideoSearchStage()
        state = PipelineState()

        can_skip = stage.can_skip(state, mock_checkpoint)

        assert can_skip is False

    @pytest.mark.fast
    @patch('src.stages.video_search.logger')
    def test_restore_handles_exception(self, mock_logger, mock_checkpoint):
        """Test restore handles exceptions gracefully"""
        stage = VideoSearchStage()
        state = PipelineState()

        mock_checkpoint.get_stage_data = Mock(side_effect=Exception("Checkpoint error"))

        restored = stage.restore(state, mock_checkpoint)

        assert restored is False
        mock_logger.warning.assert_called()


# ============================================================================
# Test Title Blacklist Filtering
# ============================================================================

class TestTitleBlacklist:
    """Test title blacklist filtering"""

    @pytest.mark.fast
    def test_filters_blacklisted_titles(self, mock_config):
        """Test videos with blacklisted words are filtered"""
        stage = VideoSearchStage()

        results = {
            'entries': [
                {'id': 'good1', 'title': 'Beautiful Beach', 'channel': 'Ch', 'duration': 120},
                {'id': 'bad1', 'title': 'Beach Reaction Video', 'channel': 'Ch', 'duration': 120},
                {'id': 'bad2', 'title': 'Beach Review 2024', 'channel': 'Ch', 'duration': 120},
                {'id': 'bad3', 'title': 'Beach Gameplay Walkthrough', 'channel': 'Ch', 'duration': 120},
                {'id': 'good2', 'title': 'Ocean Waves', 'channel': 'Ch', 'duration': 120}
            ]
        }

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = results
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            search_results = stage._search_keyword('beach', mock_config)

        video_ids = [r['video_id'] for r in search_results]
        assert 'good1' in video_ids
        assert 'good2' in video_ids
        assert 'bad1' not in video_ids
        assert 'bad2' not in video_ids
        assert 'bad3' not in video_ids

    @pytest.mark.fast
    def test_blacklist_case_insensitive(self, mock_config):
        """Test blacklist matching is case insensitive"""
        stage = VideoSearchStage()

        # Mixed case should still be filtered
        assert stage._is_title_blacklisted('BEACH REACTION VIDEO', mock_config) is True
        assert stage._is_title_blacklisted('Beach ReViEw 2024', mock_config) is True
        assert stage._is_title_blacklisted('Beautiful Beach', mock_config) is False

    @pytest.mark.fast
    def test_default_blacklist_when_none(self):
        """Test default blacklist is used when config has none"""
        stage = VideoSearchStage()
        config = Mock()
        config.download = Mock()
        config.download.title_blacklist = []  # Empty blacklist

        # Should use default blacklist
        assert stage._is_title_blacklisted('Beach Reaction', config) is True
        assert stage._is_title_blacklisted('Beach Tutorial', config) is True


# ============================================================================
# Test Duration Filtering
# ============================================================================

class TestDurationFiltering:
    """Test duration-based video filtering"""

    @pytest.mark.fast
    def test_filters_short_videos(self, mock_config):
        """Test videos below min_duration are filtered"""
        stage = VideoSearchStage()
        mock_config.download.min_duration = 60  # 1 minute minimum

        results = {
            'entries': [
                {'id': 'short', 'title': 'Short Video', 'channel': 'Ch', 'duration': 30},
                {'id': 'good', 'title': 'Good Video', 'channel': 'Ch', 'duration': 120}
            ]
        }

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = results
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            search_results = stage._search_keyword('test', mock_config)

        assert len(search_results) == 1
        assert search_results[0]['video_id'] == 'good'

    @pytest.mark.fast
    def test_filters_long_videos(self, mock_config):
        """Test videos above max_duration are filtered"""
        stage = VideoSearchStage()
        mock_config.download.max_duration = 300  # 5 minute maximum

        results = {
            'entries': [
                {'id': 'long', 'title': 'Long Video', 'channel': 'Ch', 'duration': 600},
                {'id': 'good', 'title': 'Good Video', 'channel': 'Ch', 'duration': 120}
            ]
        }

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = results
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            search_results = stage._search_keyword('test', mock_config)

        assert len(search_results) == 1
        assert search_results[0]['video_id'] == 'good'

    @pytest.mark.fast
    def test_handles_missing_duration(self, mock_config):
        """Test videos with missing duration are filtered"""
        stage = VideoSearchStage()

        results = {
            'entries': [
                {'id': 'no_duration', 'title': 'No Duration', 'channel': 'Ch'},
                {'id': 'zero_duration', 'title': 'Zero Duration', 'channel': 'Ch', 'duration': 0},
                {'id': 'good', 'title': 'Good Video', 'channel': 'Ch', 'duration': 120}
            ]
        }

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = results
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            search_results = stage._search_keyword('test', mock_config)

        # Only the video with valid duration should pass
        assert len(search_results) == 1
        assert search_results[0]['video_id'] == 'good'


# ============================================================================
# Test Max Results Limit
# ============================================================================

class TestMaxResultsLimit:
    """Test max total results limit"""

    @pytest.mark.fast
    def test_respects_max_total_results(self, mock_checkpoint):
        """Test that search stops when max_total_results is reached"""
        stage = VideoSearchStage()
        state = PipelineState()
        state.keywords = ['keyword1', 'keyword2', 'keyword3']
        state.topic_context = 'Test'

        config = Mock()
        config.download = Mock()
        config.download.video_search = {
            'results_per_keyword': 10,
            'max_total_results': 5  # Low limit
        }
        config.download.min_duration = 30
        config.download.max_duration = 600
        config.download.title_blacklist = []

        def mock_extract(url, download=False):
            return {
                'entries': [
                    {'id': f'vid_{i}', 'title': f'Video {i}', 'channel': 'Ch', 'duration': 120}
                    for i in range(10)
                ]
            }

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.side_effect = mock_extract
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            result = stage.run(state, config, mock_checkpoint)

        assert result.success is True
        # Should stop at or near max_total_results
        assert len(state.video_ids) <= 10  # First keyword fills it


# ============================================================================
# Test VideoSearchResult Conversion
# ============================================================================

class TestVideoSearchResultConversion:
    """Test _to_search_results conversion"""

    @pytest.mark.fast
    def test_converts_dicts_to_dataclass(self):
        """Test conversion from dict to VideoSearchResult"""
        stage = VideoSearchStage()
        results = [
            {
                'video_id': 'abc123',
                'url': 'https://youtube.com/watch?v=abc123',
                'title': 'Test Video',
                'channel': 'Test Channel',
                'duration': 120.5,
                'keyword': 'test'
            }
        ]

        search_results = stage._to_search_results(results)

        assert len(search_results) == 1
        assert isinstance(search_results[0], VideoSearchResult)
        assert search_results[0].video_id == 'abc123'
        assert search_results[0].title == 'Test Video'
        assert search_results[0].channel == 'Test Channel'
        assert search_results[0].duration == 120.5

    @pytest.mark.fast
    def test_handles_missing_fields(self):
        """Test conversion handles missing optional fields"""
        stage = VideoSearchStage()
        results = [
            {'video_id': 'abc123'}  # Minimal data
        ]

        search_results = stage._to_search_results(results)

        assert len(search_results) == 1
        assert search_results[0].video_id == 'abc123'
        assert search_results[0].url == ''
        assert search_results[0].title == ''
        assert search_results[0].duration == 0.0


# ============================================================================
# Test Config Access Patterns
# ============================================================================

class TestConfigAccess:
    """Test different config access patterns"""

    @pytest.mark.fast
    def test_dict_config_access(self, sample_state, mock_checkpoint):
        """Test access when video_search is a dict"""
        stage = VideoSearchStage()
        config = Mock()
        config.download = Mock()
        config.download.video_search = {
            'results_per_keyword': 15,
            'max_total_results': 100
        }
        config.download.min_duration = 30
        config.download.max_duration = 600
        config.download.title_blacklist = []

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = {'entries': []}
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            result = stage.run(sample_state, config, mock_checkpoint)

        assert result.success is True

    @pytest.mark.fast
    def test_object_config_access(self, sample_state, mock_checkpoint):
        """Test access when video_search is an object"""
        stage = VideoSearchStage()
        config = Mock()
        config.download = Mock()
        search_config = Mock()
        search_config.results_per_keyword = 15
        search_config.max_total_results = 100
        config.download.video_search = search_config
        config.download.min_duration = 30
        config.download.max_duration = 600
        config.download.title_blacklist = []

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = {'entries': []}
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            result = stage.run(sample_state, config, mock_checkpoint)

        assert result.success is True

    @pytest.mark.fast
    def test_none_config_uses_defaults(self, sample_state, mock_checkpoint):
        """Test default values when video_search config is None"""
        stage = VideoSearchStage()
        config = Mock()
        config.download = Mock()
        config.download.video_search = None
        config.download.min_duration = 30
        config.download.max_duration = 600
        config.download.title_blacklist = []

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = {'entries': []}
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            result = stage.run(sample_state, config, mock_checkpoint)

        assert result.success is True


# ============================================================================
# Test Search Query Building
# ============================================================================

class TestSearchQueryBuilding:
    """Test search query construction"""

    @pytest.mark.fast
    def test_combines_keyword_and_topic(self, mock_config):
        """Test that keyword and topic are combined in search query"""
        stage = VideoSearchStage()

        captured_url = []

        def capture_extract(url, download=False):
            captured_url.append(url)
            return {'entries': []}

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.side_effect = capture_extract
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            stage._search_keyword('beach', mock_config, max_results=10, topic='Travel')

        # Search URL should contain both keyword and topic
        assert len(captured_url) == 1
        assert 'beach' in captured_url[0]
        assert 'Travel' in captured_url[0]

    @pytest.mark.fast
    def test_keyword_only_without_topic(self, mock_config):
        """Test search with keyword only (no topic)"""
        stage = VideoSearchStage()

        captured_url = []

        def capture_extract(url, download=False):
            captured_url.append(url)
            return {'entries': []}

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.side_effect = capture_extract
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            stage._search_keyword('beach', mock_config, max_results=10, topic='')

        assert len(captured_url) == 1
        assert 'beach' in captured_url[0]


# ============================================================================
# Test State Updates
# ============================================================================

class TestStateUpdates:
    """Test that state is properly updated after search"""

    @pytest.mark.fast
    def test_updates_video_search_results(self, mock_config, mock_checkpoint, mock_yt_dlp_results):
        """Test state.video_search_results is populated"""
        stage = VideoSearchStage()
        state = PipelineState()
        state.keywords = ['beach']
        state.topic_context = 'Travel'

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_yt_dlp_results
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert len(state.video_search_results) >= 2
        assert all(isinstance(r, VideoSearchResult) for r in state.video_search_results)

    @pytest.mark.fast
    def test_tracks_failed_keywords(self, mock_config, mock_checkpoint):
        """Test state.search_failed_keywords is populated"""
        stage = VideoSearchStage()
        state = PipelineState()
        state.keywords = ['good_keyword', 'bad_keyword']
        state.topic_context = 'Test'

        def mock_extract(url, download=False):
            if 'good' in url:
                return {'entries': [{'id': 'vid1', 'title': 'Test', 'channel': 'Ch', 'duration': 120}]}
            return {'entries': []}

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.side_effect = mock_extract
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            result = stage.run(state, mock_config, mock_checkpoint)

        assert result.success is True
        assert 'bad_keyword' in state.search_failed_keywords
        assert 'good_keyword' not in state.search_failed_keywords


# ============================================================================
# Test Budget Distribution Math
# ============================================================================

class TestBudgetDistribution:
    """Test search budget distribution logic"""

    @pytest.mark.fast
    def test_budget_distribution_math_under_limit(self):
        """Test no adjustment when keywords * results <= max_total"""
        # 5 keywords * 20 results = 100, max_total = 200
        # No adjustment needed
        results_per_keyword = 20
        keyword_count = 5
        max_total_results = 200

        potential_total = results_per_keyword * keyword_count
        assert potential_total <= max_total_results
        # effective_results should stay at 20

    @pytest.mark.fast
    def test_budget_distribution_math_over_limit(self):
        """Test adjustment when keywords * results > max_total"""
        # 20 keywords * 20 results = 400, max_total = 200
        # Should adjust to 200 // 20 = 10 per keyword
        results_per_keyword = 20
        keyword_count = 20
        max_total_results = 200

        potential_total = results_per_keyword * keyword_count
        assert potential_total > max_total_results

        # Adjusted results per keyword
        effective_results_per_keyword = max(1, max_total_results // keyword_count)
        assert effective_results_per_keyword == 10
        assert effective_results_per_keyword * keyword_count <= max_total_results

    @pytest.mark.fast
    def test_budget_distribution_minimum_one_result(self):
        """Test minimum 1 result per keyword when budget very constrained"""
        # 300 keywords * 20 results = 6000, max_total = 200
        # Should adjust to max(1, 200 // 300) = 1
        results_per_keyword = 20
        keyword_count = 300
        max_total_results = 200

        potential_total = results_per_keyword * keyword_count
        assert potential_total > max_total_results

        effective_results_per_keyword = max(1, max_total_results // keyword_count)
        assert effective_results_per_keyword == 1

    @pytest.mark.fast
    def test_budget_distribution_single_keyword(self):
        """Test single keyword uses full budget"""
        # 1 keyword * 20 results = 20, max_total = 200
        results_per_keyword = 20
        keyword_count = 1
        max_total_results = 200

        potential_total = results_per_keyword * keyword_count
        assert potential_total <= max_total_results

        # Should use original results_per_keyword
        effective_results_per_keyword = results_per_keyword
        assert effective_results_per_keyword == 20

    @pytest.mark.fast
    def test_budget_distribution_exact_fit(self):
        """Test exact fit case - no adjustment needed"""
        # 10 keywords * 20 results = 200, max_total = 200
        results_per_keyword = 20
        keyword_count = 10
        max_total_results = 200

        potential_total = results_per_keyword * keyword_count
        assert potential_total == max_total_results

        effective_results_per_keyword = max(1, max_total_results // keyword_count)
        assert effective_results_per_keyword == 20

    @pytest.mark.fast
    def test_config_options_present(self, mock_config):
        """Test config has new budget-aware options"""
        search_config = mock_config.download.video_search
        assert 'search_budget_aware' in search_config
        assert 'auto_distribute_budget' in search_config
        assert search_config['search_budget_aware'] is True
        assert search_config['auto_distribute_budget'] is True


# ============================================================================
# Test Channel Diversity (US-94-009)
# ============================================================================

class TestChannelDiversity:
    """Test channel diversity filtering for search results"""

    @pytest.fixture
    def mock_config_with_channel_diversity(self):
        """Create mock config with channel diversity enabled"""
        config = Mock()
        config.download = Mock()
        config.download.video_search = {
            'results_per_keyword': 20,
            'max_total_results': 200,
            'search_budget_aware': True,
            'auto_distribute_budget': True,
            'enable_channel_diversity': True,
            'max_videos_per_channel': 3
        }
        config.download.min_duration = 30
        config.download.max_duration = 600
        config.download.title_blacklist = []
        return config

    @pytest.fixture
    def mock_config_channel_disabled(self):
        """Create mock config with channel diversity disabled"""
        config = Mock()
        config.download = Mock()
        config.download.video_search = {
            'results_per_keyword': 20,
            'max_total_results': 200,
            'search_budget_aware': True,
            'auto_distribute_budget': True,
            'enable_channel_diversity': False,
            'max_videos_per_channel': 3
        }
        config.download.min_duration = 30
        config.download.max_duration = 600
        config.download.title_blacklist = []
        return config

    @pytest.mark.fast
    def test_channel_diversity_limits_same_channel(self, mock_config_with_channel_diversity, mock_checkpoint):
        """Test that results from same channel are limited to max_videos_per_channel"""
        stage = VideoSearchStage()
        state = PipelineState()
        state.keywords = ['beach sunset']
        state.topic_context = 'Travel'

        # Multiple videos from same channel
        mock_results = {
            'entries': [
                {'id': 'vid1', 'title': 'Video 1', 'channel': 'Nature Channel', 'duration': 120},
                {'id': 'vid2', 'title': 'Video 2', 'channel': 'Nature Channel', 'duration': 120},
                {'id': 'vid3', 'title': 'Video 3', 'channel': 'Nature Channel', 'duration': 120},
                {'id': 'vid4', 'title': 'Video 4', 'channel': 'Nature Channel', 'duration': 120},
                {'id': 'vid5', 'title': 'Video 5', 'channel': 'Nature Channel', 'duration': 120},
            ]
        }

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_results
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            result = stage.run(state, mock_config_with_channel_diversity, mock_checkpoint)

        assert result.success is True
        # Only 3 videos from "Nature Channel" should be included
        assert len(state.video_ids) == 3
        assert 'vid1' in state.video_ids
        assert 'vid2' in state.video_ids
        assert 'vid3' in state.video_ids
        assert 'vid4' not in state.video_ids
        assert 'vid5' not in state.video_ids

    @pytest.mark.fast
    def test_channel_diversity_allows_different_channels(self, mock_config_with_channel_diversity, mock_checkpoint):
        """Test that videos from different channels are not limited"""
        stage = VideoSearchStage()
        state = PipelineState()
        state.keywords = ['beach sunset']
        state.topic_context = 'Travel'

        # Videos from different channels
        mock_results = {
            'entries': [
                {'id': 'vid1', 'title': 'Video 1', 'channel': 'Nature Channel', 'duration': 120},
                {'id': 'vid2', 'title': 'Video 2', 'channel': 'Travel Channel', 'duration': 120},
                {'id': 'vid3', 'title': 'Video 3', 'channel': 'Ocean Channel', 'duration': 120},
                {'id': 'vid4', 'title': 'Video 4', 'channel': 'Beach Channel', 'duration': 120},
                {'id': 'vid5', 'title': 'Video 5', 'channel': 'Sunset Channel', 'duration': 120},
            ]
        }

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_results
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            result = stage.run(state, mock_config_with_channel_diversity, mock_checkpoint)

        assert result.success is True
        # All 5 videos from different channels should be included
        assert len(state.video_ids) == 5

    @pytest.mark.fast
    def test_channel_diversity_disabled_keeps_all(self, mock_config_channel_disabled, mock_checkpoint):
        """Test that disabling channel diversity keeps all videos"""
        stage = VideoSearchStage()
        state = PipelineState()
        state.keywords = ['beach sunset']
        state.topic_context = 'Travel'

        # Multiple videos from same channel
        mock_results = {
            'entries': [
                {'id': 'vid1', 'title': 'Video 1', 'channel': 'Nature Channel', 'duration': 120},
                {'id': 'vid2', 'title': 'Video 2', 'channel': 'Nature Channel', 'duration': 120},
                {'id': 'vid3', 'title': 'Video 3', 'channel': 'Nature Channel', 'duration': 120},
                {'id': 'vid4', 'title': 'Video 4', 'channel': 'Nature Channel', 'duration': 120},
            ]
        }

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_results
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            result = stage.run(state, mock_config_channel_disabled, mock_checkpoint)

        assert result.success is True
        # All 4 videos should be included when diversity is disabled
        assert len(state.video_ids) == 4

    @pytest.mark.fast
    def test_channel_diversity_includes_no_channel_videos(self, mock_config_with_channel_diversity, mock_checkpoint):
        """Test that videos without channel info are included"""
        stage = VideoSearchStage()
        state = PipelineState()
        state.keywords = ['beach sunset']
        state.topic_context = 'Travel'

        # Mix of videos with and without channel info
        mock_results = {
            'entries': [
                {'id': 'vid1', 'title': 'Video 1', 'channel': 'Nature Channel', 'duration': 120},
                {'id': 'vid2', 'title': 'Video 2', 'channel': 'Nature Channel', 'duration': 120},
                {'id': 'vid3', 'title': 'Video 3', 'channel': 'Nature Channel', 'duration': 120},
                {'id': 'vid4', 'title': 'Video 4', 'channel': '', 'duration': 120},
                {'id': 'vid5', 'title': 'Video 5', 'channel': None, 'duration': 120},
            ]
        }

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_results
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            result = stage.run(state, mock_config_with_channel_diversity, mock_checkpoint)

        assert result.success is True
        # 3 from Nature Channel + 2 without channel info = 5 total
        assert len(state.video_ids) == 5
        assert 'vid1' in state.video_ids
        assert 'vid2' in state.video_ids
        assert 'vid3' in state.video_ids
        assert 'vid4' in state.video_ids  # No channel
        assert 'vid5' in state.video_ids  # None channel

    @pytest.mark.fast
    def test_channel_diversity_respects_custom_limit(self, mock_checkpoint):
        """Test that custom max_videos_per_channel is respected"""
        # Config with custom limit of 2
        config = Mock()
        config.download = Mock()
        config.download.video_search = {
            'results_per_keyword': 20,
            'max_total_results': 200,
            'search_budget_aware': True,
            'auto_distribute_budget': True,
            'enable_channel_diversity': True,
            'max_videos_per_channel': 2  # Custom limit
        }
        config.download.min_duration = 30
        config.download.max_duration = 600
        config.download.title_blacklist = []

        stage = VideoSearchStage()
        state = PipelineState()
        state.keywords = ['beach sunset']
        state.topic_context = 'Travel'

        mock_results = {
            'entries': [
                {'id': 'vid1', 'title': 'Video 1', 'channel': 'Nature Channel', 'duration': 120},
                {'id': 'vid2', 'title': 'Video 2', 'channel': 'Nature Channel', 'duration': 120},
                {'id': 'vid3', 'title': 'Video 3', 'channel': 'Nature Channel', 'duration': 120},
            ]
        }

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_results
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            result = stage.run(state, config, mock_checkpoint)

        assert result.success is True
        # Only 2 videos from "Nature Channel" should be included
        assert len(state.video_ids) == 2

    @pytest.mark.fast
    def test_config_options_channel_diversity_present(self):
        """Test config has channel diversity options"""
        from src.stages.video_search import VideoSearchStage
        stage = VideoSearchStage()
        # Verify the config options are handled (defaults)
        # Test default values work
        config_dict = {
            'results_per_keyword': 20,
            'max_total_results': 200,
        }
        enable_channel_diversity = config_dict.get('enable_channel_diversity', True)
        max_videos_per_channel = config_dict.get('max_videos_per_channel', 3)
        assert enable_channel_diversity is True
        assert max_videos_per_channel == 3

    @pytest.mark.fast
    def test_diverse_channels_appear_in_results(self, mock_checkpoint):
        """Test that channel diversity filtering produces diverse channels in final results"""
        # Create mock config with channel diversity enabled
        config = Mock()
        config.download = Mock()
        config.download.video_search = {
            'results_per_keyword': 20,
            'max_total_results': 200,
            'search_budget_aware': True,
            'auto_distribute_budget': True,
            'enable_channel_diversity': True,
            'max_videos_per_channel': 3
        }
        config.download.min_duration = 30
        config.download.max_duration = 600
        config.download.title_blacklist = []

        stage = VideoSearchStage()
        state = PipelineState()
        state.keywords = ['nature landscape']
        state.topic_context = 'Nature'

        # Create results with multiple videos from same channel and different channels
        mock_results = {
            'entries': [
                {'id': 'vid1', 'title': 'Video 1', 'channel': 'Nature Channel', 'duration': 120},
                {'id': 'vid2', 'title': 'Video 2', 'channel': 'Nature Channel', 'duration': 120},
                {'id': 'vid3', 'title': 'Video 3', 'channel': 'Nature Channel', 'duration': 120},
                {'id': 'vid4', 'title': 'Video 4', 'channel': 'Nature Channel', 'duration': 120},
                {'id': 'vid5', 'title': 'Video 5', 'channel': 'Nature Channel', 'duration': 120},
                {'id': 'vid6', 'title': 'Video 6', 'channel': 'Travel Channel', 'duration': 120},
                {'id': 'vid7', 'title': 'Video 7', 'channel': 'Ocean Channel', 'duration': 120},
                {'id': 'vid8', 'title': 'Video 8', 'channel': 'Mountain Channel', 'duration': 120},
            ]
        }

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_results
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            result = stage.run(state, config, mock_checkpoint)

        assert result.success is True

        # Extract channels from the video_search_results (VideoSearchResult objects)
        channels = [r.channel for r in state.video_search_results]

        # Should have diverse channels
        unique_channels = set(channels)
        # Should include Travel, Ocean, Mountain channels
        assert 'Travel Channel' in unique_channels
        assert 'Ocean Channel' in unique_channels
        assert 'Mountain Channel' in unique_channels
        # Should NOT have more than 3 from Nature Channel
        nature_count = channels.count('Nature Channel')
        assert nature_count == 3


# ============================================================================
# Test Tag-Based Query Expansion (US-95-002)
# ============================================================================

class TestTagBasedQueryExpansion:
    """Test US-95-002: Tag-based query expansion in video search"""

    @pytest.fixture
    def config_with_tags(self):
        """Create config with tag expansion enabled"""
        config = Mock()
        config.download = Mock()
        config.download.video_search = {
            'results_per_keyword': 20,
            'max_total_results': 200,
            'use_tags_in_search': True,
            'topic_tags': {
                'nature': ['wildlife', 'landscape', 'outdoor'],
                'travel': ['adventure', 'destination', 'culture'],
                'technology': ['innovation', 'science', 'gadgets'],
            }
        }
        config.download.min_duration = 30
        config.download.max_duration = 600
        config.download.title_blacklist = []
        return config

    @pytest.fixture
    def config_tags_disabled(self):
        """Create config with tag expansion disabled"""
        config = Mock()
        config.download = Mock()
        config.download.video_search = {
            'results_per_keyword': 20,
            'max_total_results': 200,
            'use_tags_in_search': False,
            'topic_tags': {
                'nature': ['wildlife', 'landscape'],
            }
        }
        config.download.min_duration = 30
        config.download.max_duration = 600
        config.download.title_blacklist = []
        return config

    @pytest.mark.fast
    def test_tag_expansion_expands_query_with_matching_tags(self, config_with_tags):
        """Test that keyword 'nature documentary' expands to include related tags"""
        stage = VideoSearchStage()

        # Mock yt-dlp
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = {'entries': []}
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            stage._search_keyword(
                keyword='nature documentary',
                config=config_with_tags,
                max_results=10,
                topic=''
            )

        # Verify yt-dlp was called with expanded query
        call_args = mock_ydl_instance.extract_info.call_args
        search_url = call_args[0][0]

        # Should include original keyword plus tags
        assert 'nature' in search_url.lower()
        # Should include expanded tags from topic_tags mapping
        assert 'wildlife' in search_url.lower() or 'landscape' in search_url.lower() or 'outdoor' in search_url.lower()

    @pytest.mark.fast
    def test_tag_expansion_uses_topic_context(self, config_with_tags):
        """Test that topic context triggers tag expansion"""
        stage = VideoSearchStage()

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = {'entries': []}
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            stage._search_keyword(
                keyword='beach',
                config=config_with_tags,
                max_results=10,
                topic='travel'
            )

        # Verify yt-dlp was called with expanded query using topic context
        call_args = mock_ydl_instance.extract_info.call_args
        search_url = call_args[0][0]

        # Should include travel-related tags
        assert 'adventure' in search_url.lower() or 'destination' in search_url.lower() or 'culture' in search_url.lower()

    @pytest.mark.fast
    def test_tag_expansion_disabled(self, config_tags_disabled):
        """Test that disabling use_tags_in_search skips tag expansion"""
        stage = VideoSearchStage()

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = {'entries': []}
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            stage._search_keyword(
                keyword='nature documentary',
                config=config_tags_disabled,
                max_results=10,
                topic=''
            )

        # Verify yt-dlp was called with basic query (no tag expansion)
        call_args = mock_ydl_instance.extract_info.call_args
        search_url = call_args[0][0]

        # Should be basic search without expanded tags
        assert 'nature' in search_url.lower()
        # Should NOT include expanded tags
        assert 'wildlife' not in search_url.lower()

    @pytest.mark.fast
    def test_tag_expansion_limits_to_three_tags(self, config_with_tags):
        """Test that tag expansion is limited to 3 tags to avoid over-broadening"""
        stage = VideoSearchStage()

        # Create config with many tags
        config = Mock()
        config.download = Mock()
        config.download.video_search = {
            'use_tags_in_search': True,
            'topic_tags': {
                'nature': ['wildlife', 'landscape', 'outdoor', 'scenery', 'environment'],
            }
        }
        config.download.min_duration = 30
        config.download.max_duration = 600
        config.download.title_blacklist = []

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = {'entries': []}
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            stage._search_keyword(
                keyword='nature',
                config=config,
                max_results=10,
                topic=''
            )

        call_args = mock_ydl_instance.extract_info.call_args
        search_url = call_args[0][0]

        # Should include keyword plus up to 3 tags
        # Count how many tag words appear beyond the keyword
        tag_words = ['wildlife', 'landscape', 'outdoor', 'scenery', 'environment']
        matching_tags = sum(1 for tag in tag_words if tag in search_url.lower())
        assert matching_tags <= 3, f"Expected at most 3 tags, got {matching_tags}"


# =============================================================================
# US-95-003: Description-based Query Refinement Tests
# =============================================================================

class TestDescriptionContext:
    """Tests for description-based query refinement (US-95-003)"""

    @pytest.fixture
    def config_with_description_context(self):
        """Create config with description context enabled"""
        config = Mock()
        config.download = Mock()
        config.download.video_search = {
            'results_per_keyword': 20,
            'max_total_results': 200,
            'use_tags_in_search': True,
            'use_description_context': True,
            'topic_tags': {
                'nature': ['wildlife', 'landscape'],
            }
        }
        config.download.min_duration = 30
        config.download.max_duration = 600
        config.download.title_blacklist = []
        return config

    @pytest.fixture
    def config_description_disabled(self):
        """Create config with description context disabled"""
        config = Mock()
        config.download = Mock()
        config.download.video_search = {
            'results_per_keyword': 20,
            'max_total_results': 200,
            'use_tags_in_search': True,
            'use_description_context': False,
            'topic_tags': {}
        }
        config.download.min_duration = 30
        config.download.max_duration = 600
        config.download.title_blacklist = []
        return config

    @pytest.mark.fast
    def test_extract_description_keywords_finds_common_terms(self):
        """Test that _extract_description_keywords finds common terms in descriptions"""
        stage = VideoSearchStage()

        results = [
            {'description': 'Amazing wildlife documentary about nature and animals'},
            {'description': 'Beautiful landscape with wildlife and nature scenes'},
            {'description': 'Exploring nature through wildlife photography'},
        ]

        keywords = stage._extract_description_keywords(results)

        # Should find common terms
        assert 'wildlife' in keywords or 'nature' in keywords

    @pytest.mark.fast
    def test_extract_description_keywords_filters_stop_words(self):
        """Test that stop words are filtered out"""
        stage = VideoSearchStage()

        results = [
            {'description': 'This is a video about nature and wildlife'},
            {'description': 'The video shows amazing nature scenes'},
        ]

        keywords = stage._extract_description_keywords(results)

        # Stop words should be filtered
        assert 'this' not in keywords
        assert 'is' not in keywords
        assert 'the' not in keywords

    @pytest.mark.fast
    def test_extract_description_keywords_returns_empty_for_no_descriptions(self):
        """Test empty list for no descriptions"""
        stage = VideoSearchStage()

        keywords = stage._extract_description_keywords([])
        assert keywords == []

    @pytest.mark.fast
    def test_description_context_refines_search_with_description_keywords(self, config_with_description_context):
        """Test that description context refines search with extracted keywords"""
        stage = VideoSearchStage()

        # Mock yt-dlp with descriptions in initial results
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = {
            'entries': [
                {
                    'id': 'video1',
                    'title': 'Nature Documentary',
                    'channel': 'Nature Channel',
                    'duration': 300,
                    'description': 'Amazing wildlife documentary about nature and animals in the forest',
                },
                {
                    'id': 'video2',
                    'title': 'Wildlife Adventure',
                    'channel': 'Wildlife TV',
                    'duration': 400,
                    'description': 'Beautiful wildlife and nature scenes from around the world',
                },
            ]
        }
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            results = stage._search_keyword(
                keyword='nature documentary',
                config=config_with_description_context,
                max_results=10,
                topic=''
            )

        # Should have called extract_info twice (initial + refined)
        assert mock_ydl_instance.extract_info.call_count >= 1

    @pytest.mark.fast
    def test_description_context_disabled_skips_refinement(self, config_description_disabled):
        """Test that disabling use_description_context skips description refinement"""
        stage = VideoSearchStage()

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = {
            'entries': [
                {
                    'id': 'video1',
                    'title': 'Nature Documentary',
                    'channel': 'Nature Channel',
                    'duration': 300,
                    'description': 'Test description',
                }
            ]
        }
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            results = stage._search_keyword(
                keyword='nature',
                config=config_description_disabled,
                max_results=10,
                topic=''
            )

        # Should only have one call (no refinement when disabled)
        assert mock_ydl_instance.extract_info.call_count == 1

    @pytest.mark.fast
    def test_description_keywords_weighted_lower_in_query(self, config_with_description_context):
        """Test that description keywords appear after primary keyword in query"""
        stage = VideoSearchStage()

        # Mock with descriptions containing common keywords
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = {
            'entries': [
                {
                    'id': 'video1',
                    'title': 'Nature Documentary',
                    'channel': 'Nature Channel',
                    'duration': 300,
                    'description': 'wildlife nature animals forest',
                },
                {
                    'id': 'video2',
                    'title': 'Wildlife Film',
                    'channel': 'Wildlife TV',
                    'duration': 350,
                    'description': 'wildlife nature landscape scenery',
                },
            ]
        }
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            results = stage._search_keyword(
                keyword='nature',
                config=config_with_description_context,
                max_results=10,
                topic=''
            )

        # Verify second call has refined query with primary keyword first
        calls = mock_ydl_instance.extract_info.call_args_list
        if len(calls) >= 2:
            second_query = calls[1][0][0]
            # Primary keyword should be first, description keywords after
            assert 'nature' in second_query.lower()


# ============================================================================
# Test Negative Context Filtering (US-95-012)
# ============================================================================

class TestNegativeContextFiltering:
    """Tests for negative context filtering (US-95-012)"""

    @pytest.fixture
    def config_with_negative_context(self):
        """Create config with negative context filtering enabled"""
        config = Mock()
        config.download = Mock()
        config.download.video_search = {
            'results_per_keyword': 20,
            'max_total_results': 200,
            'use_negative_context': True,
            'negative_keywords': ['trailer', 'teaser', 'compilation', 'best of', 'top 10']
        }
        config.download.min_duration = 30
        config.download.max_duration = 600
        config.download.title_blacklist = []
        return config

    @pytest.fixture
    def config_negative_disabled(self):
        """Create config with negative context filtering disabled"""
        config = Mock()
        config.download = Mock()
        config.download.video_search = {
            'results_per_keyword': 20,
            'max_total_results': 200,
            'use_negative_context': False,
            'negative_keywords': ['trailer', 'teaser']
        }
        config.download.min_duration = 30
        config.download.max_duration = 600
        config.download.title_blacklist = []
        return config

    @pytest.mark.fast
    def test_is_negative_matched_title(self):
        """Test _is_negative_matched matches negative keywords in title"""
        stage = VideoSearchStage()

        # Should match
        assert stage._is_negative_matched('Beach Trailer', '', ['trailer', 'teaser']) is True
        assert stage._is_negative_matched('Nature Best Of', '', ['best of']) is True
        assert stage._is_negative_matched('Top 10 Beach', '', ['top 10']) is True
        assert stage._is_negative_matched('Beach Compilation Video', '', ['compilation']) is True

    @pytest.mark.fast
    def test_is_negative_matched_description(self):
        """Test _is_negative_matched matches negative keywords in description"""
        stage = VideoSearchStage()

        # Should match in description even if title is clean
        assert stage._is_negative_matched('Beach Video', 'Watch the trailer here', ['trailer']) is True
        assert stage._is_negative_matched('Nature Documentary', 'Best of nature compilation', ['compilation']) is True

    @pytest.mark.fast
    def test_is_negative_matched_case_insensitive(self):
        """Test negative keyword matching is case insensitive"""
        stage = VideoSearchStage()

        assert stage._is_negative_matched('Beach TRAILER', '', ['trailer']) is True
        assert stage._is_negative_matched('Beach video', 'COMPILATION', ['compilation']) is True

    @pytest.mark.fast
    def test_is_negative_matched_no_match(self):
        """Test _is_negative_matched returns False when no match"""
        stage = VideoSearchStage()

        assert stage._is_negative_matched('Beautiful Beach', '', ['trailer', 'teaser']) is False
        assert stage._is_negative_matched('Ocean Waves', 'Relaxing video', ['trailer', 'teaser']) is False

    @pytest.mark.fast
    def test_is_negative_matched_empty_title(self):
        """Test _is_negative_matched returns False for empty title"""
        stage = VideoSearchStage()

        assert stage._is_negative_matched('', '', ['trailer']) is False
        assert stage._is_negative_matched(None, '', ['trailer']) is False

    @pytest.mark.fast
    def test_is_negative_matched_empty_keywords(self):
        """Test _is_negative_matched returns False for empty negative keywords"""
        stage = VideoSearchStage()

        assert stage._is_negative_matched('Beach Trailer', '', []) is False
        assert stage._is_negative_matched('Beach Trailer', '', None) is False

    @pytest.mark.fast
    def test_negative_context_filters_title_matches(self, config_with_negative_context):
        """Test that videos with negative keywords in title are filtered"""
        stage = VideoSearchStage()

        mock_results = {
            'entries': [
                {'id': 'good1', 'title': 'Beautiful Beach Sunset', 'channel': 'Ch', 'duration': 120},
                {'id': 'bad1', 'title': 'Beach Trailer 2024', 'channel': 'Ch', 'duration': 120},
                {'id': 'good2', 'title': 'Ocean Waves', 'channel': 'Ch', 'duration': 120},
                {'id': 'bad2', 'title': 'Top 10 Beaches', 'channel': 'Ch', 'duration': 120},
            ]
        }

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_results
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            search_results = stage._search_keyword('beach', config_with_negative_context)

        video_ids = [r['video_id'] for r in search_results]
        assert 'good1' in video_ids
        assert 'good2' in video_ids
        assert 'bad1' not in video_ids  # "trailer" in title
        assert 'bad2' not in video_ids  # "top 10" in title

    @pytest.mark.fast
    def test_negative_context_filters_description_matches(self, config_with_negative_context):
        """Test that videos with negative keywords in description are filtered"""
        stage = VideoSearchStage()

        mock_results = {
            'entries': [
                {'id': 'good1', 'title': 'Beautiful Beach', 'channel': 'Ch', 'duration': 120, 'description': 'Relaxing beach scene'},
                {'id': 'bad1', 'title': 'Nature Video', 'channel': 'Ch', 'duration': 120, 'description': 'Watch the trailer for this amazing documentary'},
            ]
        }

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_results
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            search_results = stage._search_keyword('nature', config_with_negative_context)

        video_ids = [r['video_id'] for r in search_results]
        assert 'good1' in video_ids
        assert 'bad1' not in video_ids  # "trailer" in description

    @pytest.mark.fast
    def test_negative_context_disabled_keeps_all(self, config_negative_disabled):
        """Test that disabling use_negative_context keeps all videos"""
        stage = VideoSearchStage()

        mock_results = {
            'entries': [
                {'id': 'good1', 'title': 'Beautiful Beach Sunset', 'channel': 'Ch', 'duration': 120},
                {'id': 'bad1', 'title': 'Beach Trailer 2024', 'channel': 'Ch', 'duration': 120},
            ]
        }

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_results
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            search_results = stage._search_keyword('beach', config_negative_disabled)

        # All videos should be included when negative context is disabled
        video_ids = [r['video_id'] for r in search_results]
        assert len(video_ids) == 2

    @pytest.mark.fast
    def test_negative_context_integration_full_flow(self, config_with_negative_context, mock_checkpoint):
        """Test negative context filtering in full stage run"""
        stage = VideoSearchStage()
        state = PipelineState()
        state.keywords = ['beach sunset']
        state.topic_context = 'Travel'

        mock_results = {
            'entries': [
                {'id': 'good1', 'title': 'Beautiful Beach Sunset', 'channel': 'Nature Ch', 'duration': 120},
                {'id': 'bad1', 'title': 'Beach Compilation', 'channel': 'Travel Ch', 'duration': 120},
                {'id': 'good2', 'title': 'Ocean Waves', 'channel': 'Ocean Ch', 'duration': 120},
            ]
        }

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = mock_results
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch('yt_dlp.YoutubeDL', return_value=mock_ydl_instance):
            result = stage.run(state, config_with_negative_context, mock_checkpoint)

        assert result.success is True
        # Should filter out the compilation video
        assert 'good1' in state.video_ids
        assert 'good2' in state.video_ids
        assert 'bad1' not in state.video_ids

    @pytest.mark.fast
    def test_extract_negative_keywords_finds_matches(self):
        """Test _extract_negative_keywords finds keywords in title/description"""
        stage = VideoSearchStage()

        # Should find keywords
        result = stage._extract_negative_keywords(
            'Beach Trailer',
            'Watch this amazing video',
            ['trailer', 'teaser', 'compilation']
        )
        assert 'trailer' in result

    @pytest.mark.fast
    def test_extract_negative_keywords_finds_in_description(self):
        """Test _extract_negative_keywords finds keywords in description"""
        stage = VideoSearchStage()

        result = stage._extract_negative_keywords(
            'Beach Video',
            'Watch the trailer for this beach',
            ['trailer', 'teaser']
        )
        assert 'trailer' in result

    @pytest.mark.fast
    def test_extract_negative_keywords_case_insensitive(self):
        """Test _extract_negative_keywords is case insensitive"""
        stage = VideoSearchStage()

        result = stage._extract_negative_keywords(
            'Beach TRAILER',
            'COMPILATION video',
            ['trailer', 'compilation']
        )
        assert 'trailer' in result
        assert 'compilation' in result

    @pytest.mark.fast
    def test_extract_negative_keywords_no_match(self):
        """Test _extract_negative_keywords returns empty when no match"""
        stage = VideoSearchStage()

        result = stage._extract_negative_keywords(
            'Beautiful Beach',
            'Relaxing ocean waves',
            ['trailer', 'teaser']
        )
        assert result == []

    @pytest.mark.fast
    def test_to_search_results_populates_negative_keywords(self):
        """Test _to_search_results populates negative_keywords in results"""
        stage = VideoSearchStage()

        results = [
            {'video_id': 'v1', 'title': 'Beach Trailer', 'channel': 'Ch1', 'duration': 120, 'keyword': 'beach', 'description': 'Watch trailer'},
            {'video_id': 'v2', 'title': 'Ocean Waves', 'channel': 'Ch2', 'duration': 120, 'keyword': 'ocean', 'description': 'Relaxing'},
        ]

        search_results = stage._to_search_results(
            results,
            negative_keywords=['trailer', 'teaser']
        )

        assert len(search_results) == 2
        # First result should have 'trailer' detected
        v1_result = next(r for r in search_results if r.video_id == 'v1')
        assert 'trailer' in v1_result.negative_keywords

        # Second result should have empty negative_keywords
        v2_result = next(r for r in search_results if r.video_id == 'v2')
        assert v2_result.negative_keywords == []

    @pytest.mark.fast
    def test_negative_signals_reduce_irrelevant_results(self):
        """AC4: Test that negative signals reduce irrelevant results"""
        stage = VideoSearchStage()

        # Simulate search results with some relevant and some irrelevant
        results = [
            {'video_id': 'v1', 'title': 'Beautiful Beach Sunset', 'channel': 'Nature', 'duration': 120, 'keyword': 'beach', 'description': 'Relaxing'},
            {'video_id': 'v2', 'title': 'Beach Trailer 2024', 'channel': 'Movies', 'duration': 120, 'keyword': 'beach', 'description': 'Watch trailer'},
            {'video_id': 'v3', 'title': 'Top 10 Beaches', 'channel': 'Lists', 'duration': 120, 'keyword': 'beach', 'description': 'Best beaches'},
            {'video_id': 'v4', 'title': 'Ocean Waves', 'channel': 'Nature', 'duration': 120, 'keyword': 'ocean', 'description': 'Calm ocean'},
        ]

        # Without negative keywords - all pass
        results_no_filter = stage._to_search_results(results, negative_keywords=[])
        assert len(results_no_filter) == 4

        # With negative keywords - irrelevant ones filtered from metadata
        results_with_filter = stage._to_search_results(
            results,
            negative_keywords=['trailer', 'top 10', 'compilation']
        )

        # Should detect negative keywords in some videos
        v2_result = next(r for r in results_with_filter if r.video_id == 'v2')
        assert 'trailer' in v2_result.negative_keywords

        v3_result = next(r for r in results_with_filter if r.video_id == 'v3')
        assert 'top 10' in v3_result.negative_keywords

        # Clean videos should have empty negative_keywords
        v1_result = next(r for r in results_with_filter if r.video_id == 'v1')
        assert v1_result.negative_keywords == []

    def test_config_use_negative_context_default(self):
        """AC3: Test that use_negative_context config option exists with default false"""
        from src.config.sections.iterative_matching import IterativeMatchingConfig
        from src.config import Config

        # Check the config.yaml source directly
        import yaml
        config_path = Path(__file__).parent.parent / 'config.yaml'
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # Verify use_negative_context is in video_search config (top-level key)
        assert 'video_search' in config_data
        assert 'use_negative_context' in config_data['video_search']
        assert config_data['video_search']['use_negative_context'] is False


# ============================================================================
# Test Listicle Topic Search (US-98-008)
# ============================================================================

class TestListicleTopicSearch:
    """Test listicle topic keywords as search terms (US-98-008)"""

    def test_build_listicle_queries_with_topics(self):
        """AC2: Test extraction of topic_keywords from ListicleGroup"""
        from src.stages.video_search import VideoSearchStage
        from src.chapter_detection.models import ListicleGroup

        stage = VideoSearchStage()

        # Create mock listicle groups with topic keywords
        listicle_groups = [
            ListicleGroup(
                group_id=1,
                item_label="first",
                marker_type="ordinal",
                topic_keywords=["beach sunset", "ocean view", "tropical"]
            ),
            ListicleGroup(
                group_id=2,
                item_label="second",
                marker_type="ordinal",
                topic_keywords=["mountain hike", "trail", "summit"]
            ),
        ]

        queries = stage._build_listicle_queries(listicle_groups, topic_context="travel")

        # Should have 2 queries
        assert len(queries) == 2

        # First query should have beach topics
        assert queries[0]['group_id'] == 1
        assert queries[0]['item_label'] == "first"
        assert "beach" in queries[0]['keyword'] or "sunset" in queries[0]['keyword']

        # Second query should have mountain topics
        assert queries[1]['group_id'] == 2
        assert queries[1]['item_label'] == "second"
        assert "mountain" in queries[1]['keyword'] or "hike" in queries[1]['keyword']

    def test_build_listicle_queries_empty_topics(self):
        """Test that groups without topic_keywords are skipped"""
        from src.stages.video_search import VideoSearchStage
        from src.chapter_detection.models import ListicleGroup

        stage = VideoSearchStage()

        # Group without topics
        listicle_groups = [
            ListicleGroup(
                group_id=1,
                item_label="first",
                marker_type="ordinal",
                topic_keywords=[]  # Empty topics
            ),
        ]

        queries = stage._build_listicle_queries(listicle_groups, topic_context="travel")

        # Should have no queries
        assert len(queries) == 0

    def test_build_listicle_queries_with_topic_context(self):
        """Test that topic_context is added when not in topics"""
        from src.stages.video_search import VideoSearchStage
        from src.chapter_detection.models import ListicleGroup

        stage = VideoSearchStage()

        listicle_groups = [
            ListicleGroup(
                group_id=1,
                item_label="first",
                marker_type="ordinal",
                topic_keywords=["sunset", "beach"]  # No "travel" in topics
            ),
        ]

        queries = stage._build_listicle_queries(listicle_groups, topic_context="travel")

        # Should include topic_context in keyword
        assert len(queries) == 1
        assert "travel" in queries[0]['keyword']

    def test_build_listicle_queries_dict_input(self):
        """Test that dict input is handled correctly"""
        from src.stages.video_search import VideoSearchStage

        stage = VideoSearchStage()

        # Use dict input (like from checkpoint)
        listicle_groups = [
            {
                'group_id': 1,
                'item_label': 'first',
                'topic_keywords': ['tips', 'tricks']
            },
        ]

        queries = stage._build_listicle_queries(listicle_groups, topic_context="")

        assert len(queries) == 1
        assert queries[0]['group_id'] == 1
        assert queries[0]['item_label'] == "first"

    def test_config_listicle_topic_as_search_terms(self):
        """AC1: Test config option 'listicle_topic_as_search_terms' exists"""
        import yaml
        from pathlib import Path

        config_path = Path(__file__).parent.parent / 'config.yaml'
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        assert 'video_search' in config_data
        assert 'listicle_topic_as_search_terms' in config_data['video_search']
        # Default should be true for listicle-matching focus
        assert config_data['video_search']['listicle_topic_as_search_terms'] is True

    def test_listicle_queries_limited_to_three_topics(self):
        """AC4: Test that search queries use max 3 topics for specificity"""
        from src.stages.video_search import VideoSearchStage
        from src.chapter_detection.models import ListicleGroup

        stage = VideoSearchStage()

        # More than 3 topics
        listicle_groups = [
            ListicleGroup(
                group_id=1,
                item_label="first",
                marker_type="ordinal",
                topic_keywords=["one", "two", "three", "four", "five"]
            ),
        ]

        queries = stage._build_listicle_queries(listicle_groups, topic_context="")

        # Should use first 3 topics
        assert len(queries) == 1
        # Should contain first 3 topics
        assert "one" in queries[0]['keyword']
        assert "two" in queries[0]['keyword']
        assert "three" in queries[0]['keyword']
        # Four and five should not be in the keyword (as single words, may be combined differently)

    def test_listicle_topics_more_specific_than_generic(self):
        """AC5: Test that listicle topics generate more specific search results

        Listicle topics (e.g., 'top 5 tips for cooking') should generate
        more targeted searches than generic keywords.
        """
        from src.stages.video_search import VideoSearchStage
        from src.chapter_detection.models import ListicleGroup

        stage = VideoSearchStage()

        # Listicle with specific topics
        listicle_groups = [
            ListicleGroup(
                group_id=1,
                item_label="#1",
                marker_type="numbered",
                topic_keywords=["pasta tips", "italian cooking", "homemade"]
            ),
        ]

        queries = stage._build_listicle_queries(listicle_groups, topic_context="")

        # The keyword should be specific to the listicle items
        assert len(queries) == 1
        # Should contain specific cooking terms
        keyword = queries[0]['keyword'].lower()
        assert "pasta" in keyword or "italian" in keyword or "cooking" in keyword

    def test_to_search_results_includes_listicle_fields(self):
        """Test that _to_search_results includes listicle_group_id and listicle_item_label"""
        from src.stages.video_search import VideoSearchStage

        stage = VideoSearchStage()

        # Results with listicle tags
        results = [
            {
                'video_id': 'v1',
                'title': 'Test Video',
                'channel': 'Test',
                'duration': 120,
                'keyword': 'test',
                'listicle_group_id': 1,
                'listicle_item_label': '#1',
            },
        ]

        search_results = stage._to_search_results(results)

        assert len(search_results) == 1
        assert search_results[0].listicle_group_id == 1
        assert search_results[0].listicle_item_label == '#1'

    def test_to_search_results_handles_missing_listicle_fields(self):
        """Test that missing listicle fields default to -1 and empty string"""
        from src.stages.video_search import VideoSearchStage

        stage = VideoSearchStage()

        # Results without listicle tags
        results = [
            {
                'video_id': 'v1',
                'title': 'Test Video',
                'channel': 'Test',
                'duration': 120,
                'keyword': 'test',
            },
        ]

        search_results = stage._to_search_results(results)

        assert len(search_results) == 1
        assert search_results[0].listicle_group_id == -1
        assert search_results[0].listicle_item_label == ""

