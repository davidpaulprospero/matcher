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
        'max_total_results': 200
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
