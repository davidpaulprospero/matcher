"""
Comprehensive tests for downloader title filter module.

Covers:
- TitleFilter initialization
- Video metadata search with yt-dlp
- LLM-based title filtering and ranking
- Gemini API integration
- Anthropic API integration
- JSON response parsing and error handling
- Batch processing
- Relevance scoring and sorting

Created: 2026-01-09 (Phase 7.1)
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest

from src.downloader.title_filter import TitleFilter


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config"""
    config = Mock()
    config.download = Mock()
    config.download.search_timeout = 60
    config.download.llm_title_filter = Mock()
    config.download.llm_title_filter.enabled = True
    config.download.llm_title_filter.provider = 'gemini'
    config.download.llm_title_filter.model = 'gemini-2.0-flash'
    config.download.llm_title_filter.min_relevance = 0.7
    config.download.llm_title_filter.batch_size = 20
    config.llm = Mock()
    config.llm.max_tokens = 2000
    return config


@pytest.fixture
def get_tier_value_func():
    """Mock tier value getter"""
    def getter(tier, key, default):
        tiers = {
            'short': {'min': 0, 'max': 60},
            'medium': {'min': 60, 'max': 300},
            'long': {'min': 300, 'max': 900},
            'longer': {'min': 900, 'max': 3600}
        }
        return tiers.get(tier, {}).get(key, default)
    return getter


@pytest.fixture
def title_filter(mock_config, get_tier_value_func):
    """Create TitleFilter instance"""
    return TitleFilter(
        config=mock_config,
        cookies_args=['--cookies', 'cookies.txt'],
        get_tier_value_func=get_tier_value_func
    )


@pytest.fixture
def sample_videos():
    """Sample video metadata"""
    return [
        {
            'id': 'abc123',
            'title': 'Amazing Beach Documentary',
            'duration': 120,
            'channel': 'NatureTV',
            'url': 'https://www.youtube.com/watch?v=abc123'
        },
        {
            'id': 'def456',
            'title': 'Beach Vlog 2024',
            'duration': 180,
            'channel': 'TravelVlogger',
            'url': 'https://www.youtube.com/watch?v=def456'
        },
        {
            'id': 'ghi789',
            'title': 'Beach Music Mix 24/7',
            'duration': 90,
            'channel': 'MusicChannel',
            'url': 'https://www.youtube.com/watch?v=ghi789'
        }
    ]


# ============================================================================
# Test Initialization
# ============================================================================

class TestTitleFilterInit:
    """Test TitleFilter initialization"""

    @pytest.mark.fast
    def test_init_basic(self, mock_config, get_tier_value_func):
        """Test basic initialization"""
        filter = TitleFilter(mock_config, ['--cookies', 'test.txt'], get_tier_value_func)

        assert filter.config == mock_config
        assert filter.download_config == mock_config.download
        assert len(filter.cookies_args) == 2
        assert filter._get_tier_value == get_tier_value_func

    @pytest.mark.fast
    def test_init_with_empty_cookies(self, mock_config, get_tier_value_func):
        """Test initialization with empty cookies"""
        filter = TitleFilter(mock_config, [], get_tier_value_func)

        assert filter.cookies_args == []


# ============================================================================
# Test Video Metadata Search
# ============================================================================

class TestVideoMetadataSearch:
    """Test searching YouTube for video metadata"""

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_search_video_metadata_success(self, mock_run, title_filter):
        """Test successful metadata search"""
        # Mock yt-dlp output
        mock_output = '\n'.join([
            json.dumps({'id': 'abc', 'title': 'Test Video 1', 'duration': 120, 'channel': 'Channel1'}),
            json.dumps({'id': 'def', 'title': 'Test Video 2', 'duration': 150, 'uploader': 'Channel2'})
        ])

        mock_run.return_value = Mock(
            returncode=0,
            stdout=mock_output,
            stderr=''
        )

        videos = title_filter.search_video_metadata('beach', 'medium', max_results=50)

        assert len(videos) == 2
        assert videos[0]['id'] == 'abc'
        assert videos[0]['title'] == 'Test Video 1'
        assert videos[0]['duration'] == 120
        assert videos[1]['channel'] == 'Channel2'  # Uses uploader fallback

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_search_video_metadata_with_tier_constraints(self, mock_run, title_filter):
        """Test search respects tier duration constraints"""
        mock_run.return_value = Mock(returncode=0, stdout='', stderr='')

        title_filter.search_video_metadata('test', 'short', max_results=10)

        # Verify command includes duration constraints
        call_args = mock_run.call_args[0][0]
        assert any('duration>0' in str(arg) and 'duration<60' in str(arg) for arg in call_args)
        assert any('!is_live' in str(arg) for arg in call_args)

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_search_video_metadata_timeout(self, mock_run, title_filter):
        """Test handling of search timeout"""
        mock_run.side_effect = subprocess.TimeoutExpired('yt-dlp', 60)

        videos = title_filter.search_video_metadata('test', 'medium')

        assert videos == []

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_search_video_metadata_error(self, mock_run, title_filter):
        """Test handling of search errors"""
        mock_run.side_effect = Exception("Network error")

        videos = title_filter.search_video_metadata('test', 'medium')

        assert videos == []

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_search_video_metadata_nonzero_return_code(self, mock_run, title_filter):
        """Test handling non-zero return code"""
        mock_run.return_value = Mock(
            returncode=1,
            stdout='',
            stderr='ERROR: Unable to download'
        )

        videos = title_filter.search_video_metadata('test', 'medium')

        assert videos == []

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_search_video_metadata_invalid_json(self, mock_run, title_filter):
        """Test handling invalid JSON in output"""
        mock_output = '\n'.join([
            json.dumps({'id': 'abc', 'title': 'Valid', 'duration': 120}),
            'invalid json line',
            json.dumps({'id': 'def', 'title': 'Also Valid', 'duration': 150})
        ])

        mock_run.return_value = Mock(
            returncode=0,
            stdout=mock_output,
            stderr=''
        )

        videos = title_filter.search_video_metadata('test', 'medium')

        # Should skip invalid line
        assert len(videos) == 2
        assert videos[0]['id'] == 'abc'
        assert videos[1]['id'] == 'def'


# ============================================================================
# Test LLM Title Filtering
# ============================================================================

class TestLLMTitleFiltering:
    """Test LLM-based title filtering"""

    @pytest.mark.fast
    def test_filter_disabled(self, sample_videos, mock_config, get_tier_value_func):
        """Test filtering when LLM filter is disabled"""
        mock_config.download.llm_title_filter.enabled = False
        filter = TitleFilter(mock_config, [], get_tier_value_func)

        result = filter.filter_titles_with_llm(sample_videos, 'beach')

        # Should return all videos unchanged
        assert result == sample_videos

    @pytest.mark.fast
    def test_filter_empty_videos(self, title_filter):
        """Test filtering empty video list"""
        result = title_filter.filter_titles_with_llm([], 'beach')

        assert result == []

    @pytest.mark.fast
    def test_filter_with_gemini_success(self, title_filter, sample_videos):
        """Test successful Gemini-based filtering"""
        # Mock LLM response
        llm_response = json.dumps([
            {"index": 1, "approve": True, "relevance": 0.9, "reason": "Documentary"},
            {"index": 2, "approve": False, "relevance": 0.0, "reason": "Vlog"},
            {"index": 3, "approve": False, "relevance": 0.0, "reason": "Music"}
        ])

        # Mock _call_gemini method directly
        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos, 'beach', topic='nature')

        # Should only approve first video
        assert len(result) == 1
        assert result[0]['title'] == 'Amazing Beach Documentary'
        assert result[0]['llm_relevance'] == 0.9

    @pytest.mark.fast
    def test_filter_relevance_sorting(self, title_filter, sample_videos):
        """Test results are sorted by relevance score"""
        llm_response = json.dumps([
            {"index": 1, "approve": True, "relevance": 0.6, "reason": "OK"},
            {"index": 2, "approve": True, "relevance": 0.9, "reason": "Great"},
            {"index": 3, "approve": True, "relevance": 0.7, "reason": "Good"}
        ])

        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos, 'beach')

        # Should be sorted by relevance (highest first)
        assert len(result) == 3
        assert result[0]['llm_relevance'] == 0.9
        assert result[1]['llm_relevance'] == 0.7
        assert result[2]['llm_relevance'] == 0.6

    @pytest.mark.fast
    def test_filter_markdown_code_block_removal(self, title_filter, sample_videos):
        """Test removal of markdown code blocks from response"""
        # LLM response with markdown code blocks
        llm_response = '```json\n' + json.dumps([
            {"index": 1, "approve": True, "relevance": 0.8, "reason": "Good"}
        ]) + '\n```'

        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos[:1], 'beach')

        assert len(result) == 1
        assert result[0]['llm_relevance'] == 0.8

    @pytest.mark.fast
    def test_filter_truncated_json_recovery(self, title_filter, sample_videos):
        """Test recovery from truncated JSON"""
        # Truncated JSON (missing closing bracket)
        llm_response = '[{"index": 1, "approve": true, "relevance": 0.8, "reason": "Good"}'

        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos[:1], 'beach')

        # Should fix and parse successfully
        assert len(result) == 1

    @pytest.mark.fast
    def test_filter_error_handling(self, title_filter, sample_videos):
        """Test error handling in filtering"""
        title_filter._call_gemini = Mock(side_effect=Exception("API error"))

        result = title_filter.filter_titles_with_llm(sample_videos, 'beach')

        # Should fail open - approve all with lower relevance
        assert len(result) == 3
        assert all(v['llm_relevance'] == 0.5 for v in result)

    @pytest.mark.fast
    def test_filter_batch_processing(self, title_filter):
        """Test batch processing of large video lists"""
        # Create 50 videos (> batch_size of 20)
        many_videos = [
            {'id': f'id{i}', 'title': f'Video {i}', 'duration': 100, 'channel': 'Test'}
            for i in range(50)
        ]

        # Mock response approving first 5 from each batch
        response_json = json.dumps([
            {"index": i, "approve": True, "relevance": 0.8, "reason": "Good"}
            for i in range(1, 6)
        ])

        title_filter._call_gemini = Mock(return_value=response_json)

        result = title_filter.filter_titles_with_llm(many_videos, 'beach')

        # Should process in 3 batches (20, 20, 10)
        # Each batch approves 5 videos = 15 total
        assert len(result) == 15


# ============================================================================
# Test Gemini API Integration
# ============================================================================

class TestGeminiIntegration:
    """Test Gemini API integration"""

    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'})
    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_call_gemini_success(self, mock_create_client, title_filter):
        """Test successful Gemini API call"""
        mock_response = Mock()
        mock_response.text = '[]'
        mock_client = Mock()
        mock_client.generate.return_value = mock_response
        mock_create_client.return_value = mock_client

        result = title_filter._call_gemini('test prompt', 'gemini-2.0-flash')

        assert result == '[]'
        mock_create_client.assert_called_once_with(
            "gemini",
            api_key='test_key',
            model='gemini-2.0-flash'
        )

    @patch.dict('os.environ', {}, clear=True)
    @pytest.mark.fast
    def test_call_gemini_no_api_key(self, title_filter):
        """Test Gemini call without API key"""
        result = title_filter._call_gemini('test prompt', 'gemini-2.0-flash')

        assert result == '[]'

    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test_key'})
    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_call_gemini_error(self, mock_create_client, title_filter):
        """Test Gemini API error handling"""
        mock_create_client.side_effect = Exception("API error")

        result = title_filter._call_gemini('test prompt', 'gemini-2.0-flash')

        assert result == '[]'


# ============================================================================
# Test Anthropic API Integration
# ============================================================================

class TestAnthropicIntegration:
    """Test Anthropic API integration"""

    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'})
    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_call_anthropic_success(self, mock_create_client, title_filter):
        """Test successful Anthropic API call"""
        mock_response = Mock()
        mock_response.text = '[]'
        mock_client = Mock()
        mock_client.generate.return_value = mock_response
        mock_create_client.return_value = mock_client

        result = title_filter._call_anthropic('test prompt', 'claude-3-haiku-20240307')

        assert result == '[]'
        mock_create_client.assert_called_once()

    @patch.dict('os.environ', {}, clear=True)
    @pytest.mark.fast
    def test_call_anthropic_no_api_key(self, title_filter):
        """Test Anthropic call without API key"""
        result = title_filter._call_anthropic('test prompt', 'claude-3-haiku-20240307')

        assert result == '[]'

    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'})
    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_call_anthropic_error(self, mock_create_client, title_filter):
        """Test Anthropic API error handling"""
        mock_create_client.side_effect = Exception("API error")

        result = title_filter._call_anthropic('test prompt', 'claude-3-haiku-20240307')

        assert result == '[]'


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases"""

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_search_empty_keyword(self, mock_run, title_filter):
        """Test search with empty keyword"""
        mock_run.return_value = Mock(returncode=0, stdout='', stderr='')

        videos = title_filter.search_video_metadata('', 'medium')

        # Should still make the call
        assert mock_run.called

    @pytest.mark.fast
    def test_filter_default_relevance(self, title_filter, sample_videos):
        """Test default relevance when not specified"""
        llm_response = json.dumps([
            {"index": 1, "approve": True, "reason": "Good"}  # No relevance field
        ])

        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos[:1], 'beach')

        # Should use default relevance of 0.7
        assert result[0]['llm_relevance'] == 0.7

    @pytest.mark.fast
    def test_filter_out_of_range_index(self, title_filter, sample_videos):
        """Test handling of out-of-range indices in LLM response"""
        llm_response = json.dumps([
            {"index": 1, "approve": True, "relevance": 0.8, "reason": "Good"},
            {"index": 99, "approve": True, "relevance": 0.9, "reason": "Invalid"}  # Out of range
        ])

        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos, 'beach')

        # Should only include valid index
        assert len(result) == 1
        assert result[0]['llm_relevance'] == 0.8

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_search_with_stderr_warnings(self, mock_run, title_filter):
        """Test search with stderr warnings"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps({'id': 'abc', 'title': 'Test', 'duration': 120}),
            stderr='WARNING: Some warning\nERROR: Some error'
        )

        videos = title_filter.search_video_metadata('test', 'medium')

        # Should still parse output despite warnings
        assert len(videos) == 1
