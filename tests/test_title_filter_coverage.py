"""
Extended coverage tests for src/downloader/title_filter.py.

This module focuses on covering the 14 missed lines identified in coverage analysis:
- Non-zero return code edge cases (lines 97-100)
- Stderr warning filtering with specific patterns (lines 103-110)
- Markdown code block edge cases (lines 227-234)
- Truncated JSON recovery branches (lines 244-260)
- Rejected video logging path (lines 271-272)
- Config max_tokens access (lines 348-349)
- Anthropic provider path (line 222)
- No llm_title_filter config attribute (line 161)

Created: 2026-01-11 (Coverage improvement session)
"""

import json
import subprocess
from unittest.mock import Mock, patch
import pytest

from src.downloader.title_filter import TitleFilter


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config with all fields"""
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
    config.llm.max_tokens = 4000
    return config


@pytest.fixture
def mock_config_no_llm():
    """Config without llm attribute"""
    config = Mock(spec=['download'])
    config.download = Mock()
    config.download.search_timeout = 60
    config.download.llm_title_filter = Mock()
    config.download.llm_title_filter.enabled = True
    config.download.llm_title_filter.provider = 'anthropic'
    config.download.llm_title_filter.model = 'claude-3-haiku-20240307'
    config.download.llm_title_filter.min_relevance = 0.7
    config.download.llm_title_filter.batch_size = 20
    # No config.llm attribute - should use default max_tokens
    return config


@pytest.fixture
def mock_config_no_filter():
    """Config without llm_title_filter attribute"""
    config = Mock()
    config.download = Mock(spec=['search_timeout'])
    config.download.search_timeout = 60
    # No llm_title_filter attribute
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
    """Sample video metadata for testing"""
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
# Test Non-Zero Return Code With Stderr (Lines 97-100)
# ============================================================================

class TestNonZeroReturnCodeWithStderr:
    """Test non-zero return code scenarios with stderr content"""

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_nonzero_return_with_short_stderr(self, mock_run, title_filter):
        """Test non-zero return code with short stderr message (line 100)"""
        # Short stderr that fits within 200 char limit
        mock_run.return_value = Mock(
            returncode=1,
            stdout='',
            stderr='ERROR: Video unavailable'
        )

        videos = title_filter.search_video_metadata('test', 'medium')

        assert videos == []

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_nonzero_return_with_long_stderr(self, mock_run, title_filter):
        """Test non-zero return code with stderr exceeding 200 chars (line 100 truncation)"""
        # Long stderr that will be truncated
        long_error = 'E' * 300
        mock_run.return_value = Mock(
            returncode=1,
            stdout='',
            stderr=long_error
        )

        videos = title_filter.search_video_metadata('test', 'medium')

        assert videos == []

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_nonzero_return_no_stderr(self, mock_run, title_filter):
        """Test non-zero return code with empty stderr (line 99 not taken)"""
        mock_run.return_value = Mock(
            returncode=1,
            stdout='',
            stderr=''
        )

        videos = title_filter.search_video_metadata('test', 'medium')

        assert videos == []


# ============================================================================
# Test Stderr Warning Filtering (Lines 103-110)
# ============================================================================

class TestStderrWarningFiltering:
    """Test filtering of stderr warnings and errors"""

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_stderr_with_download_progress(self, mock_run, title_filter):
        """Test stderr with download progress lines (filtered out)"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps({'id': 'abc', 'title': 'Test', 'duration': 120}),
            stderr='[download] Downloading video\n[youtube] Extracting info'
        )

        videos = title_filter.search_video_metadata('test', 'medium')

        # Should parse successfully, progress lines filtered
        assert len(videos) == 1

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_stderr_with_warning_keyword(self, mock_run, title_filter):
        """Test stderr with WARNING keyword (logged)"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps({'id': 'abc', 'title': 'Test', 'duration': 120}),
            stderr='WARNING: Age-restricted video'
        )

        videos = title_filter.search_video_metadata('test', 'medium')

        assert len(videos) == 1

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_stderr_with_error_keyword(self, mock_run, title_filter):
        """Test stderr with ERROR keyword (logged)"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps({'id': 'abc', 'title': 'Test', 'duration': 120}),
            stderr='ERROR: Unable to extract video data'
        )

        videos = title_filter.search_video_metadata('test', 'medium')

        assert len(videos) == 1

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_stderr_mixed_content(self, mock_run, title_filter):
        """Test stderr with mixed content (progress + warnings)"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps({'id': 'abc', 'title': 'Test', 'duration': 120}),
            stderr='[download] 100%\n[youtube] info\nWARNING: Something\nRegular line\nERROR: Issue'
        )

        videos = title_filter.search_video_metadata('test', 'medium')

        # Only WARNING and ERROR lines are logged (line 109)
        assert len(videos) == 1

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_stderr_no_warnings_or_errors(self, mock_run, title_filter):
        """Test stderr with no warnings/errors (lines 109 not logged)"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps({'id': 'abc', 'title': 'Test', 'duration': 120}),
            stderr='[download] 100%\n[youtube] Extracting info\nSome regular line'
        )

        videos = title_filter.search_video_metadata('test', 'medium')

        # No warnings/errors logged, should still parse
        assert len(videos) == 1

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_stderr_empty_lines(self, mock_run, title_filter):
        """Test stderr with empty lines"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps({'id': 'abc', 'title': 'Test', 'duration': 120}),
            stderr='\n\n\n'
        )

        videos = title_filter.search_video_metadata('test', 'medium')

        assert len(videos) == 1


# ============================================================================
# Test Markdown Code Block Edge Cases (Lines 227-234)
# ============================================================================

class TestMarkdownCodeBlockRemoval:
    """Test various markdown code block patterns"""

    @pytest.mark.fast
    def test_code_block_json_label(self, title_filter, sample_videos):
        """Test removal of ```json code blocks"""
        llm_response = '```json\n[{"index": 1, "approve": true, "relevance": 0.8}]\n```'
        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos[:1], 'beach')

        assert len(result) == 1

    @pytest.mark.fast
    def test_code_block_no_label(self, title_filter, sample_videos):
        """Test removal of ``` code blocks without language label"""
        llm_response = '```\n[{"index": 1, "approve": true, "relevance": 0.8}]\n```'
        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos[:1], 'beach')

        assert len(result) == 1

    @pytest.mark.fast
    def test_code_block_with_trailing_content(self, title_filter, sample_videos):
        """Test code block with extra trailing content after closing```"""
        llm_response = '```json\n[{"index": 1, "approve": true, "relevance": 0.8}]\n```\nSome extra text'
        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos[:1], 'beach')

        # Should still parse the JSON inside the code block
        assert len(result) == 1

    @pytest.mark.fast
    def test_code_block_missing_closing(self, title_filter, sample_videos):
        """Test code block without closing ``` (lines 232-233)"""
        llm_response = '```json\n[{"index": 1, "approve": true, "relevance": 0.8}]'
        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos[:1], 'beach')

        assert len(result) == 1

    @pytest.mark.fast
    def test_nested_code_blocks(self, title_filter, sample_videos):
        """Test multiple code block markers"""
        llm_response = '```\n```json\n[{"index": 1, "approve": true, "relevance": 0.8}]\n```'
        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos[:1], 'beach')

        assert len(result) == 1


# ============================================================================
# Test Truncated JSON Recovery (Lines 244-260)
# ============================================================================

class TestTruncatedJSONRecovery:
    """Test recovery from truncated/malformed JSON responses"""

    @pytest.mark.fast
    def test_truncated_json_single_object(self, title_filter, sample_videos):
        """Test truncated JSON with single object missing bracket (lines 247-251)"""
        # Missing closing ] bracket
        llm_response = '[{"index": 1, "approve": true, "relevance": 0.8, "reason": "Good"}'
        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos[:1], 'beach')

        assert len(result) == 1

    @pytest.mark.fast
    def test_truncated_json_multiple_objects(self, title_filter, sample_videos):
        """Test truncated JSON with multiple objects - fails because no closing ]"""
        # Truncated after first complete object - regex won't find array without ]
        llm_response = '[{"index": 1, "approve": true, "relevance": 0.9}, {"index": 2, "approve": tr'
        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos, 'beach')

        # Regex requires both [ and ] - truncated response fails open
        assert len(result) == 3
        assert all(v['llm_relevance'] == 0.5 for v in result)

    @pytest.mark.fast
    def test_truncated_json_no_complete_object(self, title_filter, sample_videos):
        """Test truncated JSON with no complete objects (lines 257-258 raise)"""
        # No closing brace at all
        llm_response = '[{"index": 1, "approve": true'
        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos, 'beach')

        # Should fail open - approve all with default relevance
        assert len(result) == 3
        assert all(v['llm_relevance'] == 0.5 for v in result)

    @pytest.mark.fast
    def test_json_ends_with_bracket_but_invalid(self, title_filter, sample_videos):
        """Test JSON that ends with ] but is still invalid (lines 259-260)"""
        # Ends with ] but content is malformed
        llm_response = '[invalid json content]'
        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos, 'beach')

        # Should fail open
        assert len(result) == 3
        assert all(v['llm_relevance'] == 0.5 for v in result)

    @pytest.mark.fast
    def test_nested_truncation_with_bracket_recovery(self, title_filter, sample_videos):
        """Test recovery from truncation where we have the closing bracket"""
        # This has a closing ] but invalid JSON - triggers truncation recovery
        llm_response = '[{"index": 1, "approve": true, "relevance": 0.8, "reason": "Documentary"}]extra'
        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos, 'beach')

        # Should recover - regex extracts valid JSON array
        assert len(result) == 1

    @pytest.mark.fast
    def test_truncated_json_with_spaces(self, title_filter, sample_videos):
        """Test truncated JSON with trailing whitespace"""
        llm_response = '[{"index": 1, "approve": true, "relevance": 0.8}   '
        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos[:1], 'beach')

        assert len(result) == 1


# ============================================================================
# Test Rejected Video Logging (Lines 271-272)
# ============================================================================

class TestRejectedVideoLogging:
    """Test logging of rejected videos"""

    @pytest.mark.fast
    def test_rejected_video_with_reason(self, title_filter, sample_videos):
        """Test rejected video logs with reason (line 272)"""
        llm_response = json.dumps([
            {"index": 1, "approve": True, "relevance": 0.9, "reason": "Documentary"},
            {"index": 2, "approve": False, "relevance": 0.0, "reason": "Music video"},
            {"index": 3, "approve": False, "relevance": 0.0, "reason": "Live stream"}
        ])
        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos, 'beach')

        # Only first video approved
        assert len(result) == 1
        assert result[0]['title'] == 'Amazing Beach Documentary'

    @pytest.mark.fast
    def test_rejected_video_no_reason(self, title_filter, sample_videos):
        """Test rejected video without reason (uses default 'No reason')"""
        llm_response = json.dumps([
            {"index": 1, "approve": True, "relevance": 0.9},
            {"index": 2, "approve": False},  # No reason provided
            {"index": 3, "approve": False}
        ])
        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos, 'beach')

        assert len(result) == 1

    @pytest.mark.fast
    def test_all_videos_rejected(self, title_filter, sample_videos):
        """Test when all videos are rejected"""
        llm_response = json.dumps([
            {"index": 1, "approve": False, "relevance": 0.0, "reason": "Music video"},
            {"index": 2, "approve": False, "relevance": 0.0, "reason": "Gaming"},
            {"index": 3, "approve": False, "relevance": 0.0, "reason": "Live stream"}
        ])
        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos, 'beach')

        assert len(result) == 0


# ============================================================================
# Test Anthropic Provider Path (Line 222)
# ============================================================================

class TestAnthropicProviderPath:
    """Test Anthropic provider selection in filter_titles_with_llm"""

    @pytest.mark.fast
    def test_anthropic_provider_selection(self, mock_config, get_tier_value_func, sample_videos):
        """Test that Anthropic provider calls _call_anthropic (line 222)"""
        mock_config.download.llm_title_filter.provider = 'anthropic'
        mock_config.download.llm_title_filter.model = 'claude-3-haiku-20240307'
        filter = TitleFilter(mock_config, [], get_tier_value_func)

        llm_response = json.dumps([
            {"index": 1, "approve": True, "relevance": 0.9, "reason": "Documentary"}
        ])
        filter._call_anthropic = Mock(return_value=llm_response)
        filter._call_gemini = Mock()  # Should NOT be called

        result = filter.filter_titles_with_llm(sample_videos[:1], 'beach')

        filter._call_anthropic.assert_called_once()
        filter._call_gemini.assert_not_called()
        assert len(result) == 1


# ============================================================================
# Test Config Max Tokens (Lines 348-349)
# ============================================================================

class TestConfigMaxTokens:
    """Test max_tokens config access in _call_anthropic"""

    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'})
    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_anthropic_uses_config_max_tokens(self, mock_create_client, mock_config, get_tier_value_func):
        """Test that max_tokens is read from config.llm (lines 348-349)"""
        mock_config.llm.max_tokens = 5000
        filter = TitleFilter(mock_config, [], get_tier_value_func)

        mock_response = Mock()
        mock_response.text = '[]'
        mock_client = Mock()
        mock_client.generate.return_value = mock_response
        mock_create_client.return_value = mock_client

        filter._call_anthropic('test prompt', 'claude-3-haiku')

        # Verify max_tokens was passed to request
        call_args = mock_client.generate.call_args
        request = call_args[0][0]  # First positional arg is the request
        assert request.max_tokens == 5000

    @patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test_key'})
    @patch('src.llm_client.create_client')
    @pytest.mark.fast
    def test_anthropic_default_max_tokens(self, mock_create_client, mock_config_no_llm, get_tier_value_func):
        """Test default max_tokens when config.llm not present"""
        filter = TitleFilter(mock_config_no_llm, [], get_tier_value_func)

        mock_response = Mock()
        mock_response.text = '[]'
        mock_client = Mock()
        mock_client.generate.return_value = mock_response
        mock_create_client.return_value = mock_client

        filter._call_anthropic('test prompt', 'claude-3-haiku')

        # Should use default 2000
        call_args = mock_client.generate.call_args
        request = call_args[0][0]
        assert request.max_tokens == 2000


# ============================================================================
# Test No LLM Filter Config (Line 161)
# ============================================================================

class TestNoLLMFilterConfig:
    """Test when llm_title_filter config is missing"""

    @pytest.mark.fast
    def test_no_llm_filter_attribute(self, mock_config_no_filter, get_tier_value_func, sample_videos):
        """Test filter_titles_with_llm when llm_title_filter attribute missing (line 161)"""
        filter = TitleFilter(mock_config_no_filter, [], get_tier_value_func)

        result = filter.filter_titles_with_llm(sample_videos, 'beach')

        # Should return all videos unchanged when config missing
        assert result == sample_videos

    @pytest.mark.fast
    def test_llm_filter_none(self, mock_config, get_tier_value_func, sample_videos):
        """Test when llm_title_filter is None"""
        mock_config.download.llm_title_filter = None
        filter = TitleFilter(mock_config, [], get_tier_value_func)

        result = filter.filter_titles_with_llm(sample_videos, 'beach')

        # Should return all videos unchanged
        assert result == sample_videos


# ============================================================================
# Test No JSON Array Found (Lines 237-239)
# ============================================================================

class TestNoJSONArrayFound:
    """Test when LLM response contains no JSON array"""

    @pytest.mark.fast
    def test_no_json_array_in_response(self, title_filter, sample_videos):
        """Test handling when no JSON array is found (line 238-239)"""
        llm_response = 'I cannot process this request due to policy restrictions.'
        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos, 'beach')

        # Should fail open
        assert len(result) == 3
        assert all(v['llm_relevance'] == 0.5 for v in result)

    @pytest.mark.fast
    def test_empty_response(self, title_filter, sample_videos):
        """Test handling empty LLM response"""
        llm_response = ''
        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos, 'beach')

        # Should fail open
        assert len(result) == 3

    @pytest.mark.fast
    def test_response_with_object_not_array(self, title_filter, sample_videos):
        """Test response with JSON object instead of array"""
        llm_response = '{"error": "Invalid request"}'
        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos, 'beach')

        # No array found, should fail open
        assert len(result) == 3


# ============================================================================
# Test Approved Video Without Index (Line 263)
# ============================================================================

class TestApprovedVideoWithoutIndex:
    """Test approved video with missing or invalid index"""

    @pytest.mark.fast
    def test_zero_index(self, title_filter, sample_videos):
        """Test handling index of 0 (converts to -1, out of range)"""
        llm_response = json.dumps([
            {"index": 0, "approve": True, "relevance": 0.9, "reason": "Test"},
            {"index": 1, "approve": True, "relevance": 0.8, "reason": "Valid"}
        ])
        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos, 'beach')

        # Index 0 becomes -1, out of range; only index 1 (0 in list) is valid
        assert len(result) == 1
        assert result[0]['title'] == 'Amazing Beach Documentary'

    @pytest.mark.fast
    def test_negative_index(self, title_filter, sample_videos):
        """Test handling negative index"""
        llm_response = json.dumps([
            {"index": -1, "approve": True, "relevance": 0.9, "reason": "Test"},
            {"index": 1, "approve": True, "relevance": 0.8, "reason": "Valid"}
        ])
        title_filter._call_gemini = Mock(return_value=llm_response)

        result = title_filter.filter_titles_with_llm(sample_videos, 'beach')

        # -1 - 1 = -2, out of range
        assert len(result) == 1


# ============================================================================
# Test Search Timeout with Cookie Detection (Lines 134-135)
# ============================================================================

class TestSearchTimeoutWithCookies:
    """Test timeout diagnostics including cookie detection"""

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_timeout_with_cookies_present(self, mock_run, title_filter):
        """Test timeout logs cookie presence (line 135)"""
        mock_run.side_effect = subprocess.TimeoutExpired(['yt-dlp'], 60)

        videos = title_filter.search_video_metadata('test', 'medium')

        assert videos == []

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_timeout_without_cookies(self, mock_run, mock_config, get_tier_value_func):
        """Test timeout logging when no cookies present"""
        filter = TitleFilter(mock_config, [], get_tier_value_func)  # Empty cookies
        mock_run.side_effect = subprocess.TimeoutExpired(['yt-dlp'], 60)

        videos = filter.search_video_metadata('test', 'medium')

        assert videos == []


# ============================================================================
# Test Gemini With GOOGLE_API_KEY (Line 304)
# ============================================================================

class TestGeminiWithGoogleAPIKey:
    """Test Gemini API key fallback to GOOGLE_API_KEY"""

    @patch.dict('os.environ', {'GOOGLE_API_KEY': 'google_key'}, clear=True)
    @patch('src.llm_client.create_client')
    @pytest.mark.requires_api
    def test_uses_google_api_key_fallback(self, mock_create_client, title_filter):
        """Test that GOOGLE_API_KEY is used when GEMINI_API_KEY not set"""
        mock_response = Mock()
        mock_response.text = '[]'
        mock_client = Mock()
        mock_client.generate.return_value = mock_response
        mock_create_client.return_value = mock_client

        result = title_filter._call_gemini('test', 'gemini-2.0-flash')

        mock_create_client.assert_called_once_with(
            "gemini",
            api_key='google_key',
            model='gemini-2.0-flash'
        )

    @patch.dict('os.environ', {'GEMINI_API_KEY': 'gemini_key', 'GOOGLE_API_KEY': 'google_key'}, clear=True)
    @patch('src.llm_client.create_client')
    @pytest.mark.requires_api
    def test_prefers_gemini_api_key(self, mock_create_client, title_filter):
        """Test that GEMINI_API_KEY takes precedence over GOOGLE_API_KEY"""
        mock_response = Mock()
        mock_response.text = '[]'
        mock_client = Mock()
        mock_client.generate.return_value = mock_response
        mock_create_client.return_value = mock_client

        result = title_filter._call_gemini('test', 'gemini-2.0-flash')

        # Should use GOOGLE_API_KEY (checked first in or expression)
        mock_create_client.assert_called_once_with(
            "gemini",
            api_key='google_key',  # GOOGLE_API_KEY is checked first
            model='gemini-2.0-flash'
        )


# ============================================================================
# Test Empty stdout Handling (Line 113)
# ============================================================================

class TestEmptyStdoutHandling:
    """Test handling of empty stdout from yt-dlp"""

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_empty_stdout(self, mock_run, title_filter):
        """Test handling when stdout is empty"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout='',
            stderr=''
        )

        videos = title_filter.search_video_metadata('test', 'medium')

        assert videos == []

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_none_stdout(self, mock_run, title_filter):
        """Test handling when stdout is None"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout=None,
            stderr=''
        )

        videos = title_filter.search_video_metadata('test', 'medium')

        assert videos == []

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_whitespace_only_stdout(self, mock_run, title_filter):
        """Test handling when stdout is only whitespace"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout='   \n   \n   ',
            stderr=''
        )

        videos = title_filter.search_video_metadata('test', 'medium')

        assert videos == []


# ============================================================================
# Test Search Timeout Configuration (Line 83)
# ============================================================================

class TestSearchTimeoutConfiguration:
    """Test search timeout configuration handling"""

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_uses_config_timeout(self, mock_run, mock_config, get_tier_value_func):
        """Test that configured timeout is used"""
        mock_config.download.search_timeout = 120
        filter = TitleFilter(mock_config, [], get_tier_value_func)

        mock_run.return_value = Mock(returncode=0, stdout='', stderr='')

        filter.search_video_metadata('test', 'medium')

        # Verify timeout parameter
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs['timeout'] == 120

    @patch('subprocess.run')
    @pytest.mark.fast
    def test_default_timeout_when_not_configured(self, mock_run, mock_config, get_tier_value_func):
        """Test default timeout when not in config"""
        # Remove search_timeout attribute
        del mock_config.download.search_timeout
        filter = TitleFilter(mock_config, [], get_tier_value_func)

        mock_run.return_value = Mock(returncode=0, stdout='', stderr='')

        filter.search_video_metadata('test', 'medium')

        # Should use default 60
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs['timeout'] == 60
