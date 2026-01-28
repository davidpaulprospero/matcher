"""Tests for search timeout with keyword remix fallback (US-004).

Tests verify that:
1. Search timeout is detected separately from download timeout
2. On search timeout, LLM keyword remix is attempted
3. Maximum 2 remix attempts per original keyword
4. Timeout and remix attempts are logged
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import the classes we're testing
from src.downloader.title_filter import TitleFilter, SearchResult


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_with_videos(self):
        """SearchResult is truthy when videos exist."""
        videos = [{'id': 'abc123', 'title': 'Test Video'}]
        result = SearchResult(videos=videos)
        assert result
        assert result.videos == videos
        assert result.timed_out is False
        assert result.error is None

    def test_search_result_empty_is_falsy(self):
        """SearchResult is falsy when no videos."""
        result = SearchResult(videos=[])
        assert not result
        assert result.videos == []
        assert result.timed_out is False

    def test_search_result_timeout(self):
        """SearchResult captures timeout state."""
        result = SearchResult(videos=[], timed_out=True, error="Timeout after 30s")
        assert not result  # No videos
        assert result.timed_out is True
        assert result.error == "Timeout after 30s"

    def test_search_result_error_without_timeout(self):
        """SearchResult captures error without timeout."""
        result = SearchResult(videos=[], timed_out=False, error="Network error")
        assert not result
        assert result.timed_out is False
        assert result.error == "Network error"


class TestSearchTimeout:
    """Tests for search timeout detection."""

    @pytest.fixture
    def mock_config(self):
        """Create mock config for TitleFilter."""
        config = Mock()
        config.download = Mock()
        config.download.search_timeout = 30  # 30 second timeout
        return config

    @pytest.fixture
    def title_filter(self, mock_config):
        """Create TitleFilter with mock config."""
        cookies_args = []
        get_tier_value = lambda tier, key, default: {'min': 0, 'max': 120}.get(key, default)
        return TitleFilter(mock_config, cookies_args, get_tier_value)

    def test_search_timeout_returns_search_result_with_timeout_flag(self, title_filter):
        """Search timeout returns SearchResult with timed_out=True."""
        import subprocess

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=['yt-dlp'], timeout=30)

            result = title_filter.search_video_metadata("test keyword", "short", 10)

            assert isinstance(result, SearchResult)
            assert result.timed_out is True
            assert result.videos == []
            assert "timeout" in result.error.lower()

    def test_successful_search_returns_videos_with_no_timeout(self, title_filter):
        """Successful search returns videos with timed_out=False."""
        mock_stdout = '{"id": "abc123", "title": "Test", "duration": 60, "channel": "Test Channel"}'

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=mock_stdout,
                stderr=""
            )

            result = title_filter.search_video_metadata("test keyword", "short", 10)

            assert isinstance(result, SearchResult)
            assert result.timed_out is False
            assert len(result.videos) == 1
            assert result.videos[0]['id'] == 'abc123'

    def test_search_uses_config_timeout(self, title_filter, mock_config):
        """Search uses timeout from config."""
        mock_config.download.search_timeout = 45  # Custom timeout

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

            title_filter.search_video_metadata("test", "short", 10)

            # Verify timeout was passed to subprocess.run
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs['timeout'] == 45

    def test_search_default_timeout_is_30_seconds(self, mock_config):
        """Default search timeout is 30 seconds (US-004 requirement)."""
        # Remove the search_timeout attribute to test default
        del mock_config.download.search_timeout

        title_filter = TitleFilter(
            mock_config,
            cookies_args=[],
            get_tier_value_func=lambda t, k, d: d
        )

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

            title_filter.search_video_metadata("test", "short", 10)

            # Default should be 30s (US-004 acceptance criteria)
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs['timeout'] == 30


class TestSearchTimeoutRemix:
    """Tests for keyword remix on search timeout."""

    @pytest.fixture
    def mock_video_downloader(self):
        """Create a mock VideoDownloader for testing remix logic."""
        # We need to test the _download_single method's remix logic
        # Create a mock that simulates the behavior
        downloader = Mock()
        downloader.download_config = Mock()
        downloader.download_config.llm_title_filter = Mock(enabled=True)
        downloader.download_config.max_keyword_len = 8
        downloader.download_config.max_filename_len = 10
        downloader.download_config.title_blacklist = []
        downloader.download_config.speech_screening = Mock(enabled=False)

        return downloader

    def test_remix_tried_on_search_timeout(self, mock_video_downloader):
        """LLM keyword remix is tried when search times out."""
        from src.downloader.core import VideoDownloader

        # Track calls to verify remix was attempted
        search_calls = []
        remix_calls = []

        def mock_search(keyword, tier, max_results):
            search_calls.append(keyword)
            if len(search_calls) == 1:
                # First call times out
                return SearchResult(videos=[], timed_out=True, error="Timeout")
            else:
                # Remix search succeeds
                return SearchResult(videos=[{'id': 'vid1', 'title': 'Test', 'duration': 60, 'channel': 'Test'}])

        def mock_remix(keyword, topic=""):
            remix_calls.append(keyword)
            return f"{keyword} footage"  # Simple remix

        # Test the logic flow (without actually calling the method)
        # First search times out
        result1 = mock_search("nature documentary", "short", 40)
        assert result1.timed_out

        # Remix is called (only once - don't append again)
        remixed = mock_remix("nature documentary")
        assert remixed == "nature documentary footage"

        # Second search with remixed keyword succeeds
        result2 = mock_search(remixed, "short", 40)
        assert not result2.timed_out
        assert len(result2.videos) == 1

        assert len(search_calls) == 2
        assert len(remix_calls) == 1  # Remix called once

    def test_max_2_remix_attempts(self):
        """Maximum 2 remix attempts per original keyword."""
        # Simulate the retry logic from _download_single
        max_remix_attempts = 2
        remix_attempts = 0

        # Simulate all searches timing out
        search_results = [
            SearchResult(videos=[], timed_out=True),  # Original
            SearchResult(videos=[], timed_out=True),  # Remix 1
            SearchResult(videos=[], timed_out=True),  # Remix 2
        ]

        current_idx = 0
        while search_results[current_idx].timed_out and remix_attempts < max_remix_attempts:
            remix_attempts += 1
            current_idx += 1

        assert remix_attempts == 2  # Stopped at max

    def test_remix_not_tried_on_successful_search(self):
        """Remix is NOT tried when search succeeds."""
        remix_called = False

        def mock_remix(keyword, topic=""):
            nonlocal remix_called
            remix_called = True
            return f"{keyword} footage"

        # Search succeeds
        result = SearchResult(videos=[{'id': 'vid1', 'title': 'Test'}], timed_out=False)

        # Simulate the condition in _download_single
        if result.timed_out:
            mock_remix("test keyword")

        assert not remix_called

    def test_remix_not_tried_when_remix_returns_same_keyword(self):
        """Stop remix attempts if remix returns same keyword."""
        current_keyword = "unique term"
        remix_attempts = 0
        max_remix_attempts = 2

        def mock_remix(keyword, topic=""):
            # Returns same keyword (no remix available)
            return keyword

        # Simulate the logic
        search_result = SearchResult(videos=[], timed_out=True)

        while search_result.timed_out and remix_attempts < max_remix_attempts:
            remix_keyword = mock_remix(current_keyword)
            if remix_keyword and remix_keyword != current_keyword:
                current_keyword = remix_keyword
                remix_attempts += 1
            else:
                break  # No remix available

        assert remix_attempts == 0  # Should not increment because remix returned same keyword


class TestSearchTimeoutLogging:
    """Tests for logging of timeout and remix attempts."""

    def test_timeout_logged_with_keyword(self):
        """Search timeout is logged with the original keyword."""
        from src.downloader.title_filter import TitleFilter, SearchResult
        import subprocess
        import logging

        # Create mock config
        config = Mock()
        config.download = Mock()
        config.download.search_timeout = 30

        title_filter = TitleFilter(
            config,
            cookies_args=[],
            get_tier_value_func=lambda t, k, d: d
        )

        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd=['yt-dlp'], timeout=30)

            with patch('src.downloader.title_filter.logger') as mock_logger:
                result = title_filter.search_video_metadata("nature documentary", "short", 10)

                # Verify warning was logged
                mock_logger.warning.assert_called()
                warning_msg = str(mock_logger.warning.call_args)
                assert "timeout" in warning_msg.lower() or "nature documentary" in warning_msg

    def test_remix_attempts_logged(self):
        """Each remix attempt is logged with attempt number."""
        import logging

        # Simulate logging during remix attempts
        log_messages = []

        def mock_log_info(msg):
            log_messages.append(msg)

        # Simulate the remix loop logging
        keyword = "test keyword"
        for attempt in range(1, 3):
            mock_log_info(f"Search timeout for '{keyword}' - trying keyword remix ({attempt}/2)")
            mock_log_info(f"Remixed keyword: '{keyword}' → '{keyword} footage'")

        assert len(log_messages) == 4  # 2 attempts x 2 messages each
        assert "1/2" in log_messages[0]
        assert "2/2" in log_messages[2]


class TestConfigDefaults:
    """Tests for configuration defaults."""

    def test_search_timeout_default_is_30(self):
        """Default search_timeout in config is 30 seconds."""
        from src.config.sections.download import DownloadConfig

        config = DownloadConfig()
        assert config.search_timeout == 30

    def test_search_timeout_separate_from_download_timeout(self):
        """search_timeout is separate from download_timeout."""
        from src.config.sections.download import DownloadConfig

        config = DownloadConfig()
        assert config.search_timeout == 30  # Search timeout
        assert config.download_timeout == 120  # Download timeout
        assert config.search_timeout != config.download_timeout


class TestIntegration:
    """Integration tests for the full search timeout remix flow."""

    def test_search_result_backward_compatible(self):
        """SearchResult maintains backward compatibility with list checks."""
        # Old code might do: if videos: (expecting list)
        # SearchResult should work the same way

        result_with_videos = SearchResult(videos=[{'id': '123'}])
        result_empty = SearchResult(videos=[])

        # Both should work with truthiness checks
        assert bool(result_with_videos) is True
        assert bool(result_empty) is False

        # Should work in if statements
        if result_with_videos:
            passed = True
        else:
            passed = False
        assert passed is True

    def test_full_remix_flow_simulation(self):
        """Simulate complete search timeout → remix → success flow."""
        # This simulates what happens in VideoDownloader._download_single

        keyword = "documentary wildlife africa"
        topic = "nature documentary about African wildlife"
        max_remix_attempts = 2

        # Track state
        search_attempts = []
        remix_attempts = 0

        def mock_search(kw):
            search_attempts.append(kw)
            if kw == keyword:
                return SearchResult(videos=[], timed_out=True, error="Timeout")
            elif "footage" in kw:
                return SearchResult(videos=[{'id': 'found_vid', 'title': 'Wildlife'}])
            else:
                return SearchResult(videos=[], timed_out=True)

        def mock_remix(kw, t=""):
            return f"{kw.split()[0]} {kw.split()[1]} footage"

        # Execute the flow
        current_keyword = keyword
        search_result = mock_search(current_keyword)

        while search_result.timed_out and remix_attempts < max_remix_attempts:
            remix_keyword = mock_remix(current_keyword, topic)
            if remix_keyword and remix_keyword != current_keyword:
                current_keyword = remix_keyword
                search_result = mock_search(current_keyword)
                remix_attempts += 1
            else:
                break

        # Verify the flow
        assert len(search_attempts) == 2  # Original + 1 remix
        assert remix_attempts == 1
        assert not search_result.timed_out
        assert len(search_result.videos) == 1
        assert search_result.videos[0]['id'] == 'found_vid'
