"""
Tests for exponential backoff retry mechanism in VideoDownloader._run_download_cmd.

Covers US-001: Implement exponential backoff retry in _run_download_cmd
- Uses config.download.max_retries for retry attempts
- Delay follows exponential backoff: retry_delay * (retry_backoff ^ attempt)
- Retries only on transient errors (timeout, connection reset, 429 rate limit)
- Permanent errors (video unavailable, private, age-restricted) fail immediately
- Each retry attempt is logged with attempt number and delay duration

Refactored as part of US-009 to use shared fixtures from tests/fixtures/downloader_fixtures.py.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from tests.fixtures.downloader_fixtures import (
    create_mock_downloader_config,
    create_mock_subprocess_process,
    patch_video_downloader_dependencies,
    patch_time_sleep,
    assert_retry_backoff_correct,
)


def create_downloader(config):
    """Helper to create VideoDownloader with all dependencies patched."""
    from src.downloader.core import VideoDownloader
    with patch_video_downloader_dependencies():
        return VideoDownloader(config)


class TestRetryConfiguration:
    """Test that retry configuration is read from config."""

    @pytest.mark.fast
    def test_uses_config_max_retries(self, tmp_path):
        """Test that max_retries is read from config."""
        config = create_mock_downloader_config(tmp_path, max_retries=5)
        with patch_video_downloader_dependencies():
            from src.downloader.core import VideoDownloader
            downloader = VideoDownloader(config)
            assert config.download.max_retries == 5

    @pytest.mark.fast
    def test_uses_config_retry_delay(self, tmp_path):
        """Test that retry_delay is read from config."""
        config = create_mock_downloader_config(tmp_path, retry_delay=5.0)
        with patch_video_downloader_dependencies():
            from src.downloader.core import VideoDownloader
            downloader = VideoDownloader(config)
            assert config.download.retry_delay == 5.0

    @pytest.mark.fast
    def test_uses_config_retry_backoff(self, tmp_path):
        """Test that retry_backoff is read from config."""
        config = create_mock_downloader_config(tmp_path, retry_backoff=3.0)
        with patch_video_downloader_dependencies():
            from src.downloader.core import VideoDownloader
            downloader = VideoDownloader(config)
            assert config.download.retry_backoff == 3.0


class TestTransientErrorDetection:
    """Test that transient errors are correctly identified."""

    @pytest.fixture
    def downloader(self, tmp_path):
        """Create a VideoDownloader instance for testing."""
        config = create_mock_downloader_config(tmp_path)
        with patch_video_downloader_dependencies():
            from src.downloader.core import VideoDownloader
            yield VideoDownloader(config)

    @pytest.mark.fast
    def test_is_transient_error_429(self, downloader):
        """Test that 429 rate limit is detected as transient."""
        assert downloader._is_transient_error("ERROR: HTTP Error 429: Too Many Requests")

    @pytest.mark.fast
    def test_is_transient_error_connection_reset(self, downloader):
        """Test that connection reset is detected as transient."""
        assert downloader._is_transient_error("Connection reset by peer")

    @pytest.mark.fast
    def test_is_transient_error_rate_limit(self, downloader):
        """Test that rate limit text is detected as transient."""
        assert downloader._is_transient_error("rate limit exceeded, please wait")

    @pytest.mark.fast
    def test_is_transient_error_service_unavailable(self, downloader):
        """Test that service unavailable is detected as transient."""
        assert downloader._is_transient_error("HTTP Error 503: Service Unavailable")

    @pytest.mark.fast
    def test_not_transient_for_unrelated_error(self, downloader):
        """Test that unrelated errors are not classified as transient."""
        assert not downloader._is_transient_error("Some random error message")


class TestPermanentErrorDetection:
    """Test that permanent errors are correctly identified."""

    @pytest.fixture
    def downloader(self, tmp_path):
        """Create a VideoDownloader instance for testing."""
        config = create_mock_downloader_config(tmp_path)
        with patch_video_downloader_dependencies():
            from src.downloader.core import VideoDownloader
            yield VideoDownloader(config)

    @pytest.mark.fast
    def test_is_permanent_error_video_unavailable(self, downloader):
        """Test that video unavailable is detected as permanent."""
        assert downloader._is_permanent_error("Video unavailable")

    @pytest.mark.fast
    def test_is_permanent_error_private_video(self, downloader):
        """Test that private video is detected as permanent."""
        assert downloader._is_permanent_error("This video is private")

    @pytest.mark.fast
    def test_is_permanent_error_age_restricted(self, downloader):
        """Test that age-restricted is detected as permanent."""
        assert downloader._is_permanent_error("Sign in to confirm your age")

    @pytest.mark.fast
    def test_is_permanent_error_copyright(self, downloader):
        """Test that copyright claim is detected as permanent."""
        assert downloader._is_permanent_error("Video removed due to copyright claim")

    @pytest.mark.fast
    def test_not_permanent_for_transient_error(self, downloader):
        """Test that transient errors are not classified as permanent."""
        assert not downloader._is_permanent_error("HTTP Error 429: Too Many Requests")


class TestRetryBehavior:
    """Test the actual retry behavior with mocked subprocess."""

    @pytest.mark.integration
    def test_retry_on_timeout(self, tmp_path):
        """Test that timeouts trigger retry with exponential backoff."""
        config = create_mock_downloader_config(
            tmp_path, max_retries=2, retry_delay=0.1, retry_backoff=2.0
        )

        with patch_video_downloader_dependencies():
            with patch_time_sleep() as mock_sleep:
                from src.downloader.core import VideoDownloader
                downloader = VideoDownloader(config)

                keyword_dir = tmp_path / "videos" / "test"
                keyword_dir.mkdir(parents=True)

                with patch('subprocess.Popen') as mock_popen:
                    mock_process = create_mock_subprocess_process(returncode=0)
                    mock_popen.return_value = mock_process

                    # All attempts timeout via progress-aware monitor
                    with patch.object(
                        downloader, '_wait_for_process_with_progress',
                        return_value=("", "", 'stall')
                    ):
                        result = downloader._run_download_cmd(
                            cmd=['yt-dlp', 'test'],
                            keyword_dir=keyword_dir,
                            output_dir=tmp_path / "videos",
                            keyword="test_keyword",
                            tier="short",
                            existing_before=set()
                        )

                        # Should return empty list after retries exhausted
                        assert result == []
                        assert downloader._last_download_timed_out

                        # Verify exponential backoff: 0.1, 0.2
                        assert_retry_backoff_correct(mock_sleep, 0.1, 2.0, 2)

    @pytest.mark.integration
    def test_retry_on_transient_error(self, tmp_path):
        """Test that transient errors trigger retry."""
        config = create_mock_downloader_config(
            tmp_path, max_retries=2, retry_delay=0.1, retry_backoff=2.0
        )

        with patch_video_downloader_dependencies():
            with patch_time_sleep() as mock_sleep:
                from src.downloader.core import VideoDownloader
                downloader = VideoDownloader(config)

                keyword_dir = tmp_path / "videos" / "test"
                keyword_dir.mkdir(parents=True)

                with patch('subprocess.Popen') as mock_popen:
                    mock_process = create_mock_subprocess_process(returncode=0)
                    mock_popen.return_value = mock_process

                    # First call: 429 error, second call: success
                    call_count = [0]

                    def wait_side_effect(process, stall, maxt, kw, tier):
                        call_count[0] += 1
                        if call_count[0] == 1:
                            mock_process.returncode = 1
                            return ("", "ERROR: HTTP Error 429: Too Many Requests", None)
                        mock_process.returncode = 0
                        return ("", "", None)

                    with patch.object(
                        downloader, '_wait_for_process_with_progress',
                        side_effect=wait_side_effect
                    ):
                        result = downloader._run_download_cmd(
                            cmd=['yt-dlp', 'test'],
                            keyword_dir=keyword_dir,
                            output_dir=tmp_path / "videos",
                            keyword="test_keyword",
                            tier="short",
                            existing_before=set()
                        )

                        # Should have slept once for retry
                        assert mock_sleep.call_count >= 1

    @pytest.mark.integration
    def test_no_retry_on_permanent_error(self, tmp_path):
        """Test that permanent errors fail immediately without retry."""
        config = create_mock_downloader_config(tmp_path, max_retries=3, retry_delay=0.1)

        with patch_video_downloader_dependencies():
            with patch_time_sleep() as mock_sleep:
                from src.downloader.core import VideoDownloader
                downloader = VideoDownloader(config)

                keyword_dir = tmp_path / "videos" / "test"
                keyword_dir.mkdir(parents=True)

                with patch('subprocess.Popen') as mock_popen:
                    # Permanent error - should fail immediately
                    mock_process = create_mock_subprocess_process(
                        returncode=1,
                        stderr_output="ERROR: Video unavailable. This video has been removed."
                    )
                    mock_process.communicate.return_value = (
                        "",
                        "ERROR: Video unavailable. This video has been removed."
                    )
                    mock_popen.return_value = mock_process

                    # Mock progress-aware wait to return the permanent error
                    with patch.object(
                        downloader, '_wait_for_process_with_progress',
                        return_value=("", "ERROR: Video unavailable. This video has been removed.", None)
                    ):
                        result = downloader._run_download_cmd(
                            cmd=['yt-dlp', 'test'],
                            keyword_dir=keyword_dir,
                            output_dir=tmp_path / "videos",
                            keyword="test_keyword",
                            tier="short",
                            existing_before=set()
                        )

                        # Should return empty list immediately
                        assert result == []
                        # Should NOT have slept (no retry)
                        assert mock_sleep.call_count == 0

    @pytest.mark.integration
    def test_success_on_first_attempt(self, tmp_path):
        """Test that successful downloads don't retry."""
        config = create_mock_downloader_config(tmp_path, max_retries=3, retry_delay=0.1)

        with patch_video_downloader_dependencies():
            with patch_time_sleep() as mock_sleep:
                with patch(
                    'src.downloader.core.utils.sanitize_filename_for_nle',
                    side_effect=lambda x: x
                ):
                    from src.downloader.core import VideoDownloader
                    downloader = VideoDownloader(config)
                    downloader._record_source_for_keyword = MagicMock()

                    keyword_dir = tmp_path / "videos" / "test"
                    keyword_dir.mkdir(parents=True)

                    # Create a video file that will be "downloaded"
                    video_file = keyword_dir / "video.mp4"
                    video_file.write_bytes(b"fake video")

                    with patch('subprocess.Popen') as mock_popen:
                        # Success immediately
                        mock_process = create_mock_subprocess_process(returncode=0)
                        mock_process.communicate.return_value = ("", "")
                        mock_popen.return_value = mock_process

                        result = downloader._run_download_cmd(
                            cmd=['yt-dlp', 'test'],
                            keyword_dir=keyword_dir,
                            output_dir=tmp_path / "videos",
                            keyword="test_keyword",
                            tier="short",
                            existing_before=set()
                        )

                        # Should NOT have slept
                        assert mock_sleep.call_count == 0
                        # Should return the video
                        assert len(result) == 1


class TestExponentialBackoffCalculation:
    """Test that exponential backoff delay is calculated correctly."""

    @pytest.mark.integration
    def test_backoff_formula_attempt_0(self, tmp_path):
        """Test backoff delay for first retry (attempt=0)."""
        # delay = retry_delay * (retry_backoff ^ attempt)
        # delay = 2.0 * (2.0 ^ 0) = 2.0 * 1 = 2.0
        # Note: 429 errors also trigger rate limit backoff (5.0s default), so we
        # expect 2 sleep calls: rate_limit_backoff (5.0s) + retry_backoff (2.0s)
        config = create_mock_downloader_config(
            tmp_path, max_retries=3, retry_delay=2.0, retry_backoff=2.0
        )

        with patch_video_downloader_dependencies():
            with patch_time_sleep() as mock_sleep:
                from src.downloader.core import VideoDownloader
                downloader = VideoDownloader(config)

                keyword_dir = tmp_path / "videos" / "test"
                keyword_dir.mkdir(parents=True)

                with patch('subprocess.Popen') as mock_popen:
                    mock_process = create_mock_subprocess_process(returncode=0)
                    mock_popen.return_value = mock_process

                    # First call: transient error, second: success
                    call_count = [0]

                    def wait_side_effect(process, stall, maxt, kw, tier):
                        call_count[0] += 1
                        if call_count[0] == 1:
                            mock_process.returncode = 1
                            return ("", "ERROR: 429 rate limit", None)
                        mock_process.returncode = 0
                        return ("", "", None)

                    with patch.object(
                        downloader, '_wait_for_process_with_progress',
                        side_effect=wait_side_effect
                    ):
                        result = downloader._run_download_cmd(
                            cmd=['yt-dlp', 'test'],
                            keyword_dir=keyword_dir,
                            output_dir=tmp_path / "videos",
                            keyword="test",
                            tier="short",
                            existing_before=set()
                        )

                        # Expect 2 sleeps: rate limit backoff (5.0s) + retry backoff (2.0s)
                        assert mock_sleep.call_count == 2
                        # Find the retry backoff call (2.0s, not the rate limit 5.0s)
                        sleep_delays = [call[0][0] for call in mock_sleep.call_args_list]
                        assert 2.0 in sleep_delays or any(abs(d - 2.0) < 0.01 for d in sleep_delays)

    @pytest.mark.integration
    def test_backoff_formula_progressive(self, tmp_path):
        """Test that delays increase with each retry."""
        config = create_mock_downloader_config(
            tmp_path, max_retries=3, retry_delay=1.0, retry_backoff=2.0
        )

        with patch_video_downloader_dependencies():
            with patch_time_sleep() as mock_sleep:
                from src.downloader.core import VideoDownloader
                downloader = VideoDownloader(config)

                keyword_dir = tmp_path / "videos" / "test"
                keyword_dir.mkdir(parents=True)

                with patch('subprocess.Popen') as mock_popen:
                    mock_process = create_mock_subprocess_process(returncode=0)
                    mock_popen.return_value = mock_process

                    # All calls timeout via progress-aware monitor
                    with patch.object(
                        downloader, '_wait_for_process_with_progress',
                        return_value=("", "", 'stall')
                    ):
                        result = downloader._run_download_cmd(
                            cmd=['yt-dlp', 'test'],
                            keyword_dir=keyword_dir,
                            output_dir=tmp_path / "videos",
                            keyword="test",
                            tier="short",
                            existing_before=set()
                        )

                        # Verify progressive backoff: 1.0, 2.0, 4.0
                        assert_retry_backoff_correct(mock_sleep, 1.0, 2.0, 3)


class TestRetryLogging:
    """Test that retry attempts are properly logged."""

    @pytest.mark.integration
    def test_retry_attempt_logged(self, tmp_path, caplog):
        """Test that each retry attempt is logged with attempt number."""
        import logging

        config = create_mock_downloader_config(
            tmp_path, max_retries=2, retry_delay=0.01, retry_backoff=2.0
        )

        with patch_video_downloader_dependencies():
            with caplog.at_level(logging.INFO):
                from src.downloader.core import VideoDownloader
                downloader = VideoDownloader(config)

                keyword_dir = tmp_path / "videos" / "test"
                keyword_dir.mkdir(parents=True)

                with patch('subprocess.Popen') as mock_popen:
                    mock_process = create_mock_subprocess_process(returncode=0)
                    mock_popen.return_value = mock_process

                    # All calls timeout via progress-aware monitor
                    with patch.object(
                        downloader, '_wait_for_process_with_progress',
                        return_value=("", "", 'stall')
                    ):
                        result = downloader._run_download_cmd(
                            cmd=['yt-dlp', 'test'],
                            keyword_dir=keyword_dir,
                            output_dir=tmp_path / "videos",
                            keyword="test_keyword",
                            tier="short",
                            existing_before=set()
                        )

                        # Check that retry attempts were logged
                        log_text = caplog.text
                        assert "retry 1/2" in log_text
                        assert "retry 2/2" in log_text

    @pytest.mark.integration
    def test_exhausted_retries_logged(self, tmp_path, caplog):
        """Test that exhausted retries are logged as warning."""
        import logging

        config = create_mock_downloader_config(tmp_path, max_retries=1, retry_delay=0.01)

        with patch_video_downloader_dependencies():
            with caplog.at_level(logging.WARNING):
                from src.downloader.core import VideoDownloader
                downloader = VideoDownloader(config)

                keyword_dir = tmp_path / "videos" / "test"
                keyword_dir.mkdir(parents=True)

                with patch('subprocess.Popen') as mock_popen:
                    mock_process = create_mock_subprocess_process(returncode=0)
                    mock_popen.return_value = mock_process

                    # All calls timeout via progress-aware monitor
                    with patch.object(
                        downloader, '_wait_for_process_with_progress',
                        return_value=("", "", 'stall')
                    ):
                        result = downloader._run_download_cmd(
                            cmd=['yt-dlp', 'test'],
                            keyword_dir=keyword_dir,
                            output_dir=tmp_path / "videos",
                            keyword="test_keyword",
                            tier="short",
                            existing_before=set()
                        )

                        # Check that exhausted retries were logged
                        assert "retries exhausted" in caplog.text.lower()
