"""Unit tests for network-aware error classification in download_segments.

US-48-006: Add network-aware error classification to retry queue for download_segments.

Tests verify:
- DNS resolution failures are classified as 'network_systemic'
- 403 Forbidden errors are classified as 'video_specific' (retryable with escalation)
- Error categories are stored on RetryItem when added to the retry queue
- get_retryable_items() excludes network_systemic items
"""

import pytest
from unittest.mock import patch

from src.stages.download_segments import classify_error_category, _is_network_failure
from src.downloader.retry_queue import RetryQueue, RetryItem, BatchRetryConfig


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def queue():
    """RetryQueue with fast config for testing."""
    config = BatchRetryConfig(
        enabled=True,
        delay_seconds=0.01,
        max_passes=2,
        jitter_factor=0.0,
    )
    return RetryQueue(config)


# =============================================================================
# classify_error_category tests
# =============================================================================

class TestClassifyErrorCategory:
    """Tests for classify_error_category function."""

    def test_dns_resolution_failure_is_network_systemic(self):
        """DNS resolution failure (getaddrinfo) classified as network_systemic."""
        error = "urllib3.exceptions.NewConnectionError: getaddrinfo failed"
        assert classify_error_category(error) == 'network_systemic'

    def test_name_not_known_is_network_systemic(self):
        """Linux DNS failure (Name or service not known) classified as network_systemic."""
        error = "socket.gaierror: [Errno -2] Name or service not known"
        assert classify_error_category(error) == 'network_systemic'

    def test_windows_dns_failure_is_network_systemic(self):
        """Windows DNS failure (Errno 11001) classified as network_systemic."""
        error = "socket.gaierror: [Errno 11001] getaddrinfo failed"
        assert classify_error_category(error) == 'network_systemic'

    def test_macos_dns_failure_is_network_systemic(self):
        """macOS DNS failure (nodename nor servname) classified as network_systemic."""
        error = "socket.gaierror: nodename nor servname provided, or not known"
        assert classify_error_category(error) == 'network_systemic'

    def test_network_unreachable_is_network_systemic(self):
        """Network is unreachable classified as network_systemic."""
        error = "OSError: [Errno 101] Network is unreachable"
        assert classify_error_category(error) == 'network_systemic'

    def test_no_address_is_network_systemic(self):
        """No address associated with hostname classified as network_systemic."""
        error = "socket.gaierror: No address associated with hostname"
        assert classify_error_category(error) == 'network_systemic'

    def test_temp_name_resolution_is_network_systemic(self):
        """Temporary failure in name resolution classified as network_systemic."""
        error = "socket.gaierror: Temporary failure in name resolution"
        assert classify_error_category(error) == 'network_systemic'

    def test_ffmpeg_network_exit_code_is_network_systemic(self):
        """ffmpeg network exit code classified as network_systemic."""
        error = "ffmpeg exited with code 4294967158"
        assert classify_error_category(error) == 'network_systemic'

    def test_403_forbidden_is_video_specific(self):
        """403 Forbidden classified as video_specific (retryable with escalation)."""
        error = "HTTP Error 403: Forbidden"
        assert classify_error_category(error) == 'video_specific'

    def test_video_unavailable_is_video_specific(self):
        """Video unavailable classified as video_specific."""
        error = "Video unavailable: This video is no longer available"
        assert classify_error_category(error) == 'video_specific'

    def test_rate_limit_is_video_specific(self):
        """Rate limit (429) classified as video_specific (retryable with backoff)."""
        error = "HTTP Error 429: Too Many Requests"
        assert classify_error_category(error) == 'video_specific'

    def test_connection_refused_is_video_specific(self):
        """Connection refused is video_specific (transient, not DNS-systemic)."""
        error = "ConnectionRefusedError: [Errno 111] Connection refused"
        assert classify_error_category(error) == 'video_specific'

    def test_generic_download_error_is_video_specific(self):
        """Generic yt-dlp download error classified as video_specific."""
        error = "ERROR: Unable to download video data"
        assert classify_error_category(error) == 'video_specific'

    def test_empty_error_is_video_specific(self):
        """Empty error message defaults to video_specific."""
        assert classify_error_category('') == 'video_specific'


# =============================================================================
# RetryQueue error_category integration tests
# =============================================================================

class TestRetryQueueErrorCategory:
    """Tests for error_category tagging in RetryQueue."""

    def test_add_with_default_category(self, queue):
        """Items added without explicit category default to video_specific."""
        queue.add('vid1', 'keyword', 'short', 'HTTP Error 403')
        assert queue.items['vid1'].error_category == 'video_specific'

    def test_add_with_network_systemic_category(self, queue):
        """Items can be tagged as network_systemic."""
        queue.add(
            'vid1', 'keyword', 'short',
            'getaddrinfo failed',
            error_category='network_systemic'
        )
        assert queue.items['vid1'].error_category == 'network_systemic'

    def test_add_with_video_specific_category(self, queue):
        """Items can be explicitly tagged as video_specific."""
        queue.add(
            'vid1', 'keyword', 'short',
            'HTTP Error 403',
            error_category='video_specific'
        )
        assert queue.items['vid1'].error_category == 'video_specific'

    def test_get_retryable_items_excludes_network_systemic(self, queue):
        """get_retryable_items() filters out network_systemic items."""
        queue.add('vid1', 'kw', 'short', '403 Forbidden', error_category='video_specific')
        queue.add('vid2', 'kw', 'short', 'getaddrinfo failed', error_category='network_systemic')
        queue.add('vid3', 'kw', 'short', 'rate limit', error_category='video_specific')

        retryable = queue.get_retryable_items()
        retryable_ids = {item.video_id for item in retryable}

        assert retryable_ids == {'vid1', 'vid3'}
        assert 'vid2' not in retryable_ids

    def test_get_retryable_items_returns_all_when_no_network_systemic(self, queue):
        """get_retryable_items() returns all items when none are network_systemic."""
        queue.add('vid1', 'kw', 'short', '403 Forbidden', error_category='video_specific')
        queue.add('vid2', 'kw', 'short', 'rate limit', error_category='video_specific')

        retryable = queue.get_retryable_items()
        assert len(retryable) == 2

    def test_get_retryable_items_returns_empty_when_all_network_systemic(self, queue):
        """get_retryable_items() returns empty when all are network_systemic."""
        queue.add('vid1', 'kw', 'short', 'DNS failed', error_category='network_systemic')
        queue.add('vid2', 'kw', 'short', 'No route', error_category='network_systemic')

        retryable = queue.get_retryable_items()
        assert len(retryable) == 0

    def test_error_category_persists_in_checkpoint(self, queue):
        """error_category is included in checkpoint serialization."""
        queue.add('vid1', 'kw', 'short', 'DNS failed', error_category='network_systemic')
        queue.add('vid2', 'kw', 'short', '403', error_category='video_specific')

        checkpoint = queue.to_checkpoint_dict()
        items = checkpoint['items']

        categories = {item['video_id']: item['error_category'] for item in items}
        assert categories['vid1'] == 'network_systemic'
        assert categories['vid2'] == 'video_specific'

    def test_error_category_restored_from_checkpoint(self, queue):
        """error_category is restored from checkpoint data."""
        checkpoint = {
            'items': [
                {
                    'video_id': 'vid1',
                    'keyword': 'kw',
                    'tier': 'short',
                    'error_message': 'DNS failed',
                    'retry_count': 0,
                    'error_category': 'network_systemic',
                },
            ],
            'current_pass': 0,
            'completed_ids': [],
            'failed_ids': [],
            'total_added': 1,
            'total_retried': 0,
        }
        queue.from_checkpoint_dict(checkpoint)

        assert queue.items['vid1'].error_category == 'network_systemic'

    def test_error_category_defaults_on_old_checkpoint(self, queue):
        """error_category defaults to video_specific for old checkpoints without it."""
        checkpoint = {
            'items': [
                {
                    'video_id': 'vid1',
                    'keyword': 'kw',
                    'tier': 'short',
                    'error_message': 'some error',
                    'retry_count': 0,
                    # no error_category field (old checkpoint)
                },
            ],
            'current_pass': 0,
            'completed_ids': [],
            'failed_ids': [],
            'total_added': 1,
            'total_retried': 0,
        }
        queue.from_checkpoint_dict(checkpoint)

        assert queue.items['vid1'].error_category == 'video_specific'

    def test_update_existing_item_updates_category(self, queue):
        """Re-adding an existing item updates its error_category."""
        queue.add('vid1', 'kw', 'short', '403', error_category='video_specific')
        assert queue.items['vid1'].error_category == 'video_specific'

        # Re-add with different category (returns False but updates)
        queue.add('vid1', 'kw', 'short', 'DNS failed', error_category='network_systemic')
        assert queue.items['vid1'].error_category == 'network_systemic'


# =============================================================================
# US-49-006: _is_network_failure() pattern validation for Python API + subprocess
# =============================================================================

class TestIsNetworkFailurePythonApiFormat:
    """Verify _is_network_failure() works with both Python API DownloadError
    format (wrapped with 'ERROR: [youtube] ID: ...') and raw subprocess stderr.

    US-49-006: Validates patterns against real-world error strings.
    """

    # --- Real-world network failure messages (should return True) ---

    def test_subprocess_stderr_dns_failure(self):
        """Raw subprocess stderr: getaddrinfo failed."""
        error = "ERROR: unable to download video data: <urlopen error [Errno 11001] getaddrinfo failed>"
        assert _is_network_failure(error) is True

    def test_python_api_download_error_dns(self):
        """Python API DownloadError wraps exception with 'ERROR: [youtube] ID:' prefix."""
        error = "ERROR: [youtube] dQw4w9WgXcQ: Unable to download API page; getaddrinfo failed"
        assert _is_network_failure(error) is True

    def test_python_api_urlerror_wrapper(self):
        """Python API: URLError wrapping a socket error."""
        error = "ERROR: [youtube] abc123: Unable to download webpage: URLError: <urlopen error [Errno 11001] getaddrinfo failed>"
        assert _is_network_failure(error) is True

    def test_urlerror_without_nested_dns(self):
        """URLError as a standalone pattern (no nested DNS error visible)."""
        error = "URLError: <urlopen error timed out>"
        assert _is_network_failure(error) is True

    def test_connection_reset_error_python_api(self):
        """Python API: ConnectionResetError during download."""
        error = "ERROR: [youtube] xyz789: Unable to download video data: ConnectionResetError: [Errno 104] Connection reset by peer"
        assert _is_network_failure(error) is True

    def test_connection_reset_error_bare(self):
        """Bare ConnectionResetError string (from repr/str of exception)."""
        error = "ConnectionResetError(104, 'Connection reset by peer')"
        assert _is_network_failure(error) is True

    def test_windows_dns_errno_11001(self):
        """Windows DNS failure: Errno 11001 in subprocess output."""
        error = "ERROR: [youtube] vid1: Unable to download webpage: <urlopen error [Errno 11001] getaddrinfo failed>"
        assert _is_network_failure(error) is True

    def test_ffmpeg_exit_code_unsigned(self):
        """ffmpeg exit code 4294967158 (0xFFFFFEC6, unsigned -314) in stderr."""
        error = "ERROR: Postprocessing: ffmpeg exited with code 4294967158"
        assert _is_network_failure(error) is True

    def test_network_unreachable_linux(self):
        """Linux: Network is unreachable (no internet)."""
        error = "ERROR: [youtube] vid2: Unable to download: OSError: [Errno 101] Network is unreachable"
        assert _is_network_failure(error) is True

    def test_temporary_name_resolution_failure(self):
        """Temporary failure in name resolution (DNS intermittent)."""
        error = "socket.gaierror: [Errno -3] Temporary failure in name resolution"
        assert _is_network_failure(error) is True

    # --- Bot-detection / video-specific messages (should return False) ---

    def test_sign_in_bot_detection_not_network(self):
        """Bot-detection 'Sign in to confirm' must NOT be classified as network failure."""
        error = "ERROR: [youtube] abc123: Sign in to confirm you're not a bot. Use --cookies-from-browser or --cookies"
        assert _is_network_failure(error) is False

    def test_http_403_forbidden_not_network(self):
        """HTTP Error 403 is access-related, NOT a network failure."""
        error = "ERROR: [youtube] def456: HTTP Error 403: Forbidden"
        assert _is_network_failure(error) is False

    def test_video_unavailable_not_network(self):
        """Video unavailable is content-specific, NOT a network failure."""
        error = "ERROR: [youtube] ghi789: Video unavailable. This video is no longer available"
        assert _is_network_failure(error) is False

    def test_age_restricted_not_network(self):
        """Age-restricted video is access-specific, NOT a network failure."""
        error = "ERROR: [youtube] jkl012: Sign in to confirm your age. This video may be inappropriate for some users."
        assert _is_network_failure(error) is False

    def test_copyright_claim_not_network(self):
        """Copyright takedown is content-specific, NOT a network failure."""
        error = "ERROR: [youtube] mno345: This video contains content from UMG, who has blocked it on copyright grounds."
        assert _is_network_failure(error) is False

    def test_429_rate_limit_not_network(self):
        """Rate limiting (429) is transient/video-specific, NOT systemic network failure."""
        error = "HTTP Error 429: Too Many Requests"
        assert _is_network_failure(error) is False
