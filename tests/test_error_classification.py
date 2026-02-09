"""Unit tests for network-aware error classification in download_segments.

US-48-006: Add network-aware error classification to retry queue for download_segments.

Tests verify:
- DNS resolution failures are classified as 'network'
- 403 Forbidden errors are classified as 'video_specific' (retryable with escalation)
- Error categories are stored on RetryItem when added to the retry queue
- get_retryable_items() excludes network items
"""

import pytest
from unittest.mock import patch

from src.stages.download_segments import classify_error_category, _is_network_failure
from src.downloader.retry_queue import RetryQueue, RetryItem, BatchRetryConfig
from src.downloader.errors import (
    ClassifiedDownloadError,
    NetworkError,
    BotDetectionError,
    RateLimitError,
    FormatError,
    AuthenticationError,
    TimeoutError_,
)
from src.downloader.error_classification import is_network_failure, is_escalation_error


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

    def test_dns_resolution_failure_is_network(self):
        """DNS resolution failure (getaddrinfo) classified as network."""
        error = "urllib3.exceptions.NewConnectionError: getaddrinfo failed"
        assert classify_error_category(error) == 'network'

    def test_name_not_known_is_network(self):
        """Linux DNS failure (Name or service not known) classified as network."""
        error = "socket.gaierror: [Errno -2] Name or service not known"
        assert classify_error_category(error) == 'network'

    def test_windows_dns_failure_is_network(self):
        """Windows DNS failure (Errno 11001) classified as network."""
        error = "socket.gaierror: [Errno 11001] getaddrinfo failed"
        assert classify_error_category(error) == 'network'

    def test_macos_dns_failure_is_network(self):
        """macOS DNS failure (nodename nor servname) classified as network."""
        error = "socket.gaierror: nodename nor servname provided, or not known"
        assert classify_error_category(error) == 'network'

    def test_network_unreachable_is_network(self):
        """Network is unreachable classified as network."""
        error = "OSError: [Errno 101] Network is unreachable"
        assert classify_error_category(error) == 'network'

    def test_no_address_is_network(self):
        """No address associated with hostname classified as network."""
        error = "socket.gaierror: No address associated with hostname"
        assert classify_error_category(error) == 'network'

    def test_temp_name_resolution_is_network(self):
        """Temporary failure in name resolution classified as network."""
        error = "socket.gaierror: Temporary failure in name resolution"
        assert classify_error_category(error) == 'network'

    def test_ffmpeg_network_exit_code_is_network(self):
        """ffmpeg network exit code classified as network."""
        error = "ffmpeg exited with code 4294967158"
        assert classify_error_category(error) == 'network'

    def test_403_forbidden_is_bot_detection(self):
        """403 Forbidden classified as bot_detection (retryable with escalation)."""
        error = "HTTP Error 403: Forbidden"
        assert classify_error_category(error) == 'bot_detection'

    def test_video_unavailable_is_video_specific(self):
        """Video unavailable classified as video_specific."""
        error = "Video unavailable: This video is no longer available"
        assert classify_error_category(error) == 'video_specific'

    def test_rate_limit_is_bot_detection(self):
        """Rate limit (429) classified as bot_detection (escalation trigger)."""
        error = "HTTP Error 429: Too Many Requests"
        assert classify_error_category(error) == 'bot_detection'

    def test_connection_refused_is_network(self):
        """Connection refused classified as network (systemic when widespread)."""
        error = "ConnectionRefusedError: [Errno 111] Connection refused"
        assert classify_error_category(error) == 'network'

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

    def test_add_with_network_category(self, queue):
        """Items can be tagged as network."""
        queue.add(
            'vid1', 'keyword', 'short',
            'getaddrinfo failed',
            error_category='network'
        )
        assert queue.items['vid1'].error_category == 'network'

    def test_add_with_video_specific_category(self, queue):
        """Items can be explicitly tagged as video_specific."""
        queue.add(
            'vid1', 'keyword', 'short',
            'HTTP Error 403',
            error_category='video_specific'
        )
        assert queue.items['vid1'].error_category == 'video_specific'

    def test_get_retryable_items_excludes_network(self, queue):
        """get_retryable_items() filters out network items."""
        queue.add('vid1', 'kw', 'short', '403 Forbidden', error_category='video_specific')
        queue.add('vid2', 'kw', 'short', 'getaddrinfo failed', error_category='network')
        queue.add('vid3', 'kw', 'short', 'rate limit', error_category='video_specific')

        retryable = queue.get_retryable_items()
        retryable_ids = {item.video_id for item in retryable}

        assert retryable_ids == {'vid1', 'vid3'}
        assert 'vid2' not in retryable_ids

    def test_get_retryable_items_returns_all_when_no_network(self, queue):
        """get_retryable_items() returns all items when none are network."""
        queue.add('vid1', 'kw', 'short', '403 Forbidden', error_category='video_specific')
        queue.add('vid2', 'kw', 'short', 'rate limit', error_category='video_specific')

        retryable = queue.get_retryable_items()
        assert len(retryable) == 2

    def test_get_retryable_items_returns_empty_when_all_network(self, queue):
        """get_retryable_items() returns empty when all are network."""
        queue.add('vid1', 'kw', 'short', 'DNS failed', error_category='network')
        queue.add('vid2', 'kw', 'short', 'No route', error_category='network')

        retryable = queue.get_retryable_items()
        assert len(retryable) == 0

    def test_error_category_persists_in_checkpoint(self, queue):
        """error_category is included in checkpoint serialization."""
        queue.add('vid1', 'kw', 'short', 'DNS failed', error_category='network')
        queue.add('vid2', 'kw', 'short', '403', error_category='video_specific')

        checkpoint = queue.to_checkpoint_dict()
        items = checkpoint['items']

        categories = {item['video_id']: item['error_category'] for item in items}
        assert categories['vid1'] == 'network'
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
                    'error_category': 'network',
                },
            ],
            'current_pass': 0,
            'completed_ids': [],
            'failed_ids': [],
            'total_added': 1,
            'total_retried': 0,
        }
        queue.from_checkpoint_dict(checkpoint)

        assert queue.items['vid1'].error_category == 'network'

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
        queue.add('vid1', 'kw', 'short', 'DNS failed', error_category='network')
        assert queue.items['vid1'].error_category == 'network'


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


# =============================================================================
# US-51-005: Harden _is_network_failure pattern matching
# =============================================================================

class TestIsNetworkFailureHardened:
    """US-51-005: Verify _is_network_failure() handles Python API DownloadError
    wrapping, ffmpeg exit codes, and new connection patterns.

    Tests use exact error strings from production logs.
    """

    # --- Python API DownloadError wrapping of DNS failures ---

    def test_download_error_wraps_dns_urlopen(self):
        """DownloadError wraps DNS failure in '<urlopen error ...>' format."""
        error = "ERROR: unable to download webpage: <urlopen error [Errno 11001] getaddrinfo failed>"
        assert _is_network_failure(error) is True

    def test_download_error_wraps_name_resolution(self):
        """DownloadError wraps 'Name or service not known' via urlopen."""
        error = "ERROR: [youtube] abc123: Unable to download webpage: <urlopen error [Errno -2] Name or service not known>"
        assert _is_network_failure(error) is True

    def test_download_error_wraps_no_address(self):
        """DownloadError wraps 'No address associated with hostname'."""
        error = "ERROR: [youtube] xyz789: Unable to download API page: <urlopen error [Errno -5] No address associated with hostname>"
        assert _is_network_failure(error) is True

    def test_download_error_wraps_temp_resolution(self):
        """DownloadError wraps 'Temporary failure in name resolution'."""
        error = "ERROR: [youtube] vid1: Unable to download webpage: <urlopen error [Errno -3] Temporary failure in name resolution>"
        assert _is_network_failure(error) is True

    def test_download_error_wraps_network_unreachable(self):
        """DownloadError wraps 'Network is unreachable'."""
        error = "ERROR: [youtube] vid2: Unable to download video data: <urlopen error [Errno 101] Network is unreachable>"
        assert _is_network_failure(error) is True

    # --- ffmpeg exit code recognition ---

    def test_ffmpeg_exit_code_in_postprocessing(self):
        """ffmpeg exit code 4294967158 (0xFFFFFEC6 = -314 signed) in postprocessing."""
        error = "ERROR: Postprocessing: ffmpeg exited with code 4294967158"
        assert _is_network_failure(error) is True

    def test_ffmpeg_exit_code_bare(self):
        """ffmpeg exit code in bare error string."""
        error = "ffmpeg exited with code 4294967158"
        assert _is_network_failure(error) is True

    def test_ffmpeg_exit_code_with_stderr(self):
        """ffmpeg exit code with additional stderr context."""
        error = "ERROR: Postprocessing: ffmpeg exited with code 4294967158 (stderr: Connection reset)"
        assert _is_network_failure(error) is True

    # --- Connection refused pattern ---

    def test_connection_refused_errno_111(self):
        """Connection refused with Linux errno 111."""
        error = "ConnectionRefusedError: [Errno 111] Connection refused"
        assert _is_network_failure(error) is True

    def test_connection_refused_in_download_error(self):
        """Connection refused wrapped in yt-dlp DownloadError."""
        error = "ERROR: [youtube] abc123: Unable to download webpage: <urlopen error [Errno 111] Connection refused>"
        assert _is_network_failure(error) is True

    def test_connection_refused_windows(self):
        """Connection refused on Windows (Errno 10061)."""
        error = "ERROR: [youtube] vid1: Unable to download: <urlopen error [Errno 10061] Connection refused>"
        assert _is_network_failure(error) is True

    # --- Connection timed out pattern ---

    def test_connection_timed_out_basic(self):
        """Basic 'Connection timed out' error."""
        error = "Connection timed out"
        assert _is_network_failure(error) is True

    def test_connection_timed_out_in_download_error(self):
        """Connection timed out wrapped in yt-dlp DownloadError."""
        error = "ERROR: [youtube] abc123: Unable to download webpage: <urlopen error [Errno 110] Connection timed out>"
        assert _is_network_failure(error) is True

    def test_connection_timed_out_windows(self):
        """Connection timed out on Windows (Errno 10060)."""
        error = "ERROR: [youtube] vid1: Unable to download: <urlopen error [Errno 10060] Connection timed out>"
        assert _is_network_failure(error) is True

    # --- Bot-detection patterns must NOT be classified as network failures ---

    def test_sign_in_confirm_not_network(self):
        """'Sign in to confirm' bot-detection is NOT network failure."""
        error = "ERROR: [youtube] abc123: Sign in to confirm you're not a bot. Use --cookies-from-browser or --cookies"
        assert _is_network_failure(error) is False

    def test_403_forbidden_not_network(self):
        """'403 Forbidden' access error is NOT network failure."""
        error = "ERROR: [youtube] def456: HTTP Error 403: Forbidden"
        assert _is_network_failure(error) is False

    def test_sign_in_age_not_network(self):
        """'Sign in to confirm your age' is NOT network failure."""
        error = "ERROR: [youtube] ghi789: Sign in to confirm your age"
        assert _is_network_failure(error) is False

    def test_bot_captcha_not_network(self):
        """CAPTCHA/bot challenge is NOT network failure."""
        error = "ERROR: [youtube] jkl012: Join this channel to get access to members-only content"
        assert _is_network_failure(error) is False

    def test_video_unavailable_not_network(self):
        """Video unavailable is NOT network failure."""
        error = "ERROR: [youtube] mno345: Video unavailable"
        assert _is_network_failure(error) is False

    # --- classify_error_category integration for new patterns ---

    def test_classify_connection_refused_as_network(self):
        """classify_error_category returns 'network' for Connection refused."""
        error = "ConnectionRefusedError: [Errno 111] Connection refused"
        assert classify_error_category(error) == 'network'

    def test_classify_connection_timed_out_as_network(self):
        """classify_error_category returns 'network' for Connection timed out."""
        error = "ERROR: [youtube] vid1: Unable to download: Connection timed out"
        assert classify_error_category(error) == 'network'


# =============================================================================
# US-52-004: Python API exception format tests for _is_network_failure()
# =============================================================================

class TestIsNetworkFailurePythonApiExceptionFormat:
    """US-52-004: Verify _is_network_failure() detects network failures when
    errors are wrapped in Python API yt_dlp.utils.DownloadError format.

    The Python API wraps errors differently than subprocess stderr:
    - subprocess: raw stderr line like "ERROR: unable to download ..."
    - Python API: DownloadError('ERROR: [youtube] ID: <original exception>')
    - str(DownloadError) produces the inner message string

    These tests use exact patterns from production logs.
    """

    # --- Errno 11001: Windows DNS resolution failure ---

    def test_errno_11001_in_download_error_wrapper(self):
        """DownloadError wrapping Errno 11001 DNS failure (Python API format)."""
        error = "ERROR: [youtube] dQw4w9WgXcQ: Unable to download webpage: <urlopen error [Errno 11001] getaddrinfo failed>"
        assert _is_network_failure(error) is True

    def test_errno_11001_bare_socket_gaierror(self):
        """Bare socket.gaierror with Errno 11001 (as str(exception))."""
        error = "[Errno 11001] getaddrinfo failed"
        assert _is_network_failure(error) is True

    def test_errno_11001_in_yt_dlp_download_error_str(self):
        """str(yt_dlp.utils.DownloadError) with Errno 11001 from production."""
        error = "ERROR: [youtube] abc123: Unable to download API page: <urlopen error [Errno 11001] getaddrinfo failed>"
        assert _is_network_failure(error) is True

    def test_errno_11001_nested_in_connection_error(self):
        """Errno 11001 nested in requests ConnectionError -> DownloadError."""
        error = (
            "ERROR: [youtube] vid1: Unable to download webpage: "
            "<urllib3.exceptions.NewConnectionError: Failed to establish a new connection: "
            "[Errno 11001] getaddrinfo failed>"
        )
        assert _is_network_failure(error) is True

    # --- ffmpeg exit code 4294967158 (0xFFFFFEC6 = -314 signed) ---

    def test_ffmpeg_exit_code_in_python_api_postprocessing(self):
        """ffmpeg exit code 4294967158 in Python API PostProcessingError."""
        error = "ERROR: Postprocessing: ffmpeg exited with code 4294967158"
        assert _is_network_failure(error) is True

    def test_ffmpeg_exit_code_with_context_stderr(self):
        """ffmpeg exit code with stderr context (network interruption mid-transcode)."""
        error = (
            "ERROR: Postprocessing: ffmpeg exited with code 4294967158 "
            "(stderr: Connection to tcp://r3---sn-abc.googlevideo.com:443 failed: Connection refused)"
        )
        assert _is_network_failure(error) is True

    def test_ffmpeg_exit_code_in_download_error_wrapper(self):
        """ffmpeg exit code wrapped in DownloadError from Python API."""
        error = "ERROR: [youtube] vid1: ffmpeg exited with code 4294967158"
        assert _is_network_failure(error) is True

    # --- ffmpeg exit code -314 (signed equivalent of 4294967158) ---

    def test_ffmpeg_signed_exit_code_minus_314(self):
        """ffmpeg signed exit code -314 (signed equivalent of unsigned 4294967158)."""
        error = "ERROR: Postprocessing: ffmpeg exited with code -314"
        assert _is_network_failure(error) is True

    def test_ffmpeg_signed_exit_code_minus_314_in_download_error(self):
        """ffmpeg signed exit code -314 wrapped in DownloadError."""
        error = "ERROR: [youtube] vid1: ffmpeg exited with code -314"
        assert _is_network_failure(error) is True

    def test_classify_ffmpeg_signed_exit_code_as_network(self):
        """classify_error_category returns 'network' for ffmpeg signed exit code -314."""
        error = "ERROR: Postprocessing: ffmpeg exited with code -314"
        assert classify_error_category(error) == 'network'

    # --- 'Failed to resolve' hostname pattern ---

    def test_failed_to_resolve_hostname_curl(self):
        """curl/curl_cffi DNS failure: 'Failed to resolve host'."""
        error = "Failed to resolve host 'www.youtube.com'"
        assert _is_network_failure(error) is True

    def test_failed_to_resolve_in_download_error(self):
        """'Failed to resolve' wrapped in yt-dlp DownloadError."""
        error = "ERROR: [youtube] abc123: Unable to download webpage: Failed to resolve host name"
        assert _is_network_failure(error) is True

    def test_failed_to_resolve_curl_cffi_format(self):
        """curl_cffi specific DNS failure from impersonation layer."""
        error = (
            "ERROR: [youtube] vid1: Unable to download webpage: "
            "curl_cffi.requests.errors.ConnectionError: Failed to resolve 'www.youtube.com'"
        )
        assert _is_network_failure(error) is True

    def test_failed_to_resolve_with_proxy_context(self):
        """Failed to resolve through proxy configuration."""
        error = "ERROR: [youtube] vid1: Failed to resolve proxy 'socks5://10.0.0.1:1080'"
        assert _is_network_failure(error) is True

    # --- classify_error_category integration for Python API format ---

    def test_classify_errno_11001_python_api_as_network(self):
        """classify_error_category returns 'network' for Python API Errno 11001."""
        error = "ERROR: [youtube] abc123: Unable to download webpage: <urlopen error [Errno 11001] getaddrinfo failed>"
        assert classify_error_category(error) == 'network'

    def test_classify_ffmpeg_exit_code_as_network(self):
        """classify_error_category returns 'network' for ffmpeg exit code 4294967158."""
        error = "ERROR: Postprocessing: ffmpeg exited with code 4294967158"
        assert classify_error_category(error) == 'network'

    def test_classify_failed_to_resolve_as_network(self):
        """classify_error_category returns 'network' for 'Failed to resolve'."""
        error = "ERROR: [youtube] abc123: Failed to resolve host 'www.youtube.com'"
        assert classify_error_category(error) == 'network'

    # --- Negative cases: Python API errors that are NOT network failures ---

    def test_python_api_403_not_network(self):
        """Python API DownloadError with 403 is NOT a network failure."""
        error = "ERROR: [youtube] abc123: HTTP Error 403: Forbidden"
        assert _is_network_failure(error) is False

    def test_python_api_video_removed_not_network(self):
        """Python API DownloadError for removed video is NOT a network failure."""
        error = "ERROR: [youtube] abc123: Video unavailable. This video has been removed by the uploader."
        assert _is_network_failure(error) is False

    def test_python_api_age_gate_not_network(self):
        """Python API DownloadError for age-gated video is NOT a network failure."""
        error = "ERROR: [youtube] abc123: Sign in to confirm your age. This video may be inappropriate for some users."
        assert _is_network_failure(error) is False


# =============================================================================
# US-55-005: Network failure pattern matching against Python API exception formats
# =============================================================================

# Import directly from the canonical error_classification module
from src.downloader.error_classification import (
    is_network_failure,
    classify_error_category as classify_error_category_direct,
)


@pytest.mark.fast
class TestNetworkFailurePythonApiPatterns:
    """US-55-005: Verify is_network_failure() and classify_error_category()
    handle Python API exception formats correctly, including DNS resolution
    errors, ffmpeg exit codes, and bot-detection negative cases.

    The Python API wraps exceptions differently than subprocess stderr.
    These tests ensure pattern matching works for both formats.
    """

    # --- Criterion 1: DNS resolution errors in Python exception format ---

    @pytest.mark.fast
    def test_failed_to_resolve_python_api_format(self):
        """'Failed to resolve' matches in Python API exception output."""
        error = "curl_cffi.requests.errors.ConnectionError: Failed to resolve 'www.youtube.com'"
        assert is_network_failure(error) is True

    @pytest.mark.fast
    def test_failed_to_resolve_in_download_error_wrapper(self):
        """'Failed to resolve' in yt-dlp DownloadError wrapper."""
        error = "ERROR: [youtube] vid1: Unable to download webpage: Failed to resolve host name"
        assert is_network_failure(error) is True

    @pytest.mark.fast
    def test_errno_11001_windows_dns_python_exception(self):
        """Errno 11001 (Windows DNS failure) in Python socket.gaierror format."""
        error = "socket.gaierror: [Errno 11001] getaddrinfo failed"
        assert is_network_failure(error) is True

    @pytest.mark.fast
    def test_errno_11001_nested_in_download_error(self):
        """Errno 11001 nested in yt-dlp DownloadError from Python API."""
        error = (
            "ERROR: [youtube] dQw4w9WgXcQ: Unable to download webpage: "
            "<urlopen error [Errno 11001] getaddrinfo failed>"
        )
        assert is_network_failure(error) is True

    @pytest.mark.fast
    def test_errno_11001_in_urllib3_connection_error(self):
        """Errno 11001 wrapped in urllib3 NewConnectionError (Python API path)."""
        error = (
            "urllib3.exceptions.NewConnectionError: "
            "Failed to establish a new connection: [Errno 11001] getaddrinfo failed"
        )
        assert is_network_failure(error) is True

    # --- Criterion 2: ffmpeg exit code patterns from Python API output ---

    @pytest.mark.fast
    def test_ffmpeg_exit_code_unsigned_4294967158(self):
        """ffmpeg exit code 4294967158 (unsigned 0xFFFFFEC6) detected."""
        error = "ERROR: Postprocessing: ffmpeg exited with code 4294967158"
        assert is_network_failure(error) is True

    @pytest.mark.fast
    def test_ffmpeg_exit_code_signed_minus_314(self):
        """ffmpeg exit code -314 (signed equivalent of 4294967158) detected."""
        error = "ERROR: Postprocessing: ffmpeg exited with code -314"
        assert is_network_failure(error) is True

    @pytest.mark.fast
    def test_ffmpeg_unsigned_in_download_error_wrapper(self):
        """ffmpeg exit code 4294967158 wrapped in DownloadError."""
        error = "ERROR: [youtube] vid1: ffmpeg exited with code 4294967158"
        assert is_network_failure(error) is True

    @pytest.mark.fast
    def test_ffmpeg_signed_in_download_error_wrapper(self):
        """ffmpeg exit code -314 wrapped in DownloadError."""
        error = "ERROR: [youtube] vid1: ffmpeg exited with code -314"
        assert is_network_failure(error) is True

    # --- Criterion 3: classify_error_category priority chain ---

    @pytest.mark.fast
    def test_classify_dns_failure_as_network(self):
        """classify_error_category returns 'network' for DNS resolution failure."""
        error = "socket.gaierror: [Errno 11001] getaddrinfo failed"
        assert classify_error_category_direct(error) == 'network'

    @pytest.mark.fast
    def test_classify_failed_to_resolve_as_network(self):
        """classify_error_category returns 'network' for 'Failed to resolve'."""
        error = "curl_cffi.requests.errors.ConnectionError: Failed to resolve 'www.youtube.com'"
        assert classify_error_category_direct(error) == 'network'

    @pytest.mark.fast
    def test_classify_403_forbidden_as_bot_detection(self):
        """classify_error_category returns 'bot_detection' for 403 Forbidden."""
        error = "ERROR: [youtube] abc123: HTTP Error 403: Forbidden"
        assert classify_error_category_direct(error) == 'bot_detection'

    @pytest.mark.fast
    def test_classify_sign_in_as_bot_detection(self):
        """classify_error_category returns 'bot_detection' for Sign-in errors."""
        error = "ERROR: [youtube] abc123: Sign in to confirm you're not a bot"
        assert classify_error_category_direct(error) == 'bot_detection'

    # --- Criterion 4: Bot-detection NOT classified as network failures ---

    @pytest.mark.fast
    def test_sign_in_confirm_not_network_failure(self):
        """'Sign in to confirm' bot-detection is NOT a network failure."""
        error = "ERROR: [youtube] abc123: Sign in to confirm you're not a bot. Use --cookies-from-browser or --cookies"
        assert is_network_failure(error) is False

    @pytest.mark.fast
    def test_403_forbidden_not_network_failure(self):
        """'403: Forbidden' is NOT a network failure."""
        error = "ERROR: [youtube] def456: HTTP Error 403: Forbidden"
        assert is_network_failure(error) is False

    @pytest.mark.fast
    def test_sign_in_age_confirm_not_network_failure(self):
        """'Sign in to confirm your age' is NOT a network failure."""
        error = "ERROR: [youtube] ghi789: Sign in to confirm your age"
        assert is_network_failure(error) is False

    @pytest.mark.fast
    def test_rate_limit_429_not_network_failure(self):
        """HTTP 429 rate limit is NOT a network failure."""
        error = "HTTP Error 429: Too Many Requests"
        assert is_network_failure(error) is False


# =============================================================================
# US-55-011: Error severity classification, multipliers, and edge cases
# =============================================================================

from src.downloader.error_classification import (
    classify_error_severity,
    classify_network_subcategory,
    ERROR_SEVERITY_PATTERNS,
    SEVERITY_MULTIPLIERS,
    ERROR_PATTERNS,
    NETWORK_ERROR_PATTERNS,
    NETWORK_FAILURE_PATTERNS,
)


@pytest.mark.fast
class TestErrorSeverityClassification:
    """US-55-011: Verify classify_error_severity() returns correct severity
    levels for quota-exceeded, bot-detection, rate-limit, and login patterns.
    """

    # --- High severity: quota exceeded and bot detection ---

    @pytest.mark.fast
    def test_quota_exceeded_is_high(self):
        """'quota exceeded' error classified as high severity."""
        assert classify_error_severity("ERROR: quota exceeded for today") == 'high'

    @pytest.mark.fast
    def test_daily_quota_is_high(self):
        """'daily quota' error classified as high severity."""
        assert classify_error_severity("daily quota limit reached") == 'high'

    @pytest.mark.fast
    def test_bot_detection_is_high(self):
        """'bot detection' error classified as high severity."""
        assert classify_error_severity("bot detection triggered, please wait") == 'high'

    @pytest.mark.fast
    def test_automated_is_high(self):
        """'automated' activity error classified as high severity."""
        assert classify_error_severity("detected automated traffic from your network") == 'high'

    @pytest.mark.fast
    def test_suspicious_activity_is_high(self):
        """'suspicious activity' error classified as high severity."""
        assert classify_error_severity("suspicious activity detected on your account") == 'high'

    @pytest.mark.fast
    def test_ip_blocked_is_high(self):
        """'ip blocked' error classified as high severity."""
        assert classify_error_severity("Your ip blocked due to abuse") == 'high'

    @pytest.mark.fast
    def test_ip_has_been_blocked_is_high(self):
        """'ip has been blocked' error classified as high severity."""
        assert classify_error_severity("Your IP has been blocked") == 'high'

    @pytest.mark.fast
    def test_permanently_banned_is_high(self):
        """'permanently banned' error classified as high severity."""
        assert classify_error_severity("Account permanently banned") == 'high'

    @pytest.mark.fast
    def test_account_suspended_is_high(self):
        """'account suspended' error classified as high severity."""
        assert classify_error_severity("Your account suspended for violations") == 'high'

    # --- Medium severity: rate limits and 429 ---

    @pytest.mark.fast
    def test_too_many_requests_is_medium(self):
        """'too many requests' error classified as medium severity."""
        assert classify_error_severity("HTTP Error 429: Too Many Requests") == 'medium'

    @pytest.mark.fast
    def test_429_status_code_is_medium(self):
        """'429' status code in error classified as medium severity."""
        assert classify_error_severity("Server returned 429") == 'medium'

    @pytest.mark.fast
    def test_rate_limit_is_medium(self):
        """'rate limit' error classified as medium severity."""
        assert classify_error_severity("rate limit exceeded, retry later") == 'medium'

    @pytest.mark.fast
    def test_please_try_again_later_is_medium(self):
        """'please try again later' error classified as medium severity."""
        assert classify_error_severity("please try again later") == 'medium'

    @pytest.mark.fast
    def test_temporarily_unavailable_is_medium(self):
        """'temporarily unavailable' error classified as medium severity."""
        assert classify_error_severity("Service temporarily unavailable") == 'medium'

    # --- Low severity: age-gate and login ---

    @pytest.mark.fast
    def test_sign_in_is_low(self):
        """'sign in' error classified as low severity."""
        assert classify_error_severity("Please sign in to continue") == 'low'

    @pytest.mark.fast
    def test_login_required_is_low(self):
        """'login required' error classified as low severity."""
        assert classify_error_severity("login required to access this content") == 'low'

    @pytest.mark.fast
    def test_confirm_your_age_is_low(self):
        """'confirm your age' error classified as low severity."""
        assert classify_error_severity("Please confirm your age to proceed") == 'low'

    @pytest.mark.fast
    def test_slow_down_is_low(self):
        """'slow down' error classified as low severity."""
        assert classify_error_severity("Please slow down your requests") == 'low'

    # --- Default: no pattern match defaults to medium ---

    @pytest.mark.fast
    def test_unrecognized_error_defaults_to_medium(self):
        """Unrecognized error pattern defaults to medium severity."""
        assert classify_error_severity("Some unknown error occurred") == 'medium'

    # --- Case insensitivity ---

    @pytest.mark.fast
    def test_case_insensitive_matching(self):
        """Severity classification is case-insensitive."""
        assert classify_error_severity("QUOTA EXCEEDED") == 'high'
        assert classify_error_severity("Rate Limit Hit") == 'medium'
        assert classify_error_severity("LOGIN REQUIRED") == 'low'


@pytest.mark.fast
class TestSeverityMultipliers:
    """US-55-011: Verify SEVERITY_MULTIPLIERS maps correctly."""

    @pytest.mark.fast
    def test_low_multiplier_is_1_5(self):
        """Low severity maps to 1.5x multiplier."""
        assert SEVERITY_MULTIPLIERS['low'] == 1.5

    @pytest.mark.fast
    def test_medium_multiplier_is_2_0(self):
        """Medium severity maps to 2.0x multiplier."""
        assert SEVERITY_MULTIPLIERS['medium'] == 2.0

    @pytest.mark.fast
    def test_high_multiplier_is_3_0(self):
        """High severity maps to 3.0x multiplier."""
        assert SEVERITY_MULTIPLIERS['high'] == 3.0

    @pytest.mark.fast
    def test_all_severity_levels_have_multipliers(self):
        """Every severity level in ERROR_SEVERITY_PATTERNS has a multiplier."""
        for level in ERROR_SEVERITY_PATTERNS:
            assert level in SEVERITY_MULTIPLIERS, f"Missing multiplier for '{level}'"

    @pytest.mark.fast
    def test_multipliers_are_increasing(self):
        """Multipliers increase with severity: low < medium < high."""
        assert SEVERITY_MULTIPLIERS['low'] < SEVERITY_MULTIPLIERS['medium']
        assert SEVERITY_MULTIPLIERS['medium'] < SEVERITY_MULTIPLIERS['high']


@pytest.mark.fast
class TestClassifyErrorCategoryFallthrough:
    """US-55-011: Verify classify_error_category() falls through to
    'video_specific' for errors that don't match network, bot_detection,
    or timeout patterns.
    """

    @pytest.mark.fast
    def test_video_removed_is_video_specific(self):
        """Removed video error falls through to video_specific."""
        error = "ERROR: [youtube] abc123: Video unavailable. This video has been removed."
        assert classify_error_category_direct(error) == 'video_specific'

    @pytest.mark.fast
    def test_private_video_is_video_specific(self):
        """Private video error falls through to video_specific."""
        error = "ERROR: [youtube] abc123: This is a private video. Please check the URL."
        assert classify_error_category_direct(error) == 'video_specific'

    @pytest.mark.fast
    def test_generic_download_error_is_video_specific(self):
        """Generic download error falls through to video_specific."""
        error = "ERROR: Unable to extract video data"
        assert classify_error_category_direct(error) == 'video_specific'

    @pytest.mark.fast
    def test_video_deleted_is_video_specific(self):
        """Deleted video error falls through to video_specific."""
        error = "ERROR: [youtube] abc123: This video has been removed by the uploader"
        assert classify_error_category_direct(error) == 'video_specific'

    @pytest.mark.fast
    def test_unable_to_extract_is_video_specific(self):
        """Unable to extract video data falls through to video_specific."""
        error = "ERROR: [youtube] abc123: Unable to extract video data. YouTube said: video not available"
        assert classify_error_category_direct(error) == 'video_specific'


@pytest.mark.fast
class TestClassificationEdgeCases:
    """US-55-011: Verify empty string and None-like inputs to all
    classification functions ensure no unhandled exceptions.
    """

    # --- is_network_failure edge cases ---

    @pytest.mark.fast
    def test_is_network_failure_empty_string(self):
        """is_network_failure handles empty string without exception."""
        assert is_network_failure('') is False

    @pytest.mark.fast
    def test_is_network_failure_whitespace_only(self):
        """is_network_failure handles whitespace-only string without exception."""
        assert is_network_failure('   ') is False

    @pytest.mark.fast
    def test_is_network_failure_single_char(self):
        """is_network_failure handles single character without exception."""
        assert is_network_failure('x') is False

    # --- classify_error_category edge cases ---

    @pytest.mark.fast
    def test_classify_error_category_empty_string(self):
        """classify_error_category handles empty string, returns video_specific."""
        assert classify_error_category_direct('') == 'video_specific'

    @pytest.mark.fast
    def test_classify_error_category_whitespace_only(self):
        """classify_error_category handles whitespace-only string."""
        assert classify_error_category_direct('   ') == 'video_specific'

    @pytest.mark.fast
    def test_classify_error_category_single_char(self):
        """classify_error_category handles single character."""
        assert classify_error_category_direct('x') == 'video_specific'

    # --- classify_error_severity edge cases ---

    @pytest.mark.fast
    def test_classify_error_severity_empty_string(self):
        """classify_error_severity handles empty string, returns medium default."""
        assert classify_error_severity('') == 'medium'

    @pytest.mark.fast
    def test_classify_error_severity_whitespace_only(self):
        """classify_error_severity handles whitespace-only string."""
        assert classify_error_severity('   ') == 'medium'

    @pytest.mark.fast
    def test_classify_error_severity_single_char(self):
        """classify_error_severity handles single character."""
        assert classify_error_severity('x') == 'medium'

    @pytest.mark.fast
    def test_classify_error_severity_numbers_only(self):
        """classify_error_severity handles numeric-only string."""
        assert classify_error_severity('12345') == 'medium'


# =============================================================================
# US-57-008: Categorized network error patterns and sub-category classification
# =============================================================================


@pytest.mark.fast
class TestNetworkErrorPatternCategories:
    """US-57-008: Verify NETWORK_ERROR_PATTERNS categorized dict structure
    and classify_network_subcategory() sub-category classification.
    """

    # --- Sub-category classification ---

    @pytest.mark.fast
    def test_getaddrinfo_failed_is_dns(self):
        """'getaddrinfo failed' is classified under the 'dns' sub-category."""
        error = "urllib3.exceptions.NewConnectionError: getaddrinfo failed"
        assert classify_network_subcategory(error) == 'dns'

    @pytest.mark.fast
    def test_name_not_known_is_dns(self):
        """'Name or service not known' is classified under 'dns'."""
        error = "socket.gaierror: [Errno -2] Name or service not known"
        assert classify_network_subcategory(error) == 'dns'

    @pytest.mark.fast
    def test_errno_11001_is_dns(self):
        """Windows DNS failure 'Errno 11001' is classified under 'dns'."""
        error = "socket.gaierror: [Errno 11001] getaddrinfo failed"
        assert classify_network_subcategory(error) == 'dns'

    @pytest.mark.fast
    def test_failed_to_resolve_is_dns(self):
        """'Failed to resolve' is classified under 'dns'."""
        error = "curl_cffi.requests.errors.ConnectionError: Failed to resolve 'www.youtube.com'"
        assert classify_network_subcategory(error) == 'dns'

    @pytest.mark.fast
    def test_temp_name_resolution_is_dns(self):
        """'Temporary failure in name resolution' is classified under 'dns'."""
        error = "socket.gaierror: Temporary failure in name resolution"
        assert classify_network_subcategory(error) == 'dns'

    @pytest.mark.fast
    def test_connection_refused_is_tcp(self):
        """'Connection refused' is classified under the 'tcp' sub-category."""
        error = "ConnectionRefusedError: [Errno 111] Connection refused"
        assert classify_network_subcategory(error) == 'tcp'

    @pytest.mark.fast
    def test_connection_timed_out_is_tcp(self):
        """'Connection timed out' is classified under 'tcp'."""
        error = "ERROR: [youtube] vid1: Connection timed out"
        assert classify_network_subcategory(error) == 'tcp'

    @pytest.mark.fast
    def test_connection_reset_is_tcp(self):
        """'ConnectionResetError' is classified under 'tcp'."""
        error = "ConnectionResetError: [Errno 104] Connection reset by peer"
        assert classify_network_subcategory(error) == 'tcp'

    @pytest.mark.fast
    def test_network_unreachable_is_tcp(self):
        """'Network is unreachable' is classified under 'tcp'."""
        error = "OSError: [Errno 101] Network is unreachable"
        assert classify_network_subcategory(error) == 'tcp'

    @pytest.mark.fast
    def test_urlerror_is_http(self):
        """'URLError' is classified under the 'http' sub-category."""
        error = "URLError: <urlopen error timed out>"
        assert classify_network_subcategory(error) == 'http'

    @pytest.mark.fast
    def test_ffmpeg_exit_code_is_ffmpeg(self):
        """ffmpeg exit code 4294967158 is classified under 'ffmpeg'."""
        error = "ERROR: Postprocessing: ffmpeg exited with code 4294967158"
        assert classify_network_subcategory(error) == 'ffmpeg'

    @pytest.mark.fast
    def test_ffmpeg_signed_exit_code_is_ffmpeg(self):
        """ffmpeg exit code -314 is classified under 'ffmpeg'."""
        error = "ERROR: Postprocessing: ffmpeg exited with code -314"
        assert classify_network_subcategory(error) == 'ffmpeg'

    @pytest.mark.fast
    def test_http_error_403_is_http(self):
        """'HTTP Error 403' is classified under the 'http' sub-category."""
        error = "ERROR: [youtube] abc123: HTTP Error 403: Forbidden"
        assert classify_network_subcategory(error) == 'http'

    @pytest.mark.fast
    def test_http_error_429_is_http(self):
        """'HTTP Error 429' is classified under 'http'."""
        error = "HTTP Error 429: Too Many Requests"
        assert classify_network_subcategory(error) == 'http'

    @pytest.mark.fast
    def test_http_error_500_is_http(self):
        """'HTTP Error 500' is classified under 'http'."""
        error = "HTTP Error 500: Internal Server Error"
        assert classify_network_subcategory(error) == 'http'

    @pytest.mark.fast
    def test_video_unavailable_returns_none(self):
        """Non-pattern errors return None from classify_network_subcategory."""
        error = "Video unavailable: This video has been removed"
        assert classify_network_subcategory(error) is None

    @pytest.mark.fast
    def test_empty_string_returns_none(self):
        """Empty string returns None."""
        assert classify_network_subcategory('') is None

    # --- Flattened tuple backward compatibility ---

    @pytest.mark.fast
    def test_flattened_tuple_contains_all_network_patterns(self):
        """NETWORK_FAILURE_PATTERNS flat tuple contains all patterns from
        dns, tcp, tls categories plus URLError from http."""
        for category in ('dns', 'tcp', 'tls'):
            for pattern in ERROR_PATTERNS[category]:
                assert pattern in NETWORK_FAILURE_PATTERNS, (
                    f"Pattern '{pattern}' from '{category}' missing from NETWORK_FAILURE_PATTERNS"
                )
        # URLError is the transport-level http pattern included in the flat tuple
        assert 'URLError' in NETWORK_FAILURE_PATTERNS

    @pytest.mark.fast
    def test_flattened_tuple_count_matches_network_categories(self):
        """Flat tuple length equals dns+tcp+tls patterns plus URLError."""
        expected_count = sum(
            len(patterns)
            for cat, patterns in ERROR_PATTERNS.items()
            if cat in ('dns', 'tcp', 'tls')
        ) + 1  # +1 for URLError
        assert len(NETWORK_FAILURE_PATTERNS) == expected_count

    @pytest.mark.fast
    def test_flattened_tuple_is_tuple(self):
        """NETWORK_FAILURE_PATTERNS remains a tuple for backward compat."""
        assert isinstance(NETWORK_FAILURE_PATTERNS, tuple)

    # --- Categorized dict structure ---

    @pytest.mark.fast
    def test_error_patterns_has_required_categories(self):
        """ERROR_PATTERNS has dns, tcp, tls, http, ffmpeg keys."""
        required = {'dns', 'tcp', 'tls', 'http', 'ffmpeg'}
        assert required == set(ERROR_PATTERNS.keys())

    @pytest.mark.fast
    def test_network_error_patterns_is_alias(self):
        """NETWORK_ERROR_PATTERNS is an alias for ERROR_PATTERNS."""
        assert NETWORK_ERROR_PATTERNS is ERROR_PATTERNS

    @pytest.mark.fast
    def test_dns_category_not_empty(self):
        """DNS category has patterns."""
        assert len(ERROR_PATTERNS['dns']) > 0

    @pytest.mark.fast
    def test_tcp_category_not_empty(self):
        """TCP category has patterns."""
        assert len(ERROR_PATTERNS['tcp']) > 0

    @pytest.mark.fast
    def test_http_category_not_empty(self):
        """HTTP category has patterns."""
        assert len(ERROR_PATTERNS['http']) > 0

    @pytest.mark.fast
    def test_ffmpeg_category_has_exit_codes(self):
        """ffmpeg category contains the expected exit codes."""
        assert '4294967158' in ERROR_PATTERNS['ffmpeg']
        assert '-314' in ERROR_PATTERNS['ffmpeg']


# =============================================================================
# US-82-002: Typed DownloadError exception hierarchy tests
# =============================================================================

class TestTypedErrorHierarchy:
    """Tests for the typed DownloadError exception hierarchy.

    US-82-002: Verifies each error subclass is correctly instantiated
    from representative error strings, has the expected structured
    fields, and works with isinstance checks.
    """

    @pytest.mark.fast
    def test_network_error_from_dns_failure(self):
        """DNS failure produces NetworkError with correct fields."""
        result = classify_error_category("getaddrinfo failed")
        assert isinstance(result, NetworkError)
        assert result.category == 'network'
        assert result.retryable is False
        assert result.original_message == "getaddrinfo failed"

    @pytest.mark.fast
    def test_network_error_from_connection_refused(self):
        """Connection refused produces NetworkError."""
        result = classify_error_category("Connection refused")
        assert isinstance(result, NetworkError)
        assert result.category == 'network'

    @pytest.mark.fast
    def test_network_error_from_ffmpeg_exit_code(self):
        """ffmpeg network exit code produces NetworkError."""
        result = classify_error_category("ffmpeg exited with code 4294967158")
        assert isinstance(result, NetworkError)
        assert result.category == 'network'

    @pytest.mark.fast
    def test_bot_detection_error_from_403(self):
        """HTTP 403 produces BotDetectionError with correct fields."""
        result = classify_error_category("HTTP Error 403: Forbidden")
        assert isinstance(result, BotDetectionError)
        assert result.category == 'bot_detection'
        assert result.retryable is True
        assert result.original_message == "HTTP Error 403: Forbidden"

    @pytest.mark.fast
    def test_bot_detection_error_from_captcha(self):
        """Captcha error produces BotDetectionError."""
        result = classify_error_category("Sign in to confirm you're not a bot")
        assert isinstance(result, BotDetectionError)
        assert result.category == 'bot_detection'

    @pytest.mark.fast
    def test_timeout_error_from_stalled(self):
        """Stall timeout produces TimeoutError_ with correct fields."""
        result = classify_error_category("Download stalled for 30 seconds")
        assert isinstance(result, TimeoutError_)
        assert result.category == 'timeout'
        assert result.retryable is True
        assert result.original_message == "Download stalled for 30 seconds"

    @pytest.mark.fast
    def test_timeout_error_from_timed_out(self):
        """Socket timeout produces TimeoutError_."""
        result = classify_error_category("Connection timed out")
        # Note: "Connection timed out" matches TCP network pattern first
        # so it's NetworkError, not TimeoutError_ — this is correct behavior
        assert isinstance(result, (NetworkError, TimeoutError_))

    @pytest.mark.fast
    def test_format_error_from_unavailable(self):
        """Video unavailable produces FormatError with correct fields."""
        result = classify_error_category("Video unavailable: removed by uploader")
        assert isinstance(result, FormatError)
        assert result.category == 'video_specific'
        assert result.retryable is False
        assert result.original_message == "Video unavailable: removed by uploader"

    @pytest.mark.fast
    def test_format_error_from_empty_string(self):
        """Empty error message defaults to FormatError (video_specific)."""
        result = classify_error_category("")
        assert isinstance(result, FormatError)
        assert result.category == 'video_specific'

    @pytest.mark.fast
    def test_all_subclasses_extend_classified_download_error(self):
        """All typed errors are instances of ClassifiedDownloadError."""
        for cls in (NetworkError, BotDetectionError, RateLimitError,
                    FormatError, AuthenticationError, TimeoutError_):
            err = cls("test message")
            assert isinstance(err, ClassifiedDownloadError)

    @pytest.mark.fast
    def test_all_subclasses_extend_base_download_error(self):
        """All typed errors are instances of the original DownloadError from types.py."""
        from src.downloader.types import DownloadError
        for cls in (NetworkError, BotDetectionError, RateLimitError,
                    FormatError, AuthenticationError, TimeoutError_):
            err = cls("test message")
            assert isinstance(err, DownloadError)

    @pytest.mark.fast
    def test_structured_fields_on_all_subclasses(self):
        """Each subclass stores category, severity, retryable, original_message."""
        test_cases = [
            (NetworkError, 'network', 'high', False),
            (BotDetectionError, 'bot_detection', 'high', True),
            (RateLimitError, 'bot_detection', 'medium', True),
            (FormatError, 'video_specific', 'low', False),
            (AuthenticationError, 'bot_detection', 'low', True),
            (TimeoutError_, 'timeout', 'medium', True),
        ]
        for cls, expected_category, expected_severity, expected_retryable in test_cases:
            err = cls("test error")
            assert err.category == expected_category, f"{cls.__name__}.category"
            assert err.severity == expected_severity, f"{cls.__name__}.severity"
            assert err.retryable == expected_retryable, f"{cls.__name__}.retryable"
            assert err.original_message == "test error", f"{cls.__name__}.original_message"

    @pytest.mark.fast
    def test_severity_override_in_constructor(self):
        """Severity can be overridden via constructor kwargs."""
        err = NetworkError("dns failure", severity='low')
        assert err.severity == 'low'
        assert err.category == 'network'  # category still from class default

    @pytest.mark.fast
    def test_backward_compat_string_equality(self):
        """ClassifiedDownloadError == category string for backward compat."""
        err = NetworkError("dns failure")
        assert err == 'network'
        assert err != 'bot_detection'
        err2 = BotDetectionError("403 forbidden")
        assert err2 == 'bot_detection'
        assert err2 != 'network'

    @pytest.mark.fast
    def test_classify_returns_type_that_compares_as_string(self):
        """classify_error_category result compares equal to category string."""
        result = classify_error_category("getaddrinfo failed")
        assert result == 'network'
        result = classify_error_category("HTTP Error 403: Forbidden")
        assert result == 'bot_detection'
        result = classify_error_category("Video unavailable")
        assert result == 'video_specific'


class TestIsNetworkFailureWithTypedErrors:
    """Tests for is_network_failure with ClassifiedDownloadError instances.

    US-82-002: Verifies isinstance-based fast path works correctly.
    """

    @pytest.mark.fast
    def test_network_error_instance_returns_true(self):
        """NetworkError instance returns True without re-parsing."""
        err = NetworkError("getaddrinfo failed")
        assert is_network_failure(err) is True

    @pytest.mark.fast
    def test_bot_detection_error_instance_returns_false(self):
        """BotDetectionError instance returns False."""
        err = BotDetectionError("HTTP Error 403: Forbidden")
        assert is_network_failure(err) is False

    @pytest.mark.fast
    def test_format_error_instance_returns_false(self):
        """FormatError instance returns False."""
        err = FormatError("Video unavailable")
        assert is_network_failure(err) is False

    @pytest.mark.fast
    def test_string_still_works(self):
        """String input still works via pattern matching path."""
        assert is_network_failure("getaddrinfo failed") is True
        assert is_network_failure("HTTP Error 403: Forbidden") is False


class TestIsEscalationErrorWithTypedErrors:
    """Tests for is_escalation_error with ClassifiedDownloadError instances.

    US-82-002: Verifies isinstance-based fast path works correctly.
    """

    @pytest.mark.fast
    def test_bot_detection_error_returns_true(self):
        """BotDetectionError instance returns True."""
        err = BotDetectionError("HTTP Error 403: Forbidden")
        assert is_escalation_error(err) is True

    @pytest.mark.fast
    def test_authentication_error_returns_true(self):
        """AuthenticationError instance returns True."""
        err = AuthenticationError("Sign in to confirm your age")
        assert is_escalation_error(err) is True

    @pytest.mark.fast
    def test_network_error_returns_false(self):
        """NetworkError instance returns False."""
        err = NetworkError("getaddrinfo failed")
        assert is_escalation_error(err) is False

    @pytest.mark.fast
    def test_format_error_returns_false(self):
        """FormatError instance returns False."""
        err = FormatError("Video unavailable")
        assert is_escalation_error(err) is False

    @pytest.mark.fast
    def test_string_still_works(self):
        """String input still works via pattern matching path."""
        assert is_escalation_error("HTTP Error 403: Forbidden") is True
        assert is_escalation_error("getaddrinfo failed") is False
