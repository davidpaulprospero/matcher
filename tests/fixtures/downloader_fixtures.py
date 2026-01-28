"""
Shared Fixtures for Downloader Tests

US-009: Consolidate duplicate test patterns in download tests

This module provides reusable mock factories and fixtures for testing
the downloader module. These fixtures eliminate duplication across
test_downloader_*.py files.

Usage:
    from tests.fixtures.downloader_fixtures import (
        create_mock_downloader_config,
        create_mock_download_checkpoint,
        create_mock_subprocess_process,
        create_sample_downloaded_videos,
        create_duration_tiers_config,
    )

    def test_something(tmp_path):
        config = create_mock_downloader_config(tmp_path)
        # ... test code ...

Pytest fixtures are also available when this module is imported into conftest.py:
    @pytest.fixture
    def mock_downloader_config(tmp_path):
        return create_mock_downloader_config(tmp_path)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch


# =============================================================================
# Mock Config Factories
# =============================================================================

def create_mock_downloader_config(
    tmp_path: Path,
    *,
    davinci_mode: bool = False,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    retry_backoff: float = 2.0,
    stall_timeout: float = 0,
    download_timeout: int = 120,
    max_concurrent: int = 4,
    cookies: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    **overrides,
) -> MagicMock:
    """
    Create a mock config object for downloader testing.

    Provides a fully configured mock config with sensible defaults for
    download operations. All download-related settings are available as
    keyword arguments.

    Args:
        tmp_path: Pytest tmp_path fixture for temporary directories
        davinci_mode: Whether to enable DaVinci mode (default: False)
        max_retries: Maximum retry attempts (default: 3)
        retry_delay: Initial retry delay seconds (default: 2.0)
        retry_backoff: Retry delay multiplier (default: 2.0)
        stall_timeout: Stall detection timeout (default: 0 = disabled)
        download_timeout: Download timeout seconds (default: 120)
        max_concurrent: Max concurrent downloads (default: 4)
        cookies: Path to cookies file (default: None)
        cookies_from_browser: Browser to extract cookies from (default: None)
        **overrides: Additional config overrides

    Returns:
        MagicMock: Configured mock config object

    Example:
        >>> config = create_mock_downloader_config(tmp_path, max_retries=5)
        >>> assert config.download.max_retries == 5
    """
    mock_config = MagicMock()

    # Base paths
    mock_config.cache_dir = str(tmp_path / ".cache")
    mock_config.downloaded_videos_dir = str(tmp_path / "videos")

    # Download config section
    mock_config.download = MagicMock()
    mock_config.download.davinci_mode = davinci_mode
    mock_config.download.cookies = cookies
    mock_config.download.cookies_from_browser = cookies_from_browser
    mock_config.download.cookies_path = None  # Explicitly set to None to avoid MagicMock behavior
    mock_config.download.download_timeout = download_timeout
    mock_config.download.download_timeouts = {}
    mock_config.download.stall_timeout = stall_timeout
    mock_config.download.delete_original = False
    mock_config.download.max_retries = max_retries
    mock_config.download.retry_delay = retry_delay
    mock_config.download.retry_backoff = retry_backoff
    mock_config.download.max_concurrent = max_concurrent
    mock_config.download.skip_existing = True
    mock_config.download.video_quality = "best"

    # Impersonation config
    mock_config.download.impersonation = MagicMock()
    mock_config.download.impersonation.enabled = False
    mock_config.download.impersonation.preferred_targets = []

    # Extractor args config
    mock_config.download.extractor_args = MagicMock()
    mock_config.download.extractor_args.enabled = False
    mock_config.download.extractor_args.player_clients = []
    mock_config.download.extractor_args.escalation_threshold = 2
    mock_config.download.extractor_args.cooldown_seconds = 300

    # Fallback config
    mock_config.download.fallback = MagicMock()
    mock_config.download.fallback.enabled = False

    # Rate limiting / budget config (set to None to disable advanced features)
    mock_config.download.rate_limit_budget = None
    mock_config.download.cookie_rotation = None
    mock_config.download.vpn = None
    mock_config.download.rate_limit = None

    # LLM config (for components that check it)
    mock_config.llm = MagicMock()
    mock_config.llm.provider = 'gemini'
    mock_config.llm.model = 'gemini-pro'

    # Apply any additional overrides
    for key, value in overrides.items():
        if hasattr(mock_config.download, key):
            setattr(mock_config.download, key, value)
        elif hasattr(mock_config, key):
            setattr(mock_config, key, value)

    return mock_config


def create_duration_tiers_config(mock_config: MagicMock) -> MagicMock:
    """
    Add duration tiers configuration to a mock config.

    Adds the standard 4 duration tiers (short, medium, long, longer)
    with typical values.

    Args:
        mock_config: Mock config to add duration tiers to

    Returns:
        MagicMock: The config with duration_tiers added

    Example:
        >>> config = create_mock_downloader_config(tmp_path)
        >>> config = create_duration_tiers_config(config)
        >>> assert config.duration_tiers.short.min_seconds == 20
    """
    mock_config.duration_tiers = Mock()

    # Short tier (20-120 seconds)
    mock_config.duration_tiers.short = Mock()
    mock_config.duration_tiers.short.min_seconds = 20
    mock_config.duration_tiers.short.max_seconds = 120
    mock_config.duration_tiers.short.videos_per_keyword = 8
    mock_config.duration_tiers.short.max_total = 0

    # Medium tier (120-600 seconds)
    mock_config.duration_tiers.medium = Mock()
    mock_config.duration_tiers.medium.min_seconds = 120
    mock_config.duration_tiers.medium.max_seconds = 600
    mock_config.duration_tiers.medium.videos_per_keyword = 5
    mock_config.duration_tiers.medium.max_total = 0

    # Long tier (600-1800 seconds)
    mock_config.duration_tiers.long = Mock()
    mock_config.duration_tiers.long.min_seconds = 600
    mock_config.duration_tiers.long.max_seconds = 1800
    mock_config.duration_tiers.long.videos_per_keyword = 3
    mock_config.duration_tiers.long.max_total = 0

    # Longer tier (1800+ seconds)
    mock_config.duration_tiers.longer = Mock()
    mock_config.duration_tiers.longer.min_seconds = 1800
    mock_config.duration_tiers.longer.max_seconds = 7200
    mock_config.duration_tiers.longer.videos_per_keyword = 2
    mock_config.duration_tiers.longer.max_total = 0

    return mock_config


# =============================================================================
# Mock Checkpoint Factory
# =============================================================================

def create_mock_download_checkpoint(
    *,
    downloaded_count: int = 0,
    keywords_completed: Optional[List[str]] = None,
    current_keyword: Optional[str] = None,
    errors: Optional[List[str]] = None,
) -> Mock:
    """
    Create a mock checkpoint object for download tracking.

    Args:
        downloaded_count: Number of videos downloaded (default: 0)
        keywords_completed: List of completed keywords (default: [])
        current_keyword: Currently processing keyword (default: None)
        errors: List of errors encountered (default: [])

    Returns:
        Mock: Configured checkpoint mock

    Example:
        >>> checkpoint = create_mock_download_checkpoint(downloaded_count=5)
        >>> assert checkpoint.downloaded_count == 5
    """
    mock_checkpoint = Mock()
    mock_checkpoint.downloaded_count = downloaded_count
    mock_checkpoint.keywords_completed = keywords_completed or []
    mock_checkpoint.current_keyword = current_keyword
    mock_checkpoint.errors = errors or []
    mock_checkpoint.save = Mock()
    mock_checkpoint.load = Mock()

    return mock_checkpoint


# =============================================================================
# Mock Subprocess Factory
# =============================================================================

def create_mock_subprocess_process(
    *,
    returncode: int = 0,
    stdout_output: str = "",
    stderr_output: str = "",
    poll_returns: Optional[List[Optional[int]]] = None,
) -> MagicMock:
    """
    Create a mock subprocess.Popen process.

    Provides a configurable mock for subprocess testing with
    stdout/stderr pipes and returncode.

    Args:
        returncode: Process return code (default: 0)
        stdout_output: Output for stdout.readline() (default: "")
        stderr_output: Output for stderr.readline() (default: "")
        poll_returns: Sequence of poll() return values (default: [returncode])

    Returns:
        MagicMock: Configured process mock

    Example:
        >>> proc = create_mock_subprocess_process(returncode=1, stderr_output="Error!")
        >>> assert proc.poll() == 1
        >>> assert proc.stderr.readline() == "Error!"
    """
    mock_process = MagicMock()

    # Return code
    mock_process.returncode = returncode

    # Kill and terminate
    mock_process.kill = MagicMock()
    mock_process.terminate = MagicMock()

    # Poll behavior
    if poll_returns is not None:
        mock_process.poll.side_effect = poll_returns + [returncode]
    else:
        mock_process.poll.return_value = returncode

    # Stdout/stderr
    mock_process.stdout = MagicMock()
    if stdout_output:
        mock_process.stdout.readline.side_effect = [stdout_output, ""]
    else:
        mock_process.stdout.readline.return_value = ""

    mock_process.stderr = MagicMock()
    if stderr_output:
        mock_process.stderr.readline.side_effect = [stderr_output, ""]
    else:
        mock_process.stderr.readline.return_value = ""

    # Wait
    mock_process.wait = MagicMock(return_value=returncode)

    return mock_process


# =============================================================================
# Sample Data Factories
# =============================================================================

def create_sample_downloaded_videos(
    count: int = 3,
    keyword: str = "travel",
    duration_tier: str = "medium",
) -> List[Dict[str, Any]]:
    """
    Create sample DownloadedVideo-like dicts for testing.

    Returns list of dicts with DownloadedVideo fields. Use with
    DownloadedVideo(**item) to create actual objects.

    Args:
        count: Number of videos to create (default: 3)
        keyword: Keyword for all videos (default: "travel")
        duration_tier: Duration tier for all videos (default: "medium")

    Returns:
        List[Dict]: List of video data dicts

    Example:
        >>> from src.state import DownloadedVideo
        >>> videos = create_sample_downloaded_videos(count=2)
        >>> downloaded = [DownloadedVideo(**v) for v in videos]
        >>> assert len(downloaded) == 2
    """
    videos = []
    for i in range(count):
        videos.append({
            "file": f"/videos/{keyword}/video_{i}.mp4",
            "url": f"https://youtube.com/watch?v=vid{i}",
            "title": f"{keyword.title()} Video {i}",
            "duration": 120.0 + (i * 30),
            "duration_tier": duration_tier,
            "keyword": keyword,
        })
    return videos


def create_temp_video_directory(
    tmp_path: Path,
    keyword: str = "test_keyword",
    *,
    create_partial_files: bool = False,
    video_count: int = 0,
) -> Path:
    """
    Create a temporary video directory structure for testing.

    Creates the standard directory structure used by the downloader
    with optional partial/temporary files.

    Args:
        tmp_path: Pytest tmp_path fixture
        keyword: Keyword directory name (default: "test_keyword")
        create_partial_files: Create .part and .ytdl files (default: False)
        video_count: Number of empty video files to create (default: 0)

    Returns:
        Path: Path to the keyword directory

    Example:
        >>> keyword_dir = create_temp_video_directory(tmp_path, "travel")
        >>> assert keyword_dir.exists()
    """
    output_dir = tmp_path / "videos"
    keyword_dir = output_dir / keyword
    keyword_dir.mkdir(parents=True, exist_ok=True)

    if create_partial_files:
        (keyword_dir / "video1.mp4.part").write_text("partial data")
        (keyword_dir / "video1.mp4.ytdl").write_text("ytdl state")

    for i in range(video_count):
        (keyword_dir / f"video_{i}.mp4").write_bytes(b"fake video data")

    return keyword_dir


# =============================================================================
# Patch Context Managers
# =============================================================================

def patch_video_downloader_dependencies():
    """
    Context manager to patch all VideoDownloader dependencies.

    Patches CheckpointManager, TranscodingManager, TitleFilter,
    SpeechScreener, SearchOptimizer, AudioFirstPipeline, and
    get_cookies_args.

    Returns:
        Nested context manager for all patches

    Example:
        >>> with patch_video_downloader_dependencies():
        ...     downloader = VideoDownloader(config)
        ...     # All dependencies are mocked
    """
    from contextlib import ExitStack

    stack = ExitStack()

    patches = [
        patch('src.downloader.core.CheckpointManager'),
        patch('src.downloader.core.TranscodingManager'),
        patch('src.downloader.core.TitleFilter'),
        patch('src.downloader.core.SpeechScreener'),
        patch('src.downloader.core.SearchOptimizer'),
        patch('src.downloader.core.AudioFirstPipeline'),
        patch('src.downloader.core.utils.get_cookies_args', return_value=[]),
    ]

    for p in patches:
        stack.enter_context(p)

    return stack


def patch_time_sleep():
    """
    Context manager to patch time.sleep for retry testing.

    Returns the mock object so call counts can be verified.

    Returns:
        Patch context manager

    Example:
        >>> with patch_time_sleep() as mock_sleep:
        ...     # Code that calls time.sleep
        ...     assert mock_sleep.call_count == 2
    """
    return patch('time.sleep')


# =============================================================================
# Assertion Helpers
# =============================================================================

def assert_download_successful(
    result: Any,
    *,
    expected_count: Optional[int] = None,
    min_count: int = 1,
) -> None:
    """
    Assert that a download operation was successful.

    Validates the result of a download operation, checking for
    expected video count and no critical errors.

    Args:
        result: Result from download operation (list or dict)
        expected_count: Exact expected video count (optional)
        min_count: Minimum expected videos (default: 1)

    Raises:
        AssertionError: If download result is invalid

    Example:
        >>> result = downloader.download_videos(["travel"])
        >>> assert_download_successful(result, min_count=3)
    """
    if isinstance(result, list):
        count = len(result)
    elif isinstance(result, dict):
        count = result.get("count", len(result.get("videos", [])))
    else:
        count = 1 if result else 0

    if expected_count is not None:
        assert count == expected_count, (
            f"Expected {expected_count} downloads, got {count}"
        )
    else:
        assert count >= min_count, (
            f"Expected at least {min_count} downloads, got {count}"
        )


def assert_retry_backoff_correct(
    mock_sleep: MagicMock,
    initial_delay: float,
    backoff: float,
    expected_calls: int,
) -> None:
    """
    Assert that retry backoff delays are correct.

    Validates that time.sleep was called with expected exponential
    backoff values.

    Args:
        mock_sleep: Mocked time.sleep
        initial_delay: First delay value expected
        backoff: Backoff multiplier
        expected_calls: Number of sleep calls expected

    Raises:
        AssertionError: If backoff is incorrect

    Example:
        >>> with patch_time_sleep() as mock_sleep:
        ...     # Code that retries with backoff
        ...     assert_retry_backoff_correct(mock_sleep, 2.0, 2.0, 3)
    """
    assert mock_sleep.call_count == expected_calls, (
        f"Expected {expected_calls} sleep calls, got {mock_sleep.call_count}"
    )

    calls = mock_sleep.call_args_list
    for i, call in enumerate(calls):
        expected_delay = initial_delay * (backoff ** i)
        actual_delay = call[0][0]
        assert abs(actual_delay - expected_delay) < 0.01, (
            f"Call {i}: expected delay {expected_delay}, got {actual_delay}"
        )


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Config factories
    'create_mock_downloader_config',
    'create_duration_tiers_config',
    # Checkpoint factory
    'create_mock_download_checkpoint',
    # Subprocess factory
    'create_mock_subprocess_process',
    # Sample data factories
    'create_sample_downloaded_videos',
    'create_temp_video_directory',
    # Patch helpers
    'patch_video_downloader_dependencies',
    'patch_time_sleep',
    # Assertion helpers
    'assert_download_successful',
    'assert_retry_backoff_correct',
]
