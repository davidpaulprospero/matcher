"""Test mode configuration for pipeline testing.

Controls pipeline behavior when running in test mode (--test-mode flag).
Limits video/segment counts and skips expensive operations for fast iteration.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def is_mock_rate_limits_enabled(test_mode_config: TestModeConfig = None) -> bool:
    """Check if mock rate limits should be enabled.

    Checks both the environment variable MOCK_RATE_LIMITS=1 and the config option.
    Environment variable takes precedence if set.

    Args:
        test_mode_config: Optional TestModeConfig to check. If None, only checks env var.

    Returns:
        True if mock rate limits should be enabled
    """
    # Check environment variable first (takes precedence)
    env_value = os.environ.get('MOCK_RATE_LIMITS', '').lower()
    if env_value in ('1', 'true', 'yes'):
        return True
    if env_value in ('0', 'false', 'no'):
        return False

    # Fall back to config setting
    if test_mode_config is not None:
        return getattr(test_mode_config, 'mock_rate_limits', False)

    return False


def get_mock_delay_seconds(test_mode_config: TestModeConfig = None) -> float:
    """Get the mock delay seconds from config or environment variable.

    Environment variable MOCK_DELAY_SECONDS takes precedence if set.

    Args:
        test_mode_config: Optional TestModeConfig to check

    Returns:
        Mock delay in seconds (default: 0.01)
    """
    # Check environment variable first
    env_value = os.environ.get('MOCK_DELAY_SECONDS')
    if env_value is not None:
        try:
            return float(env_value)
        except ValueError:
            pass  # Fall back to config

    # Fall back to config setting
    if test_mode_config is not None:
        return getattr(test_mode_config, 'mock_delay_seconds', 0.01)

    return 0.01


def is_mock_youtube_api_enabled(test_mode_config: TestModeConfig = None) -> bool:
    """Check if mock YouTube API should be enabled.

    Checks both the environment variable MOCK_YOUTUBE_API=1 and the config option.
    Environment variable takes precedence if set.

    When enabled, YouTubeAPIClient will use recorded fixtures instead of making
    actual API calls. This is useful for testing without quota consumption.

    Args:
        test_mode_config: Optional TestModeConfig to check. If None, only checks env var.

    Returns:
        True if mock YouTube API should be enabled
    """
    # Check environment variable first (takes precedence)
    env_value = os.environ.get('MOCK_YOUTUBE_API', '').lower()
    if env_value in ('1', 'true', 'yes'):
        return True
    if env_value in ('0', 'false', 'no'):
        return False

    # Fall back to config setting
    if test_mode_config is not None:
        return getattr(test_mode_config, 'mock_youtube_api', False)

    return False


@dataclass
class TestModeConfig:
    """Configuration for test mode - enables fast pipeline testing.

    When enabled, reduces resource usage by limiting video counts
    and skipping expensive operations like embeddings and iterative matching.
    """
    # Maximum number of videos to process per query
    max_videos: int = 3

    # Maximum number of segments to generate matches for
    max_segments: int = 10

    # Skip embedding computation (use pre-computed or skip entirely)
    skip_embeddings: bool = True

    # Skip iterative matching (run only initial matching pass)
    skip_iterative: bool = True

    # Maximum number of segments to download
    max_downloads: int = 3

    # Enable mock delays for rate limiting (skip actual delays in test mode)
    mock_rate_limits: bool = True

    # Mock delay duration in seconds (when mock_rate_limits is enabled)
    mock_delay_seconds: float = 0.01

    # Enable mock YouTube API responses (use fixtures instead of real API calls)
    mock_youtube_api: bool = False


__all__ = ['TestModeConfig', 'is_mock_rate_limits_enabled', 'get_mock_delay_seconds', 'is_mock_youtube_api_enabled']
