"""
Pytest configuration for test suite.

Provides fixtures and marks for integration tests that require
external resources (videos, API keys, etc.).
"""

import pytest
from pathlib import Path
from typing import List


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "unit: marks tests as fast unit tests (no external resources, <1s)"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (require external resources)"
    )
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (>5 seconds)"
    )


@pytest.fixture
def srt_path(tmp_path) -> Path:
    """Fixture for SRT file path (integration tests)."""
    pytest.skip("Integration test - requires actual SRT file")


@pytest.fixture
def config(tmp_path):
    """Fixture for config."""
    from unittest.mock import Mock, MagicMock

    mock_config = Mock()
    mock_config.cache = Mock(cache_dir=str(tmp_path / "cache"))
    mock_config.output = Mock(output_dir=str(tmp_path / "output"))
    mock_config.transcription = Mock(model="base", language="en")
    mock_config.matching = Mock(
        min_confidence=0.5,
        location_matching=Mock(enabled=False, geonames_username="")
    )
    mock_config.keyword = Mock(max_keywords=10, min_keyword_length=3)
    mock_config.download = Mock(root_dir=str(tmp_path / "downloads"))

    # Create directories
    (tmp_path / "cache").mkdir(exist_ok=True)
    (tmp_path / "output").mkdir(exist_ok=True)
    (tmp_path / "downloads").mkdir(exist_ok=True)

    return mock_config


@pytest.fixture
def output_dir(tmp_path) -> Path:
    """Fixture for output directory."""
    output = tmp_path / "output"
    output.mkdir(parents=True, exist_ok=True)
    return output


@pytest.fixture
def cookies_path() -> str:
    """Fixture for cookies path (integration tests)."""
    pytest.skip("Integration test - requires cookies file")


@pytest.fixture
def video_paths(tmp_path) -> List[str]:
    """Fixture for video paths (integration tests)."""
    pytest.skip("Integration test - requires downloaded videos")


@pytest.fixture
def cache_dir(tmp_path) -> Path:
    """Fixture for cache directory."""
    cache = tmp_path / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


@pytest.fixture
def video_path(tmp_path) -> str:
    """Fixture for single video path (integration tests)."""
    pytest.skip("Integration test - requires actual video file")


@pytest.fixture
def runner():
    """Fixture for test runner (integration tests)."""
    pytest.skip("Integration test - requires runner setup")


@pytest.fixture
def texts():
    """Fixture for text list."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "The early bird catches the worm."
    ]


@pytest.fixture
def voiceover_segments():
    """Fixture for voiceover segments."""
    return [
        {"index": 0, "text": "Welcome to the tutorial", "start": 0.0, "end": 2.5},
        {"index": 1, "text": "We will learn about Python", "start": 2.5, "end": 5.0},
        {"index": 2, "text": "Let's get started", "start": 5.0, "end": 7.0},
    ]


@pytest.fixture
def video_segments():
    """Fixture for video segments."""
    return [
        {"file": "video1.mp4", "text": "Python tutorial intro", "start": 0.0, "end": 10.0, "duration_tier": "short"},
        {"file": "video2.mp4", "text": "Learning programming", "start": 0.0, "end": 15.0, "duration_tier": "medium"},
        {"file": "video3.mp4", "text": "Getting started guide", "start": 0.0, "end": 8.0, "duration_tier": "short"},
    ]


@pytest.fixture
def vo_embeddings():
    """Fixture for voiceover embeddings."""
    import numpy as np
    # Return mock 384-dimensional embeddings (common embedding size)
    np.random.seed(42)
    return np.random.rand(3, 384).astype(np.float32)


@pytest.fixture
def video_embeddings():
    """Fixture for video embeddings."""
    import numpy as np
    # Return mock 384-dimensional embeddings (common embedding size)
    np.random.seed(123)
    return np.random.rand(3, 384).astype(np.float32)


@pytest.fixture
def matches():
    """Fixture for match results."""
    from unittest.mock import Mock
    return [
        Mock(
            vo_index=0,
            video_file="video1.mp4",
            start=0.0,
            end=2.5,
            confidence=0.85,
            strategy="primary"
        ),
        Mock(
            vo_index=1,
            video_file="video2.mp4",
            start=0.0,
            end=5.0,
            confidence=0.72,
            strategy="primary"
        ),
        Mock(
            vo_index=2,
            video_file="video3.mp4",
            start=0.0,
            end=3.0,
            confidence=0.78,
            strategy="primary"
        ),
    ]


@pytest.fixture
def temp_dir(tmp_path):
    """Fixture for temporary directory."""
    # Simple temp directory that doesn't require external resources
    return tmp_path
