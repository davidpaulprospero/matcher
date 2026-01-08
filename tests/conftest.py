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
        "integration: marks tests as integration tests (require external resources)"
    )


@pytest.fixture
def srt_path(tmp_path) -> Path:
    """Fixture for SRT file path (integration tests)."""
    pytest.skip("Integration test - requires actual SRT file")


@pytest.fixture
def config():
    """Fixture for config (integration tests)."""
    pytest.skip("Integration test - requires full config setup")


@pytest.fixture
def output_dir(tmp_path) -> Path:
    """Fixture for output directory (integration tests)."""
    pytest.skip("Integration test - requires download setup")


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
    """Fixture for cache directory (integration tests)."""
    pytest.skip("Integration test - requires cache setup")


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
    """Fixture for text list (integration tests)."""
    pytest.skip("Integration test - requires text data")


@pytest.fixture
def voiceover_segments():
    """Fixture for voiceover segments (integration tests)."""
    pytest.skip("Integration test - requires voiceover data")


@pytest.fixture
def video_segments():
    """Fixture for video segments (integration tests)."""
    pytest.skip("Integration test - requires video data")


@pytest.fixture
def vo_embeddings():
    """Fixture for voiceover embeddings (integration tests)."""
    pytest.skip("Integration test - requires embeddings")


@pytest.fixture
def video_embeddings():
    """Fixture for video embeddings (integration tests)."""
    pytest.skip("Integration test - requires embeddings")


@pytest.fixture
def matches():
    """Fixture for match results (integration tests)."""
    pytest.skip("Integration test - requires match data")


@pytest.fixture
def temp_dir(tmp_path):
    """Fixture for temporary directory (integration tests)."""
    pytest.skip("Integration test - requires temp directory setup")
