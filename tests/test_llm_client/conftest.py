"""
Pytest configuration and shared fixtures for LLM client tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def mock_api_key():
    """Provide a mock API key for testing."""
    return "test_api_key_12345"


@pytest.fixture
def sample_json_response():
    """Provide sample JSON response for testing."""
    return '{"status": "success", "data": {"id": 1, "name": "test"}}'


@pytest.fixture
def sample_json_array_response():
    """Provide sample JSON array response for testing."""
    return '[{"id": 1, "name": "first"}, {"id": 2, "name": "second"}]'


@pytest.fixture(autouse=True)
def cleanup_env_vars(monkeypatch):
    """Cleanup environment variables after each test."""
    # This runs after each test to prevent env var pollution
    yield
    # Cleanup is automatic with monkeypatch
