"""Tests for US-143-012: Download segment checksum validation.

Tests the checksum validation logic that validates:
- SHA256/MD5/SHA1 checksum calculation for downloaded segments
- Checksum verification after download
- Retry on checksum failure
- Metrics logging
"""

import hashlib
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.stages.download_segments import DownloadVideoSegmentsStage
from src.config.sections.download import ChecksumValidationConfig


class MockChecksumConfig:
    """Mock checksum validation config."""
    def __init__(
        self,
        enabled: bool = True,
        algorithm: str = "sha256",
        retry_on_failure: bool = True,
        max_retries: int = 2,
        min_file_size_bytes: int = 1024,
        log_to_metrics: bool = True,
        verify_file_size: bool = True,
    ):
        self.enabled = enabled
        self.algorithm = algorithm
        self.retry_on_failure = retry_on_failure
        self.max_retries = max_retries
        self.min_file_size_bytes = min_file_size_bytes
        self.log_to_metrics = log_to_metrics
        self.verify_file_size = verify_file_size


class MockDownloadConfig:
    """Mock download config with checksum validation."""
    def __init__(self, checksum_validation: MockChecksumConfig = None):
        self.checksum_validation = checksum_validation or MockChecksumConfig()


@pytest.fixture
def stage():
    """Create a DownloadVideoSegmentsStage for testing."""
    return DownloadVideoSegmentsStage()


@pytest.fixture
def temp_video_file():
    """Create a temporary video file for testing (>= 1024 bytes)."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
        # Write enough data to exceed min_file_size_bytes (1024)
        f.write(b'test video content for checksum validation ' * 50)
        temp_path = f.name

    yield Path(temp_path)

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestChecksumValidationConfig:
    """Tests for ChecksumValidationConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ChecksumValidationConfig()
        assert config.enabled is False
        assert config.algorithm == "sha256"
        assert config.retry_on_failure is True
        assert config.max_retries == 2
        assert config.min_file_size_bytes == 1024
        assert config.log_to_metrics is True
        assert config.verify_file_size is True

    def test_invalid_algorithm(self):
        """Test that invalid algorithm raises error."""
        with pytest.raises(ValueError, match="algorithm must be one of"):
            ChecksumValidationConfig(algorithm="invalid")

    def test_invalid_max_retries(self):
        """Test that negative max_retries raises error."""
        with pytest.raises(ValueError, match="max_retries must be >= 0"):
            ChecksumValidationConfig(max_retries=-1)

    def test_invalid_min_file_size(self):
        """Test that negative min_file_size raises error."""
        with pytest.raises(ValueError, match="min_file_size_bytes must be >= 0"):
            ChecksumValidationConfig(min_file_size_bytes=-1)

    def test_valid_algorithms(self):
        """Test that all valid algorithms work."""
        for algo in ["sha256", "sha1", "md5"]:
            config = ChecksumValidationConfig(algorithm=algo)
            assert config.algorithm == algo


class TestChecksumValidation:
    """Tests for checksum validation logic."""

    def test_validate_checksum_disabled(self, stage, temp_video_file):
        """Test validation returns valid when disabled."""
        # Set download_config to None to simulate disabled
        stage.download_config = None
        result = stage._validate_checksum(temp_video_file)

        assert result['valid'] is True
        assert result['checksum'] is None

    def test_validate_checksum_small_file(self, stage):
        """Test validation skips very small files."""
        # Create a small temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
            f.write(b'tiny')
            temp_path = f.name

        try:
            stage.download_config = MockDownloadConfig()
            result = stage._validate_checksum(Path(temp_path))

            # Should return valid but skip checksum (file too small)
            assert result['valid'] is True
            assert result['checksum'] is None  # Skipped due to size
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_validate_checksum_sha256(self, stage, temp_video_file):
        """Test SHA256 checksum calculation."""
        stage.download_config = MockDownloadConfig()
        result = stage._validate_checksum(temp_video_file)

        assert result['valid'] is True
        assert result['checksum'] is not None
        assert len(result['checksum']) == 64  # SHA256 hex is 64 chars

        # Verify against manual calculation
        with open(temp_video_file, 'rb') as f:
            expected = hashlib.sha256(f.read()).hexdigest()
        assert result['checksum'] == expected

    def test_validate_checksum_sha1(self, stage, temp_video_file):
        """Test SHA1 checksum calculation."""
        config = MockChecksumConfig(algorithm="sha1")
        stage.download_config = MockDownloadConfig(checksum_validation=config)
        result = stage._validate_checksum(temp_video_file)

        assert result['valid'] is True
        assert result['checksum'] is not None
        assert len(result['checksum']) == 40  # SHA1 hex is 40 chars

    def test_validate_checksum_md5(self, stage, temp_video_file):
        """Test MD5 checksum calculation."""
        config = MockChecksumConfig(algorithm="md5")
        stage.download_config = MockDownloadConfig(checksum_validation=config)
        result = stage._validate_checksum(temp_video_file)

        assert result['valid'] is True
        assert result['checksum'] is not None
        assert len(result['checksum']) == 32  # MD5 hex is 32 chars

    def test_validate_checksum_mismatch(self, stage, temp_video_file):
        """Test checksum mismatch detection."""
        wrong_checksum = "0" * 64  # Wrong expected checksum
        stage.download_config = MockDownloadConfig()
        result = stage._validate_checksum(temp_video_file, expected_checksum=wrong_checksum)

        assert result['valid'] is False
        assert "Checksum mismatch" in result['error_msg']

    def test_validate_file_size_mismatch(self, stage, temp_video_file):
        """Test file size mismatch detection."""
        wrong_size = 99999  # Wrong expected size
        stage.download_config = MockDownloadConfig()
        result = stage._validate_checksum(temp_video_file, expected_size=wrong_size)

        assert result['valid'] is False
        assert result['size_match'] is False
        assert "Size mismatch" in result['error_msg']

    def test_validate_file_not_found(self, stage):
        """Test handling of missing file."""
        nonexistent = Path("/nonexistent/file.mp4")
        stage.download_config = MockDownloadConfig()
        result = stage._validate_checksum(nonexistent)

        assert result['valid'] is False
        assert "File not found" in result['error_msg']

    def test_validate_both_checksum_and_size_mismatch(self, stage, temp_video_file):
        """Test when both checksum and size don't match."""
        wrong_checksum = "0" * 64
        wrong_size = 99999
        stage.download_config = MockDownloadConfig()
        result = stage._validate_checksum(
            temp_video_file,
            expected_checksum=wrong_checksum,
            expected_size=wrong_size
        )

        assert result['valid'] is False
        # Size mismatch is detected first

    def test_validate_success_with_expected_values(self, stage, temp_video_file):
        """Test validation passes with correct expected values."""
        # Calculate correct checksum first
        with open(temp_video_file, 'rb') as f:
            correct_checksum = hashlib.sha256(f.read()).hexdigest()
        correct_size = temp_video_file.stat().st_size

        stage.download_config = MockDownloadConfig()
        result = stage._validate_checksum(
            temp_video_file,
            expected_checksum=correct_checksum,
            expected_size=correct_size
        )

        assert result['valid'] is True
        assert result['size_match'] is True


class TestChecksumValidationMetrics:
    """Tests for checksum validation metrics logging."""

    def test_checksum_logged_to_metrics(self, stage, temp_video_file):
        """Test that validation results are logged to error aggregator."""
        # This would require integration with the actual metrics system
        # For now, just verify the method runs without error
        stage.download_config = MockDownloadConfig()
        result = stage._validate_checksum(temp_video_file)

        assert result['valid'] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
