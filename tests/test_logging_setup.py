"""
Tests for src/cli/logging_setup.py

Covers:
- setup_logging function with various configurations
- Console and file handler creation
- Log level configuration
- Error handling for file creation failures
- FFmpeg debug log setup
"""

import logging
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from src.cli.logging_setup import setup_logging


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create a mock config object."""
    config = Mock()
    config.logging = Mock()
    config.logging.log_level = "INFO"
    config.output = Mock()
    config.output.output_dir = "/tmp/test_output"
    return config


@pytest.fixture
def temp_dir():
    """Create a temporary directory that handles cleanup properly."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    # Cleanup - close all handlers first
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        try:
            handler.close()
        except:
            pass
        root_logger.removeHandler(handler)
    # Now try to remove directory
    import shutil
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except:
        pass


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging state before and after each test."""
    # Clear handlers before test
    root_logger = logging.getLogger()
    old_handlers = root_logger.handlers[:]
    old_level = root_logger.level

    yield

    # Cleanup after test
    for handler in root_logger.handlers[:]:
        try:
            handler.close()
        except:
            pass
        root_logger.removeHandler(handler)

    # Restore original state
    root_logger.setLevel(old_level)


def cleanup_logger(logger):
    """Helper to cleanup a logger's file handlers."""
    root = logging.getLogger()
    for handler in root.handlers[:]:
        try:
            handler.close()
        except:
            pass
        root.removeHandler(handler)


# ============================================================================
# Test Basic Setup
# ============================================================================

class TestSetupLoggingBasic:
    """Test basic setup_logging functionality."""

    def test_setup_logging_returns_logger(self, mock_config, temp_dir):
        """Test that setup_logging returns a logger instance."""
        logger = setup_logging(mock_config, output_dir=temp_dir)
        assert isinstance(logger, logging.Logger)
        cleanup_logger(logger)

    def test_setup_logging_creates_log_directory(self, mock_config, temp_dir):
        """Test that setup_logging creates the logs subdirectory."""
        logger = setup_logging(mock_config, output_dir=temp_dir)
        logs_dir = temp_dir / "logs"
        assert logs_dir.exists()
        assert logs_dir.is_dir()
        cleanup_logger(logger)

    def test_setup_logging_creates_log_files(self, mock_config, temp_dir):
        """Test that setup_logging creates both log files."""
        logger = setup_logging(mock_config, output_dir=temp_dir, run_timestamp="test123")
        cleanup_logger(logger)

        logs_dir = temp_dir / "logs"
        normal_log = logs_dir / "run_test123.log"
        verbose_log = logs_dir / "run_test123_verbose.log"

        assert normal_log.exists()
        assert verbose_log.exists()

    def test_setup_logging_log_paths_attribute(self, mock_config, temp_dir):
        """Test that logger has log_paths attribute."""
        logger = setup_logging(mock_config, output_dir=temp_dir, run_timestamp="test456")

        assert hasattr(logger, 'log_paths')
        assert 'normal' in logger.log_paths
        assert 'verbose' in logger.log_paths
        assert 'ffmpeg_debug' in logger.log_paths
        assert 'test456' in logger.log_paths['normal']
        assert 'test456' in logger.log_paths['verbose']
        cleanup_logger(logger)


# ============================================================================
# Test Log Levels
# ============================================================================

class TestSetupLoggingLevels:
    """Test log level configuration."""

    def test_setup_logging_info_level(self, mock_config, temp_dir):
        """Test INFO log level configuration."""
        mock_config.logging.log_level = "INFO"
        logger = setup_logging(mock_config, output_dir=temp_dir)

        root_logger = logging.getLogger()
        console_handlers = [h for h in root_logger.handlers
                          if isinstance(h, logging.StreamHandler)
                          and not isinstance(h, logging.FileHandler)]
        assert len(console_handlers) >= 1
        assert console_handlers[0].level == logging.INFO
        cleanup_logger(logger)

    def test_setup_logging_debug_level(self, mock_config, temp_dir):
        """Test DEBUG log level configuration."""
        mock_config.logging.log_level = "DEBUG"
        logger = setup_logging(mock_config, output_dir=temp_dir)

        root_logger = logging.getLogger()
        console_handlers = [h for h in root_logger.handlers
                          if isinstance(h, logging.StreamHandler)
                          and not isinstance(h, logging.FileHandler)]
        assert len(console_handlers) >= 1
        assert console_handlers[0].level == logging.DEBUG
        cleanup_logger(logger)

    def test_setup_logging_warning_level(self, mock_config, temp_dir):
        """Test WARNING log level configuration."""
        mock_config.logging.log_level = "WARNING"
        logger = setup_logging(mock_config, output_dir=temp_dir)

        root_logger = logging.getLogger()
        console_handlers = [h for h in root_logger.handlers
                          if isinstance(h, logging.StreamHandler)
                          and not isinstance(h, logging.FileHandler)]
        assert len(console_handlers) >= 1
        assert console_handlers[0].level == logging.WARNING
        cleanup_logger(logger)

    def test_setup_logging_invalid_level_defaults_to_info(self, mock_config, temp_dir):
        """Test that invalid log level defaults to INFO."""
        mock_config.logging.log_level = "INVALID_LEVEL"
        logger = setup_logging(mock_config, output_dir=temp_dir)

        root_logger = logging.getLogger()
        console_handlers = [h for h in root_logger.handlers
                          if isinstance(h, logging.StreamHandler)
                          and not isinstance(h, logging.FileHandler)]
        assert len(console_handlers) >= 1
        assert console_handlers[0].level == logging.INFO
        cleanup_logger(logger)

    def test_setup_logging_lowercase_level(self, mock_config, temp_dir):
        """Test that lowercase log levels are handled."""
        mock_config.logging.log_level = "debug"
        logger = setup_logging(mock_config, output_dir=temp_dir)

        root_logger = logging.getLogger()
        console_handlers = [h for h in root_logger.handlers
                          if isinstance(h, logging.StreamHandler)
                          and not isinstance(h, logging.FileHandler)]
        assert len(console_handlers) >= 1
        assert console_handlers[0].level == logging.DEBUG
        cleanup_logger(logger)


# ============================================================================
# Test Handlers
# ============================================================================

class TestSetupLoggingHandlers:
    """Test handler creation and configuration."""

    def test_setup_logging_clears_existing_handlers(self, mock_config, temp_dir):
        """Test that existing handlers are cleared."""
        root_logger = logging.getLogger()
        # Add a dummy handler
        dummy_handler = logging.StreamHandler()
        root_logger.addHandler(dummy_handler)

        logger = setup_logging(mock_config, output_dir=temp_dir)

        # Should have new handlers, not including the dummy
        assert dummy_handler not in root_logger.handlers
        cleanup_logger(logger)

    def test_setup_logging_creates_console_handler(self, mock_config, temp_dir):
        """Test that console handler is created."""
        logger = setup_logging(mock_config, output_dir=temp_dir)

        root_logger = logging.getLogger()
        console_handlers = [h for h in root_logger.handlers
                          if isinstance(h, logging.StreamHandler)
                          and not isinstance(h, logging.FileHandler)]
        assert len(console_handlers) >= 1
        cleanup_logger(logger)

    def test_setup_logging_creates_file_handlers(self, mock_config, temp_dir):
        """Test that file handlers are created."""
        logger = setup_logging(mock_config, output_dir=temp_dir)

        root_logger = logging.getLogger()
        file_handlers = [h for h in root_logger.handlers
                        if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) >= 2  # Normal and verbose
        cleanup_logger(logger)

    def test_setup_logging_normal_handler_info_level(self, mock_config, temp_dir):
        """Test that normal file handler has INFO level."""
        logger = setup_logging(mock_config, output_dir=temp_dir, run_timestamp="test")

        root_logger = logging.getLogger()
        file_handlers = [h for h in root_logger.handlers
                        if isinstance(h, logging.FileHandler)]

        # Find normal handler (not verbose)
        normal_handlers = [h for h in file_handlers if 'verbose' not in h.baseFilename]
        assert len(normal_handlers) >= 1
        assert normal_handlers[0].level == logging.INFO
        cleanup_logger(logger)

    def test_setup_logging_verbose_handler_debug_level(self, mock_config, temp_dir):
        """Test that verbose file handler has DEBUG level."""
        logger = setup_logging(mock_config, output_dir=temp_dir, run_timestamp="test")

        root_logger = logging.getLogger()
        file_handlers = [h for h in root_logger.handlers
                        if isinstance(h, logging.FileHandler)]

        # Find verbose handler
        verbose_handlers = [h for h in file_handlers if 'verbose' in h.baseFilename]
        assert len(verbose_handlers) >= 1
        assert verbose_handlers[0].level == logging.DEBUG
        cleanup_logger(logger)


# ============================================================================
# Test Timestamp Generation
# ============================================================================

class TestSetupLoggingTimestamp:
    """Test timestamp handling."""

    def test_setup_logging_custom_timestamp(self, mock_config, temp_dir):
        """Test that custom timestamp is used."""
        logger = setup_logging(mock_config, output_dir=temp_dir, run_timestamp="custom_ts")

        assert 'custom_ts' in logger.log_paths['normal']
        assert 'custom_ts' in logger.log_paths['verbose']
        cleanup_logger(logger)

    def test_setup_logging_auto_timestamp(self, mock_config, temp_dir):
        """Test that timestamp is auto-generated when not provided."""
        logger = setup_logging(mock_config, output_dir=temp_dir)

        # Should have a timestamp in the format YYYYMMDD_HHMMSS
        normal_path = logger.log_paths['normal']
        assert 'run_' in normal_path
        filename = Path(normal_path).name
        assert filename.startswith('run_')
        assert len(filename) > len('run_.log')
        cleanup_logger(logger)


# ============================================================================
# Test Output Directory
# ============================================================================

class TestSetupLoggingOutputDir:
    """Test output directory handling."""

    def test_setup_logging_creates_nested_directories(self, mock_config, temp_dir):
        """Test that nested directories are created."""
        nested_dir = temp_dir / "deep" / "nested" / "path"
        logger = setup_logging(mock_config, output_dir=nested_dir)

        logs_dir = nested_dir / "logs"
        assert logs_dir.exists()
        cleanup_logger(logger)

    def test_setup_logging_uses_config_output_dir(self, mock_config, temp_dir):
        """Test that config output_dir is used when output_dir is None."""
        mock_config.output.output_dir = str(temp_dir)
        logger = setup_logging(mock_config, output_dir=None)

        logs_dir = temp_dir / "logs"
        assert logs_dir.exists()
        cleanup_logger(logger)

    def test_setup_logging_string_output_dir(self, mock_config, temp_dir):
        """Test that string output_dir is converted to Path."""
        logger = setup_logging(mock_config, output_dir=str(temp_dir))

        logs_dir = temp_dir / "logs"
        assert logs_dir.exists()
        cleanup_logger(logger)


# ============================================================================
# Test Error Handling
# ============================================================================

class TestSetupLoggingErrorHandling:
    """Test error handling in setup_logging."""

    def test_setup_logging_normal_file_error(self, mock_config, temp_dir, capsys):
        """Test handling of normal log file creation error."""
        call_count = [0]
        original_file_handler = logging.FileHandler

        def mock_file_handler(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:  # First call (normal) fails
                raise PermissionError("Permission denied")
            return original_file_handler(*args, **kwargs)

        with patch('logging.FileHandler', side_effect=mock_file_handler):
            logger = setup_logging(mock_config, output_dir=temp_dir)
            cleanup_logger(logger)

        captured = capsys.readouterr()
        assert "Warning: Could not create normal log file" in captured.out

    def test_setup_logging_verbose_file_error(self, mock_config, temp_dir, capsys):
        """Test handling of verbose log file creation error."""
        call_count = [0]
        original_file_handler = logging.FileHandler

        def mock_file_handler(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:  # Second call (verbose) fails
                raise PermissionError("Permission denied")
            return original_file_handler(*args, **kwargs)

        with patch('logging.FileHandler', side_effect=mock_file_handler):
            logger = setup_logging(mock_config, output_dir=temp_dir)
            cleanup_logger(logger)

        captured = capsys.readouterr()
        assert "Warning: Could not create verbose log file" in captured.out

    def test_setup_logging_ffmpeg_setup_error(self, mock_config, temp_dir, capsys):
        """Test handling of FFmpeg setup error."""
        mock_utils = MagicMock()
        mock_utils.setup_ffmpeg_debug_log = MagicMock(side_effect=RuntimeError("FFmpeg error"))

        with patch.dict('sys.modules', {'src.utils': mock_utils}):
            logger = setup_logging(mock_config, output_dir=temp_dir)
            cleanup_logger(logger)

        captured = capsys.readouterr()
        assert "Warning: Could not create FFmpeg debug log" in captured.out


# ============================================================================
# Test FFmpeg Integration
# ============================================================================

class TestSetupLoggingFFmpeg:
    """Test FFmpeg debug log integration."""

    def test_setup_logging_with_ffmpeg(self, mock_config, temp_dir):
        """Test setup_logging when FFmpeg utils are available."""
        logger = setup_logging(mock_config, output_dir=temp_dir)
        # Should have ffmpeg_debug in log_paths (even if None)
        assert 'ffmpeg_debug' in logger.log_paths
        cleanup_logger(logger)

    def test_setup_logging_ffmpeg_path_stored(self, mock_config, temp_dir):
        """Test that FFmpeg path is stored when successful."""
        mock_ffmpeg_path = str(temp_dir / "logs" / "ffmpeg.log")
        mock_utils = MagicMock()
        mock_utils.setup_ffmpeg_debug_log = MagicMock(return_value=mock_ffmpeg_path)

        with patch.dict('sys.modules', {'src.utils': mock_utils}):
            logger = setup_logging(mock_config, output_dir=temp_dir)
            assert logger.log_paths['ffmpeg_debug'] == mock_ffmpeg_path
            cleanup_logger(logger)


# ============================================================================
# Test Logging Output
# ============================================================================

class TestSetupLoggingOutput:
    """Test actual logging output."""

    def test_setup_logging_writes_to_normal_log(self, mock_config, temp_dir):
        """Test that INFO messages are written to normal log."""
        logger = setup_logging(mock_config, output_dir=temp_dir, run_timestamp="test")

        # Log a message
        test_logger = logging.getLogger("test_module")
        test_logger.info("Test info message")

        # Flush and close handlers
        cleanup_logger(logger)

        # Check normal log
        normal_log = temp_dir / "logs" / "run_test.log"
        content = normal_log.read_text()
        assert "Test info message" in content

    def test_setup_logging_writes_to_verbose_log(self, mock_config, temp_dir):
        """Test that DEBUG messages are written to verbose log."""
        logger = setup_logging(mock_config, output_dir=temp_dir, run_timestamp="test")

        # Log a debug message
        test_logger = logging.getLogger("test_module")
        test_logger.debug("Test debug message")

        # Flush and close handlers
        cleanup_logger(logger)

        # Check verbose log
        verbose_log = temp_dir / "logs" / "run_test_verbose.log"
        content = verbose_log.read_text()
        assert "Test debug message" in content

    def test_setup_logging_debug_not_in_normal_log(self, mock_config, temp_dir):
        """Test that DEBUG messages are NOT in normal log (only INFO+)."""
        mock_config.logging.log_level = "INFO"
        logger = setup_logging(mock_config, output_dir=temp_dir, run_timestamp="test")

        # Log a debug message
        test_logger = logging.getLogger("test_module")
        test_logger.debug("Secret debug message")

        # Flush and close handlers
        cleanup_logger(logger)

        # Check normal log - should NOT contain debug message
        normal_log = temp_dir / "logs" / "run_test.log"
        content = normal_log.read_text()
        assert "Secret debug message" not in content


# ============================================================================
# Test Root Logger Configuration
# ============================================================================

class TestSetupLoggingRootLogger:
    """Test root logger configuration."""

    def test_setup_logging_root_level_debug(self, mock_config, temp_dir):
        """Test that root logger level is set to DEBUG to capture all."""
        logger = setup_logging(mock_config, output_dir=temp_dir)

        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG
        cleanup_logger(logger)

    def test_setup_logging_init_message(self, mock_config, temp_dir):
        """Test that initialization message is logged."""
        logger = setup_logging(mock_config, output_dir=temp_dir, run_timestamp="test")

        # Flush and close handlers
        cleanup_logger(logger)

        # Check that init message was logged
        normal_log = temp_dir / "logs" / "run_test.log"
        content = normal_log.read_text()
        assert "Logging initialized" in content
