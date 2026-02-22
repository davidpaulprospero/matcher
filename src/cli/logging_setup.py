"""
Logging setup utilities for the matcher pipeline.

Extracted from main.py (Jan 2026).

Features:
- Dual log files (normal + verbose)
- Console output matching config level
- FFmpeg debug log capture
- Correlation ID support for request tracing (US-159-005)
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from ..config import Config


class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    US-162-006: Outputs logs as JSON lines with correlation_id, timestamp, level, message, and context.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Build context from record attributes
        context: Dict[str, Any] = {}

        # Add correlation_id if present
        if hasattr(record, 'correlation_id') and record.correlation_id != '-':
            context['correlation_id'] = record.correlation_id

        # Add module and line number for debugging
        if record.name:
            context['logger'] = record.name
        if record.lineno:
            context['line'] = record.lineno

        # Build the JSON log entry
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
        }

        # Add correlation_id at top level for easier filtering
        if hasattr(record, 'correlation_id') and record.correlation_id != '-':
            log_entry['correlation_id'] = record.correlation_id

        # Add context if present
        if context:
            log_entry['context'] = context

        # Add error code if present in message
        msg = record.getMessage()
        if '[ERROR]' in msg or '[DL-' in msg or '[MATCH-' in msg or '[SEARCH-' in msg:
            # Extract error code from message
            import re
            error_match = re.search(r'\[([A-Z]+-\d+)\]', msg)
            if error_match:
                log_entry['error_code'] = error_match.group(1)

        return json.dumps(log_entry)


class CorrelationIdFilter(logging.Filter):
    """Logging filter that adds correlation ID to log records.

    US-159-005: Enables structured logging with correlation IDs for request tracing.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation_id to the log record if available.

        Args:
            record: The log record to filter.

        Returns:
            True to allow the record through.
        """
        try:
            # Try to import from pipeline module - will fail if not yet imported
            from ..pipeline import get_correlation_id
            correlation_id = get_correlation_id()
            record.correlation_id = correlation_id if correlation_id else '-'
        except (ImportError, Exception):
            # If pipeline module not available or no correlation ID set
            record.correlation_id = '-'
        return True


def get_correlation_id_formatter(format_string: str, datefmt: str) -> logging.Formatter:
    """Create a formatter that includes correlation ID in the output.

    Args:
        format_string: The format string for the formatter.
        datefmt: The date format string.

    Returns:
        A formatter that includes correlation_id.
    """
    # Insert correlation_id into the format string if not already present
    if '%(correlation_id)s' not in format_string:
        format_string = '%(asctime)s - %(correlation_id)s - ' + format_string.lstrip('%(asctime)s - ')

    formatter = logging.Formatter(format_string, datefmt=datefmt)
    formatter.addFilter(CorrelationIdFilter())
    return formatter


def setup_logging(
    config: 'Config',
    output_dir: Path = None,
    run_timestamp: str = None,
    json_logs: bool = False
) -> logging.Logger:
    """
    Setup logging with dual log files per run.

    Creates two log files:
    1. run_{timestamp}.log - Normal logging (INFO level, matches console)
    2. run_{timestamp}_verbose.log - Verbose logging (DEBUG level, everything)

    Args:
        config: Configuration object
        output_dir: Directory to save log files (default: project output dir)
        run_timestamp: Timestamp string for filenames (default: auto-generated)

    Returns:
        Logger instance with log_paths attribute
    """
    # Get log level from config
    log_level = getattr(logging, config.logging.log_level.upper(), logging.INFO)

    # Generate timestamp if not provided
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine output directory
    if output_dir is None:
        output_dir = Path(config.output.output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create logs subdirectory
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Log file paths
    normal_log_path = logs_dir / f"run_{run_timestamp}.log"
    verbose_log_path = logs_dir / f"run_{run_timestamp}_verbose.log"

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all levels

    # Clear any existing handlers
    root_logger.handlers.clear()

    # Console handler (matches config level)
    # US-159-005: Add correlation ID filter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.addFilter(CorrelationIdFilter())
    console_format = logging.Formatter(
        '%(asctime)s - %(correlation_id)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # Normal log file handler (INFO level)
    # US-159-005: Add correlation ID to log format
    try:
        normal_handler = logging.FileHandler(normal_log_path, encoding='utf-8')
        normal_handler.setLevel(logging.INFO)
        normal_handler.addFilter(CorrelationIdFilter())
        normal_format = logging.Formatter(
            '%(asctime)s - %(correlation_id)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        normal_handler.setFormatter(normal_format)
        root_logger.addHandler(normal_handler)
    except Exception as e:
        print(f"  Warning: Could not create normal log file: {e}")

    # Verbose log file handler (DEBUG level - everything)
    # US-159-005: Add correlation ID to log format
    try:
        verbose_handler = logging.FileHandler(verbose_log_path, encoding='utf-8')
        verbose_handler.setLevel(logging.DEBUG)
        verbose_handler.addFilter(CorrelationIdFilter())
        verbose_format = logging.Formatter(
            '%(asctime)s - %(correlation_id)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        verbose_handler.setFormatter(verbose_format)
        root_logger.addHandler(verbose_handler)
    except Exception as e:
        print(f"  Warning: Could not create verbose log file: {e}")

    # JSON log file handler (US-162-006: Structured JSON logging for machine parsing)
    json_log_path = None
    if json_logs:
        try:
            json_log_path = logs_dir / f"run_{run_timestamp}.jsonl"
            json_handler = logging.FileHandler(json_log_path, encoding='utf-8')
            json_handler.setLevel(logging.DEBUG)
            json_handler.setFormatter(JsonFormatter())
            root_logger.addHandler(json_handler)
        except Exception as e:
            print(f"  Warning: Could not create JSON log file: {e}")

    # Setup FFmpeg debug log (captures H.264 decoder warnings from OpenCV)
    ffmpeg_debug_path = None
    try:
        from ..utils import setup_ffmpeg_debug_log
        ffmpeg_debug_path = setup_ffmpeg_debug_log(logs_dir)
    except Exception as e:
        print(f"  Warning: Could not create FFmpeg debug log: {e}")

    # Log startup info
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized")
    logger.debug(f"Normal log: {normal_log_path}")
    logger.debug(f"Verbose log: {verbose_log_path}")
    if ffmpeg_debug_path:
        logger.debug(f"FFmpeg debug log: {ffmpeg_debug_path}")
    if json_log_path:
        logger.debug(f"JSON log: {json_log_path}")

    # Store paths for reference
    logger.log_paths = {
        'normal': str(normal_log_path),
        'verbose': str(verbose_log_path),
        'ffmpeg_debug': str(ffmpeg_debug_path) if ffmpeg_debug_path else None,
        'json': str(json_log_path) if json_log_path else None
    }

    return logger
