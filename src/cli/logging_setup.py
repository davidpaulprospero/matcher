"""
Logging setup utilities for the matcher pipeline.

Extracted from main.py (Jan 2026).

Features:
- Dual log files (normal + verbose)
- Console output matching config level
- FFmpeg debug log capture
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..config import Config


def setup_logging(
    config: 'Config',
    output_dir: Path = None,
    run_timestamp: str = None
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
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # Normal log file handler (INFO level)
    try:
        normal_handler = logging.FileHandler(normal_log_path, encoding='utf-8')
        normal_handler.setLevel(logging.INFO)
        normal_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        normal_handler.setFormatter(normal_format)
        root_logger.addHandler(normal_handler)
    except Exception as e:
        print(f"  Warning: Could not create normal log file: {e}")

    # Verbose log file handler (DEBUG level - everything)
    try:
        verbose_handler = logging.FileHandler(verbose_log_path, encoding='utf-8')
        verbose_handler.setLevel(logging.DEBUG)
        verbose_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        verbose_handler.setFormatter(verbose_format)
        root_logger.addHandler(verbose_handler)
    except Exception as e:
        print(f"  Warning: Could not create verbose log file: {e}")

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

    # Store paths for reference
    logger.log_paths = {
        'normal': str(normal_log_path),
        'verbose': str(verbose_log_path),
        'ffmpeg_debug': str(ffmpeg_debug_path) if ffmpeg_debug_path else None
    }

    return logger
