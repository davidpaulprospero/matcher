"""
Structured logging utilities for the download fallback system.

Provides consistent, detailed logging that makes debugging easy by:
- Using structured log formats with context
- Including timing information
- Capturing full error details with stack traces
- Tagging logs by tier/component for filtering
"""

from __future__ import annotations

import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional

# Configure module logger
logger = logging.getLogger("fallback")


class LogLevel(Enum):
    """Log levels for fallback operations."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class FallbackLogContext:
    """Context for a fallback operation."""
    tier: str
    operation: str
    video_id: str
    instance: Optional[str] = None
    attempt: int = 1
    max_attempts: int = 1
    start_time: float = field(default_factory=time.time)
    extra: dict = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.time() - self.start_time) * 1000

    @property
    def prefix(self) -> str:
        """Get log prefix with context."""
        parts = [f"[{self.tier}]"]
        if self.instance:
            parts.append(f"[{self._short_instance}]")
        parts.append(f"[{self.video_id}]")
        if self.max_attempts > 1:
            parts.append(f"[{self.attempt}/{self.max_attempts}]")
        return " ".join(parts)

    @property
    def _short_instance(self) -> str:
        """Get shortened instance URL for logging."""
        if not self.instance:
            return ""
        # Extract domain from URL
        instance = self.instance.replace("https://", "").replace("http://", "")
        if "/" in instance:
            instance = instance.split("/")[0]
        return instance[:25]


class FallbackLogger:
    """
    Structured logger for fallback operations.

    Usage:
        log = FallbackLogger("TIER_2_TRANSCRIPT_API")
        log.start_operation("fetch", "dQw4w9WgXcQ")
        log.info("Fetching transcript...")
        log.success("Got 150 segments")
        # or
        log.error("Rate limited", error=e)
    """

    def __init__(self, tier: str):
        self.tier = tier
        self._context: Optional[FallbackLogContext] = None
        self._logger = logging.getLogger(f"fallback.{tier.lower()}")

    def start_operation(
        self,
        operation: str,
        video_id: str,
        instance: Optional[str] = None,
        attempt: int = 1,
        max_attempts: int = 1,
        **extra
    ) -> FallbackLogContext:
        """Start a new operation context."""
        self._context = FallbackLogContext(
            tier=self.tier,
            operation=operation,
            video_id=video_id,
            instance=instance,
            attempt=attempt,
            max_attempts=max_attempts,
            extra=extra,
        )
        return self._context

    def _log(self, level: int, msg: str, **kwargs):
        """Internal log method."""
        ctx = self._context
        if ctx:
            prefix = ctx.prefix
            elapsed = f" [{ctx.elapsed_ms:.0f}ms]" if ctx.elapsed_ms > 0 else ""
            full_msg = f"{prefix}{elapsed} {msg}"
        else:
            full_msg = f"[{self.tier}] {msg}"

        # Add extra context
        extra = kwargs.pop("extra", {})
        if ctx:
            extra.update(ctx.extra)

        self._logger.log(level, full_msg, extra=extra, **kwargs)

    def debug(self, msg: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, error: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception details."""
        if error:
            error_type = type(error).__name__
            error_msg = str(error)
            msg = f"{msg} | {error_type}: {error_msg}"

            # Log full traceback at debug level
            tb = traceback.format_exc()
            if tb and "NoneType" not in tb:
                self._log(logging.DEBUG, f"Traceback:\n{tb}")

        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, error: Optional[Exception] = None, **kwargs):
        """Log critical message."""
        if error:
            error_type = type(error).__name__
            error_msg = str(error)
            msg = f"{msg} | {error_type}: {error_msg}"

            # Always log traceback for critical errors
            tb = traceback.format_exc()
            if tb and "NoneType" not in tb:
                self._log(logging.ERROR, f"Traceback:\n{tb}")

        self._log(logging.CRITICAL, msg, **kwargs)

    def success(self, msg: str, **kwargs):
        """Log success message (info level with SUCCESS tag)."""
        self._log(logging.INFO, f"SUCCESS: {msg}", **kwargs)

    def failure(self, msg: str, reason: str = "", **kwargs):
        """Log failure message (warning level with FAILED tag)."""
        full_msg = f"FAILED: {msg}"
        if reason:
            full_msg += f" | Reason: {reason}"
        self._log(logging.WARNING, full_msg, **kwargs)

    def rate_limited(self, source: str = "YouTube", **kwargs):
        """Log rate limit detection."""
        self._log(
            logging.WARNING,
            f"RATE_LIMITED: Detected rate limiting from {source} (HTTP 429)",
            **kwargs
        )

    def request(
        self,
        method: str,
        url: str,
        status: Optional[int] = None,
        size: Optional[int] = None,
        **kwargs
    ):
        """Log HTTP request details."""
        url_short = url[:80] + "..." if len(url) > 80 else url
        parts = [f"REQUEST: {method} {url_short}"]
        if status is not None:
            parts.append(f"-> {status}")
        if size is not None:
            parts.append(f"({size} bytes)")
        self._log(logging.DEBUG, " ".join(parts), **kwargs)

    def response(
        self,
        status: int,
        size: int,
        content_type: Optional[str] = None,
        **kwargs
    ):
        """Log HTTP response details."""
        parts = [f"RESPONSE: HTTP {status}"]
        parts.append(f"({size} bytes)")
        if content_type:
            parts.append(f"[{content_type}]")
        self._log(logging.DEBUG, " ".join(parts), **kwargs)

    def parsing(self, format: str, segments: int, **kwargs):
        """Log parsing results."""
        self._log(logging.DEBUG, f"PARSED: {format} format -> {segments} segments", **kwargs)

    def instance_status(self, instance: str, status: str, **kwargs):
        """Log instance health status."""
        self._log(logging.DEBUG, f"INSTANCE: {instance} -> {status}", **kwargs)


@contextmanager
def log_operation(
    log: FallbackLogger,
    operation: str,
    video_id: str,
    instance: Optional[str] = None,
    **extra
):
    """
    Context manager for logging an operation with timing.

    Usage:
        log = FallbackLogger("TIER_2")
        with log_operation(log, "fetch", "abc123") as ctx:
            # do work
            log.info("Working...")
        # Automatically logs completion with timing
    """
    ctx = log.start_operation(operation, video_id, instance=instance, **extra)
    log.debug(f"Starting {operation}...")

    try:
        yield ctx
    except Exception as e:
        log.error(f"Exception during {operation}", error=e)
        raise
    finally:
        log.debug(f"Completed {operation} in {ctx.elapsed_ms:.0f}ms")


def log_tier_attempt(tier: str):
    """
    Decorator to log tier fetch attempts.

    Usage:
        @log_tier_attempt("TIER_2_TRANSCRIPT_API")
        def fetch(self, video_id: str) -> CaptionResult:
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, video_id: str, *args, **kwargs):
            log = FallbackLogger(tier)
            log.start_operation("fetch", video_id)
            log.info(f"Attempting {tier}...")

            try:
                result = func(self, video_id, *args, **kwargs)

                # Check result
                if hasattr(result, "success"):
                    if result.success:
                        segments = len(result.segments) if hasattr(result, "segments") else 0
                        log.success(f"Got {segments} segments")
                    else:
                        log.failure("Fetch failed", reason=getattr(result, "error", "Unknown"))

                return result

            except Exception as e:
                log.error("Exception during fetch", error=e)
                raise

        return wrapper
    return decorator


# Pre-configured loggers for each tier
TIER_LOGGERS = {
    "YTDLP": FallbackLogger("TIER_1_YTDLP"),
    "TRANSCRIPT_API": FallbackLogger("TIER_2_TRANSCRIPT_API"),
    "INNERTUBE": FallbackLogger("TIER_3_INNERTUBE"),
    "INVIDIOUS": FallbackLogger("TIER_4_INVIDIOUS"),
    "PIPED": FallbackLogger("TIER_5_PIPED"),
    "WHISPER": FallbackLogger("TIER_6_WHISPER"),
    "CHAIN": FallbackLogger("FALLBACK_CHAIN"),
    "VIDEO_INVIDIOUS": FallbackLogger("VIDEO_TIER_2_INVIDIOUS"),
    "VIDEO_PIPED": FallbackLogger("VIDEO_TIER_3_PIPED"),
    "VIDEO_COBALT": FallbackLogger("VIDEO_TIER_4_COBALT"),
}


def get_tier_logger(tier: str) -> FallbackLogger:
    """Get logger for a specific tier."""
    return TIER_LOGGERS.get(tier, FallbackLogger(tier))


def configure_fallback_logging(
    level: int = logging.INFO,
    format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handler: Optional[logging.Handler] = None,
):
    """
    Configure logging for the fallback system.

    Args:
        level: Logging level (default INFO)
        format: Log format string
        handler: Optional custom handler
    """
    fallback_logger = logging.getLogger("fallback")
    fallback_logger.setLevel(level)

    if not fallback_logger.handlers:
        if handler:
            fallback_logger.addHandler(handler)
        else:
            console = logging.StreamHandler()
            console.setLevel(level)
            console.setFormatter(logging.Formatter(format))
            fallback_logger.addHandler(console)


# Utility functions for common log patterns
def log_chain_start(video_id: str, skip_tiers: list = None):
    """Log start of fallback chain."""
    log = get_tier_logger("CHAIN")
    log.start_operation("fallback_chain", video_id)
    skip_str = f" (skipping: {skip_tiers})" if skip_tiers else ""
    log.info(f"Starting fallback chain for {video_id}{skip_str}")


def log_chain_result(video_id: str, success: bool, tier_used: str, stats: dict):
    """Log result of fallback chain."""
    log = get_tier_logger("CHAIN")

    if success:
        log.success(f"Completed via {tier_used}")
    else:
        log.failure(f"All tiers failed for {video_id}")

    # Log stats summary
    stats_str = ", ".join(f"{k}: {v['success']}/{v['success']+v['failure']}" for k, v in stats.items() if v['success'] + v['failure'] > 0)
    if stats_str:
        log.info(f"Tier stats: {stats_str}")


def log_http_error(log: FallbackLogger, url: str, status: int, body: str = ""):
    """Log HTTP error with details."""
    url_short = url[:60] + "..." if len(url) > 60 else url

    if status == 429:
        log.rate_limited()
        log.warning(f"HTTP 429 from {url_short}")
    elif status >= 500:
        log.warning(f"HTTP {status} (server error) from {url_short}")
    elif status >= 400:
        log.warning(f"HTTP {status} (client error) from {url_short}")

    # Log body preview for debugging
    if body and len(body) < 500:
        log.debug(f"Response body: {body[:200]}")
