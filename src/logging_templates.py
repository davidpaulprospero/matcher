"""
Standardized logging message templates for consistent logging across pipeline stages.

This module provides template functions that ensure consistent log message format
across all stages of the pipeline. Each template includes:
- Stage name and identifier
- Correlation ID (when available)
- Error codes (for error messages)
- Context-specific information

Usage:
    from src.logging_templates import (
        log_stage_start,
        log_stage_complete,
        log_stage_skip,
        log_error_with_context
    )

    # Stage start
    log_stage_start(logger, "VIDEO_SEARCH", total_items=50)

    # Stage complete
    log_stage_complete(logger, "VIDEO_SEARCH", items_processed=50, items_succeeded=48)

    # Stage skip
    log_stage_skip(logger, "VIDEO_SEARCH", reason="already completed")

    # Error with context
    log_error_with_context(logger, "DL-001", "Video download failed", video_id="abc123")
"""

import logging
from typing import Any, Optional, Dict, List


def get_correlation_id() -> Optional[str]:
    """
    Get the current correlation ID from the pipeline context.
    Returns None if no correlation ID is set.

    This function imports the correlation ID variable from pipeline.py
    to ensure proper context propagation across the pipeline.
    """
    try:
        # Import from pipeline to get the actual context variable
        from src.pipeline import get_correlation_id as pipeline_get_correlation_id
        return pipeline_get_correlation_id()
    except ImportError:
        # Fallback if pipeline import fails
        return None
    except Exception:
        return None


def _format_correlation(correlation_id: Optional[str]) -> str:
    """Format correlation ID into log prefix."""
    if correlation_id:
        return f"[corr:{correlation_id}]"
    return ""


def log_stage_start(
    logger: logging.Logger,
    stage_name: str,
    correlation_id: Optional[str] = None,
    **context: Any
) -> None:
    """
    Log the start of a stage execution.

    Args:
        logger: The logger instance
        stage_name: Name of the stage (e.g., "VIDEO_SEARCH", "CAPTION")
        correlation_id: Optional correlation ID for request tracing (auto-fetched if not provided)
        **context: Additional context (e.g., total_items=50, keywords=["term1"])
    """
    # Auto-fetch correlation ID if not explicitly provided
    if correlation_id is None:
        correlation_id = get_correlation_id()
    corr = _format_correlation(correlation_id)
    base_msg = f"[{stage_name}] Stage started{corr}"

    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        logger.info(f"{base_msg} - {context_str}")
    else:
        logger.info(base_msg)


def log_stage_complete(
    logger: logging.Logger,
    stage_name: str,
    elapsed_seconds: Optional[float] = None,
    correlation_id: Optional[str] = None,
    **context: Any
) -> None:
    """
    Log the completion of a stage execution.

    Args:
        logger: The logger instance
        stage_name: Name of the stage
        elapsed_seconds: Time taken for stage execution (seconds)
        correlation_id: Optional correlation ID for request tracing (auto-fetched if not provided)
        **context: Context metrics (e.g., items_processed=50, items_succeeded=48,
                   memory_percent=45.2, cpu_percent=78.5)
    """
    # Auto-fetch correlation ID if not provided
    if correlation_id is None:
        correlation_id = get_correlation_id()
    corr = _format_correlation(correlation_id)
    base_msg = f"[{stage_name}] Stage complete{corr}"

    # Build context string with timing prominently displayed
    ctx_parts = []
    if elapsed_seconds is not None:
        ctx_parts.append(f"elapsed={elapsed_seconds:.1f}s")

    # Add resource usage if present in context
    memory_percent = context.pop('memory_percent', None)
    cpu_percent = context.pop('cpu_percent', None)
    if memory_percent is not None:
        ctx_parts.append(f"memory={memory_percent:.1f}%")
    if cpu_percent is not None:
        ctx_parts.append(f"cpu={cpu_percent:.1f}%")

    # Add remaining context items
    for k, v in context.items():
        ctx_parts.append(f"{k}={v}")

    if ctx_parts:
        context_str = ", ".join(ctx_parts)
        logger.info(f"{base_msg} - {context_str}")
    else:
        logger.info(base_msg)


def log_stage_skip(
    logger: logging.Logger,
    stage_name: str,
    reason: str,
    correlation_id: Optional[str] = None,
    **context: Any
) -> None:
    """
    Log when a stage is skipped.

    Args:
        logger: The logger instance
        stage_name: Name of the stage
        reason: Reason for skipping (e.g., "already completed", "dry-run")
        correlation_id: Optional correlation ID for request tracing (auto-fetched if not provided)
        **context: Additional context
    """
    # Auto-fetch correlation ID if not explicitly provided
    if correlation_id is None:
        correlation_id = get_correlation_id()
    corr = _format_correlation(correlation_id)

    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        logger.info(f"[{stage_name}] Stage skipped{corr} - {reason} - {context_str}")
    else:
        logger.info(f"[{stage_name}] Stage skipped{corr} - {reason}")


def log_error_with_context(
    logger: logging.Logger,
    error_code: str,
    message: str,
    correlation_id: Optional[str] = None,
    **context: Any
) -> None:
    """
    Log an error with standardized format including error code and context.

    Args:
        logger: The logger instance
        error_code: Error code (e.g., "DL-001", "MATCH-002")
        message: Human-readable error message
        correlation_id: Optional correlation ID for request tracing (auto-fetched if not provided)
        **context: Error context (e.g., video_id="abc123", segment_id=5)
    """
    # Auto-fetch correlation ID if not explicitly provided
    if correlation_id is None:
        correlation_id = get_correlation_id()
    corr = _format_correlation(correlation_id)
    base_msg = f"[{error_code}]{corr} {message}"

    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        logger.error(f"{base_msg} - {context_str}")
    else:
        logger.error(base_msg)


def log_progress(
    logger: logging.Logger,
    stage_name: str,
    progress_pct: float,
    current: int,
    total: int,
    correlation_id: Optional[str] = None,
    **context: Any
) -> None:
    """
    Log progress percentage for long-running stages.

    Args:
        logger: The logger instance
        stage_name: Name of the stage
        progress_pct: Percentage complete (0-100)
        current: Current item number
        total: Total items
        correlation_id: Optional correlation ID for request tracing (auto-fetched if not provided)
        **context: Additional context (e.g., keyword="search term")
    """
    # Auto-fetch correlation ID if not explicitly provided
    if correlation_id is None:
        correlation_id = get_correlation_id()
    corr = _format_correlation(correlation_id)
    base_msg = f"[{stage_name}] Progress: {progress_pct:.0f}% ({current}/{total}){corr}"

    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        logger.info(f"{base_msg} - {context_str}")
    else:
        logger.info(base_msg)


def log_rate_limit(
    logger: logging.Logger,
    operation: str,
    resource: str,
    action: str,
    correlation_id: Optional[str] = None,
    **context: Any
) -> None:
    """
    Log rate limit related events.

    Args:
        logger: The logger instance
        operation: Operation type (e.g., "cookie_rotation", "vpn_rotation")
        resource: Resource affected (e.g., "youtube_api", "download")
        action: Action taken (e.g., "exhausted", "rotating", "recovered")
        correlation_id: Optional correlation ID for request tracing (auto-fetched if not provided)
        **context: Additional context (e.g., attempt=3, country="US")
    """
    # Auto-fetch correlation ID if not explicitly provided
    if correlation_id is None:
        correlation_id = get_correlation_id()
    corr = _format_correlation(correlation_id)
    base_msg = f"[RATE-LIMIT:{operation}] {resource}: {action}{corr}"

    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        logger.warning(f"{base_msg} - {context_str}")
    else:
        logger.warning(base_msg)


def log_match_context(
    logger: logging.Logger,
    level: int,
    message: str,
    segment_id: Optional[int] = None,
    video_id: Optional[str] = None,
    time_range: Optional[tuple] = None,
    correlation_id: Optional[str] = None,
    **context: Any
) -> None:
    """
    Log matching-related messages with full context (video ID, segment, time range).

    Args:
        logger: The logger instance
        level: Logging level (10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR)
        message: Log message
        segment_id: Voiceover segment ID
        video_id: YouTube video ID
        time_range: Tuple of (start_time, end_time)
        correlation_id: Optional correlation ID for request tracing (auto-fetched if not provided)
        **context: Additional context
    """
    # Auto-fetch correlation ID if not explicitly provided
    if correlation_id is None:
        correlation_id = get_correlation_id()
    corr = _format_correlation(correlation_id)

    # Build context string
    ctx_parts = []
    if segment_id is not None:
        ctx_parts.append(f"seg={segment_id}")
    if video_id:
        ctx_parts.append(f"vid={video_id}")
    if time_range:
        ctx_parts.append(f"time={time_range[0]:.1f}-{time_range[1]:.1f}")
    for k, v in context.items():
        ctx_parts.append(f"{k}={v}")

    context_str = f"[SEGMENT] {' '.join(ctx_parts)}" if ctx_parts else "[SEGMENT]"

    full_msg = f"{context_str}{corr} {message}"

    # Use the appropriate log level
    if level >= 40:
        logger.error(full_msg)
    elif level >= 30:
        logger.warning(full_msg)
    elif level >= 20:
        logger.info(full_msg)
    else:
        logger.debug(full_msg)


# HEAL- error codes for self-healing agent actions
# These are used with log_error_with_context for healing failures

# HEAL-xxx: Self-Healing Agent Errors
# | Prefix | Component | Description |
# |--------|-----------|-------------|
# | HEAL-001 | Runner | Healing attempt failed |
# | HEAL-002 | Runner | No healer available for error |
# | HEAL-003 | Runner | Maximum healing attempts exceeded |
# | HEAL-004 | Runner | Manual intervention required |
# | HEAL-005 | Orchestrator | Coordinate heal failed |
# | HEAL-006 | Healer | Healer raised exception |
# | HEAL-007 | Healer | Healer could not fix error |


def log_healing_attempt(
    logger: logging.Logger,
    healer_name: str,
    target: str,
    attempt: int,
    max_attempts: int,
    correlation_id: Optional[str] = None,
    **context: Any
) -> None:
    """
    Log a healing attempt with structured format.

    Args:
        logger: The logger instance
        healer_name: Name of the healer attempting the fix
        target: Target stage/component being healed
        attempt: Current attempt number
        max_attempts: Maximum attempts allowed
        correlation_id: Optional correlation ID for request tracing
        **context: Additional context (e.g., error_type, error_message)
    """
    if correlation_id is None:
        correlation_id = get_correlation_id()
    corr = _format_correlation(correlation_id)

    base_msg = f"[HEALER] Attempt {healer_name} on {target}{corr}"

    if context:
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        logger.info(f"{base_msg} (attempt {attempt}/{max_attempts}) - {context_str}")
    else:
        logger.info(f"{base_msg} (attempt {attempt}/{max_attempts})")


def log_healing_success(
    logger: logging.Logger,
    healer_name: str,
    target: str,
    action: str,
    affected_components: Optional[List[str]] = None,
    duration_ms: float = 0,
    correlation_id: Optional[str] = None,
    **context: Any
) -> None:
    """
    Log a successful healing action with details.

    Args:
        logger: The logger instance
        healer_name: Name of the healer
        target: Target stage/component that was healed
        action: Action taken (e.g., "RETRY", "MODIFY_CONFIG", "RESTORE")
        affected_components: List of components that were affected
        duration_ms: Time taken for healing
        correlation_id: Optional correlation ID for request tracing
        **context: Additional context (e.g., files_modified, config_changes)
    """
    if correlation_id is None:
        correlation_id = get_correlation_id()
    corr = _format_correlation(correlation_id)

    comp_str = ", ".join(affected_components) if affected_components else "none"

    base_msg = f"[HEALER] Success: {healer_name} healed {target}{corr}"

    ctx_parts = [f"action={action}", f"components={comp_str}"]
    if duration_ms > 0:
        ctx_parts.append(f"duration={duration_ms:.1f}ms")

    if context:
        for k, v in context.items():
            ctx_parts.append(f"{k}={v}")

    logger.info(f"{base_msg} - {', '.join(ctx_parts)}")


def log_healing_failure(
    logger: logging.Logger,
    healer_name: str,
    target: str,
    error_message: str,
    correlation_id: Optional[str] = None,
    **context: Any
) -> None:
    """
    Log a failed healing attempt using log_error_with_context.

    Args:
        logger: The logger instance
        healer_name: Name of the healer
        target: Target stage/component
        error_message: Error message from the healer
        correlation_id: Optional correlation ID for request tracing
        **context: Additional context
    """
    log_error_with_context(
        logger,
        "HEAL-001",
        f"Healing failed: {healer_name} could not fix {target}",
        correlation_id=correlation_id,
        healer=healer_name,
        target=target,
        error=error_message,
        **context
    )


def log_no_healer_available(
    logger: logging.Logger,
    target: str,
    error_type: str,
    error_message: str,
    correlation_id: Optional[str] = None
) -> None:
    """
    Log when no healer is available for an error.

    Args:
        logger: The logger instance
        target: Target stage/component
        error_type: Type of error that occurred
        error_message: Error message
        correlation_id: Optional correlation ID for request tracing
    """
    log_error_with_context(
        logger,
        "HEAL-002",
        f"No healer available for error in {target}",
        correlation_id=correlation_id,
        target=target,
        error_type=error_type,
        error=error_message
    )


def log_max_healing_exceeded(
    logger: logging.Logger,
    target: str,
    max_attempts: int,
    correlation_id: Optional[str] = None,
    **context: Any
) -> None:
    """
    Log when maximum healing attempts are exceeded.

    Args:
        logger: The logger instance
        target: Target stage/component
        max_attempts: Maximum attempts that were tried
        correlation_id: Optional correlation ID for request tracing
        **context: Additional context
    """
    log_error_with_context(
        logger,
        "HEAL-003",
        f"Maximum healing attempts exceeded for {target}",
        correlation_id=correlation_id,
        target=target,
        max_attempts=max_attempts,
        **context
    )


def log_manual_intervention(
    logger: logging.Logger,
    target: str,
    reason: str,
    correlation_id: Optional[str] = None,
    **context: Any
) -> None:
    """
    Log fallback to manual intervention.

    Args:
        logger: The logger instance
        target: Target stage/component
        reason: Reason why manual intervention is required
        correlation_id: Optional correlation ID for request tracing
        **context: Additional context
    """
    log_error_with_context(
        logger,
        "HEAL-004",
        f"Manual intervention required: {reason}",
        correlation_id=correlation_id,
        target=target,
        reason=reason,
        **context
    )


def log_healing_action_details(
    logger: logging.Logger,
    healer_name: str,
    target: str,
    action: str,
    files_modified: Optional[List[str]] = None,
    config_changes: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None
) -> None:
    """
    Log detailed healing action information.

    Args:
        logger: The logger instance
        healer_name: Name of the healer
        target: Target stage/component
        action: Action taken
        files_modified: List of files that were modified
        config_changes: Dictionary of config changes made
        correlation_id: Optional correlation ID for request tracing
    """
    if correlation_id is None:
        correlation_id = get_correlation_id()
    corr = _format_correlation(correlation_id)

    logger.info(f"[HEALER] Action details for {healer_name} on {target}{corr}")

    if files_modified:
        for f in files_modified:
            logger.info(f"[HEALER]   Modified: {f}")

    if config_changes:
        for k, v in config_changes.items():
            logger.info(f"[HEALER]   Config: {k} = {v}")
