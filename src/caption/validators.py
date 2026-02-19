"""
Caption configuration validators.

Provides functions for validating language codes, caption configuration,
timestamps, and duration bounds.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

from .constants import ISO_639_1_CODES
from .exceptions import ConfigValidationError

logger = logging.getLogger(__name__)

# Supported caption formats
CAPTION_FORMATS = {'srt', 'vtt', 'json3', 'srv1', 'srv2', 'srv3', 'ttml'}

# Valid timestamp range (0 to 24 hours in seconds)
MAX_TIMESTAMP_SECONDS = 24 * 60 * 60  # 24 hours

# Reasonable duration bounds (0.1 seconds to 4 hours)
MIN_DURATION_SECONDS = 0.1
MAX_DURATION_SECONDS = 4 * 60 * 60  # 4 hours


def is_valid_language_code(code: str) -> bool:
    """Check if a language code is a valid ISO 639-1 code (US-005).

    Args:
        code: Language code to validate.

    Returns:
        True if valid, False otherwise.

    Examples:
        >>> is_valid_language_code('en')
        True
        >>> is_valid_language_code('xyz')
        False
        >>> is_valid_language_code('ENG')
        False
    """
    if not isinstance(code, str):
        return False
    # ISO 639-1 codes are exactly 2 lowercase letters
    return code.lower() in ISO_639_1_CODES


def validate_language_config(
    preferred_language: str,
    fallback_languages: List[str],
    raise_on_error: bool = True
) -> List[str]:
    """Validate language configuration for caption fetching (US-005).

    Checks:
    1. All language codes are valid ISO 639-1 (2-letter codes)
    2. No duplicates in fallback_languages
    3. preferred_language is not repeated in fallback_languages (redundant)

    Args:
        preferred_language: Primary language code (e.g., 'en').
        fallback_languages: List of fallback language codes.
        raise_on_error: If True, raise ConfigValidationError on first error.
                       If False, return list of warning messages.

    Returns:
        Empty list if valid, or list of warning/error messages.

    Raises:
        ConfigValidationError: If raise_on_error=True and validation fails.

    Examples:
        >>> validate_language_config('en', ['es', 'pt', 'fr'])
        []
        >>> validate_language_config('xyz', [])  # Invalid code
        ConfigValidationError: Invalid language configuration for 'preferred_language': xyz
        >>> validate_language_config('en', ['en', 'es'])  # Redundant
        ['Warning: preferred_language "en" also in fallback_languages (redundant)']
    """
    issues = []

    # Check preferred_language
    if not is_valid_language_code(preferred_language):
        msg = f'"{preferred_language}" is not a valid ISO 639-1 language code'
        suggestion = 'Use a 2-letter code like "en", "es", "fr", "de", "pt", "zh", "ja"'
        if raise_on_error:
            raise ConfigValidationError(
                field='preferred_language',
                value=preferred_language,
                reason=msg,
                suggestion=suggestion
            )
        issues.append(f"Error: {msg}. {suggestion}")

    # Check fallback_languages
    seen = set()
    for i, lang in enumerate(fallback_languages):
        # Check validity
        if not is_valid_language_code(lang):
            msg = f'"{lang}" at position {i} is not a valid ISO 639-1 language code'
            suggestion = 'Use 2-letter codes like "en", "es", "fr", "de", "pt", "zh", "ja"'
            if raise_on_error:
                raise ConfigValidationError(
                    field='fallback_languages',
                    value=lang,
                    reason=msg,
                    suggestion=suggestion
                )
            issues.append(f"Error: {msg}. {suggestion}")

        # Check for duplicates within fallback_languages
        lang_lower = lang.lower()
        if lang_lower in seen:
            msg = f'"{lang}" appears multiple times in fallback_languages'
            suggestion = 'Remove duplicate entries'
            if raise_on_error:
                raise ConfigValidationError(
                    field='fallback_languages',
                    value=lang,
                    reason=msg,
                    suggestion=suggestion
                )
            issues.append(f"Error: {msg}. {suggestion}")
        seen.add(lang_lower)

    # Check if preferred_language is in fallback_languages (warn, not error)
    if preferred_language.lower() in {lang.lower() for lang in fallback_languages}:
        msg = (f'preferred_language "{preferred_language}" also in '
               f'fallback_languages (redundant, will be tried twice)')
        logger.warning(msg)
        issues.append(f"Warning: {msg}")

    return issues


def validate_caption_format(fmt: str, raise_on_error: bool = False) -> bool:
    """Validate a caption format string (US-005).

    Args:
        fmt: Caption format to validate (e.g., 'srt', 'vtt', 'json3').
        raise_on_error: If True, raise ConfigValidationError on invalid format.

    Returns:
        True if valid format, False otherwise.

    Raises:
        ConfigValidationError: If raise_on_error=True and format is invalid.

    Examples:
        >>> validate_caption_format('srt')
        True
        >>> validate_caption_format('json3')
        True
        >>> validate_caption_format('invalid')
        False
    """
    if not isinstance(fmt, str):
        msg = f"Caption format must be a string, got {type(fmt).__name__}"
        if raise_on_error:
            raise ConfigValidationError(
                field='caption_format',
                value=fmt,
                reason=msg,
                suggestion=f"Use one of: {', '.join(sorted(CAPTION_FORMATS))}"
            )
        return False

    fmt_lower = fmt.lower()
    if fmt_lower in CAPTION_FORMATS:
        return True

    msg = f"'{fmt}' is not a supported caption format"
    if raise_on_error:
        raise ConfigValidationError(
            field='caption_format',
            value=fmt,
            reason=msg,
            suggestion=f"Supported formats: {', '.join(sorted(CAPTION_FORMATS))}"
        )
    return False


def validate_timestamp(
    timestamp: float,
    raise_on_error: bool = False
) -> Tuple[bool, str]:
    """Validate a timestamp value (US-005).

    Args:
        timestamp: Timestamp in seconds to validate.
        raise_on_error: If True, raise ConfigValidationError on invalid timestamp.

    Returns:
        Tuple of (is_valid, error_message).

    Raises:
        ConfigValidationError: If raise_on_error=True and timestamp is invalid.

    Examples:
        >>> validate_timestamp(0.0)
        (True, '')
        >>> validate_timestamp(3600.5)
        (True, '')
        >>> validate_timestamp(-1.0)
        (False, 'Timestamp must be >= 0')
        >>> validate_timestamp(90000.0)
        (False, 'Timestamp exceeds maximum (24 hours)')
    """
    # Check type
    if not isinstance(timestamp, (int, float)):
        msg = f"Timestamp must be a number, got {type(timestamp).__name__}"
        if raise_on_error:
            raise ConfigValidationError(
                field='timestamp',
                value=timestamp,
                reason=msg,
                suggestion="Provide timestamp as seconds (float or int)"
            )
        return False, msg

    # Check for NaN and infinity
    import math
    if math.isnan(timestamp) or math.isinf(timestamp):
        msg = "Timestamp cannot be NaN or infinity"
        if raise_on_error:
            raise ConfigValidationError(
                field='timestamp',
                value=timestamp,
                reason=msg,
                suggestion="Provide a finite timestamp value"
            )
        return False, msg

    # Check lower bound
    if timestamp < 0:
        msg = "Timestamp must be >= 0"
        if raise_on_error:
            raise ConfigValidationError(
                field='timestamp',
                value=timestamp,
                reason=msg,
                suggestion="Timestamps cannot be negative"
            )
        return False, msg

    # Check upper bound (24 hours)
    if timestamp > MAX_TIMESTAMP_SECONDS:
        msg = f"Timestamp exceeds maximum ({MAX_TIMESTAMP_SECONDS} seconds = 24 hours)"
        if raise_on_error:
            raise ConfigValidationError(
                field='timestamp',
                value=timestamp,
                reason=msg,
                suggestion=f"Timestamp must be <= {MAX_TIMESTAMP_SECONDS}"
            )
        return False, msg

    return True, ""


def validate_duration_bounds(
    duration: float,
    min_duration: float = MIN_DURATION_SECONDS,
    max_duration: float = MAX_DURATION_SECONDS,
    raise_on_error: bool = False
) -> Tuple[bool, str]:
    """Validate a duration value is within reasonable bounds (US-005).

    Args:
        duration: Duration in seconds to validate.
        min_duration: Minimum allowed duration (default 0.1 seconds).
        max_duration: Maximum allowed duration (default 4 hours = 14400 seconds).
        raise_on_error: If True, raise ConfigValidationError on invalid duration.

    Returns:
        Tuple of (is_valid, error_message).

    Raises:
        ConfigValidationError: If raise_on_error=True and duration is invalid.

    Examples:
        >>> validate_duration_bounds(30.0)
        (True, '')
        >>> validate_duration_bounds(0.05)
        (False, 'Duration must be >= 0.1 seconds')
        >>> validate_duration_bounds(20000.0)
        (False, 'Duration exceeds maximum (14400 seconds)')
    """
    # Check type
    if not isinstance(duration, (int, float)):
        msg = f"Duration must be a number, got {type(duration).__name__}"
        if raise_on_error:
            raise ConfigValidationError(
                field='duration',
                value=duration,
                reason=msg,
                suggestion="Provide duration as seconds (float or int)"
            )
        return False, msg

    # Check for NaN and infinity
    import math
    if math.isnan(duration) or math.isinf(duration):
        msg = "Duration cannot be NaN or infinity"
        if raise_on_error:
            raise ConfigValidationError(
                field='duration',
                value=duration,
                reason=msg,
                suggestion="Provide a finite duration value"
            )
        return False, msg

    # Check lower bound
    if duration < min_duration:
        msg = f"Duration must be >= {min_duration} seconds"
        if raise_on_error:
            raise ConfigValidationError(
                field='duration',
                value=duration,
                reason=msg,
                suggestion=f"Duration must be at least {min_duration} seconds"
            )
        return False, msg

    # Check upper bound
    if duration > max_duration:
        msg = f"Duration exceeds maximum ({max_duration} seconds)"
        if raise_on_error:
            raise ConfigValidationError(
                field='duration',
                value=duration,
                reason=msg,
                suggestion=f"Duration must be <= {max_duration} seconds"
            )
        return False, msg

    return True, ""
