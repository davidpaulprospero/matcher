"""
Caption configuration validators.

Provides functions for validating language codes and caption configuration.
"""

from __future__ import annotations

import logging
from typing import List

from .constants import ISO_639_1_CODES
from .exceptions import ConfigValidationError

logger = logging.getLogger(__name__)


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
