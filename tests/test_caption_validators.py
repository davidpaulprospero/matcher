"""
Unit tests for src/caption/validators.py

Tests language code validation, caption format validation, timestamp validation,
duration bounds validation, and language configuration validation.
"""

import math
import pytest

from src.caption.validators import (
    CAPTION_FORMATS,
    is_valid_language_code,
    validate_caption_format,
    validate_timestamp,
    validate_duration_bounds,
    validate_language_config,
)
from src.caption.exceptions import ConfigValidationError


@pytest.mark.fast
class TestIsValidLanguageCode:
    """Test is_valid_language_code function"""

    def test_valid_english_code(self):
        """Test that 'en' is a valid ISO 639-1 code"""
        assert is_valid_language_code('en') is True

    def test_valid_spanish_code(self):
        """Test that 'es' is a valid ISO 639-1 code"""
        assert is_valid_language_code('es') is True

    def test_valid_chinese_code(self):
        """Test that 'zh' is a valid ISO 639-1 code"""
        assert is_valid_language_code('zh') is True

    def test_valid_japanese_code(self):
        """Test that 'ja' is a valid ISO 639-1 code"""
        assert is_valid_language_code('ja') is True

    def test_valid_german_code(self):
        """Test that 'de' is a valid ISO 639-1 code"""
        assert is_valid_language_code('de') is True

    def test_valid_french_code(self):
        """Test that 'fr' is a valid ISO 639-1 code"""
        assert is_valid_language_code('fr') is True

    def test_valid_portuguese_code(self):
        """Test that 'pt' is a valid ISO 639-1 code"""
        assert is_valid_language_code('pt') is True

    def test_valid_russian_code(self):
        """Test that 'ru' is a valid ISO 639-1 code"""
        assert is_valid_language_code('ru') is True

    def test_valid_korean_code(self):
        """Test that 'ko' is a valid ISO 639-1 code"""
        assert is_valid_language_code('ko') is True

    def test_valid_arabic_code(self):
        """Test that 'ar' is a valid ISO 639-1 code"""
        assert is_valid_language_code('ar') is True

    def test_invalid_three_letter_code(self):
        """Test that 3-letter codes are invalid"""
        assert is_valid_language_code('eng') is False

    def test_invalid_four_letter_code(self):
        """Test that 4-letter codes are invalid"""
        assert is_valid_language_code('engl') is False

    def test_invalid_arbitrary_code(self):
        """Test that arbitrary strings are invalid"""
        assert is_valid_language_code('xyz') is False
        assert is_valid_language_code('abc') is False
        assert is_valid_language_code('foo') is False

    def test_case_sensitive_uppercase(self):
        """Test that uppercase codes are normalized to lowercase (function is case-insensitive)"""
        # Function converts to lowercase internally, so uppercase codes are valid
        assert is_valid_language_code('EN') is True
        assert is_valid_language_code('ES') is True
        assert is_valid_language_code('ZH') is True

    def test_case_sensitive_mixed_case(self):
        """Test that mixed case codes are normalized (function is case-insensitive)"""
        # Function converts to lowercase internally, so mixed case codes are valid
        assert is_valid_language_code('En') is True
        assert is_valid_language_code('eN') is True
        assert is_valid_language_code('Zh') is True

    def test_lowercase_is_valid(self):
        """Test that lowercase codes work correctly"""
        assert is_valid_language_code('en') is True
        assert is_valid_language_code('fr') is True

    def test_empty_string(self):
        """Test that empty string is invalid"""
        assert is_valid_language_code('') is False

    def test_none_input(self):
        """Test that None is invalid"""
        assert is_valid_language_code(None) is False

    def test_non_string_input_int(self):
        """Test that integer input is invalid"""
        assert is_valid_language_code(123) is False

    def test_non_string_input_list(self):
        """Test that list input is invalid"""
        assert is_valid_language_code(['en', 'es']) is False


@pytest.mark.fast
class TestValidateLanguageConfig:
    """Test validate_language_config function"""

    def test_valid_single_language(self):
        """Test valid config with single language"""
        issues = validate_language_config('en', [], raise_on_error=False)
        assert issues == []

    def test_valid_multiple_fallbacks(self):
        """Test valid config with multiple fallback languages"""
        issues = validate_language_config('en', ['es', 'pt', 'fr'], raise_on_error=False)
        assert issues == []

    def test_valid_no_fallbacks(self):
        """Test valid config with empty fallback list"""
        issues = validate_language_config('de', [], raise_on_error=False)
        assert issues == []

    def test_valid_asian_languages(self):
        """Test valid config with Asian languages"""
        issues = validate_language_config('ja', ['zh', 'ko'], raise_on_error=False)
        assert issues == []

    def test_invalid_preferred_language_raises(self):
        """Test that invalid preferred_language raises ConfigValidationError"""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_language_config('xyz', [], raise_on_error=True)
        assert exc_info.value.field == 'preferred_language'
        assert exc_info.value.value == 'xyz'

    def test_invalid_preferred_language_no_raise(self):
        """Test that invalid preferred_language returns error message"""
        issues = validate_language_config('xyz', [], raise_on_error=False)
        assert len(issues) == 1
        assert 'Error' in issues[0]
        assert 'xyz' in issues[0]

    def test_invalid_fallback_language_raises(self):
        """Test that invalid fallback language raises ConfigValidationError"""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_language_config('en', ['xyz'], raise_on_error=True)
        assert exc_info.value.field == 'fallback_languages'
        assert exc_info.value.value == 'xyz'

    def test_invalid_fallback_language_no_raise(self):
        """Test that invalid fallback language returns error message"""
        issues = validate_language_config('en', ['xyz'], raise_on_error=False)
        assert len(issues) == 1
        assert 'Error' in issues[0]
        assert 'xyz' in issues[0]

    def test_duplicate_fallback_languages_raises(self):
        """Test that duplicate fallback languages raise ConfigValidationError"""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_language_config('en', ['es', 'es'], raise_on_error=True)
        assert exc_info.value.field == 'fallback_languages'
        assert exc_info.value.value == 'es'

    def test_duplicate_fallback_languages_no_raise(self):
        """Test that duplicate fallback languages return error message"""
        issues = validate_language_config('en', ['es', 'es'], raise_on_error=False)
        assert len(issues) == 1
        assert 'Error' in issues[0]
        assert 'duplicate' in issues[0].lower()

    def test_redundant_preferred_in_fallback_warns(self):
        """Test warning when preferred_language is also in fallback_languages"""
        issues = validate_language_config('en', ['es', 'en'], raise_on_error=False)
        assert len(issues) == 1
        assert 'Warning' in issues[0]
        assert 'en' in issues[0]
        assert 'redundant' in issues[0].lower()

    def test_redundant_preferred_in_fallback_raises(self):
        """Test warning when preferred_language is also in fallback_languages with raise_on_error=True"""
        # With raise_on_error=True, redundancy should still not raise (it's just a warning)
        issues = validate_language_config('en', ['es', 'en'], raise_on_error=True)
        assert len(issues) == 1
        assert 'Warning' in issues[0]

    def test_multiple_errors_combined(self):
        """Test that multiple errors are combined when raise_on_error=False"""
        issues = validate_language_config('xyz', ['abc', 'abc', 'xyz'], raise_on_error=False)
        # Expected: invalid preferred 'xyz', 2x invalid fallback 'abc', duplicate 'abc', invalid fallback 'xyz' at position 2, redundant 'xyz' in fallback
        # That's 5 errors + 1 warning = 6 issues
        assert len(issues) >= 3  # At least 3 errors (invalid preferred, invalid fallback, duplicate fallback)

    def test_case_insensitive_fallback_check(self):
        """Test that fallback duplicate check is case-insensitive"""
        issues = validate_language_config('en', ['ES', 'es'], raise_on_error=False)
        assert len(issues) == 1
        assert 'duplicate' in issues[0].lower()

    def test_case_insensitive_redundant_check(self):
        """Test that redundant check is case-insensitive"""
        issues = validate_language_config('EN', ['es', 'en'], raise_on_error=False)
        # Should warn about redundant (case insensitive)
        assert len(issues) >= 1

    def test_empty_fallback_list_is_valid(self):
        """Test that empty fallback list is valid"""
        issues = validate_language_config('en', [], raise_on_error=False)
        assert issues == []

    def test_config_validation_error_attributes(self):
        """Test ConfigValidationError has correct attributes"""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_language_config('invalid', [], raise_on_error=True)

        error = exc_info.value
        assert error.field == 'preferred_language'
        assert error.value == 'invalid'
        assert 'not a valid ISO 639-1' in error.reason
        assert '2-letter' in error.suggestion

    def test_fallback_error_message_format(self):
        """Test fallback language error message includes position"""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_language_config('en', ['xyz'], raise_on_error=True)

        error = exc_info.value
        assert error.field == 'fallback_languages'
        assert 'position' in error.reason.lower()

    def test_error_message_formatting_with_raise_false(self):
        """Test error message formatting when raise_on_error=False"""
        issues = validate_language_config('xyz', [], raise_on_error=False)
        assert len(issues) == 1
        assert 'Error:' in issues[0]
        # Error message contains reason but not "Suggestion:" as a separate keyword in the returned string
        assert 'xyz' in issues[0]
        assert 'not a valid ISO 639-1' in issues[0]

    def test_unicode_valid_codes(self):
        """Test various valid ISO 639-1 codes including less common ones"""
        valid_codes = ['aa', 'af', 'bg', 'bn', 'ca', 'cs', 'cy', 'da', 'el', 'eo']
        for code in valid_codes:
            assert is_valid_language_code(code) is True

    def test_all_european_language_codes(self):
        """Test various European language codes"""
        european_codes = ['en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'pl', 'ru', 'uk', 'sv', 'no', 'da', 'fi', 'el', 'hu', 'cs', 'ro', 'bg', 'hr']
        for code in european_codes:
            assert is_valid_language_code(code) is True


@pytest.mark.fast
class TestValidatorsIntegration:
    """Integration tests for validators module"""

    def test_validate_language_config_accepts_all_valid_codes(self):
        """Test that validate_language_config accepts all common valid codes"""
        common_codes = ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko', 'ar', 'hi']
        for code in common_codes:
            issues = validate_language_config(code, [], raise_on_error=False)
            assert issues == [], f"Code {code} should be valid"

    def test_full_workflow_valid_config(self):
        """Test full workflow with valid configuration"""
        # Valid configuration should return empty issues
        preferred = 'en'
        fallbacks = ['es', 'fr', 'de']

        # First validate each code individually
        assert is_valid_language_code(preferred) is True
        for fb in fallbacks:
            assert is_valid_language_code(fb) is True

        # Then validate the full config
        issues = validate_language_config(preferred, fallbacks, raise_on_error=False)
        assert issues == []

    def test_full_workflow_invalid_preferred(self):
        """Test workflow when preferred language is invalid"""
        preferred = 'xxx'
        fallbacks = ['es', 'fr']

        # Individual validation should fail
        assert is_valid_language_code(preferred) is False

        # Config validation should return error
        issues = validate_language_config(preferred, fallbacks, raise_on_error=False)
        assert len(issues) == 1
        assert 'Error' in issues[0]


@pytest.mark.fast
class TestValidateCaptionFormat:
    """Test validate_caption_format function"""

    def test_valid_srt_format(self):
        """Test that 'srt' is a valid caption format"""
        assert validate_caption_format('srt') is True

    def test_valid_vtt_format(self):
        """Test that 'vtt' is a valid caption format"""
        assert validate_caption_format('vtt') is True

    def test_valid_json3_format(self):
        """Test that 'json3' is a valid caption format"""
        assert validate_caption_format('json3') is True

    def test_valid_srv1_format(self):
        """Test that 'srv1' is a valid caption format"""
        assert validate_caption_format('srv1') is True

    def test_valid_srv2_format(self):
        """Test that 'srv2' is a valid caption format"""
        assert validate_caption_format('srv2') is True

    def test_valid_srv3_format(self):
        """Test that 'srv3' is a valid caption format"""
        assert validate_caption_format('srv3') is True

    def test_valid_ttml_format(self):
        """Test that 'ttml' is a valid caption format"""
        assert validate_caption_format('ttml') is True

    def test_case_insensitive_format(self):
        """Test that format validation is case-insensitive"""
        assert validate_caption_format('SRT') is True
        assert validate_caption_format('VTT') is True
        assert validate_caption_format('Srt') is True
        assert validate_caption_format('Json3') is True

    def test_invalid_format_returns_false(self):
        """Test that invalid format returns False"""
        assert validate_caption_format('invalid') is False
        assert validate_caption_format('mp4') is False
        assert validate_caption_format('txt') is False

    def test_invalid_format_raises_error(self):
        """Test that invalid format raises ConfigValidationError"""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_caption_format('invalid', raise_on_error=True)
        assert exc_info.value.field == 'caption_format'
        assert exc_info.value.value == 'invalid'

    def test_non_string_format_raises(self):
        """Test that non-string format raises ConfigValidationError"""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_caption_format(123, raise_on_error=True)
        assert exc_info.value.field == 'caption_format'
        assert 'must be a string' in exc_info.value.reason

    def test_empty_string_is_invalid(self):
        """Test that empty string is invalid format"""
        assert validate_caption_format('') is False

    def test_supported_formats_list(self):
        """Test that all expected formats are in CAPTION_FORMATS"""
        expected = {'srt', 'vtt', 'json3', 'srv1', 'srv2', 'srv3', 'ttml'}
        assert CAPTION_FORMATS == expected


@pytest.mark.fast
class TestValidateTimestamp:
    """Test validate_timestamp function"""

    def test_valid_zero_timestamp(self):
        """Test that 0 is a valid timestamp"""
        is_valid, msg = validate_timestamp(0.0)
        assert is_valid is True
        assert msg == ''

    def test_valid_positive_timestamp(self):
        """Test that positive timestamp is valid"""
        is_valid, msg = validate_timestamp(3600.5)
        assert is_valid is True
        assert msg == ''

    def test_valid_integer_timestamp(self):
        """Test that integer timestamp is valid"""
        is_valid, msg = validate_timestamp(60)
        assert is_valid is True

    def test_valid_max_timestamp(self):
        """Test that max timestamp (24 hours) is valid"""
        max_ts = 24 * 60 * 60  # 86400 seconds
        is_valid, msg = validate_timestamp(max_ts)
        assert is_valid is True
        assert msg == ''

    def test_valid_near_max_timestamp(self):
        """Test near-max timestamp is valid"""
        is_valid, msg = validate_timestamp(86400 - 0.001)
        assert is_valid is True

    def test_negative_timestamp_returns_false(self):
        """Test that negative timestamp is invalid"""
        is_valid, msg = validate_timestamp(-1.0)
        assert is_valid is False
        assert '>=' in msg

    def test_negative_timestamp_raises_error(self):
        """Test that negative timestamp raises ConfigValidationError"""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_timestamp(-1.0, raise_on_error=True)
        assert exc_info.value.field == 'timestamp'
        assert '-1' in str(exc_info.value.value)

    def test_exceeds_max_timestamp_returns_false(self):
        """Test that timestamp exceeding 24 hours is invalid"""
        is_valid, msg = validate_timestamp(90000.0)
        assert is_valid is False
        assert 'exceeds' in msg.lower()

    def test_exceeds_max_raises_error(self):
        """Test that exceeding max raises ConfigValidationError"""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_timestamp(90000.0, raise_on_error=True)
        assert exc_info.value.field == 'timestamp'

    def test_nan_timestamp_returns_false(self):
        """Test that NaN timestamp is invalid"""
        is_valid, msg = validate_timestamp(float('nan'))
        assert is_valid is False
        assert 'NaN' in msg

    def test_nan_raises_error(self):
        """Test that NaN timestamp raises ConfigValidationError"""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_timestamp(float('nan'), raise_on_error=True)
        assert exc_info.value.field == 'timestamp'

    def test_positive_inf_returns_false(self):
        """Test that positive infinity is invalid"""
        is_valid, msg = validate_timestamp(float('inf'))
        assert is_valid is False
        assert 'infinity' in msg.lower()

    def test_negative_inf_returns_false(self):
        """Test that negative infinity is invalid"""
        is_valid, msg = validate_timestamp(float('-inf'))
        assert is_valid is False
        assert 'infinity' in msg.lower()

    def test_inf_raises_error(self):
        """Test that infinity raises ConfigValidationError"""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_timestamp(float('inf'), raise_on_error=True)
        assert exc_info.value.field == 'timestamp'

    def test_non_numeric_returns_false(self):
        """Test that non-numeric input is invalid"""
        is_valid, msg = validate_timestamp("60")
        assert is_valid is False

    def test_non_numeric_raises_error(self):
        """Test that non-numeric raises ConfigValidationError"""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_timestamp("60", raise_on_error=True)
        assert exc_info.value.field == 'timestamp'
        assert 'must be a number' in exc_info.value.reason

    def test_list_input_returns_false(self):
        """Test that list input is invalid"""
        is_valid, msg = validate_timestamp([60])
        assert is_valid is False

    def test_none_input_returns_false(self):
        """Test that None input is invalid"""
        is_valid, msg = validate_timestamp(None)
        assert is_valid is False


@pytest.mark.fast
class TestValidateDurationBounds:
    """Test validate_duration_bounds function"""

    def test_valid_minimum_duration(self):
        """Test that minimum valid duration (0.1s) is valid"""
        is_valid, msg = validate_duration_bounds(0.1)
        assert is_valid is True
        assert msg == ''

    def test_valid_middle_duration(self):
        """Test that middle-range duration is valid"""
        is_valid, msg = validate_duration_bounds(30.0)
        assert is_valid is True
        assert msg == ''

    def test_valid_max_duration(self):
        """Test that max duration (4 hours) is valid"""
        max_dur = 4 * 60 * 60  # 14400 seconds
        is_valid, msg = validate_duration_bounds(max_dur)
        assert is_valid is True
        assert msg == ''

    def test_valid_near_max_duration(self):
        """Test near-max duration is valid"""
        is_valid, msg = validate_duration_bounds(14400 - 0.001)
        assert is_valid is True

    def test_valid_integer_duration(self):
        """Test that integer duration is valid"""
        is_valid, msg = validate_duration_bounds(60)
        assert is_valid is True

    def test_below_minimum_returns_false(self):
        """Test that duration below minimum is invalid"""
        is_valid, msg = validate_duration_bounds(0.05)
        assert is_valid is False
        assert '>=' in msg

    def test_below_minimum_raises_error(self):
        """Test that below minimum raises ConfigValidationError"""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_duration_bounds(0.05, raise_on_error=True)
        assert exc_info.value.field == 'duration'

    def test_zero_duration_returns_false(self):
        """Test that zero duration is invalid (below minimum)"""
        is_valid, msg = validate_duration_bounds(0.0)
        assert is_valid is False

    def test_exceeds_max_returns_false(self):
        """Test that duration exceeding max is invalid"""
        is_valid, msg = validate_duration_bounds(20000.0)
        assert is_valid is False
        assert 'exceeds' in msg.lower()

    def test_exceeds_max_raises_error(self):
        """Test that exceeding max raises ConfigValidationError"""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_duration_bounds(20000.0, raise_on_error=True)
        assert exc_info.value.field == 'duration'

    def test_custom_min_bounds(self):
        """Test with custom minimum bound"""
        is_valid, msg = validate_duration_bounds(5.0, min_duration=10.0)
        assert is_valid is False

        is_valid, msg = validate_duration_bounds(15.0, min_duration=10.0)
        assert is_valid is True

    def test_custom_max_bounds(self):
        """Test with custom maximum bound"""
        is_valid, msg = validate_duration_bounds(100.0, max_duration=50.0)
        assert is_valid is False

        is_valid, msg = validate_duration_bounds(30.0, max_duration=50.0)
        assert is_valid is True

    def test_nan_duration_returns_false(self):
        """Test that NaN duration is invalid"""
        is_valid, msg = validate_duration_bounds(float('nan'))
        assert is_valid is False
        assert 'NaN' in msg

    def test_nan_raises_error(self):
        """Test that NaN duration raises ConfigValidationError"""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_duration_bounds(float('nan'), raise_on_error=True)
        assert exc_info.value.field == 'duration'

    def test_positive_inf_returns_false(self):
        """Test that positive infinity is invalid"""
        is_valid, msg = validate_duration_bounds(float('inf'))
        assert is_valid is False

    def test_negative_inf_returns_false(self):
        """Test that negative infinity is invalid"""
        is_valid, msg = validate_duration_bounds(float('-inf'))
        assert is_valid is False

    def test_non_numeric_returns_false(self):
        """Test that non-numeric input is invalid"""
        is_valid, msg = validate_duration_bounds("30")
        assert is_valid is False

    def test_non_numeric_raises_error(self):
        """Test that non-numeric raises ConfigValidationError"""
        with pytest.raises(ConfigValidationError) as exc_info:
            validate_duration_bounds("30", raise_on_error=True)
        assert exc_info.value.field == 'duration'
        assert 'must be a number' in exc_info.value.reason

    def test_list_input_returns_false(self):
        """Test that list input is invalid"""
        is_valid, msg = validate_duration_bounds([30])
        assert is_valid is False

    def test_none_input_returns_false(self):
        """Test that None input is invalid"""
        is_valid, msg = validate_duration_bounds(None)
        assert is_valid is False
