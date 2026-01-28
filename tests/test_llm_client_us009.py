"""
US-009: Unit tests for LLM client _normalize_confidence and _is_permanent_error.

Tests cover:
- AC1: String label mapping (high, medium, low, very_high, very_low)
- AC2: Numeric string parsing, float/int passthrough, invalid input returns None
- AC3: Permanent error pattern detection (case-insensitive)
- AC4: Transient error patterns return False
- AC5: Edge cases for float range handling
"""

import pytest
from src.llm_client.cache import _normalize_confidence
from src.llm_client.retry import _is_permanent_error


# --- AC1: String label mapping ---

class TestNormalizeConfidenceStringLabelsUS009:
    """AC1: Test _normalize_confidence() converts string labels to correct float values."""

    def test_high_maps_to_0_9(self):
        assert _normalize_confidence('high') == 0.9

    def test_medium_maps_to_0_7(self):
        assert _normalize_confidence('medium') == 0.7

    def test_low_maps_to_0_5(self):
        assert _normalize_confidence('low') == 0.5

    def test_very_high_maps_to_0_95(self):
        assert _normalize_confidence('very_high') == 0.95

    def test_very_low_maps_to_0_3(self):
        assert _normalize_confidence('very_low') == 0.3

    def test_med_alias_maps_to_0_7(self):
        """The implementation also supports 'med' as an alias for medium."""
        assert _normalize_confidence('med') == 0.7

    def test_case_insensitive_high(self):
        assert _normalize_confidence('HIGH') == 0.9

    def test_case_insensitive_mixed(self):
        assert _normalize_confidence('Medium') == 0.7

    def test_whitespace_stripped(self):
        assert _normalize_confidence('  high  ') == 0.9


# --- AC2: Numeric strings, float/int passthrough, invalid input ---

class TestNormalizeConfidenceNumericUS009:
    """AC2: Test parsing numeric strings, float/int passthrough, and invalid input."""

    def test_numeric_string_0_85(self):
        assert _normalize_confidence('0.85') == 0.85

    def test_numeric_string_0(self):
        assert _normalize_confidence('0') == 0.0

    def test_numeric_string_1(self):
        assert _normalize_confidence('1') == 1.0

    def test_numeric_string_0_5(self):
        assert _normalize_confidence('0.5') == 0.5

    def test_float_passthrough(self):
        assert _normalize_confidence(0.75) == 0.75

    def test_int_passthrough(self):
        result = _normalize_confidence(1)
        assert result == 1.0
        assert isinstance(result, float)

    def test_int_zero_passthrough(self):
        result = _normalize_confidence(0)
        assert result == 0.0
        assert isinstance(result, float)

    def test_none_returns_none(self):
        assert _normalize_confidence(None) is None

    def test_empty_string_returns_none(self):
        assert _normalize_confidence('') is None

    def test_invalid_string_returns_none(self):
        assert _normalize_confidence('invalid') is None

    def test_non_numeric_string_returns_none(self):
        assert _normalize_confidence('not_a_number') is None

    def test_list_returns_none(self):
        assert _normalize_confidence([0.5]) is None

    def test_dict_returns_none(self):
        assert _normalize_confidence({'value': 0.5}) is None


# --- AC3: Permanent error detection ---

class TestIsPermanentErrorTrueUS009:
    """AC3: Test _is_permanent_error() returns True for permanent error patterns."""

    def test_model_not_found(self):
        assert _is_permanent_error('model not found') is True

    def test_invalid_api_key(self):
        assert _is_permanent_error('invalid api key') is True

    def test_unauthorized(self):
        assert _is_permanent_error('unauthorized') is True

    def test_permission_denied(self):
        assert _is_permanent_error('permission denied') is True

    def test_authentication_failed(self):
        assert _is_permanent_error('authentication failed') is True

    def test_case_insensitive_model_not_found(self):
        assert _is_permanent_error('Model Not Found') is True

    def test_case_insensitive_invalid_api_key(self):
        assert _is_permanent_error('INVALID API KEY') is True

    def test_case_insensitive_unauthorized(self):
        assert _is_permanent_error('Unauthorized') is True

    def test_api_key_with_underscore(self):
        assert _is_permanent_error('invalid api_key provided') is True

    def test_invalid_key(self):
        assert _is_permanent_error('invalid key for this service') is True

    def test_embedded_in_longer_message(self):
        assert _is_permanent_error('Error: authentication failed for provider gemini') is True


# --- AC4: Transient errors return False ---

class TestIsPermanentErrorFalseUS009:
    """AC4: Test _is_permanent_error() returns False for transient errors."""

    def test_timeout(self):
        assert _is_permanent_error('timeout') is False

    def test_connection_refused(self):
        assert _is_permanent_error('connection refused') is False

    def test_rate_limit_exceeded(self):
        assert _is_permanent_error('rate limit exceeded') is False

    def test_service_unavailable(self):
        assert _is_permanent_error('service unavailable') is False

    def test_internal_server_error(self):
        assert _is_permanent_error('internal server error') is False

    def test_http_429(self):
        assert _is_permanent_error('HTTP Error 429: Too Many Requests') is False

    def test_connection_reset(self):
        assert _is_permanent_error('connection reset by peer') is False

    def test_dns_resolution_failed(self):
        assert _is_permanent_error('DNS resolution failed') is False

    def test_empty_string(self):
        assert _is_permanent_error('') is False


# --- AC5: Edge cases for float range ---

class TestNormalizeConfidenceEdgeCasesUS009:
    """AC5: Test edge cases for float values in and outside 0-1 range."""

    def test_float_0_0_passthrough(self):
        assert _normalize_confidence(0.0) == 0.0

    def test_float_1_0_passthrough(self):
        assert _normalize_confidence(1.0) == 1.0

    def test_float_0_5_passthrough(self):
        assert _normalize_confidence(0.5) == 0.5

    def test_float_0_001_passthrough(self):
        assert _normalize_confidence(0.001) == 0.001

    def test_float_0_999_passthrough(self):
        assert _normalize_confidence(0.999) == 0.999

    def test_value_above_1_returned_as_float(self):
        """Values >1.0 are returned as-is (implementation does not clamp)."""
        result = _normalize_confidence(1.5)
        assert result == 1.5

    def test_negative_value_returned_as_float(self):
        """Negative values are returned as-is (implementation does not clamp)."""
        result = _normalize_confidence(-0.5)
        assert result == -0.5

    def test_large_int_returned_as_float(self):
        """Large int values are converted to float as-is."""
        result = _normalize_confidence(100)
        assert result == 100.0
        assert isinstance(result, float)

    def test_bool_true_treated_as_int(self):
        """Python bool is subclass of int, so True -> 1.0."""
        result = _normalize_confidence(True)
        assert result == 1.0

    def test_bool_false_treated_as_int(self):
        """Python bool is subclass of int, so False -> 0.0."""
        result = _normalize_confidence(False)
        assert result == 0.0
