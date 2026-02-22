"""Unit tests for YouTube API error code mapping (US-156-010).

Tests verify:
- ERROR_CODE_MAPPING dictionary contains YouTube-specific error codes
- get_error_category() returns correct error type, description, severity, and retryable
- get_error_description() returns human-readable descriptions
- get_error_type_for_code() returns correct error type strings
- Error codes 100-110 are properly mapped to YouTubeAPIError subclasses

US-157-005: Added tests for error_code_to_action mapping and recovery path selection.
"""

import pytest

from src.downloader.errors import (
    ERROR_CODE_MAPPING,
    ERROR_CODE_TO_ACTION,
    error_code_to_action,
    get_error_category,
    get_error_description,
    get_error_type_for_code,
    get_action_for_error_type,
    YouTubeAPIError,
    YouTubeAPIErrorCode,
    ErrorAction,
)


class TestErrorCodeMapping:
    """Tests for ERROR_CODE_MAPPING dictionary."""

    def test_error_code_mapping_exists(self):
        """Verify ERROR_CODE_MAPPING dictionary exists and contains entries."""
        assert ERROR_CODE_MAPPING is not None
        assert isinstance(ERROR_CODE_MAPPING, dict)
        assert len(ERROR_CODE_MAPPING) > 0

    def test_error_code_100_maps_to_permission_denied(self):
        """Error code 100 should map to permission_denied error type."""
        result = get_error_category(100)
        assert result["error_type"] == "permission_denied"
        assert "API is not enabled" in result["description"]
        assert result["severity"] == "critical"
        assert result["retryable"] is False

    def test_error_code_101_maps_to_not_found(self):
        """Error code 101 should map to not_found error type."""
        result = get_error_category(101)
        assert result["error_type"] == "not_found"
        assert "not found" in result["description"].lower()
        assert result["severity"] == "low"
        assert result["retryable"] is False

    def test_error_code_102_maps_to_invalid_parameter(self):
        """Error code 102 should map to invalid_parameter error type."""
        result = get_error_category(102)
        assert result["error_type"] == "invalid_parameter"
        assert "invalid parameter" in result["description"].lower()
        assert result["severity"] == "medium"
        assert result["retryable"] is False

    def test_error_code_103_maps_to_permission_denied(self):
        """Error code 103 should map to permission_denied error type."""
        result = get_error_category(103)
        assert result["error_type"] == "permission_denied"
        assert "Permission denied" in result["description"]
        assert result["severity"] == "high"
        assert result["retryable"] is False

    def test_error_code_200_maps_to_missing_parameter(self):
        """Error code 200 should map to missing_parameter error type."""
        result = get_error_category(200)
        assert result["error_type"] == "missing_parameter"
        assert "missing" in result["description"].lower()
        assert result["severity"] == "medium"
        assert result["retryable"] is False

    def test_error_code_400_maps_to_invalid_filter(self):
        """Error code 400 should map to invalid_filter error type."""
        result = get_error_category(400)
        assert result["error_type"] == "invalid_filter"
        assert "filter" in result["description"].lower()
        assert result["severity"] == "medium"
        assert result["retryable"] is False

    def test_error_code_401_maps_to_invalid_key(self):
        """Error code 401 should map to invalid_key error type."""
        result = get_error_category(401)
        assert result["error_type"] == YouTubeAPIError.TYPE_INVALID_KEY
        assert "API key is invalid" in result["description"]
        assert result["severity"] == "critical"
        assert result["retryable"] is False

    def test_error_code_402_maps_to_quota_exceeded(self):
        """Error code 402 should map to quota_exceeded error type."""
        result = get_error_category(402)
        assert result["error_type"] == YouTubeAPIError.TYPE_QUOTA_EXCEEDED
        assert "quota exceeded" in result["description"].lower()
        assert result["severity"] == "high"
        assert result["retryable"] is False

    def test_error_code_403_maps_to_rate_limited(self):
        """Error code 403 should map to rate_limited error type."""
        result = get_error_category(403)
        assert result["error_type"] == YouTubeAPIError.TYPE_RATE_LIMITED
        assert "rate limit" in result["description"].lower()
        assert result["severity"] == "medium"
        assert result["retryable"] is True

    def test_error_code_404_maps_to_not_found(self):
        """Error code 404 should map to not_found error type."""
        result = get_error_category(404)
        assert result["error_type"] == "not_found"
        assert "not found" in result["description"].lower()
        assert result["severity"] == "low"
        assert result["retryable"] is False

    def test_error_code_500_maps_to_temporary_error(self):
        """Error code 500 should map to temporary_error (retryable)."""
        result = get_error_category(500)
        assert result["error_type"] == "temporary_error"
        assert "internal server error" in result["description"].lower()
        assert result["severity"] == "low"
        assert result["retryable"] is True

    def test_error_code_503_maps_to_temporary_error(self):
        """Error code 503 should map to temporary_error (retryable)."""
        result = get_error_category(503)
        assert result["error_type"] == "temporary_error"
        assert "backend error" in result["description"].lower()
        assert result["severity"] == "low"
        assert result["retryable"] is True


class TestGetErrorDescription:
    """Tests for get_error_description() function."""

    def test_get_error_description_returns_string(self):
        """Verify get_error_description returns a string."""
        desc = get_error_description(401)
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_get_error_description_401(self):
        """Verify error description for code 401."""
        desc = get_error_description(401)
        assert "API key is invalid" in desc

    def test_get_error_description_402(self):
        """Verify error description for code 402."""
        desc = get_error_description(402)
        assert "quota exceeded" in desc.lower()

    def test_get_error_description_403(self):
        """Verify error description for code 403."""
        desc = get_error_description(403)
        assert "rate limit" in desc.lower()

    def test_get_error_description_unknown(self):
        """Verify error description for unknown code returns fallback."""
        desc = get_error_description(999)
        assert "Unknown" in desc


class TestGetErrorTypeForCode:
    """Tests for get_error_type_for_code() function."""

    def test_get_error_type_for_code_returns_string(self):
        """Verify get_error_type_for_code returns a string."""
        error_type = get_error_type_for_code(401)
        assert isinstance(error_type, str)

    def test_get_error_type_for_code_401(self):
        """Verify error type for code 401."""
        error_type = get_error_type_for_code(401)
        assert error_type == YouTubeAPIError.TYPE_INVALID_KEY

    def test_get_error_type_for_code_402(self):
        """Verify error type for code 402."""
        error_type = get_error_type_for_code(402)
        assert error_type == YouTubeAPIError.TYPE_QUOTA_EXCEEDED

    def test_get_error_type_for_code_403(self):
        """Verify error type for code 403."""
        error_type = get_error_type_for_code(403)
        assert error_type == YouTubeAPIError.TYPE_RATE_LIMITED

    def test_get_error_type_for_code_unknown(self):
        """Verify error type for unknown code returns 'unknown'."""
        error_type = get_error_type_for_code(999)
        assert error_type == "unknown"


class TestYouTubeAPIErrorCodeEnum:
    """Tests for YouTubeAPIErrorCode enum."""

    def test_error_code_enum_values(self):
        """Verify enum contains expected error codes."""
        assert YouTubeAPIErrorCode.ERR_API_DISABLED.value == 100
        assert YouTubeAPIErrorCode.ERR_NOT_FOUND.value == 101
        assert YouTubeAPIErrorCode.ERR_INVALID_PARAMETER.value == 102
        assert YouTubeAPIErrorCode.ERR_PERMISSION_DENIED.value == 103
        assert YouTubeAPIErrorCode.ERR_QUOTA_EXCEEDED.value == 402
        assert YouTubeAPIErrorCode.ERR_RATE_LIMIT_EXCEEDED.value == 403
        assert YouTubeAPIErrorCode.ERR_RESOURCE_NOT_FOUND.value == 404
        assert YouTubeAPIErrorCode.ERR_INTERNAL_ERROR.value == 500
        assert YouTubeAPIErrorCode.ERR_BACKEND_ERROR.value == 503


class TestYouTubeAPIErrorWithCode:
    """Tests for YouTubeAPIError with error_code field."""

    def test_youtube_api_error_accepts_error_code(self):
        """Verify YouTubeAPIError accepts error_code parameter."""
        error = YouTubeAPIError(
            "Test error",
            error_code=401,
            endpoint="search"
        )
        assert error.error_code == 401

    def test_youtube_api_error_str_includes_error_code(self):
        """Verify YouTubeAPIError __str__ includes error code."""
        error = YouTubeAPIError(
            "Test error",
            error_code=401
        )
        error_str = str(error)
        assert "401" in error_str

    def test_youtube_api_error_str_includes_description(self):
        """Verify YouTubeAPIError __str__ includes human-readable description."""
        error = YouTubeAPIError(
            "Test error",
            error_code=402
        )
        error_str = str(error)
        # Should include error code in format [Error 402]
        assert "402" in error_str or "quota" in error_str.lower()

    def test_youtube_api_error_with_none_code(self):
        """Verify YouTubeAPIError handles None error_code gracefully."""
        error = YouTubeAPIError("Test error", error_code=None)
        assert error.error_code is None
        error_str = str(error)
        # Should not include "[Error None]"
        assert "[Error None]" not in error_str


# =============================================================================
# US-157-005: Error Code to Action Mapping Tests
# =============================================================================


class TestErrorActionEnum:
    """Tests for ErrorAction enum values."""

    def test_error_action_values(self):
        """Verify ErrorAction enum has expected values."""
        assert ErrorAction.RETRY.value == "retry"
        assert ErrorAction.BACKOFF.value == "backoff"
        assert ErrorAction.FALLBACK.value == "fallback"
        assert ErrorAction.ABORT.value == "abort"


class TestErrorCodeToAction:
    """Tests for ERROR_CODE_TO_ACTION mapping."""

    def test_error_code_to_action_exists(self):
        """Verify ERROR_CODE_TO_ACTION dictionary exists and contains entries."""
        assert ERROR_CODE_TO_ACTION is not None
        assert isinstance(ERROR_CODE_TO_ACTION, dict)
        assert len(ERROR_CODE_TO_ACTION) > 0

    def test_error_code_100_action_abort(self):
        """Error code 100 (API disabled) should result in ABORT."""
        action = error_code_to_action(100)
        assert action == ErrorAction.ABORT

    def test_error_code_101_action_abort(self):
        """Error code 101 (not found) should result in ABORT."""
        action = error_code_to_action(101)
        assert action == ErrorAction.ABORT

    def test_error_code_102_action_abort(self):
        """Error code 102 (invalid parameter) should result in ABORT."""
        action = error_code_to_action(102)
        assert action == ErrorAction.ABORT

    def test_error_code_103_action_abort(self):
        """Error code 103 (permission denied) should result in ABORT."""
        action = error_code_to_action(103)
        assert action == ErrorAction.ABORT

    def test_error_code_200_action_abort(self):
        """Error code 200 (missing parameter) should result in ABORT."""
        action = error_code_to_action(200)
        assert action == ErrorAction.ABORT

    def test_error_code_400_action_abort(self):
        """Error code 400 (invalid filter) should result in ABORT."""
        action = error_code_to_action(400)
        assert action == ErrorAction.ABORT

    def test_error_code_401_action_fallback(self):
        """Error code 401 (invalid key) should result in FALLBACK."""
        action = error_code_to_action(401)
        assert action == ErrorAction.FALLBACK

    def test_error_code_402_action_fallback(self):
        """Error code 402 (quota exceeded) should result in FALLBACK."""
        action = error_code_to_action(402)
        assert action == ErrorAction.FALLBACK

    def test_error_code_403_action_backoff(self):
        """Error code 403 (rate limited) should result in BACKOFF."""
        action = error_code_to_action(403)
        assert action == ErrorAction.BACKOFF

    def test_error_code_404_action_abort(self):
        """Error code 404 (not found) should result in ABORT."""
        action = error_code_to_action(404)
        assert action == ErrorAction.ABORT

    def test_error_code_500_action_backoff(self):
        """Error code 500 (internal error) should result in BACKOFF."""
        action = error_code_to_action(500)
        assert action == ErrorAction.BACKOFF

    def test_error_code_501_action_abort(self):
        """Error code 501 (not implemented) should result in ABORT."""
        action = error_code_to_action(501)
        assert action == ErrorAction.ABORT

    def test_error_code_503_action_backoff(self):
        """Error code 503 (backend error) should result in BACKOFF."""
        action = error_code_to_action(503)
        assert action == ErrorAction.BACKOFF

    def test_unknown_error_code_defaults_to_backoff(self):
        """Unknown error code should default to BACKOFF."""
        action = error_code_to_action(999)
        assert action == ErrorAction.BACKOFF


class TestErrorTypeToAction:
    """Tests for get_action_for_error_type function."""

    def test_rate_limited_returns_backoff(self):
        """Error type 'rate_limited' should return BACKOFF action."""
        action = get_action_for_error_type("rate_limited")
        assert action == ErrorAction.BACKOFF

    def test_quota_exceeded_returns_fallback(self):
        """Error type 'quota_exceeded' should return FALLBACK action."""
        action = get_action_for_error_type("quota_exceeded")
        assert action == ErrorAction.FALLBACK

    def test_daily_quota_exceeded_returns_fallback(self):
        """Error type 'daily_quota_exceeded' should return FALLBACK action."""
        action = get_action_for_error_type("daily_quota_exceeded")
        assert action == ErrorAction.FALLBACK

    def test_invalid_key_returns_fallback(self):
        """Error type 'invalid_key' should return FALLBACK action."""
        action = get_action_for_error_type("invalid_key")
        assert action == ErrorAction.FALLBACK

    def test_not_found_returns_abort(self):
        """Error type 'not_found' should return ABORT action."""
        action = get_action_for_error_type("not_found")
        assert action == ErrorAction.ABORT

    def test_temporary_error_returns_backoff(self):
        """Error type 'temporary_error' should return BACKOFF action."""
        action = get_action_for_error_type("temporary_error")
        assert action == ErrorAction.BACKOFF

    def test_network_error_returns_backoff(self):
        """Error type 'network_error' should return BACKOFF action."""
        action = get_action_for_error_type("network_error")
        assert action == ErrorAction.BACKOFF

    def test_geo_blocked_returns_fallback(self):
        """Error type 'geo_blocked' should return FALLBACK action."""
        action = get_action_for_error_type("geo_blocked")
        assert action == ErrorAction.FALLBACK

    def test_premium_required_returns_abort(self):
        """Error type 'premium_required' should return ABORT action."""
        action = get_action_for_error_type("premium_required")
        assert action == ErrorAction.ABORT

    def test_device_limit_returns_retry(self):
        """Error type 'device_limit' should return RETRY action."""
        action = get_action_for_error_type("device_limit")
        assert action == ErrorAction.RETRY

    def test_disabled_project_returns_abort(self):
        """Error type 'disabled_project' should return ABORT action."""
        action = get_action_for_error_type("disabled_project")
        assert action == ErrorAction.ABORT

    def test_invalid_project_returns_abort(self):
        """Error type 'invalid_project' should return ABORT action."""
        action = get_action_for_error_type("invalid_project")
        assert action == ErrorAction.ABORT

    def test_unknown_error_type_defaults_to_backoff(self):
        """Unknown error type should default to BACKOFF action."""
        action = get_action_for_error_type("unknown")
        assert action == ErrorAction.BACKOFF


class TestRecoveryPathSelection:
    """Tests for recovery path selection based on error classification."""

    def test_quota_error_triggers_fallback_path(self):
        """Quota exceeded error should trigger fallback path."""
        action = error_code_to_action(402)
        # Fallback path means switch to yt-dlp
        assert action == ErrorAction.FALLBACK

    def test_rate_limit_triggers_backoff_path(self):
        """Rate limit error should trigger backoff and retry path."""
        action = error_code_to_action(403)
        # Backoff path means exponential backoff retry
        assert action == ErrorAction.BACKOFF

    def test_invalid_key_triggers_fallback(self):
        """Invalid API key error should trigger fallback."""
        action = error_code_to_action(401)
        # Fallback path means switch to yt-dlp
        assert action == ErrorAction.FALLBACK

    def test_not_found_triggers_abort(self):
        """Not found error should abort current item."""
        action = error_code_to_action(101)
        assert action == ErrorAction.ABORT
        action = error_code_to_action(404)
        assert action == ErrorAction.ABORT

    def test_server_error_triggers_backoff(self):
        """5xx server errors should trigger backoff retry."""
        action_500 = error_code_to_action(500)
        action_503 = error_code_to_action(503)
        assert action_500 == ErrorAction.BACKOFF
        assert action_503 == ErrorAction.BACKOFF

    def test_critical_config_error_triggers_abort(self):
        """Critical configuration errors should abort operation."""
        # API disabled
        action = error_code_to_action(100)
        assert action == ErrorAction.ABORT
        # Invalid parameter (code bug)
        action = error_code_to_action(102)
        assert action == ErrorAction.ABORT


class TestErrorDistribution:
    """Tests for error distribution metrics (US-157-005)."""

    def test_error_metrics_tracker_exists_in_error_classification(self):
        """Verify ErrorMetricsTracker is available in error_classification module."""
        from src.downloader.error_classification import ErrorMetricsTracker, get_error_metrics
        tracker = get_error_metrics()
        assert tracker is not None
        assert isinstance(tracker, ErrorMetricsTracker)

    def test_error_metrics_can_record_errors(self):
        """Verify error metrics tracker can record errors."""
        from src.downloader.error_classification import get_error_metrics
        tracker = get_error_metrics()
        tracker.reset()
        tracker.record_error("rate_limit")
        tracker.record_error("rate_limit")
        tracker.record_error("quota_exceeded")
        counts = tracker.get_counts()
        assert counts.get("rate_limit") == 2
        assert counts.get("quota_exceeded") == 1

    def test_error_metrics_get_percentages(self):
        """Verify error metrics can return percentages."""
        from src.downloader.error_classification import get_error_metrics
        tracker = get_error_metrics()
        tracker.reset()
        tracker.record_error("rate_limit")
        tracker.record_error("rate_limit")
        tracker.record_error("rate_limit")
        tracker.record_error("quota_exceeded")
        percentages = tracker.get_percentages()
        assert percentages.get("rate_limit") == 75.0
        assert percentages.get("quota_exceeded") == 25.0

    def test_error_metrics_summary_includes_error_type(self):
        """Verify error metrics summary includes error type distribution."""
        from src.downloader.error_classification import get_error_metrics
        tracker = get_error_metrics()
        tracker.reset()
        # Record category first
        tracker.record_error("quota_exceeded")
        # Then record error type
        tracker.record_error_type("quota_exceeded")
        tracker.record_error_type("daily_quota_exceeded")
        summary = tracker.get_summary()
        assert "by_error_type" in summary
        assert summary["by_error_type"].get("quota_exceeded") == 1
        assert summary["by_error_type"].get("daily_quota_exceeded") == 1
