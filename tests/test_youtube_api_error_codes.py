"""Unit tests for YouTube API error code mapping (US-156-010).

Tests verify:
- ERROR_CODE_MAPPING contains valid entries for error codes 100-110
- get_error_category() returns correct error type, description, severity
- get_error_description() returns human-readable descriptions
- get_error_type_for_code() returns error type strings
- YouTubeAPIError includes error_code in string representation
- Unknown error codes return default values
"""

import pytest
from src.downloader.errors import (
    YouTubeAPIError,
    YouTubeAPIErrorCode,
    ERROR_CODE_MAPPING,
    get_error_category,
    get_error_description,
    get_error_type_for_code,
)


class TestYouTubeAPIErrorCodeMapping:
    """Tests for ERROR_CODE_MAPPING dictionary."""

    def test_error_code_100_maps_to_permission_denied(self):
        """Error code 100 should map to permission_denied error type."""
        result = get_error_category(100)
        assert result["error_type"] == YouTubeAPIError.TYPE_PERMISSION_DENIED
        assert result["severity"] == "critical"
        assert result["retryable"] is False

    def test_error_code_101_maps_to_not_found(self):
        """Error code 101 should map to not_found error type."""
        result = get_error_category(101)
        assert result["error_type"] == "not_found"
        assert result["severity"] == "low"
        assert result["retryable"] is False

    def test_error_code_102_maps_to_invalid_parameter(self):
        """Error code 102 should map to invalid_parameter error type."""
        result = get_error_category(102)
        assert result["error_type"] == "invalid_parameter"
        assert result["severity"] == "medium"
        assert result["retryable"] is False

    def test_error_code_103_maps_to_permission_denied(self):
        """Error code 103 should map to permission_denied error type."""
        result = get_error_category(103)
        assert result["error_type"] == YouTubeAPIError.TYPE_PERMISSION_DENIED
        assert result["severity"] == "high"
        assert result["retryable"] is False

    def test_error_code_200_maps_to_missing_parameter(self):
        """Error code 200 should map to missing_parameter error type."""
        result = get_error_category(200)
        assert result["error_type"] == "missing_parameter"
        assert result["severity"] == "medium"
        assert result["retryable"] is False

    def test_error_code_400_maps_to_invalid_filter(self):
        """Error code 400 should map to invalid_filter error type."""
        result = get_error_category(400)
        assert result["error_type"] == "invalid_filter"
        assert result["severity"] == "medium"
        assert result["retryable"] is False

    def test_error_code_401_maps_to_invalid_key(self):
        """Error code 401 should map to invalid_key error type."""
        result = get_error_category(401)
        assert result["error_type"] == YouTubeAPIError.TYPE_INVALID_KEY
        assert result["severity"] == "critical"
        assert result["retryable"] is False

    def test_error_code_402_maps_to_quota_exceeded(self):
        """Error code 402 should map to quota_exceeded error type."""
        result = get_error_category(402)
        assert result["error_type"] == YouTubeAPIError.TYPE_QUOTA_EXCEEDED
        assert result["severity"] == "high"
        assert result["retryable"] is False

    def test_error_code_403_maps_to_rate_limited(self):
        """Error code 403 should map to rate_limited error type."""
        result = get_error_category(403)
        assert result["error_type"] == YouTubeAPIError.TYPE_RATE_LIMITED
        assert result["severity"] == "medium"
        assert result["retryable"] is True

    def test_error_code_404_maps_to_not_found(self):
        """Error code 404 should map to not_found error type."""
        result = get_error_category(404)
        assert result["error_type"] == "not_found"
        assert result["severity"] == "low"
        assert result["retryable"] is False

    def test_error_code_500_maps_to_temporary_error(self):
        """Error code 500 should map to temporary_error with retryable=True."""
        result = get_error_category(500)
        assert result["error_type"] == "temporary_error"
        assert result["severity"] == "low"
        assert result["retryable"] is True

    def test_error_code_503_maps_to_temporary_error(self):
        """Error code 503 should map to temporary_error with retryable=True."""
        result = get_error_category(503)
        assert result["error_type"] == "temporary_error"
        assert result["severity"] == "low"
        assert result["retryable"] is True


class TestGetErrorCategory:
    """Tests for get_error_category() function."""

    def test_returns_dict_with_all_keys(self):
        """get_error_category should return dict with error_type, description, severity, retryable."""
        result = get_error_category(401)
        assert "error_type" in result
        assert "description" in result
        assert "severity" in result
        assert "retryable" in result

    def test_unknown_error_code_returns_defaults(self):
        """Unknown error code should return default values."""
        result = get_error_category(999)
        assert result["error_type"] == "unknown"
        assert result["description"] == "Unknown YouTube API error code: 999"
        assert result["severity"] == "medium"
        assert result["retryable"] is True

    def test_negative_error_code_returns_defaults(self):
        """Negative error code should return default values."""
        result = get_error_category(-1)
        assert result["error_type"] == "unknown"
        assert result["retryable"] is True


class TestGetErrorDescription:
    """Tests for get_error_description() function."""

    def test_returns_description_for_known_code(self):
        """Should return description for known error code."""
        desc = get_error_description(402)
        assert "quota" in desc.lower()
        assert "midnight" in desc.lower()

    def test_returns_unknown_for_unknown_code(self):
        """Should return unknown message for unknown error code."""
        desc = get_error_description(999)
        assert "Unknown" in desc


class TestGetErrorTypeForCode:
    """Tests for get_error_type_for_code() function."""

    def test_returns_error_type_string(self):
        """Should return error type string for known code."""
        error_type = get_error_type_for_code(403)
        assert error_type == YouTubeAPIError.TYPE_RATE_LIMITED

    def test_returns_unknown_for_unknown_code(self):
        """Should return 'unknown' for unknown error code."""
        error_type = get_error_type_for_code(999)
        assert error_type == "unknown"


class TestYouTubeAPIErrorWithCode:
    """Tests for YouTubeAPIError with error_code field."""

    def test_error_code_field_stored(self):
        """YouTubeAPIError should store error_code in attribute."""
        error = YouTubeAPIError(
            "Test error",
            error_code=402,
        )
        assert error.error_code == 402

    def test_error_code_none_by_default(self):
        """error_code should be None by default."""
        error = YouTubeAPIError("Test error")
        assert error.error_code is None

    def test_str_includes_error_code(self):
        """__str__ should include error code when present."""
        error = YouTubeAPIError(
            "API quota exceeded",
            error_code=402,
        )
        error_str = str(error)
        assert "402" in error_str

    def test_str_without_error_code(self):
        """__str__ should not include error code when None."""
        error = YouTubeAPIError("Test error")
        error_str = str(error)
        assert "Error" not in error_str or "None" not in error_str

    def test_error_code_403_includes_rate_limit_guidance(self):
        """Error 403 should include rate limit guidance when error_type is set."""
        error = YouTubeAPIError(
            "Rate limit exceeded",
            error_type=YouTubeAPIError.TYPE_RATE_LIMITED,
            error_code=403,
        )
        assert error.error_type == YouTubeAPIError.TYPE_RATE_LIMITED
        assert error.error_code == 403
        assert "wait" in error.actionable_guidance.lower()

    def test_error_code_401_includes_key_guidance(self):
        """Error 401 should include API key guidance when error_type is set."""
        error = YouTubeAPIError(
            "Invalid API key",
            error_type=YouTubeAPIError.TYPE_INVALID_KEY,
            error_code=401,
        )
        assert error.error_type == YouTubeAPIError.TYPE_INVALID_KEY
        assert error.error_code == 401
        assert "key" in error.actionable_guidance.lower()


class TestYouTubeAPIErrorCodeEnum:
    """Tests for YouTubeAPIErrorCode enum."""

    def test_error_code_enum_values(self):
        """Verify error code enum values are correct."""
        assert YouTubeAPIErrorCode.ERR_API_DISABLED.value == 100
        assert YouTubeAPIErrorCode.ERR_NOT_FOUND.value == 101
        assert YouTubeAPIErrorCode.ERR_INVALID_PARAMETER.value == 102
        assert YouTubeAPIErrorCode.ERR_PERMISSION_DENIED.value == 103
        assert YouTubeAPIErrorCode.ERR_INVALID_API_KEY.value == 401
        assert YouTubeAPIErrorCode.ERR_QUOTA_EXCEEDED.value == 402
        assert YouTubeAPIErrorCode.ERR_RATE_LIMIT_EXCEEDED.value == 403
        assert YouTubeAPIErrorCode.ERR_RESOURCE_NOT_FOUND.value == 404
        assert YouTubeAPIErrorCode.ERR_INTERNAL_ERROR.value == 500
        assert YouTubeAPIErrorCode.ERR_BACKEND_ERROR.value == 503


# =============================================================================
# US-157-005: Error Code to Action Mapping Tests
# =============================================================================


from src.downloader.errors import (
    ErrorAction,
    error_code_to_action,
    get_action_for_error_type,
)


class TestErrorCodeToAction:
    """Tests for get_error_action() function."""

    def test_error_code_100_abort(self):
        """Error code 100 (API disabled) should abort."""
        action = error_code_to_action(100)
        assert action == ErrorAction.ABORT

    def test_error_code_101_abort(self):
        """Error code 101 (not found) should abort."""
        action = error_code_to_action(101)
        assert action == ErrorAction.ABORT

    def test_error_code_102_abort(self):
        """Error code 102 (invalid parameter) should abort."""
        action = error_code_to_action(102)
        assert action == ErrorAction.ABORT

    def test_error_code_103_abort(self):
        """Error code 103 (permission denied) should abort."""
        action = error_code_to_action(103)
        assert action == ErrorAction.ABORT

    def test_error_code_200_abort(self):
        """Error code 200 (missing parameter) should abort."""
        action = error_code_to_action(200)
        assert action == ErrorAction.ABORT

    def test_error_code_400_abort(self):
        """Error code 400 (invalid filter) should abort."""
        action = error_code_to_action(400)
        assert action == ErrorAction.ABORT

    def test_error_code_401_fallback(self):
        """Error code 401 (invalid key) should fallback."""
        action = error_code_to_action(401)
        assert action == ErrorAction.FALLBACK

    def test_error_code_402_fallback(self):
        """Error code 402 (quota exceeded) should fallback."""
        action = error_code_to_action(402)
        assert action == ErrorAction.FALLBACK

    def test_error_code_403_backoff(self):
        """Error code 403 (rate limited) should backoff."""
        action = error_code_to_action(403)
        assert action == ErrorAction.BACKOFF

    def test_error_code_404_abort(self):
        """Error code 404 (not found) should abort."""
        action = error_code_to_action(404)
        assert action == ErrorAction.ABORT

    def test_error_code_500_backoff(self):
        """Error code 500 (internal error) should backoff."""
        action = error_code_to_action(500)
        assert action == ErrorAction.BACKOFF

    def test_error_code_503_backoff(self):
        """Error code 503 (backend error) should backoff."""
        action = error_code_to_action(503)
        assert action == ErrorAction.BACKOFF

    def test_unknown_error_code_backoff(self):
        """Unknown error codes should default to backoff."""
        action = error_code_to_action(999)
        assert action == ErrorAction.BACKOFF


class TestGetActionForErrorType:
    """Tests for get_action_for_error_type() function."""

    def test_quota_exceeded_fallback(self):
        """quota_exceeded error type should fallback."""
        action = get_action_for_error_type('quota_exceeded')
        assert action == ErrorAction.FALLBACK

    def test_daily_quota_exceeded_fallback(self):
        """daily_quota_exceeded error type should fallback."""
        action = get_action_for_error_type('daily_quota_exceeded')
        assert action == ErrorAction.FALLBACK

    def test_invalid_key_fallback(self):
        """invalid_key error type should fallback."""
        action = get_action_for_error_type('invalid_key')
        assert action == ErrorAction.FALLBACK

    def test_rate_limited_backoff(self):
        """rate_limited error type should backoff."""
        action = get_action_for_error_type('rate_limited')
        assert action == ErrorAction.BACKOFF

    def test_per_second_limit_retry(self):
        """per_second_limit error type should retry."""
        action = get_action_for_error_type('per_second_limit')
        assert action == ErrorAction.RETRY

    def test_temporary_error_backoff(self):
        """temporary_error error type should backoff."""
        action = get_action_for_error_type('temporary_error')
        assert action == ErrorAction.BACKOFF

    def test_timeout_backoff(self):
        """timeout error type should backoff."""
        action = get_action_for_error_type('timeout')
        assert action == ErrorAction.BACKOFF

    def test_not_found_abort(self):
        """not_found error type should abort."""
        action = get_action_for_error_type('not_found')
        assert action == ErrorAction.ABORT

    def test_permission_denied_abort(self):
        """permission_denied error type should abort."""
        action = get_action_for_error_type('permission_denied')
        assert action == ErrorAction.ABORT

    def test_geo_blocked_fallback(self):
        """geo_blocked error type should fallback."""
        action = get_action_for_error_type('geo_blocked')
        assert action == ErrorAction.FALLBACK

    def test_network_error_backoff(self):
        """network_error should backoff."""
        action = get_action_for_error_type('network_error')
        assert action == ErrorAction.BACKOFF

    def test_invalid_project_abort(self):
        """invalid_project error type should abort."""
        action = get_action_for_error_type('invalid_project')
        assert action == ErrorAction.ABORT

    def test_disabled_project_abort(self):
        """disabled_project error type should abort."""
        action = get_action_for_error_type('disabled_project')
        assert action == ErrorAction.ABORT


class TestErrorActionEnum:
    """Tests for ErrorAction enum."""

    def test_error_action_retry_value(self):
        """ErrorAction.RETRY should have correct value."""
        assert ErrorAction.RETRY.value == "retry"

    def test_error_action_backoff_value(self):
        """ErrorAction.BACKOFF should have correct value."""
        assert ErrorAction.BACKOFF.value == "backoff"

    def test_error_action_fallback_value(self):
        """ErrorAction.FALLBACK should have correct value."""
        assert ErrorAction.FALLBACK.value == "fallback"

    def test_error_action_abort_value(self):
        """ErrorAction.ABORT should have correct value."""
        assert ErrorAction.ABORT.value == "abort"


# =============================================================================
# US-158-008: Tests for Error Reason Mapping
# =============================================================================


from src.downloader.errors import (
    get_error_reason_info,
    parse_youtube_api_error_response,
    get_error_domain_info,
)


class TestYouTubeAPIErrorReasons:
    """Tests for YouTube API error reason mappings (US-158-008)."""

    def test_not_found_reason(self):
        """Error reason 'notFound' should map to not_found error type."""
        result = get_error_reason_info("notFound")
        assert result["error_type"] == "not_found"
        assert result["severity"] == "low"
        assert result["retryable"] is False

    def test_closed_reason(self):
        """Error reason 'closed' should map to closed error type."""
        result = get_error_reason_info("closed")
        assert result["error_type"] == "closed"
        assert result["severity"] == "low"
        assert result["retryable"] is False

    def test_embedding_disabled_reason(self):
        """Error reason 'embeddingDisabled' should map to embedding_disabled error type."""
        result = get_error_reason_info("embeddingDisabled")
        assert result["error_type"] == "embedding_disabled"
        assert result["severity"] == "low"
        assert result["retryable"] is False

    def test_upload_failed_reason(self):
        """Error reason 'uploadFailed' should map to upload_failed error type."""
        result = get_error_reason_info("uploadFailed")
        assert result["error_type"] == "upload_failed"
        assert result["severity"] == "medium"
        assert result["retryable"] is True

    def test_invalid_request_reason(self):
        """Error reason 'invalidRequest' should map to invalid_request error type."""
        result = get_error_reason_info("invalidRequest")
        assert result["error_type"] == "invalid_request"
        assert result["severity"] == "medium"
        assert result["retryable"] is False

    def test_quota_exceeded_reason(self):
        """Error reason 'quotaExceeded' should map to quota_exceeded error type."""
        result = get_error_reason_info("quotaExceeded")
        assert result["error_type"] == "quota_exceeded"
        assert result["severity"] == "high"
        assert result["retryable"] is False

    def test_daily_limit_exceeded_reason(self):
        """Error reason 'dailyLimitExceeded' should map to daily_quota_exceeded error type."""
        result = get_error_reason_info("dailyLimitExceeded")
        assert result["error_type"] == "daily_quota_exceeded"
        assert result["severity"] == "high"

    def test_rate_limit_exceeded_reason(self):
        """Error reason 'rateLimitExceeded' should map to rate_limited error type."""
        result = get_error_reason_info("rateLimitExceeded")
        assert result["error_type"] == "rate_limited"
        assert result["severity"] == "medium"
        assert result["retryable"] is True

    def test_access_not_configured_reason(self):
        """Error reason 'accessNotConfigured' should map to permission_denied error type."""
        result = get_error_reason_info("accessNotConfigured")
        assert result["error_type"] == "permission_denied"
        assert result["severity"] == "critical"

    def test_forbidden_reason(self):
        """Error reason 'forbidden' should map to permission_denied error type."""
        result = get_error_reason_info("forbidden")
        assert result["error_type"] == "permission_denied"
        assert result["severity"] == "high"

    def test_unknown_reason_returns_defaults(self):
        """Unknown error reason should return default values."""
        result = get_error_reason_info("unknownReason")
        assert result["error_type"] == "unknown"
        assert result["severity"] == "medium"
        assert result["retryable"] is True
        assert "unknownReason" in result["description"]


class TestParseYouTubeAPIErrorResponse:
    """Tests for parsing YouTube API error responses (US-158-008)."""

    def test_parse_full_error_response(self):
        """Should parse complete error response with all fields."""
        error_response = {
            "error": {
                "code": 404,
                "message": "Not Found",
                "errors": [
                    {
                        "domain": "youtube.search",
                        "reason": "notFound",
                        "message": "Not Found",
                    }
                ],
            }
        }
        result = parse_youtube_api_error_response(error_response)
        assert result["code"] == 404
        assert result["reason"] == "notFound"
        assert result["domain"] == "youtube.search"
        assert result["message"] == "Not Found"

    def test_parse_error_with_multiple_errors(self):
        """Should extract from first error in array."""
        error_response = {
            "error": {
                "code": 403,
                "message": "Quota exceeded",
                "errors": [
                    {"reason": "quotaExceeded", "domain": "youtube"},
                    {"reason": "somethingElse", "domain": "other"},
                ],
            }
        }
        result = parse_youtube_api_error_response(error_response)
        assert result["code"] == 403
        assert result["reason"] == "quotaExceeded"
        assert result["domain"] == "youtube"

    def test_parse_empty_error(self):
        """Should handle empty error response."""
        result = parse_youtube_api_error_response({})
        assert result["code"] is None
        assert result["reason"] is None
        assert result["domain"] is None

    def test_parse_minimal_error(self):
        """Should handle minimal error response."""
        error_response = {"error": {"code": 500, "message": "Server Error"}}
        result = parse_youtube_api_error_response(error_response)
        assert result["code"] == 500
        assert result["message"] == "Server Error"
        assert result["reason"] is None


class TestErrorDomainInfo:
    """Tests for error domain information (US-158-008)."""

    def test_youtube_search_domain(self):
        """Should return readable description for youtube.search."""
        assert get_error_domain_info("youtube.search") == "YouTube Search API"

    def test_youtube_video_domain(self):
        """Should return readable description for youtube.video."""
        assert get_error_domain_info("youtube.video") == "YouTube Video API"

    def test_youtube_channel_domain(self):
        """Should return readable description for youtube.channel."""
        assert get_error_domain_info("youtube.channel") == "YouTube Channel API"

    def test_youtube_upload_domain(self):
        """Should return readable description for youtube.upload."""
        assert get_error_domain_info("youtube.upload") == "YouTube Upload API"

    def test_resumable_upload_domain(self):
        """Should return readable description for resumableUpload."""
        assert get_error_domain_info("resumableUpload") == "Resumable Upload API"

    def test_quota_error_domain(self):
        """Should return readable description for quotaError."""
        assert get_error_domain_info("quotaError") == "Quota Error"

    def test_unknown_domain_returns_default(self):
        """Unknown domain should return default description."""
        result = get_error_domain_info("unknown.domain")
        assert "Unknown domain" in result
        assert "unknown.domain" in result
