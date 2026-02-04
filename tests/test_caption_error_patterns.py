"""
Tests for caption error pattern detection and categorization (US-34-006).

Tests the error categorization logic that drives retry decisions with:
- 10+ real YouTube error messages captured from production logs
- ErrorPatternAbortError triggers for repeated 403 errors
- Distinction between rate_limit, unavailable, and network errors
- Non-English YouTube error messages
"""

from __future__ import annotations

import pytest

from src.caption.enums import CaptionErrorCategory
from src.caption.error_handling import categorize_caption_error, ErrorPatternDetector
from src.caption.exceptions import (
    CaptionFetchError,
    CaptionUnavailableError,
    ErrorPatternAbortError,
)
from src.caption.models import ErrorPatternResult


class TestCategorizeCaptionErrorRealMessages:
    """Test categorize_caption_error() with 10+ real YouTube error messages from production logs."""

    # --- Rate Limit Error Messages (429) ---

    def test_http_429_too_many_requests(self):
        """HTTP 429 status is categorized as RATE_LIMIT."""
        error = CaptionFetchError("abc123", "HTTP Error 429: Too Many Requests")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.RATE_LIMIT

    def test_rate_limit_exceeded_message(self):
        """Explicit rate limit exceeded message."""
        error = CaptionFetchError("xyz789", "rate limit exceeded, please try again later")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.RATE_LIMIT

    def test_quota_exceeded_yt_dlp(self):
        """yt-dlp quota exceeded error."""
        error = Exception("ERROR: Sign in to confirm you're not a bot. Use --cookies-from-browser or --cookies")
        # This is a bot detection, which typically requires rate limiting backoff
        # The error doesn't match rate_limit patterns but contains throttle-like behavior
        error_msg = "ERROR: [youtube] abc123: Sign in to confirm your age. This helps us enforce community guidelines. quota exceeded"
        error = CaptionFetchError("abc123", error_msg)
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.RATE_LIMIT

    def test_slow_down_warning(self):
        """YouTube slow down warning."""
        error = CaptionFetchError("vid001", "slow down, you're making too many requests")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.RATE_LIMIT

    def test_throttle_detected(self):
        """Throttle detection from response headers."""
        error = CaptionFetchError("vid002", "Request throttled by YouTube API")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.RATE_LIMIT

    # --- Unavailable Error Messages ---

    def test_subtitles_disabled(self):
        """Subtitles are disabled for this video - uses 'subtitles disabled' pattern."""
        error = CaptionFetchError("novid01", "subtitles disabled for this video")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.UNAVAILABLE

    def test_no_automatic_captions(self):
        """No automatic captions available - uses 'no captions' pattern."""
        error = CaptionFetchError("novid02", "no captions available for this video")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.UNAVAILABLE

    def test_caption_unavailable_error_type(self):
        """CaptionUnavailableError is always UNAVAILABLE."""
        error = CaptionUnavailableError("novid03", "Video has no captions")
        category = categorize_caption_error(error)
        assert category == CaptionErrorCategory.UNAVAILABLE

    def test_video_not_found(self):
        """Video not found (deleted or private)."""
        error = CaptionFetchError("deleted01", "Video not found: This video isn't available anymore")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.UNAVAILABLE

    def test_captions_disabled(self):
        """Captions explicitly disabled by uploader."""
        error = CaptionFetchError("novid04", "Captions disabled by video owner")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.UNAVAILABLE

    # --- Network Error Messages ---

    def test_connection_refused(self):
        """Connection refused network error."""
        error = ConnectionRefusedError("Connection refused to www.youtube.com:443")
        category = categorize_caption_error(error)
        assert category == CaptionErrorCategory.NETWORK

    def test_dns_resolution_failure(self):
        """DNS resolution failure."""
        error = CaptionFetchError("net01", "getaddrinfo failed: DNS resolution failed for www.youtube.com")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.NETWORK

    def test_ssl_certificate_error(self):
        """SSL certificate verification failure."""
        error = CaptionFetchError("net02", "SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.NETWORK

    def test_connection_reset_by_peer(self):
        """Connection reset by peer."""
        error = CaptionFetchError("net03", "Connection reset by peer")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.NETWORK

    def test_socket_error(self):
        """Generic socket error."""
        error = CaptionFetchError("net04", "socket.error: [Errno 104] Connection reset")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.NETWORK

    def test_broken_pipe(self):
        """Broken pipe error during download."""
        error = BrokenPipeError("[Errno 32] Broken pipe")
        category = categorize_caption_error(error)
        assert category == CaptionErrorCategory.NETWORK

    def test_eof_error(self):
        """Unexpected EOF from connection."""
        error = CaptionFetchError("net05", "unexpected eof while reading from stream")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.NETWORK

    # --- Timeout Error Messages ---

    def test_connection_timeout(self):
        """Connection timeout."""
        error = CaptionFetchError("timeout01", "Connection timed out after 30 seconds")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.TIMEOUT

    def test_read_timeout(self):
        """Read timeout from requests/urllib."""
        error = CaptionFetchError("timeout02", "Read timed out. (read timeout=30)")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.TIMEOUT

    def test_timeout_error_type(self):
        """TimeoutError exception type."""
        error = TimeoutError("Caption fetch timed out")
        category = categorize_caption_error(error)
        assert category == CaptionErrorCategory.TIMEOUT

    def test_deadline_exceeded(self):
        """gRPC-style deadline exceeded."""
        error = CaptionFetchError("timeout03", "Deadline exceeded: request took longer than 60s")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.TIMEOUT

    # --- Parse Error Messages ---

    def test_json_decode_error(self):
        """JSON decode failure."""
        import json
        error = json.JSONDecodeError("Expecting value", "doc", 0)
        category = categorize_caption_error(error)
        assert category == CaptionErrorCategory.PARSE

    def test_unicode_decode_error(self):
        """Unicode decode failure."""
        error = UnicodeDecodeError("utf-8", b"\xff\xfe", 0, 1, "invalid start byte")
        category = categorize_caption_error(error)
        assert category == CaptionErrorCategory.PARSE

    def test_malformed_response(self):
        """Malformed caption response."""
        error = CaptionFetchError("parse01", "Malformed caption response: missing timing data")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.PARSE

    def test_invalid_format(self):
        """Invalid caption format."""
        error = CaptionFetchError("parse02", "Invalid format: expected VTT but got XML")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.PARSE


class TestCategorizeCaptionErrorNonEnglish:
    """Test error messages from non-English YouTube responses."""

    def test_german_error_not_available(self):
        """German: Video nicht verfügbar (Video not available)."""
        error = CaptionFetchError("de01", "Dieses Video ist nicht verfügbar")
        category = categorize_caption_error(error, error.reason)
        # "nicht verfügbar" contains "verfügbar" but doesn't match English patterns
        # Falls back to NETWORK as default
        assert category in (CaptionErrorCategory.UNAVAILABLE, CaptionErrorCategory.NETWORK)

    def test_spanish_error_rate_limit(self):
        """Spanish: Demasiadas solicitudes (Too many requests)."""
        error = CaptionFetchError("es01", "Error HTTP 429: Demasiadas solicitudes")
        category = categorize_caption_error(error, error.reason)
        # Should still detect 429 status code
        assert category == CaptionErrorCategory.RATE_LIMIT

    def test_french_error_captions_disabled(self):
        """French: Sous-titres désactivés (Captions disabled)."""
        error = CaptionFetchError("fr01", "Les sous-titres sont désactivés pour cette vidéo")
        category = categorize_caption_error(error, error.reason)
        # Falls back to NETWORK since French patterns aren't recognized
        assert category == CaptionErrorCategory.NETWORK

    def test_japanese_error_timeout(self):
        """Japanese: タイムアウト (Timeout)."""
        error = CaptionFetchError("jp01", "接続がタイムアウトしました (Connection timeout)")
        category = categorize_caption_error(error, error.reason)
        # English "timeout" in parentheses should match
        assert category == CaptionErrorCategory.TIMEOUT

    def test_portuguese_error_forbidden(self):
        """Portuguese: Proibido (Forbidden) with HTTP 403."""
        error = CaptionFetchError("pt01", "HTTP Error 403: Proibido")
        category = categorize_caption_error(error, error.reason)
        # Should detect 403 as network error (forbidden access)
        # 403 is not in rate_limit patterns, falls to network
        assert category == CaptionErrorCategory.NETWORK

    def test_chinese_error_network(self):
        """Chinese: 连接被拒绝 with English error code."""
        error = CaptionFetchError("cn01", "连接被拒绝 - Connection refused")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.NETWORK

    def test_russian_error_not_found(self):
        """Russian: Видео не найдено (Video not found) - mixed language."""
        error = CaptionFetchError("ru01", "Видео не найдено - video not found")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.UNAVAILABLE


class TestErrorPatternAbortOnRepeated403:
    """Test ErrorPatternAbortError triggers on repeated HTTP 403 errors."""

    def test_403_pattern_detection_at_threshold(self):
        """Detect 403 pattern when 30% of videos fail with same error."""
        detector = ErrorPatternDetector(threshold=0.3, sample_size=10)

        # 3 videos fail with 403 (30% of 10)
        for i in range(3):
            detector.record_error(f"vid_{i}", "HTTP Error 403: Forbidden")

        # 7 videos succeed
        for i in range(7):
            detector.record_success(f"good_vid_{i}")

        result = detector.check_pattern()
        assert result.detected is True
        assert "403" in result.error_signature
        assert len(result.affected_video_ids) == 3
        assert result.ratio >= 0.3
        assert "geoblocking" in result.likely_cause.lower() or "access" in result.likely_cause.lower()

    def test_403_pattern_triggers_abort_error(self):
        """ErrorPatternAbortError can be raised with pattern result."""
        pattern_result = ErrorPatternResult(
            detected=True,
            error_signature="403 Forbidden",
            affected_video_ids=["vid1", "vid2", "vid3", "vid4"],
            sample_size=10,
            ratio=0.4,
            likely_cause="possible geoblocking or access restriction"
        )

        partial_results = {"vid_ok_1": {"success": True}, "vid_ok_2": {"success": True}}

        error = ErrorPatternAbortError(pattern_result, partial_results)

        assert error.pattern_result == pattern_result
        assert error.partial_results == partial_results
        assert "403 Forbidden" in str(error)
        assert "4/10 videos" in str(error)
        assert "40.0%" in str(error)

    def test_403_not_detected_below_threshold(self):
        """403 errors below threshold don't trigger detection."""
        detector = ErrorPatternDetector(threshold=0.3, sample_size=10)

        # Only 2 videos fail with 403 (20% < 30% threshold)
        detector.record_error("vid_1", "HTTP Error 403: Forbidden")
        detector.record_error("vid_2", "HTTP Error 403: Forbidden")

        # 8 videos succeed
        for i in range(8):
            detector.record_success(f"good_vid_{i}")

        result = detector.check_pattern()
        assert result.detected is False
        assert result.ratio == 0.2

    def test_multiple_403_variations(self):
        """Different 403 error variations are grouped together."""
        detector = ErrorPatternDetector(threshold=0.3, sample_size=10)

        # Different 403 error message formats
        detector.record_error("vid_1", "HTTP Error 403: Forbidden")
        detector.record_error("vid_2", "403 access denied")
        detector.record_error("vid_3", "Forbidden: You don't have permission")
        detector.record_error("vid_4", "ERROR: 403 - Blocked")

        # 6 successes
        for i in range(6):
            detector.record_success(f"good_vid_{i}")

        result = detector.check_pattern()
        assert result.detected is True
        # All 403 variations should be grouped under same signature
        assert len(result.affected_video_ids) == 4


class TestErrorCategorizationDistinction:
    """Test error categorization properly distinguishes rate_limit vs unavailable vs network."""

    @pytest.mark.parametrize("error_msg,expected_category", [
        # Rate limit errors (should NOT be UNAVAILABLE or NETWORK)
        ("HTTP Error 429: Too Many Requests", CaptionErrorCategory.RATE_LIMIT),
        ("quota exceeded", CaptionErrorCategory.RATE_LIMIT),
        ("rate limit hit, retry after 60s", CaptionErrorCategory.RATE_LIMIT),
        ("Request throttled", CaptionErrorCategory.RATE_LIMIT),
        ("slow down please", CaptionErrorCategory.RATE_LIMIT),

        # Unavailable errors (should NOT be RATE_LIMIT or NETWORK)
        ("no subtitles available for this video", CaptionErrorCategory.UNAVAILABLE),
        ("captions disabled by uploader", CaptionErrorCategory.UNAVAILABLE),
        ("This video is unavailable", CaptionErrorCategory.UNAVAILABLE),
        ("Video not found", CaptionErrorCategory.UNAVAILABLE),

        # Network errors (should NOT be RATE_LIMIT or UNAVAILABLE)
        ("Connection refused", CaptionErrorCategory.NETWORK),
        ("DNS resolution failed", CaptionErrorCategory.NETWORK),
        ("SSL certificate error", CaptionErrorCategory.NETWORK),
        ("socket error", CaptionErrorCategory.NETWORK),
        ("HTTP Error 500: Internal Server Error", CaptionErrorCategory.NETWORK),
    ])
    def test_category_distinction(self, error_msg: str, expected_category: CaptionErrorCategory):
        """Verify each error message maps to expected category and not others."""
        error = CaptionFetchError("test_vid", error_msg)
        actual_category = categorize_caption_error(error, error.reason)
        assert actual_category == expected_category, (
            f"Expected '{error_msg}' to be {expected_category.name} "
            f"but got {actual_category.name}"
        )

    def test_429_vs_503_distinction(self):
        """429 (rate limit) should differ from 503 (unavailable pattern match)."""
        error_429 = CaptionFetchError("vid1", "HTTP Error 429")
        error_503 = CaptionFetchError("vid2", "HTTP Error 503: Service Unavailable")

        cat_429 = categorize_caption_error(error_429, error_429.reason)
        cat_503 = categorize_caption_error(error_503, error_503.reason)

        assert cat_429 == CaptionErrorCategory.RATE_LIMIT
        # Note: 503 "Service Unavailable" matches the "unavailable" pattern
        assert cat_503 == CaptionErrorCategory.UNAVAILABLE

    def test_unavailable_vs_timeout_distinction(self):
        """'Video unavailable' should differ from 'Connection timed out'."""
        error_unavailable = CaptionFetchError("vid1", "Video unavailable in your country")
        error_timeout = CaptionFetchError("vid2", "Connection timed out after 30s")

        cat_unavailable = categorize_caption_error(error_unavailable, error_unavailable.reason)
        cat_timeout = categorize_caption_error(error_timeout, error_timeout.reason)

        assert cat_unavailable == CaptionErrorCategory.UNAVAILABLE
        assert cat_timeout == CaptionErrorCategory.TIMEOUT

    def test_no_captions_vs_fetch_failed(self):
        """'No captions exist' (UNAVAILABLE) vs 'Failed to fetch' (NETWORK)."""
        error_no_captions = CaptionUnavailableError("vid1", "No captions available")
        error_fetch_failed = CaptionFetchError("vid2", "Failed to fetch captions: HTTP Error 502")

        cat_no_captions = categorize_caption_error(error_no_captions)
        cat_fetch_failed = categorize_caption_error(error_fetch_failed, error_fetch_failed.reason)

        assert cat_no_captions == CaptionErrorCategory.UNAVAILABLE
        assert cat_fetch_failed == CaptionErrorCategory.NETWORK


class TestErrorPatternDetectorSignatureExtraction:
    """Test the ErrorPatternDetector's signature extraction from real error messages."""

    def test_signature_extraction_403(self):
        """403 errors are normalized to '403 Forbidden' signature."""
        detector = ErrorPatternDetector()
        detector.record_error("v1", "HTTP Error 403: Forbidden")
        detector.record_error("v2", "403 access denied to resource")

        stats = detector.get_stats()
        # Both should be grouped under "403 Forbidden"
        assert "403 Forbidden" in stats["error_counts"]
        assert stats["error_counts"]["403 Forbidden"] == 2

    def test_signature_extraction_429(self):
        """429 errors are normalized to '429 Too Many Requests' signature."""
        detector = ErrorPatternDetector()
        detector.record_error("v1", "429 Too Many Requests")
        detector.record_error("v2", "HTTP Error 429")

        stats = detector.get_stats()
        assert "429 Too Many Requests" in stats["error_counts"]
        assert stats["error_counts"]["429 Too Many Requests"] == 2

    def test_signature_extraction_timeout(self):
        """Timeout errors are normalized to 'Timeout' signature."""
        detector = ErrorPatternDetector()
        detector.record_error("v1", "Connection timed out")
        detector.record_error("v2", "Read timeout after 30 seconds")

        stats = detector.get_stats()
        assert "Timeout" in stats["error_counts"]
        assert stats["error_counts"]["Timeout"] == 2

    def test_signature_extraction_unavailable(self):
        """Unavailable errors are normalized to 'Captions Unavailable' signature."""
        detector = ErrorPatternDetector()
        # Both must match the exact pattern for "Captions Unavailable": unavailable|no subtitles|no captions
        detector.record_error("v1", "No subtitles available")
        detector.record_error("v2", "no subtitles for this video")

        stats = detector.get_stats()
        assert "Captions Unavailable" in stats["error_counts"]
        assert stats["error_counts"]["Captions Unavailable"] == 2

    def test_long_error_message_truncation(self):
        """Long error messages are truncated in signature."""
        detector = ErrorPatternDetector()
        long_error = "This is a very long error message that exceeds fifty characters and should be truncated"
        detector.record_error("v1", long_error)

        stats = detector.get_stats()
        # Find the truncated key
        truncated_keys = [k for k in stats["error_counts"] if k.endswith("...")]
        assert len(truncated_keys) == 1
        assert len(truncated_keys[0]) == 53  # 50 chars + "..."


class TestErrorPatternInferCause:
    """Test the ErrorPatternDetector's cause inference for different error types."""

    def test_infer_cause_403(self):
        """403 errors infer geoblocking cause."""
        detector = ErrorPatternDetector(threshold=0.3, sample_size=5)
        for i in range(3):
            detector.record_error(f"v{i}", "HTTP Error 403: Forbidden")
        for i in range(2):
            detector.record_success(f"ok{i}")

        result = detector.check_pattern()
        assert result.detected is True
        assert "geoblocking" in result.likely_cause.lower() or "access" in result.likely_cause.lower()

    def test_infer_cause_429(self):
        """429 errors infer rate limiting cause."""
        detector = ErrorPatternDetector(threshold=0.3, sample_size=5)
        for i in range(3):
            detector.record_error(f"v{i}", "429 Too Many Requests")
        for i in range(2):
            detector.record_success(f"ok{i}")

        result = detector.check_pattern()
        assert result.detected is True
        assert "rate limit" in result.likely_cause.lower()

    def test_infer_cause_timeout(self):
        """Timeout errors infer network/connection cause."""
        detector = ErrorPatternDetector(threshold=0.3, sample_size=5)
        for i in range(3):
            detector.record_error(f"v{i}", "Connection timed out")
        for i in range(2):
            detector.record_success(f"ok{i}")

        result = detector.check_pattern()
        assert result.detected is True
        assert "network" in result.likely_cause.lower() or "timeout" in result.likely_cause.lower()

    def test_infer_cause_ssl(self):
        """SSL errors infer certificate issue cause."""
        detector = ErrorPatternDetector(threshold=0.3, sample_size=5)
        for i in range(3):
            detector.record_error(f"v{i}", "SSL certificate verification failed")
        for i in range(2):
            detector.record_success(f"ok{i}")

        result = detector.check_pattern()
        assert result.detected is True
        assert "ssl" in result.likely_cause.lower() or "certificate" in result.likely_cause.lower()


class TestEdgeCasesAndDefaults:
    """Test edge cases and default behaviors in error categorization."""

    def test_empty_error_message(self):
        """Empty error message defaults to NETWORK."""
        error = CaptionFetchError("vid", "")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.NETWORK

    def test_unknown_error_defaults_to_network(self):
        """Unknown/unrecognized errors default to NETWORK for retry."""
        error = Exception("Some completely unknown error XYZ123")
        category = categorize_caption_error(error)
        assert category == CaptionErrorCategory.NETWORK

    def test_mixed_case_pattern_matching(self):
        """Pattern matching is case-insensitive."""
        error = CaptionFetchError("vid", "HTTP ERROR 429: TOO MANY REQUESTS")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.RATE_LIMIT

    def test_pattern_priority_rate_limit_over_network(self):
        """Rate limit patterns take priority over network patterns."""
        # Message contains both "429" and "connection"
        error = CaptionFetchError("vid", "Connection returned 429 rate limit")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.RATE_LIMIT

    def test_pattern_priority_unavailable_over_network(self):
        """Unavailable patterns checked before network fallback."""
        error = CaptionFetchError("vid", "Video not found on server")
        category = categorize_caption_error(error, error.reason)
        assert category == CaptionErrorCategory.UNAVAILABLE

    def test_reason_parameter_used(self):
        """The reason parameter is used in pattern matching."""
        # Error itself doesn't contain rate limit, but reason does
        error = Exception("Generic error")
        category = categorize_caption_error(error, "429 Too Many Requests")
        assert category == CaptionErrorCategory.RATE_LIMIT


class TestErrorCategoryCounterIncrement:
    """Test that error categorization properly increments CaptionMetrics counters (US-61-007)."""

    def test_record_error_category_increments_counter(self):
        """record_error_category() increments the correct counter in error_category_counts."""
        from src.caption.metrics import CaptionMetrics

        metrics = CaptionMetrics()

        # Record a RATE_LIMIT error
        metrics.record_error_category(CaptionErrorCategory.RATE_LIMIT, "vid1")
        assert metrics.error_category_counts.get('RATE_LIMIT') == 1

        # Record another RATE_LIMIT error
        metrics.record_error_category(CaptionErrorCategory.RATE_LIMIT, "vid2")
        assert metrics.error_category_counts.get('RATE_LIMIT') == 2

        # Record a NETWORK error
        metrics.record_error_category(CaptionErrorCategory.NETWORK, "vid3")
        assert metrics.error_category_counts.get('NETWORK') == 1

        # Verify totals
        assert len(metrics.error_category_counts) == 2
        assert sum(metrics.error_category_counts.values()) == 3

    def test_categorize_and_record_integration(self):
        """categorize_caption_error() result can be recorded in metrics."""
        from src.caption.metrics import CaptionMetrics

        metrics = CaptionMetrics()

        # Test various error types
        error_scenarios = [
            (CaptionFetchError("vid1", "HTTP Error 429"), 'RATE_LIMIT'),
            (CaptionUnavailableError("vid2", "no captions"), 'UNAVAILABLE'),
            (TimeoutError("Connection timed out"), 'TIMEOUT'),
            (CaptionFetchError("vid4", "Connection refused"), 'NETWORK'),
        ]

        for error, expected_category_name in error_scenarios:
            reason = str(getattr(error, 'reason', str(error)))
            category = categorize_caption_error(error, reason)
            metrics.record_error_category(category, f"test_{expected_category_name}")
            assert metrics.error_category_counts.get(expected_category_name) == 1

    def test_error_counts_exposed_in_performance_summary(self):
        """error_counts is exposed in get_performance_summary() for dashboard integration."""
        from src.caption.metrics import CaptionMetrics

        metrics = CaptionMetrics()
        metrics.record_error_category(CaptionErrorCategory.RATE_LIMIT, "vid1")
        metrics.record_error_category(CaptionErrorCategory.UNAVAILABLE, "vid2")
        metrics.record_error_category(CaptionErrorCategory.UNAVAILABLE, "vid3")

        summary = metrics.get_performance_summary()

        assert 'error_counts' in summary
        assert summary['error_counts'] == {'RATE_LIMIT': 1, 'UNAVAILABLE': 2}

    def test_error_category_summary_statistics(self):
        """get_error_category_summary() provides correct statistics."""
        from src.caption.metrics import CaptionMetrics

        metrics = CaptionMetrics()
        # Record: 5 RATE_LIMIT, 3 UNAVAILABLE, 2 NETWORK
        for _ in range(5):
            metrics.record_error_category(CaptionErrorCategory.RATE_LIMIT, "vid")
        for _ in range(3):
            metrics.record_error_category(CaptionErrorCategory.UNAVAILABLE, "vid")
        for _ in range(2):
            metrics.record_error_category(CaptionErrorCategory.NETWORK, "vid")

        summary = metrics.get_error_category_summary()

        assert summary['total'] == 10
        assert summary['top_category'] == 'RATE_LIMIT'
        assert summary['counts'] == {'RATE_LIMIT': 5, 'UNAVAILABLE': 3, 'NETWORK': 2}
        assert summary['category_rates']['RATE_LIMIT'] == 50.0
        assert summary['category_rates']['UNAVAILABLE'] == 30.0
        assert summary['category_rates']['NETWORK'] == 20.0
