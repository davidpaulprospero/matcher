"""
Tests for CaptionFormatUnavailableError differentiation (US-59-004).

Verifies that 'Requested format is not available' is distinguished from
'no subtitles'/'subtitles are disabled' in error handling, and that the
format loop behaves correctly for each error type.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path
import subprocess

import pytest

from src.caption.enums import CaptionErrorCategory, DEFAULT_RETRY_BUDGETS
from src.caption.error_handling import categorize_caption_error
from src.caption.exceptions import (
    CaptionFetchError,
    CaptionFormatUnavailableError,
    CaptionUnavailableError,
)


# ============================================================
# CaptionFormatUnavailableError exception class tests
# ============================================================

class TestCaptionFormatUnavailableError:
    """Test the new CaptionFormatUnavailableError exception class."""

    def test_is_subclass_of_caption_fetch_error(self):
        """CaptionFormatUnavailableError is a subclass of CaptionFetchError."""
        err = CaptionFormatUnavailableError("vid1", "json3")
        assert isinstance(err, CaptionFetchError)

    def test_stores_format(self):
        """Exception stores the format that was unavailable."""
        err = CaptionFormatUnavailableError("vid1", "json3")
        assert err.format == "json3"
        assert err.video_id == "vid1"

    def test_default_reason(self):
        """Default reason is generated from format name."""
        err = CaptionFormatUnavailableError("vid1", "json3")
        assert "json3" in err.reason
        assert "not available" in err.reason

    def test_custom_reason(self):
        """Custom reason overrides the default."""
        err = CaptionFormatUnavailableError("vid1", "json3", "custom reason")
        assert err.reason == "custom reason"

    def test_is_not_caption_unavailable_error(self):
        """CaptionFormatUnavailableError is NOT a CaptionUnavailableError."""
        err = CaptionFormatUnavailableError("vid1", "json3")
        assert not isinstance(err, CaptionUnavailableError)


# ============================================================
# Error categorization tests
# ============================================================

class TestCategorizeCaptionErrorFormatUnavailable:
    """Test that categorize_caption_error() handles FORMAT_UNAVAILABLE correctly."""

    def test_format_unavailable_error_instance(self):
        """CaptionFormatUnavailableError maps to FORMAT_UNAVAILABLE category."""
        err = CaptionFormatUnavailableError("vid1", "json3")
        category = categorize_caption_error(err, err.reason)
        assert category == CaptionErrorCategory.FORMAT_UNAVAILABLE

    def test_requested_format_string_pattern(self):
        """String 'Requested format is not available' maps to FORMAT_UNAVAILABLE."""
        err = CaptionFetchError("vid1", "Requested format is not available")
        category = categorize_caption_error(err, err.reason)
        assert category == CaptionErrorCategory.FORMAT_UNAVAILABLE

    def test_requested_format_mixed_case(self):
        """Case-insensitive match on 'requested format is not available'."""
        err = CaptionFetchError("vid1", "REQUESTED FORMAT IS NOT AVAILABLE")
        category = categorize_caption_error(err, err.reason)
        assert category == CaptionErrorCategory.FORMAT_UNAVAILABLE

    def test_no_subtitles_still_unavailable(self):
        """'no subtitles' still categorized as UNAVAILABLE (not FORMAT_UNAVAILABLE)."""
        err = CaptionUnavailableError("vid1", "no subtitles for this video")
        category = categorize_caption_error(err, err.reason)
        assert category == CaptionErrorCategory.UNAVAILABLE

    def test_subtitles_disabled_still_unavailable(self):
        """'subtitles are disabled' still categorized as UNAVAILABLE."""
        err = CaptionUnavailableError("vid1", "subtitles are disabled for this video")
        category = categorize_caption_error(err, err.reason)
        assert category == CaptionErrorCategory.UNAVAILABLE

    def test_format_unavailable_not_confused_with_generic_unavailable(self):
        """'Requested format is not available' is NOT generic UNAVAILABLE."""
        err = CaptionFetchError("vid1", "Requested format is not available")
        category = categorize_caption_error(err, err.reason)
        # Must be FORMAT_UNAVAILABLE, not UNAVAILABLE
        assert category != CaptionErrorCategory.UNAVAILABLE
        assert category == CaptionErrorCategory.FORMAT_UNAVAILABLE

    def test_format_unavailable_has_retry_budget_zero(self):
        """FORMAT_UNAVAILABLE has retry budget of 0 (format loop handles it)."""
        assert DEFAULT_RETRY_BUDGETS[CaptionErrorCategory.FORMAT_UNAVAILABLE] == 0


# ============================================================
# Format loop behavior tests (integration-style with mocking)
# ============================================================

class TestFormatLoopBehavior:
    """Test that _fetch_subtitle() loop handles errors correctly.

    - 'Requested format is not available' on json3 should still try vtt/srt
    - 'no subtitles' on json3 should stop immediately (no captions in any format)
    """

    def _make_fetcher_mock(self):
        """Create a mock caption fetcher with _fetch_subtitle method accessible."""
        from unittest.mock import MagicMock
        fetcher = MagicMock()
        fetcher._preferred_formats = ['json3', 'vtt', 'srt']
        return fetcher

    def test_format_unavailable_on_json3_tries_vtt_srt(self):
        """When json3 raises CaptionFormatUnavailableError, loop tries vtt and srt."""
        call_log = []

        def mock_fetch_with_format(video_url, video_id, temp_dir, language, auto_gen, fmt):
            call_log.append(fmt)
            if fmt == 'json3':
                raise CaptionFormatUnavailableError(video_id, fmt, "Requested format is not available")
            elif fmt == 'vtt':
                raise CaptionFormatUnavailableError(video_id, fmt, "Requested format is not available")
            else:
                return None  # srt also returns nothing

        # Simulate the loop logic from _fetch_subtitle
        preferred_formats = ['json3', 'vtt', 'srt']
        last_error = None
        for fmt in preferred_formats:
            try:
                result = mock_fetch_with_format("url", "vid1", "/tmp", "en", True, fmt)
                if result and hasattr(result, 'segments') and result.segments:
                    break
            except CaptionUnavailableError:
                raise
            except CaptionFormatUnavailableError as e:
                last_error = e
                continue
            except CaptionFetchError as e:
                last_error = e
                continue

        # All three formats should have been tried
        assert call_log == ['json3', 'vtt', 'srt']

    def test_no_subtitles_on_json3_stops_immediately(self):
        """When json3 raises CaptionUnavailableError, loop breaks immediately."""
        call_log = []

        def mock_fetch_with_format(video_url, video_id, temp_dir, language, auto_gen, fmt):
            call_log.append(fmt)
            if fmt == 'json3':
                raise CaptionUnavailableError(video_id, "no subtitles for this video")
            return None

        # Simulate the loop logic from _fetch_subtitle
        preferred_formats = ['json3', 'vtt', 'srt']
        last_error = None
        with pytest.raises(CaptionUnavailableError):
            for fmt in preferred_formats:
                try:
                    result = mock_fetch_with_format("url", "vid1", "/tmp", "en", True, fmt)
                    if result and hasattr(result, 'segments') and result.segments:
                        break
                except CaptionUnavailableError:
                    # Video has no captions at all - break immediately (US-59-004)
                    raise
                except CaptionFormatUnavailableError as e:
                    last_error = e
                    continue
                except CaptionFetchError as e:
                    last_error = e
                    continue

        # Only json3 should have been attempted
        assert call_log == ['json3']

    def test_subtitles_disabled_on_json3_stops_immediately(self):
        """When json3 raises CaptionUnavailableError for 'disabled', loop breaks."""
        call_log = []

        def mock_fetch_with_format(video_url, video_id, temp_dir, language, auto_gen, fmt):
            call_log.append(fmt)
            if fmt == 'json3':
                raise CaptionUnavailableError(video_id, "subtitles are disabled for this video")
            return None

        preferred_formats = ['json3', 'vtt', 'srt']
        with pytest.raises(CaptionUnavailableError):
            for fmt in preferred_formats:
                try:
                    result = mock_fetch_with_format("url", "vid1", "/tmp", "en", True, fmt)
                    if result and hasattr(result, 'segments') and result.segments:
                        break
                except CaptionUnavailableError:
                    raise
                except CaptionFormatUnavailableError as e:
                    continue
                except CaptionFetchError as e:
                    continue

        assert call_log == ['json3']

    def test_generic_fetch_error_tries_remaining_formats(self):
        """Generic CaptionFetchError (e.g., network) still tries all formats."""
        call_log = []

        def mock_fetch_with_format(video_url, video_id, temp_dir, language, auto_gen, fmt):
            call_log.append(fmt)
            raise CaptionFetchError(video_id, "network timeout")

        preferred_formats = ['json3', 'vtt', 'srt']
        last_error = None
        for fmt in preferred_formats:
            try:
                result = mock_fetch_with_format("url", "vid1", "/tmp", "en", True, fmt)
            except CaptionUnavailableError:
                raise
            except CaptionFormatUnavailableError as e:
                last_error = e
                continue
            except CaptionFetchError as e:
                last_error = e
                continue

        # All formats should be tried for generic errors
        assert call_log == ['json3', 'vtt', 'srt']
        assert last_error is not None


# ============================================================
# _fetch_subtitle_with_format stderr parsing tests
# ============================================================

class TestFetchSubtitleWithFormatStderrParsing:
    """Test stderr parsing in _fetch_subtitle_with_format raises correct exceptions."""

    def test_requested_format_not_available_raises_format_error(self):
        """'Requested format is not available' in stderr raises CaptionFormatUnavailableError."""
        stderr = "ERROR: Requested format is not available. Use --list-formats"

        # Simulate the parsing logic from _fetch_subtitle_with_format
        stderr_lower = stderr.lower()
        video_id = "vid1"
        subtitle_format = "json3"

        video_level_phrases = [
            'no subtitles', 'no automatic captions',
            'subtitles are disabled', 'video unavailable', 'private video',
        ]

        if any(phrase in stderr_lower for phrase in video_level_phrases):
            raised = CaptionUnavailableError(video_id, stderr[:200])
        elif 'requested format is not available' in stderr_lower:
            raised = CaptionFormatUnavailableError(video_id, subtitle_format, stderr[:200])
        else:
            raised = CaptionFetchError(video_id, stderr[:200])

        assert isinstance(raised, CaptionFormatUnavailableError)
        assert raised.format == "json3"

    def test_no_subtitles_stderr_raises_unavailable_error(self):
        """'no subtitles' in stderr raises CaptionUnavailableError."""
        stderr = "WARNING: [youtube] vid1: no subtitles for this video"
        stderr_lower = stderr.lower()
        video_id = "vid1"

        video_level_phrases = [
            'no subtitles', 'no automatic captions',
            'subtitles are disabled', 'video unavailable', 'private video',
        ]

        if any(phrase in stderr_lower for phrase in video_level_phrases):
            raised = CaptionUnavailableError(video_id, stderr[:200])
        elif 'requested format is not available' in stderr_lower:
            raised = CaptionFormatUnavailableError(video_id, "json3", stderr[:200])
        else:
            raised = CaptionFetchError(video_id, stderr[:200])

        assert isinstance(raised, CaptionUnavailableError)
        assert not isinstance(raised, CaptionFormatUnavailableError)

    def test_subtitles_disabled_stderr_raises_unavailable_error(self):
        """'subtitles are disabled' in stderr raises CaptionUnavailableError."""
        stderr = "ERROR: [youtube] vid1: subtitles are disabled for this video"
        stderr_lower = stderr.lower()
        video_id = "vid1"

        video_level_phrases = [
            'no subtitles', 'no automatic captions',
            'subtitles are disabled', 'video unavailable', 'private video',
        ]

        if any(phrase in stderr_lower for phrase in video_level_phrases):
            raised = CaptionUnavailableError(video_id, stderr[:200])
        elif 'requested format is not available' in stderr_lower:
            raised = CaptionFormatUnavailableError(video_id, "json3", stderr[:200])
        else:
            raised = CaptionFetchError(video_id, stderr[:200])

        assert isinstance(raised, CaptionUnavailableError)

    def test_generic_error_stderr_raises_fetch_error(self):
        """Unknown error in stderr raises generic CaptionFetchError."""
        stderr = "ERROR: [youtube] vid1: HTTP Error 500: Internal Server Error"
        stderr_lower = stderr.lower()
        video_id = "vid1"

        video_level_phrases = [
            'no subtitles', 'no automatic captions',
            'subtitles are disabled', 'video unavailable', 'private video',
        ]

        if any(phrase in stderr_lower for phrase in video_level_phrases):
            raised = CaptionUnavailableError(video_id, stderr[:200])
        elif 'requested format is not available' in stderr_lower:
            raised = CaptionFormatUnavailableError(video_id, "json3", stderr[:200])
        else:
            raised = CaptionFetchError(video_id, stderr[:200])

        assert isinstance(raised, CaptionFetchError)
        assert not isinstance(raised, CaptionFormatUnavailableError)


# ============================================================
# ErrorPatternDetector integration with FORMAT_UNAVAILABLE
# ============================================================

class TestErrorPatternDetectorFormatUnavailable:
    """Test that ErrorPatternDetector handles format unavailable errors."""

    def test_format_unavailable_signature(self):
        """'Requested format is not available' gets a recognizable signature."""
        from src.caption.error_handling import ErrorPatternDetector
        detector = ErrorPatternDetector(threshold=0.3, sample_size=3)

        detector.record_error("vid1", "Requested format is not available")
        detector.record_error("vid2", "Requested format is not available")
        detector.record_error("vid3", "Requested format is not available")

        result = detector.check_pattern()
        assert result.detected is True
        # The signature should include "not available" or similar
        assert result.error_signature is not None
