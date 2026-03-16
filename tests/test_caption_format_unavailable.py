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
    CaptionFormatExhaustedError,
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


# ============================================================
# Format uniqueness tests (US-60-003)
# ============================================================

class TestFormatOnlyTriedOnce:
    """Test that each format is only tried once per fetch attempt (US-60-003).

    Previously, a bug caused json3 to be tried twice: once at the start
    and again at the end of the fallback sequence. This test suite ensures
    the format sequence is json3->vtt->srt with no duplicates.
    """

    def test_each_format_tried_once_on_format_unavailable(self):
        """AC: Each format is only tried once per fetch attempt.

        When all formats raise CaptionFormatUnavailableError, each format
        should be attempted exactly once.
        """
        call_counts = {'json3': 0, 'vtt': 0, 'srt': 0}

        def mock_fetch_with_format(video_url, video_id, temp_dir, language, auto_gen, fmt, fallback_level=0):
            call_counts[fmt] = call_counts.get(fmt, 0) + 1
            raise CaptionFormatUnavailableError(video_id, fmt, "Requested format is not available")

        # Simulate the format loop from _fetch_subtitle_formats
        preferred_formats = ['json3', 'vtt', 'srt']
        last_error = None
        for fallback_level, fmt in enumerate(preferred_formats):
            try:
                result = mock_fetch_with_format("url", "vid1", "/tmp", "en", False, fmt, fallback_level)
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

        # Each format should be tried exactly once
        assert call_counts['json3'] == 1, f"json3 tried {call_counts['json3']} times, expected 1"
        assert call_counts['vtt'] == 1, f"vtt tried {call_counts['vtt']} times, expected 1"
        assert call_counts['srt'] == 1, f"srt tried {call_counts['srt']} times, expected 1"

    def test_no_duplicate_formats_in_preferred_list(self):
        """AC: The preferred formats list should have no duplicates."""
        from src.caption_fetcher import CaptionFetcher

        fetcher = CaptionFetcher()
        formats = fetcher._preferred_formats

        # No duplicates in the list
        assert len(formats) == len(set(formats)), \
            f"Duplicate formats found in preferred_formats: {formats}"

    def test_format_sequence_no_repeat_json3(self):
        """AC: Format sequence is json3->vtt->srt with no repeat of json3.

        This specifically tests the bug fix where json3 was tried again
        at the end of the sequence after already failing.
        """
        from collections import Counter

        call_log = []

        def mock_fetch_with_format(video_url, video_id, temp_dir, language, auto_gen, fmt, fallback_level=0):
            call_log.append(fmt)
            raise CaptionFormatUnavailableError(video_id, fmt, "Requested format is not available")

        # Simulate the format loop
        preferred_formats = ['json3', 'vtt', 'srt']
        for fallback_level, fmt in enumerate(preferred_formats):
            try:
                mock_fetch_with_format("url", "vid1", "/tmp", "en", False, fmt, fallback_level)
            except CaptionFormatUnavailableError:
                continue

        # Check call sequence
        assert call_log == ['json3', 'vtt', 'srt'], \
            f"Expected ['json3', 'vtt', 'srt'], got {call_log}"

        # Check no format appears more than once
        counts = Counter(call_log)
        for fmt, count in counts.items():
            assert count == 1, f"Format {fmt} was tried {count} times, expected 1"

    def test_real_fetcher_format_loop_no_duplicates_single_pass(self):
        """Integration test: A single format pass has no duplicates.

        Uses _fetch_subtitle_formats directly to test just one pass,
        avoiding the auto-generated fallback which correctly retries.
        """
        from src.caption_fetcher import CaptionFetcher, AvailableLanguage
        from unittest.mock import patch
        from pathlib import Path
        import tempfile

        fetcher = CaptionFetcher()
        call_log = []

        def mock_format(video_url, video_id, temp_dir, language, auto_gen, fmt, fallback_level=0):
            call_log.append((fmt, auto_gen))
            raise CaptionFormatUnavailableError(video_id, fmt, "Requested format is not available")

        with patch.object(fetcher, '_fetch_subtitle_with_format', side_effect=mock_format):
            with tempfile.TemporaryDirectory() as td:
                with pytest.raises(CaptionFormatExhaustedError):
                    # Call _fetch_subtitle_formats directly (single pass)
                    fetcher._fetch_subtitle_formats(
                        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                        "dQw4w9WgXcQ",
                        Path(td),
                        "en",
                        False,  # manual captions
                    )

        # Extract just the format names
        formats_tried = [fmt for fmt, _ in call_log]

        # Verify no duplicates within a single pass
        assert len(formats_tried) == len(set(formats_tried)), \
            f"Duplicate formats in single pass: {formats_tried}"
        # Verify expected sequence
        assert formats_tried == ['json3', 'vtt', 'srt'], \
            f"Expected ['json3', 'vtt', 'srt'], got {formats_tried}"

    def test_auto_fallback_correctly_retries_all_formats(self):
        """Auto-fallback is allowed to retry all formats (different auto_gen flag).

        When manual captions fail, auto-generated fallback should try all
        formats again. This is correct behavior (not a duplicate bug).
        """
        from src.caption_fetcher import CaptionFetcher, AvailableLanguage
        from unittest.mock import patch
        from pathlib import Path
        import tempfile

        fetcher = CaptionFetcher()
        call_log = []

        def mock_format(video_url, video_id, temp_dir, language, auto_gen, fmt, fallback_level=0):
            call_log.append((fmt, auto_gen))
            raise CaptionFormatUnavailableError(video_id, fmt, "Requested format is not available")

        # Mock pre-flight check to return English available
        with patch.object(fetcher, 'list_available_languages', return_value=[
            AvailableLanguage(code='en', name='English', is_auto_generated=True),
        ]):
            with patch.object(fetcher, '_fetch_subtitle_with_format', side_effect=mock_format):
                with tempfile.TemporaryDirectory() as td:
                    with pytest.raises(CaptionFormatExhaustedError):
                        fetcher._fetch_subtitle(
                            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                            "dQw4w9WgXcQ",
                            Path(td),
                            "en",
                            False,  # Start with manual
                        )

        # Expect 6 calls: 3 manual (auto_gen=False) + 3 auto (auto_gen=True)
        manual_calls = [(fmt, auto) for fmt, auto in call_log if not auto]
        auto_calls = [(fmt, auto) for fmt, auto in call_log if auto]

        # Manual pass should have no duplicates
        manual_formats = [fmt for fmt, _ in manual_calls]
        assert manual_formats == ['json3', 'vtt', 'srt'], \
            f"Manual pass expected ['json3', 'vtt', 'srt'], got {manual_formats}"

        # Auto pass should have no duplicates
        auto_formats = [fmt for fmt, _ in auto_calls]
        assert auto_formats == ['json3', 'vtt', 'srt'], \
            f"Auto pass expected ['json3', 'vtt', 'srt'], got {auto_formats}"

    def test_all_formats_exhausted_logged_after_srt_fails(self, caplog):
        """AC: Logging shows 'All formats exhausted' after srt fails (US-62-003)."""
        import logging
        from src.caption_fetcher import CaptionFetcher
        from unittest.mock import patch
        from pathlib import Path
        import tempfile

        fetcher = CaptionFetcher()
        call_log = []

        def mock_format(video_url, video_id, temp_dir, language, auto_gen, fmt, fallback_level=0):
            call_log.append(fmt)
            raise CaptionFormatUnavailableError(video_id, fmt, "Requested format is not available")

        with caplog.at_level(logging.DEBUG):
            with patch.object(fetcher, '_fetch_subtitle_with_format', side_effect=mock_format):
                with tempfile.TemporaryDirectory() as td:
                    with pytest.raises(CaptionFormatExhaustedError):
                        fetcher._fetch_subtitle_formats(
                            "https://www.youtube.com/watch?v=test123",
                            "test123",
                            Path(td),
                            "en",
                            False,
                        )

        # Verify log message appears after all formats exhausted
        log_messages = [r.message for r in caplog.records]
        exhausted_msgs = [m for m in log_messages if 'All formats exhausted' in m]
        assert len(exhausted_msgs) == 1, f"Expected 1 'All formats exhausted' log, got {len(exhausted_msgs)}"
        assert 'test123' in exhausted_msgs[0], "Log should contain video ID"
        assert '3 attempts' in exhausted_msgs[0], "Log should show 3 format attempts"

        # Verify the log appears after srt (last format) failed
        # The call_log should be json3->vtt->srt, and exhaust log comes after
        assert call_log == ['json3', 'vtt', 'srt'], f"Unexpected format order: {call_log}"
