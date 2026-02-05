"""Tests for US-67-002: Eliminate duplicate format retry in caption fallback sequence.

Verifies that:
1. Caption format fallback sequence does NOT repeat any format
2. When all formats fail, total yt-dlp invocations == len(unique_formats)
3. CaptionFormatExhaustedError is raised when all unique formats fail
4. Each format is tried exactly once then CaptionFormatExhaustedError raised
5. No subprocess is spawned for a format that already failed in same cycle
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.caption.exceptions import (
    CaptionFetchError,
    CaptionFormatExhaustedError,
    CaptionFormatUnavailableError,
    CaptionUnavailableError,
)
from src.caption_fetcher import CaptionFetcher


# ============================================================
# AC1: Format fallback sequence does NOT repeat any format
# ============================================================

class TestNoFormatDuplicatesInSequence:
    """AC: Caption format fallback sequence does NOT repeat any format."""

    def test_default_preferred_formats_have_no_duplicates(self):
        """Default _preferred_formats list has no duplicates."""
        fetcher = CaptionFetcher()
        formats = fetcher._preferred_formats
        assert len(formats) == len(set(formats)), \
            f"Duplicate formats found: {formats}"

    def test_config_with_duplicate_formats_gets_deduplicated(self):
        """Config providing ['json3', 'vtt', 'srt', 'json3'] is deduplicated."""
        config = MagicMock()
        config.download.caption_first.max_retries = 3
        config.download.caption_first.retry_delay = 2.0
        config.download.caption_first.timeout = 60
        config.download.caption_first.preferred_formats = ['json3', 'vtt', 'srt', 'json3']
        config.download.caption_first.adaptive_format_order = False
        config.download.caption_first.format_timeouts = None

        fetcher = CaptionFetcher(config=config)
        assert fetcher._preferred_formats == ['json3', 'vtt', 'srt'], \
            f"Expected ['json3', 'vtt', 'srt'], got {fetcher._preferred_formats}"

    def test_config_with_multiple_duplicates_deduplicated(self):
        """['json3', 'json3', 'vtt', 'vtt', 'srt'] becomes ['json3', 'vtt', 'srt']."""
        config = MagicMock()
        config.download.caption_first.max_retries = 3
        config.download.caption_first.retry_delay = 2.0
        config.download.caption_first.timeout = 60
        config.download.caption_first.preferred_formats = ['json3', 'json3', 'vtt', 'vtt', 'srt']
        config.download.caption_first.adaptive_format_order = False
        config.download.caption_first.format_timeouts = None

        fetcher = CaptionFetcher(config=config)
        assert fetcher._preferred_formats == ['json3', 'vtt', 'srt']

    def test_dedup_preserves_order(self):
        """Deduplication preserves the first occurrence order."""
        config = MagicMock()
        config.download.caption_first.max_retries = 3
        config.download.caption_first.retry_delay = 2.0
        config.download.caption_first.timeout = 60
        config.download.caption_first.preferred_formats = ['srt', 'json3', 'vtt', 'json3', 'srt']
        config.download.caption_first.adaptive_format_order = False
        config.download.caption_first.format_timeouts = None

        fetcher = CaptionFetcher(config=config)
        assert fetcher._preferred_formats == ['srt', 'json3', 'vtt']


# ============================================================
# AC2: Total yt-dlp invocations == len(unique_formats)
# ============================================================

class TestExactInvocationCount:
    """AC: When all formats fail, total yt-dlp invocations is exactly len(unique_formats)."""

    def test_three_unique_formats_exactly_three_invocations(self):
        """With ['json3', 'vtt', 'srt'], exactly 3 subprocess calls are made."""
        fetcher = CaptionFetcher()
        call_log = []

        def mock_format(video_url, video_id, temp_dir, language,
                        auto_gen, fmt, fallback_level=0):
            call_log.append(fmt)
            raise CaptionFormatUnavailableError(video_id, fmt, "not available")

        with patch.object(fetcher, '_fetch_subtitle_with_format', side_effect=mock_format):
            with tempfile.TemporaryDirectory() as td:
                with pytest.raises(CaptionFormatExhaustedError):
                    fetcher._fetch_subtitle_formats(
                        "https://www.youtube.com/watch?v=test",
                        "test",
                        Path(td),
                        "en",
                        False,
                    )

        assert len(call_log) == 3, \
            f"Expected exactly 3 invocations, got {len(call_log)}: {call_log}"

    def test_duplicate_config_still_only_unique_invocations(self):
        """Config with ['json3', 'vtt', 'srt', 'json3'] still makes only 3 calls."""
        fetcher = CaptionFetcher()
        # Simulate a config that had duplicates (even though init deduplicates,
        # test the _fetch_subtitle_formats method directly with duplicates)
        fetcher._preferred_formats = ['json3', 'vtt', 'srt', 'json3']
        call_log = []

        def mock_format(video_url, video_id, temp_dir, language,
                        auto_gen, fmt, fallback_level=0):
            call_log.append(fmt)
            raise CaptionFormatUnavailableError(video_id, fmt, "not available")

        with patch.object(fetcher, '_fetch_subtitle_with_format', side_effect=mock_format):
            with tempfile.TemporaryDirectory() as td:
                with pytest.raises(CaptionFormatExhaustedError):
                    fetcher._fetch_subtitle_formats(
                        "https://www.youtube.com/watch?v=test",
                        "test",
                        Path(td),
                        "en",
                        False,
                    )

        # Should only try 3 unique formats, not 4
        assert len(call_log) == 3, \
            f"Expected 3 invocations (deduplicated), got {len(call_log)}: {call_log}"
        assert call_log == ['json3', 'vtt', 'srt']


# ============================================================
# AC3: CaptionFormatExhaustedError raised when all unique formats fail
# ============================================================

class TestCaptionFormatExhaustedError:
    """AC: New CaptionFormatExhaustedError is raised when all unique formats fail."""

    def test_is_subclass_of_caption_fetch_error(self):
        """CaptionFormatExhaustedError is a subclass of CaptionFetchError."""
        err = CaptionFormatExhaustedError("vid1", formats_tried=["json3", "vtt", "srt"])
        assert isinstance(err, CaptionFetchError)

    def test_stores_formats_tried(self):
        """Exception stores the list of formats that were attempted."""
        err = CaptionFormatExhaustedError("vid1", formats_tried=["json3", "vtt", "srt"])
        assert err.formats_tried == ["json3", "vtt", "srt"]
        assert err.video_id == "vid1"

    def test_stores_formats_skipped(self):
        """Exception stores the count of skipped formats."""
        err = CaptionFormatExhaustedError(
            "vid1", formats_tried=["vtt"], formats_skipped=2
        )
        assert err.formats_skipped == 2

    def test_not_a_caption_unavailable_error(self):
        """CaptionFormatExhaustedError is NOT a CaptionUnavailableError."""
        err = CaptionFormatExhaustedError("vid1", formats_tried=["json3"])
        assert not isinstance(err, CaptionUnavailableError)

    def test_raised_from_fetch_subtitle_formats_all_fail(self):
        """_fetch_subtitle_formats raises CaptionFormatExhaustedError when all fail."""
        fetcher = CaptionFetcher()

        def mock_format(video_url, video_id, temp_dir, language,
                        auto_gen, fmt, fallback_level=0):
            raise CaptionFormatUnavailableError(video_id, fmt, "not available")

        with patch.object(fetcher, '_fetch_subtitle_with_format', side_effect=mock_format):
            with tempfile.TemporaryDirectory() as td:
                with pytest.raises(CaptionFormatExhaustedError) as exc_info:
                    fetcher._fetch_subtitle_formats(
                        "https://www.youtube.com/watch?v=test",
                        "test",
                        Path(td),
                        "en",
                        False,
                    )

                err = exc_info.value
                assert err.formats_tried == ["json3", "vtt", "srt"]
                assert err.formats_skipped == 0

    def test_raised_with_generic_fetch_errors(self):
        """CaptionFormatExhaustedError raised when all formats fail with CaptionFetchError."""
        fetcher = CaptionFetcher()

        def mock_format(video_url, video_id, temp_dir, language,
                        auto_gen, fmt, fallback_level=0):
            raise CaptionFetchError(video_id, f"timeout on {fmt}")

        with patch.object(fetcher, '_fetch_subtitle_with_format', side_effect=mock_format):
            with tempfile.TemporaryDirectory() as td:
                with pytest.raises(CaptionFormatExhaustedError) as exc_info:
                    fetcher._fetch_subtitle_formats(
                        "https://www.youtube.com/watch?v=test",
                        "test",
                        Path(td),
                        "en",
                        False,
                    )

                assert exc_info.value.formats_tried == ["json3", "vtt", "srt"]


# ============================================================
# AC4: Each format tried exactly once then CaptionFormatExhaustedError
# ============================================================

class TestEachFormatTriedOnce:
    """AC: Unit test verifies fallback tries each format exactly once then raises."""

    def test_each_format_called_exactly_once(self):
        """Each format in sequence is attempted exactly once."""
        from collections import Counter
        fetcher = CaptionFetcher()
        call_log = []

        def mock_format(video_url, video_id, temp_dir, language,
                        auto_gen, fmt, fallback_level=0):
            call_log.append(fmt)
            raise CaptionFormatUnavailableError(video_id, fmt, "not available")

        with patch.object(fetcher, '_fetch_subtitle_with_format', side_effect=mock_format):
            with tempfile.TemporaryDirectory() as td:
                with pytest.raises(CaptionFormatExhaustedError):
                    fetcher._fetch_subtitle_formats(
                        "https://www.youtube.com/watch?v=test",
                        "test",
                        Path(td),
                        "en",
                        False,
                    )

        counts = Counter(call_log)
        for fmt, count in counts.items():
            assert count == 1, f"Format {fmt} was tried {count} times, expected 1"
        assert call_log == ['json3', 'vtt', 'srt']


# ============================================================
# AC5: No subprocess spawned for a format that already failed
# ============================================================

class TestNoSubprocessForAlreadyFailedFormat:
    """AC: No subprocess is spawned for a format that already failed in same cycle."""

    def test_failed_format_not_retried_in_same_cycle(self):
        """When _preferred_formats has duplicates, each format is tried only once."""
        fetcher = CaptionFetcher()
        # Force duplicates in preferred list (even though init deduplicates)
        fetcher._preferred_formats = ['json3', 'vtt', 'json3', 'srt', 'vtt']
        subprocess_calls = []

        def mock_format(video_url, video_id, temp_dir, language,
                        auto_gen, fmt, fallback_level=0):
            subprocess_calls.append(fmt)
            raise CaptionFormatUnavailableError(video_id, fmt, "not available")

        with patch.object(fetcher, '_fetch_subtitle_with_format', side_effect=mock_format):
            with tempfile.TemporaryDirectory() as td:
                with pytest.raises(CaptionFormatExhaustedError):
                    fetcher._fetch_subtitle_formats(
                        "https://www.youtube.com/watch?v=test",
                        "test",
                        Path(td),
                        "en",
                        False,
                    )

        # Deduplication in _fetch_subtitle_formats should prevent duplicate calls
        assert subprocess_calls == ['json3', 'vtt', 'srt'], \
            f"Expected ['json3', 'vtt', 'srt'], got {subprocess_calls}"

    def test_successful_early_format_stops_sequence(self):
        """When a format succeeds, no further formats are tried."""
        fetcher = CaptionFetcher()
        call_log = []

        def mock_format(video_url, video_id, temp_dir, language,
                        auto_gen, fmt, fallback_level=0):
            call_log.append(fmt)
            if fmt == 'json3':
                raise CaptionFormatUnavailableError(video_id, fmt, "not available")
            # vtt succeeds
            result = MagicMock()
            result.segments = [MagicMock()]
            return result

        with patch.object(fetcher, '_fetch_subtitle_with_format', side_effect=mock_format):
            with tempfile.TemporaryDirectory() as td:
                result = fetcher._fetch_subtitle_formats(
                    "https://www.youtube.com/watch?v=test",
                    "test",
                    Path(td),
                    "en",
                    False,
                )

        # Should stop after vtt succeeds, srt never attempted
        assert call_log == ['json3', 'vtt'], \
            f"Expected ['json3', 'vtt'], got {call_log}"
        assert result is not None
