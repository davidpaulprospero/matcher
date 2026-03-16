"""
Tests for US-59-002: Pre-flight list-subs check in _fetch_subtitle().

Verifies that _fetch_subtitle() calls list_available_languages() before
iterating formats, eliminating wasted subprocess calls for videos with
no captions.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
import tempfile

from src.caption_fetcher import (
    CaptionFetcher,
    CaptionResult,
    CaptionSegment,
    CaptionUnavailableError,
    CaptionFetchError,
    AvailableLanguage,
)


@pytest.mark.fast
class TestPreflightListSubsCheck:
    """Tests for pre-flight list-subs check wired into _fetch_subtitle (US-59-002)."""

    def _make_fetcher(self):
        """Create a CaptionFetcher with no config."""
        return CaptionFetcher()

    def test_no_captions_triggers_single_ytdlp_call(self):
        """Video with no captions triggers exactly 1 yt-dlp invocation (list-subs).

        AC: Unit test verifies that a video with no captions triggers exactly
        1 yt-dlp invocation (list-subs) instead of 3+ format attempts.
        """
        fetcher = self._make_fetcher()

        with patch('src.caption_fetcher.subprocess.run') as mock_run:
            # list-subs returns no languages
            mock_run.return_value = Mock(
                stdout="",
                stderr="",
                returncode=0,
            )

            with tempfile.TemporaryDirectory() as td:
                with pytest.raises(CaptionUnavailableError) as exc_info:
                    fetcher._fetch_subtitle(
                        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                        "dQw4w9WgXcQ",
                        Path(td),
                        "en",
                        False,
                    )

                assert "Pre-flight check" in exc_info.value.reason
                assert "no captions available" in exc_info.value.reason

            # Exactly 1 subprocess call: the list-subs pre-flight check
            assert mock_run.call_count == 1
            cmd_args = mock_run.call_args[0][0]
            assert '--list-subs' in cmd_args

    def test_preflight_returns_empty_raises_immediately(self):
        """AC: If list_available_languages() returns empty list,
        _fetch_subtitle() raises CaptionUnavailableError immediately
        without trying any format.
        """
        fetcher = self._make_fetcher()

        with patch.object(fetcher, 'list_available_languages', return_value=[]):
            with patch.object(fetcher, '_fetch_subtitle_with_format') as mock_format:
                with tempfile.TemporaryDirectory() as td:
                    with pytest.raises(CaptionUnavailableError) as exc_info:
                        fetcher._fetch_subtitle(
                            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                            "dQw4w9WgXcQ",
                            Path(td),
                            "en",
                            False,
                        )

                    assert "no captions available" in exc_info.value.reason

                # _fetch_subtitle_with_format should NOT have been called
                mock_format.assert_not_called()

    def test_preflight_language_not_available(self):
        """AC: If list_available_languages() returns languages but none match
        the requested language, _fetch_subtitle() raises CaptionUnavailableError
        with descriptive message listing available languages.
        """
        fetcher = self._make_fetcher()

        available = [
            AvailableLanguage(code='es', name='Spanish', is_auto_generated=False),
            AvailableLanguage(code='fr', name='French', is_auto_generated=True),
        ]

        with patch.object(fetcher, 'list_available_languages', return_value=available):
            with patch.object(fetcher, '_fetch_subtitle_with_format') as mock_format:
                with tempfile.TemporaryDirectory() as td:
                    with pytest.raises(CaptionUnavailableError) as exc_info:
                        fetcher._fetch_subtitle(
                            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                            "dQw4w9WgXcQ",
                            Path(td),
                            "en",
                            False,
                        )

                    assert "language 'en' not available" in exc_info.value.reason
                    assert "es" in exc_info.value.reason
                    assert "fr" in exc_info.value.reason

                mock_format.assert_not_called()

    def test_preflight_language_available_proceeds_to_format_loop(self):
        """When requested language is available, format loop proceeds normally."""
        fetcher = self._make_fetcher()

        available = [
            AvailableLanguage(code='en', name='English', is_auto_generated=False),
            AvailableLanguage(code='es', name='Spanish', is_auto_generated=True),
        ]

        expected_result = CaptionResult(
            video_id="dQw4w9WgXcQ",
            language="en",
            is_auto_generated=False,
            segments=[
                CaptionSegment(index=0, start_time=0.0, end_time=1.0, text="Hello")
            ],
        )

        with patch.object(fetcher, 'list_available_languages', return_value=available):
            with patch.object(
                fetcher, '_fetch_subtitle_with_format', return_value=expected_result
            ) as mock_format:
                with tempfile.TemporaryDirectory() as td:
                    result = fetcher._fetch_subtitle(
                        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                        "dQw4w9WgXcQ",
                        Path(td),
                        "en",
                        False,
                    )

                assert result is expected_result
                # Format loop should have been called (at least once)
                assert mock_format.call_count >= 1

    def test_preflight_cached_across_calls(self):
        """AC: Pre-flight check is skipped when caption cache already has
        a valid entry for the video (avoids redundant yt-dlp call).

        The _preflight_lang_cache stores results per video_id so that
        repeated calls for the same video (different lang/auto combos)
        only trigger one list-subs call.
        """
        fetcher = self._make_fetcher()

        available = [
            AvailableLanguage(code='en', name='English', is_auto_generated=False),
        ]

        with patch.object(
            fetcher, 'list_available_languages', return_value=available
        ) as mock_list:
            with patch.object(
                fetcher,
                '_fetch_subtitle_with_format',
                return_value=CaptionResult(
                    video_id="dQw4w9WgXcQ",
                    language="en",
                    is_auto_generated=False,
                    segments=[
                        CaptionSegment(
                            index=0, start_time=0.0, end_time=1.0, text="Hi"
                        )
                    ],
                ),
            ):
                with tempfile.TemporaryDirectory() as td:
                    td_path = Path(td)
                    # First call - should invoke list_available_languages
                    fetcher._fetch_subtitle(
                        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                        "dQw4w9WgXcQ",
                        td_path,
                        "en",
                        False,
                    )
                    # Second call same video - should use cache
                    fetcher._fetch_subtitle(
                        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                        "dQw4w9WgXcQ",
                        td_path,
                        "en",
                        True,
                    )

                # list_available_languages called exactly once despite 2 _fetch_subtitle calls
                assert mock_list.call_count == 1

    def test_preflight_failure_falls_through_to_format_loop(self):
        """When list_available_languages() raises CaptionFetchError (network error),
        _fetch_subtitle should fall through to the format iteration loop.
        """
        fetcher = self._make_fetcher()

        expected_result = CaptionResult(
            video_id="dQw4w9WgXcQ",
            language="en",
            is_auto_generated=False,
            segments=[
                CaptionSegment(index=0, start_time=0.0, end_time=1.0, text="Hello")
            ],
        )

        with patch.object(
            fetcher,
            'list_available_languages',
            side_effect=CaptionFetchError("dQw4w9WgXcQ", "Timeout"),
        ):
            with patch.object(
                fetcher, '_fetch_subtitle_with_format', return_value=expected_result
            ) as mock_format:
                with tempfile.TemporaryDirectory() as td:
                    result = fetcher._fetch_subtitle(
                        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                        "dQw4w9WgXcQ",
                        Path(td),
                        "en",
                        False,
                    )

                assert result is expected_result
                # Format loop should have been called despite pre-flight failure
                assert mock_format.call_count >= 1

    def test_preflight_different_videos_each_get_own_check(self):
        """Different video IDs each get their own list-subs check."""
        fetcher = self._make_fetcher()

        available = [
            AvailableLanguage(code='en', name='English', is_auto_generated=False),
        ]

        with patch.object(
            fetcher, 'list_available_languages', return_value=available
        ) as mock_list:
            with patch.object(
                fetcher,
                '_fetch_subtitle_with_format',
                return_value=CaptionResult(
                    video_id="vid1",
                    language="en",
                    is_auto_generated=False,
                    segments=[
                        CaptionSegment(
                            index=0, start_time=0.0, end_time=1.0, text="Hi"
                        )
                    ],
                ),
            ):
                with tempfile.TemporaryDirectory() as td:
                    td_path = Path(td)
                    fetcher._fetch_subtitle(
                        "https://www.youtube.com/watch?v=abc12345678",
                        "abc12345678",
                        td_path,
                        "en",
                        False,
                    )
                    fetcher._fetch_subtitle(
                        "https://www.youtube.com/watch?v=def12345678",
                        "def12345678",
                        td_path,
                        "en",
                        False,
                    )

                # Each video gets its own list-subs call
                assert mock_list.call_count == 2


@pytest.mark.fast
class TestListAvailableSubtitles:
    """Tests for list_available_subtitles() structured result (US-60-002)."""

    def _make_fetcher(self):
        """Create a CaptionFetcher with no config."""
        return CaptionFetcher()

    def test_returns_structured_result_with_all_fields(self):
        """AC: list_available_subtitles returns structured result with
        available_manual_languages, available_auto_languages, has_any_subtitles.
        """
        from src.caption_fetcher import SubtitleAvailabilityResult
        fetcher = self._make_fetcher()

        available = [
            AvailableLanguage(code='en', name='English', is_auto_generated=False),
            AvailableLanguage(code='es', name='Spanish', is_auto_generated=False),
            AvailableLanguage(code='en', name='English (auto-generated)', is_auto_generated=True),
            AvailableLanguage(code='fr', name='French (auto-generated)', is_auto_generated=True),
        ]

        with patch.object(fetcher, 'list_available_languages', return_value=available):
            result = fetcher.list_available_subtitles("dQw4w9WgXcQ")

            assert isinstance(result, SubtitleAvailabilityResult)
            assert result.video_id == "dQw4w9WgXcQ"
            assert result.has_any_subtitles is True
            assert 'en' in result.available_manual_languages
            assert 'es' in result.available_manual_languages
            assert 'en' in result.available_auto_languages
            assert 'fr' in result.available_auto_languages
            assert len(result.all_languages) == 4

    def test_has_any_subtitles_false_when_empty(self):
        """AC: has_any_subtitles is False when no subtitles exist."""
        fetcher = self._make_fetcher()

        with patch.object(fetcher, 'list_available_languages', return_value=[]):
            result = fetcher.list_available_subtitles("noSubsVideo")

            assert result.has_any_subtitles is False
            assert result.available_manual_languages == []
            assert result.available_auto_languages == []

    def test_has_language_method_checks_correctly(self):
        """Test the has_language helper method."""
        from src.caption_fetcher import SubtitleAvailabilityResult
        fetcher = self._make_fetcher()

        available = [
            AvailableLanguage(code='en', name='English', is_auto_generated=False),
            AvailableLanguage(code='de', name='German (auto)', is_auto_generated=True),
        ]

        with patch.object(fetcher, 'list_available_languages', return_value=available):
            result = fetcher.list_available_subtitles("testVideo")

            # Test has_language method
            assert result.has_language('en') is True
            assert result.has_language('EN') is True  # Case insensitive
            assert result.has_language('en', manual_only=True) is True
            assert result.has_language('de') is True
            assert result.has_language('de', manual_only=True) is False  # Only auto
            assert result.has_language('fr') is False  # Not available

    def test_to_dict_serialization(self):
        """Test that result can be serialized to dict for caching."""
        from src.caption_fetcher import SubtitleAvailabilityResult
        fetcher = self._make_fetcher()

        available = [
            AvailableLanguage(code='en', name='English', is_auto_generated=False),
        ]

        with patch.object(fetcher, 'list_available_languages', return_value=available):
            result = fetcher.list_available_subtitles("testVideo")
            serialized = result.to_dict()

            assert serialized['video_id'] == "testVideo"
            assert serialized['has_any_subtitles'] is True
            assert 'en' in serialized['available_manual_languages']

    def test_preflight_enables_fail_fast_no_format_attempts(self):
        """AC: Unit test verifies pre-flight check prevents unnecessary format attempts.

        When list_available_subtitles returns empty, no format attempts should occur.
        """
        fetcher = self._make_fetcher()

        with patch('src.caption_fetcher.subprocess.run') as mock_run:
            # list-subs returns empty (no captions)
            mock_run.return_value = Mock(
                stdout="",
                stderr="",
                returncode=0,
            )

            result = fetcher.list_available_subtitles("noSubsVideo")
            assert result.has_any_subtitles is False

            # Only 1 subprocess call - the list-subs check
            assert mock_run.call_count == 1
            cmd_args = mock_run.call_args[0][0]
            assert '--list-subs' in cmd_args
            # No format download attempts should have been made
            assert '--write-sub' not in cmd_args
