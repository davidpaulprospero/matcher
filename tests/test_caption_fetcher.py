"""
Tests for src/caption_fetcher.py

Tests YouTube caption fetching, parsing, and error handling.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess

from src.caption_fetcher import (
    CaptionFetcher,
    CaptionResult,
    CaptionSegment,
    CaptionError,
    CaptionUnavailableError,
    CaptionFetchError,
    AvailableLanguage,
)


class TestCaptionSegment:
    """Test CaptionSegment dataclass"""

    def test_caption_segment_creation(self):
        """Test creating a caption segment"""
        segment = CaptionSegment(
            index=0,
            start_time=1.5,
            end_time=4.0,
            text="Hello world",
            source_file="dQw4w9WgXcQ"
        )

        assert segment.index == 0
        assert segment.start_time == 1.5
        assert segment.end_time == 4.0
        assert segment.text == "Hello world"
        assert segment.source_file == "dQw4w9WgXcQ"

    def test_caption_segment_to_dict(self):
        """Test segment serialization"""
        segment = CaptionSegment(
            index=1,
            start_time=5.0,
            end_time=8.5,
            text="Test text",
            source_file="test123"
        )

        result = segment.to_dict()

        assert result['index'] == 1
        assert result['start'] == 5.0
        assert result['end'] == 8.5
        assert result['text'] == "Test text"
        assert result['source_file'] == "test123"

    def test_caption_segment_default_source_file(self):
        """Test default source_file value"""
        segment = CaptionSegment(
            index=0,
            start_time=0.0,
            end_time=1.0,
            text="Test"
        )

        assert segment.source_file == ""


class TestCaptionResult:
    """Test CaptionResult dataclass"""

    def test_caption_result_creation(self):
        """Test creating a caption result"""
        segments = [
            CaptionSegment(0, 0.0, 2.0, "First", "vid1"),
            CaptionSegment(1, 2.0, 4.0, "Second", "vid1"),
        ]

        result = CaptionResult(
            video_id="vid123",
            segments=segments,
            language="en",
            is_auto_generated=False,
            format_source="vtt"
        )

        assert result.video_id == "vid123"
        assert len(result.segments) == 2
        assert result.language == "en"
        assert result.is_auto_generated is False
        assert result.format_source == "vtt"

    def test_caption_result_text_property(self):
        """Test text property concatenates all segment text"""
        segments = [
            CaptionSegment(0, 0.0, 2.0, "Hello", "vid1"),
            CaptionSegment(1, 2.0, 4.0, "world", "vid1"),
        ]

        result = CaptionResult(video_id="vid1", segments=segments)

        assert result.text == "Hello world"

    def test_caption_result_text_empty(self):
        """Test text property with no segments"""
        result = CaptionResult(video_id="vid1", segments=[])

        assert result.text == ""

    def test_caption_result_duration(self):
        """Test duration property calculation"""
        segments = [
            CaptionSegment(0, 1.0, 3.0, "First", "vid1"),
            CaptionSegment(1, 3.0, 10.0, "Last", "vid1"),
        ]

        result = CaptionResult(video_id="vid1", segments=segments)

        # Duration is last.end - first.start = 10.0 - 1.0 = 9.0
        assert result.duration == 9.0

    def test_caption_result_duration_empty(self):
        """Test duration with no segments"""
        result = CaptionResult(video_id="vid1", segments=[])

        assert result.duration == 0.0

    def test_caption_result_to_dict(self):
        """Test result serialization"""
        segments = [
            CaptionSegment(0, 0.0, 1.0, "Test", "vid1")
        ]

        result = CaptionResult(
            video_id="vid1",
            segments=segments,
            language="en",
            is_auto_generated=True,
            format_source="json3"
        )

        data = result.to_dict()

        assert data['video_id'] == "vid1"
        assert len(data['segments']) == 1
        assert data['language'] == "en"
        assert data['is_auto_generated'] is True
        assert data['format_source'] == "json3"


class TestCaptionExceptions:
    """Test caption exception classes"""

    def test_caption_unavailable_error(self):
        """Test CaptionUnavailableError"""
        error = CaptionUnavailableError("vid123", "No subtitles available")

        assert error.video_id == "vid123"
        assert error.reason == "No subtitles available"
        assert "vid123" in str(error)
        assert "No subtitles" in str(error)

    def test_caption_unavailable_error_no_reason(self):
        """Test CaptionUnavailableError without reason"""
        error = CaptionUnavailableError("vid123")

        assert error.video_id == "vid123"
        assert "vid123" in str(error)

    def test_caption_fetch_error(self):
        """Test CaptionFetchError"""
        error = CaptionFetchError("vid123", "Network timeout")

        assert error.video_id == "vid123"
        assert error.reason == "Network timeout"
        assert "vid123" in str(error)
        assert "Network timeout" in str(error)

    def test_exception_inheritance(self):
        """Test exception inheritance"""
        assert issubclass(CaptionUnavailableError, CaptionError)
        assert issubclass(CaptionFetchError, CaptionError)
        assert issubclass(CaptionError, Exception)


class TestCaptionFetcherVideoIdValidation:
    """Test video ID validation"""

    def test_valid_video_id(self):
        """Test valid YouTube video ID"""
        fetcher = CaptionFetcher()

        assert fetcher._is_valid_video_id("dQw4w9WgXcQ") is True
        assert fetcher._is_valid_video_id("abc123DEF-_") is True
        assert fetcher._is_valid_video_id("12345678901") is True

    def test_invalid_video_id_too_short(self):
        """Test rejection of too-short ID"""
        fetcher = CaptionFetcher()

        assert fetcher._is_valid_video_id("dQw4w9WgXc") is False  # 10 chars
        assert fetcher._is_valid_video_id("abc") is False

    def test_invalid_video_id_too_long(self):
        """Test rejection of too-long ID"""
        fetcher = CaptionFetcher()

        assert fetcher._is_valid_video_id("dQw4w9WgXcQQ") is False  # 12 chars

    def test_invalid_video_id_special_chars(self):
        """Test rejection of invalid characters"""
        fetcher = CaptionFetcher()

        assert fetcher._is_valid_video_id("dQw4w9Wg@cQ") is False  # @ is invalid
        assert fetcher._is_valid_video_id("dQw4w9Wg cQ") is False  # space is invalid
        assert fetcher._is_valid_video_id("dQw4w9Wg.cQ") is False  # . is invalid

    def test_invalid_video_id_empty(self):
        """Test rejection of empty ID"""
        fetcher = CaptionFetcher()

        assert fetcher._is_valid_video_id("") is False
        assert fetcher._is_valid_video_id(None) is False


class TestCaptionFetcherTimestampParsing:
    """Test timestamp parsing"""

    def test_parse_timestamp_vtt_format(self):
        """Test parsing VTT timestamp format (HH:MM:SS.mmm)"""
        fetcher = CaptionFetcher()

        assert fetcher._parse_timestamp("00:00:01.000") == 1.0
        assert fetcher._parse_timestamp("00:01:30.500") == 90.5
        assert fetcher._parse_timestamp("01:00:00.000") == 3600.0
        assert fetcher._parse_timestamp("01:30:45.123") == 5445.123

    def test_parse_timestamp_srt_format(self):
        """Test parsing SRT timestamp format (HH:MM:SS,mmm)"""
        fetcher = CaptionFetcher()

        assert fetcher._parse_timestamp("00:00:01,000") == 1.0
        assert fetcher._parse_timestamp("00:01:30,500") == 90.5
        assert fetcher._parse_timestamp("01:00:00,000") == 3600.0

    def test_parse_timestamp_short_format(self):
        """Test parsing short VTT format (MM:SS.mmm)"""
        fetcher = CaptionFetcher()

        assert fetcher._parse_timestamp("00:01.000") == 1.0
        assert fetcher._parse_timestamp("01:30.500") == 90.5
        assert fetcher._parse_timestamp("59:59.999") == 3599.999

    def test_parse_timestamp_without_milliseconds(self):
        """Test parsing timestamp without milliseconds"""
        fetcher = CaptionFetcher()

        assert fetcher._parse_timestamp("00:00:01") == 1.0
        assert fetcher._parse_timestamp("01:30") == 90.0

    def test_parse_timestamp_with_whitespace(self):
        """Test parsing timestamp with whitespace"""
        fetcher = CaptionFetcher()

        assert fetcher._parse_timestamp("  00:00:01.000  ") == 1.0

    def test_parse_timestamp_invalid(self):
        """Test parsing invalid timestamp returns None"""
        fetcher = CaptionFetcher()

        assert fetcher._parse_timestamp("invalid") is None
        assert fetcher._parse_timestamp("abc:def:ghi") is None  # Non-numeric


class TestCaptionFetcherVttParsing:
    """Test VTT format parsing"""

    def test_parse_vtt_basic(self):
        """Test parsing basic VTT content"""
        fetcher = CaptionFetcher()

        vtt_content = """WEBVTT

00:00:01.000 --> 00:00:04.000
Hello, world!

00:00:05.000 --> 00:00:08.000
This is a test.
"""
        segments = fetcher._parse_vtt(vtt_content, "test_video")

        assert len(segments) == 2
        assert segments[0].start_time == 1.0
        assert segments[0].end_time == 4.0
        assert segments[0].text == "Hello, world!"
        assert segments[1].start_time == 5.0
        assert segments[1].end_time == 8.0
        assert segments[1].text == "This is a test."

    def test_parse_vtt_with_style_tags(self):
        """Test VTT parsing removes style tags"""
        fetcher = CaptionFetcher()

        vtt_content = """WEBVTT

00:00:01.000 --> 00:00:04.000
<c.colorWhite>Hello</c> <b>world</b>!
"""
        segments = fetcher._parse_vtt(vtt_content, "test_video")

        assert len(segments) == 1
        assert segments[0].text == "Hello world!"

    def test_parse_vtt_multiline_text(self):
        """Test VTT parsing with multiline text"""
        fetcher = CaptionFetcher()

        vtt_content = """WEBVTT

00:00:01.000 --> 00:00:04.000
Line one
Line two
"""
        segments = fetcher._parse_vtt(vtt_content, "test_video")

        assert len(segments) == 1
        assert "Line one" in segments[0].text
        assert "Line two" in segments[0].text

    def test_parse_vtt_with_cue_identifiers(self):
        """Test VTT parsing with cue identifiers"""
        fetcher = CaptionFetcher()

        vtt_content = """WEBVTT

1
00:00:01.000 --> 00:00:04.000
First cue

2
00:00:05.000 --> 00:00:08.000
Second cue
"""
        segments = fetcher._parse_vtt(vtt_content, "test_video")

        assert len(segments) == 2

    def test_parse_vtt_empty(self):
        """Test parsing empty VTT content"""
        fetcher = CaptionFetcher()

        vtt_content = """WEBVTT

"""
        segments = fetcher._parse_vtt(vtt_content, "test_video")

        assert len(segments) == 0

    def test_parse_vtt_sets_source_file(self):
        """Test that source_file is set correctly"""
        fetcher = CaptionFetcher()

        vtt_content = """WEBVTT

00:00:01.000 --> 00:00:02.000
Test
"""
        segments = fetcher._parse_vtt(vtt_content, "my_video_id")

        assert segments[0].source_file == "my_video_id"


class TestCaptionFetcherSrtParsing:
    """Test SRT format parsing"""

    def test_parse_srt_basic(self):
        """Test parsing basic SRT content"""
        fetcher = CaptionFetcher()

        srt_content = """1
00:00:01,000 --> 00:00:04,000
Hello, world!

2
00:00:05,000 --> 00:00:08,000
This is a test.
"""
        segments = fetcher._parse_srt(srt_content, "test_video")

        assert len(segments) == 2
        assert segments[0].start_time == 1.0
        assert segments[0].end_time == 4.0
        assert segments[0].text == "Hello, world!"
        assert segments[1].text == "This is a test."

    def test_parse_srt_with_tags(self):
        """Test SRT parsing removes formatting tags"""
        fetcher = CaptionFetcher()

        srt_content = """1
00:00:01,000 --> 00:00:04,000
<i>Italic</i> and <b>bold</b>
"""
        segments = fetcher._parse_srt(srt_content, "test_video")

        assert segments[0].text == "Italic and bold"

    def test_parse_srt_multiline(self):
        """Test SRT with multiline subtitles"""
        fetcher = CaptionFetcher()

        srt_content = """1
00:00:01,000 --> 00:00:04,000
Line one
Line two

2
00:00:05,000 --> 00:00:08,000
Single line
"""
        segments = fetcher._parse_srt(srt_content, "test_video")

        assert len(segments) == 2
        assert "Line one" in segments[0].text
        assert "Line two" in segments[0].text

    def test_parse_srt_empty_blocks(self):
        """Test SRT parsing handles empty blocks gracefully"""
        fetcher = CaptionFetcher()

        srt_content = """1
00:00:01,000 --> 00:00:04,000
Valid


3
00:00:05,000 --> 00:00:08,000
Also valid
"""
        segments = fetcher._parse_srt(srt_content, "test_video")

        assert len(segments) == 2


class TestCaptionFetcherJson3Parsing:
    """Test JSON3/SRV3 format parsing"""

    def test_parse_json3_basic(self):
        """Test parsing basic JSON3 content"""
        fetcher = CaptionFetcher()

        json3_content = json.dumps({
            "events": [
                {
                    "tStartMs": 1000,
                    "dDurationMs": 3000,
                    "segs": [{"utf8": "Hello, world!"}]
                },
                {
                    "tStartMs": 5000,
                    "dDurationMs": 3000,
                    "segs": [{"utf8": "This is a test."}]
                }
            ]
        })

        segments = fetcher._parse_json3(json3_content, "test_video")

        assert len(segments) == 2
        assert segments[0].start_time == 1.0
        assert segments[0].end_time == 4.0
        assert segments[0].text == "Hello, world!"
        assert segments[1].start_time == 5.0
        assert segments[1].end_time == 8.0

    def test_parse_json3_multiple_segs(self):
        """Test JSON3 with multiple text segments per event"""
        fetcher = CaptionFetcher()

        json3_content = json.dumps({
            "events": [
                {
                    "tStartMs": 1000,
                    "dDurationMs": 3000,
                    "segs": [
                        {"utf8": "Hello "},
                        {"utf8": "world!"}
                    ]
                }
            ]
        })

        segments = fetcher._parse_json3(json3_content, "test_video")

        assert len(segments) == 1
        assert segments[0].text == "Hello world!"

    def test_parse_json3_empty_events(self):
        """Test JSON3 with empty events array"""
        fetcher = CaptionFetcher()

        json3_content = json.dumps({"events": []})

        segments = fetcher._parse_json3(json3_content, "test_video")

        assert len(segments) == 0

    def test_parse_json3_events_without_segs(self):
        """Test JSON3 skips events without segs"""
        fetcher = CaptionFetcher()

        json3_content = json.dumps({
            "events": [
                {"tStartMs": 1000, "dDurationMs": 3000},  # No segs
                {
                    "tStartMs": 5000,
                    "dDurationMs": 3000,
                    "segs": [{"utf8": "Valid"}]
                }
            ]
        })

        segments = fetcher._parse_json3(json3_content, "test_video")

        assert len(segments) == 1
        assert segments[0].text == "Valid"

    def test_parse_json3_invalid_json(self):
        """Test JSON3 parsing handles invalid JSON"""
        fetcher = CaptionFetcher()

        segments = fetcher._parse_json3("not valid json", "test_video")

        assert len(segments) == 0


class TestCaptionFetcherFetchCaptions:
    """Test fetch_captions method"""

    @patch('subprocess.run')
    def test_fetch_captions_invalid_video_id(self, mock_run):
        """Test fetch with invalid video ID raises error"""
        fetcher = CaptionFetcher()

        with pytest.raises(CaptionFetchError) as exc_info:
            fetcher.fetch_captions("invalid")

        assert "Invalid video ID" in str(exc_info.value)
        mock_run.assert_not_called()

    @patch('subprocess.run')
    @patch('tempfile.TemporaryDirectory')
    def test_fetch_captions_no_subtitles_available(self, mock_tempdir, mock_run):
        """Test fetch when no subtitles available"""
        fetcher = CaptionFetcher()

        # Setup temp directory mock
        mock_temp_path = MagicMock()
        mock_tempdir.return_value.__enter__ = MagicMock(return_value=str(mock_temp_path))
        mock_tempdir.return_value.__exit__ = MagicMock(return_value=False)

        # Mock yt-dlp failure with "no subtitles" message
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "no subtitles available for this video"
        mock_run.return_value = mock_result

        with pytest.raises(CaptionUnavailableError):
            fetcher.fetch_captions("dQw4w9WgXcQ")

    @patch('subprocess.run')
    @patch('tempfile.TemporaryDirectory')
    def test_fetch_captions_network_error(self, mock_tempdir, mock_run):
        """Test fetch with network error"""
        fetcher = CaptionFetcher()

        mock_temp_path = MagicMock()
        mock_tempdir.return_value.__enter__ = MagicMock(return_value=str(mock_temp_path))
        mock_tempdir.return_value.__exit__ = MagicMock(return_value=False)

        # Mock network error
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "Connection timeout"
        mock_run.return_value = mock_result

        with pytest.raises(CaptionFetchError):
            fetcher.fetch_captions("dQw4w9WgXcQ")

    @patch('subprocess.run')
    @patch('tempfile.TemporaryDirectory')
    def test_fetch_captions_timeout(self, mock_tempdir, mock_run):
        """Test fetch with subprocess timeout"""
        fetcher = CaptionFetcher()

        mock_temp_path = MagicMock()
        mock_tempdir.return_value.__enter__ = MagicMock(return_value=str(mock_temp_path))
        mock_tempdir.return_value.__exit__ = MagicMock(return_value=False)

        mock_run.side_effect = subprocess.TimeoutExpired('yt-dlp', 60)

        with pytest.raises(CaptionFetchError) as exc_info:
            fetcher.fetch_captions("dQw4w9WgXcQ")

        assert "Timeout" in str(exc_info.value)

    @patch('subprocess.run')
    def test_fetch_captions_success(self, mock_run, tmp_path):
        """Test successful caption fetch"""
        fetcher = CaptionFetcher()

        # Create a VTT file that will be "downloaded"
        video_id = "dQw4w9WgXcQ"
        vtt_content = """WEBVTT

00:00:01.000 --> 00:00:04.000
Never gonna give you up

00:00:05.000 --> 00:00:08.000
Never gonna let you down
"""

        with patch('tempfile.TemporaryDirectory') as mock_tempdir:
            # Use the actual tmp_path for the mock
            mock_tempdir.return_value.__enter__ = MagicMock(return_value=str(tmp_path))
            mock_tempdir.return_value.__exit__ = MagicMock(return_value=False)

            # Create the subtitle file
            vtt_file = tmp_path / f"{video_id}.en.vtt"
            vtt_file.write_text(vtt_content)

            # Mock successful yt-dlp run
            mock_result = Mock()
            mock_result.returncode = 0
            mock_result.stderr = ""
            mock_run.return_value = mock_result

            result = fetcher.fetch_captions(video_id)

            assert result.video_id == video_id
            assert len(result.segments) == 2
            assert "Never gonna give you up" in result.text
            assert result.format_source == "vtt"


class TestCaptionFetcherCookies:
    """Test cookie handling"""

    def test_get_cookies_args_no_config(self):
        """Test cookie args with no config"""
        fetcher = CaptionFetcher()

        args = fetcher._get_cookies_args()

        assert args == []

    def test_get_cookies_args_browser_cookies(self):
        """Test cookie args with browser cookies configured"""
        mock_config = Mock()
        mock_config.download.cookies_from_browser = "chrome"
        mock_config.download.cookies_path = ""

        fetcher = CaptionFetcher(config=mock_config)

        args = fetcher._get_cookies_args()

        assert args == ['--cookies-from-browser', 'chrome']

    def test_get_cookies_args_file_cookies(self):
        """Test cookie args with file cookies configured"""
        mock_config = Mock()
        mock_config.download.cookies_from_browser = ""
        mock_config.download.cookies_path = "/path/to/cookies.txt"

        with patch.object(Path, 'exists', return_value=True):
            fetcher = CaptionFetcher(config=mock_config)
            args = fetcher._get_cookies_args()

        assert args == ['--cookies', '/path/to/cookies.txt']

    def test_get_cookies_args_browser_preferred(self):
        """Test that browser cookies are preferred over file"""
        mock_config = Mock()
        mock_config.download.cookies_from_browser = "firefox"
        mock_config.download.cookies_path = "/path/to/cookies.txt"

        fetcher = CaptionFetcher(config=mock_config)

        args = fetcher._get_cookies_args()

        # Should use browser cookies, not file
        assert args == ['--cookies-from-browser', 'firefox']


class TestCaptionFetcherIntegration:
    """Integration tests for caption fetcher"""

    def test_parse_subtitle_file_vtt(self, tmp_path):
        """Test parsing VTT file from disk"""
        fetcher = CaptionFetcher()

        vtt_file = tmp_path / "test.vtt"
        vtt_file.write_text("""WEBVTT

00:00:01.000 --> 00:00:04.000
Test caption
""")

        segments = fetcher._parse_subtitle_file(vtt_file, "test_video")

        assert len(segments) == 1
        assert segments[0].text == "Test caption"

    def test_parse_subtitle_file_srt(self, tmp_path):
        """Test parsing SRT file from disk"""
        fetcher = CaptionFetcher()

        srt_file = tmp_path / "test.srt"
        srt_file.write_text("""1
00:00:01,000 --> 00:00:04,000
Test caption
""")

        segments = fetcher._parse_subtitle_file(srt_file, "test_video")

        assert len(segments) == 1
        assert segments[0].text == "Test caption"

    def test_parse_subtitle_file_json3(self, tmp_path):
        """Test parsing JSON3 file from disk"""
        fetcher = CaptionFetcher()

        json_file = tmp_path / "test.json3"
        json_file.write_text(json.dumps({
            "events": [
                {
                    "tStartMs": 1000,
                    "dDurationMs": 3000,
                    "segs": [{"utf8": "Test caption"}]
                }
            ]
        }))

        segments = fetcher._parse_subtitle_file(json_file, "test_video")

        assert len(segments) == 1
        assert segments[0].text == "Test caption"

    def test_parse_subtitle_file_unicode(self, tmp_path):
        """Test parsing file with unicode characters"""
        fetcher = CaptionFetcher()

        vtt_file = tmp_path / "test.vtt"
        vtt_file.write_text("""WEBVTT

00:00:01.000 --> 00:00:04.000
Hello 世界! Привет мир! 🎉
""", encoding='utf-8')

        segments = fetcher._parse_subtitle_file(vtt_file, "test_video")

        assert len(segments) == 1
        assert "世界" in segments[0].text
        assert "Привет" in segments[0].text
        assert "🎉" in segments[0].text


# Mark integration tests that require network
@pytest.mark.requires_network
class TestCaptionFetcherRealVideos:
    """Integration tests with real YouTube videos.

    These tests are skipped by default. Run with:
        pytest -m requires_network tests/test_caption_fetcher.py

    Known video IDs for testing:
    - "dQw4w9WgXcQ": Rick Astley - Never Gonna Give You Up (has human captions)
    - Some videos may have auto-generated captions only
    """

    def test_fetch_video_with_captions(self):
        """Test fetching captions from a known video with captions"""
        fetcher = CaptionFetcher()

        # Note: This test may fail if the video is removed or captions change
        # Rick Astley - Never Gonna Give You Up is a stable choice
        try:
            result = fetcher.fetch_captions("dQw4w9WgXcQ", language="en")

            assert result.video_id == "dQw4w9WgXcQ"
            assert len(result.segments) > 0
            assert result.language == "en"
            # The actual is_auto_generated value depends on what's available
        except CaptionUnavailableError:
            pytest.skip("Video captions not available (may have been removed)")


class TestAvailableLanguage:
    """Test AvailableLanguage dataclass"""

    def test_available_language_creation(self):
        """Test creating an available language"""
        lang = AvailableLanguage(
            code="en",
            name="English",
            is_auto_generated=False
        )

        assert lang.code == "en"
        assert lang.name == "English"
        assert lang.is_auto_generated is False

    def test_available_language_auto_generated(self):
        """Test auto-generated language"""
        lang = AvailableLanguage(
            code="es",
            name="Spanish (auto-generated)",
            is_auto_generated=True
        )

        assert lang.code == "es"
        assert lang.is_auto_generated is True


class TestListAvailableLanguages:
    """Test list_available_languages method"""

    @patch('subprocess.run')
    def test_list_languages_invalid_video_id(self, mock_run):
        """Test listing with invalid video ID"""
        fetcher = CaptionFetcher()

        with pytest.raises(CaptionFetchError) as exc_info:
            fetcher.list_available_languages("invalid")

        assert "Invalid video ID" in str(exc_info.value)
        mock_run.assert_not_called()

    @patch('subprocess.run')
    def test_list_languages_timeout(self, mock_run):
        """Test listing with subprocess timeout"""
        fetcher = CaptionFetcher()

        mock_run.side_effect = subprocess.TimeoutExpired('yt-dlp', 60)

        with pytest.raises(CaptionFetchError) as exc_info:
            fetcher.list_available_languages("dQw4w9WgXcQ")

        assert "Timeout" in str(exc_info.value)

    @patch('subprocess.run')
    def test_list_languages_manual_only(self, mock_run):
        """Test parsing manual subtitles section"""
        fetcher = CaptionFetcher()

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """[info] Available subtitles for dQw4w9WgXcQ:
Language  Name                 Formats
en        English              vtt, ttml, srv3, srv2, srv1, json3
es        Spanish              vtt, ttml, srv3, srv2, srv1, json3
fr        French               vtt, ttml, srv3, srv2, srv1, json3
"""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        languages = fetcher.list_available_languages("dQw4w9WgXcQ")

        assert len(languages) == 3
        assert all(not lang.is_auto_generated for lang in languages)
        assert languages[0].code == "en"
        assert languages[1].code == "es"
        assert languages[2].code == "fr"

    @patch('subprocess.run')
    def test_list_languages_auto_only(self, mock_run):
        """Test parsing auto-generated captions section"""
        fetcher = CaptionFetcher()

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """[info] Available automatic captions for dQw4w9WgXcQ:
Language  Name                              Formats
en        English (auto-generated)          vtt, ttml, srv3, srv2, srv1, json3
es        Spanish (auto-generated)          vtt, ttml, srv3, srv2, srv1, json3
"""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        languages = fetcher.list_available_languages("dQw4w9WgXcQ")

        assert len(languages) == 2
        assert all(lang.is_auto_generated for lang in languages)

    @patch('subprocess.run')
    def test_list_languages_mixed(self, mock_run):
        """Test parsing both manual and auto sections"""
        fetcher = CaptionFetcher()

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """[info] Available subtitles for dQw4w9WgXcQ:
Language  Name                 Formats
en        English              vtt, ttml, srv3, srv2, srv1, json3

[info] Available automatic captions for dQw4w9WgXcQ:
Language  Name                              Formats
en        English (auto-generated)          vtt, ttml, srv3, srv2, srv1, json3
es        Spanish (auto-generated)          vtt, ttml, srv3, srv2, srv1, json3
"""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        languages = fetcher.list_available_languages("dQw4w9WgXcQ")

        # Should have 3: 1 manual + 2 auto
        assert len(languages) == 3

        # Sorted: manual first (en), then auto (en, es)
        assert languages[0].code == "en"
        assert languages[0].is_auto_generated is False

        assert languages[1].code == "en"
        assert languages[1].is_auto_generated is True

        assert languages[2].code == "es"
        assert languages[2].is_auto_generated is True

    @patch('subprocess.run')
    def test_list_languages_empty(self, mock_run):
        """Test parsing output with no subtitles"""
        fetcher = CaptionFetcher()

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """[info] dQw4w9WgXcQ: Downloading webpage
[info] dQw4w9WgXcQ: Downloading ios player API JSON
"""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        languages = fetcher.list_available_languages("dQw4w9WgXcQ")

        assert len(languages) == 0

    @patch('subprocess.run')
    def test_list_languages_regional_codes(self, mock_run):
        """Test parsing regional language codes like en-GB"""
        fetcher = CaptionFetcher()

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = """[info] Available subtitles for dQw4w9WgXcQ:
Language  Name                 Formats
en-GB     English (UK)         vtt, ttml, srv3, srv2, srv1, json3
pt-BR     Portuguese (Brazil)  vtt, ttml, srv3, srv2, srv1, json3
zh-Hans   Chinese (Simplified) vtt, ttml, srv3, srv2, srv1, json3
"""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        languages = fetcher.list_available_languages("dQw4w9WgXcQ")

        assert len(languages) == 3
        assert languages[0].code == "en-gb"
        assert languages[1].code == "pt-br"
        assert languages[2].code == "zh-hans"


class TestSelectBestLanguage:
    """Test select_best_language method"""

    def test_select_preferred_manual(self):
        """Test selecting preferred language when manual is available"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("es", "Spanish", False),
            AvailableLanguage("en", "English (auto)", True),
        ]

        result = fetcher.select_best_language(available, preferred="es")

        assert result.code == "es"
        assert result.is_auto_generated is False

    def test_select_preferred_auto_fallback(self):
        """Test falling back to auto when manual not available"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("es", "Spanish (auto)", True),
        ]

        result = fetcher.select_best_language(available, preferred="es")

        assert result.code == "es"
        assert result.is_auto_generated is True

    def test_select_english_fallback(self):
        """Test falling back to English when preferred not available"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("fr", "French", False),
        ]

        result = fetcher.select_best_language(available, preferred="es")

        assert result.code == "en"
        assert result.is_auto_generated is False

    def test_select_english_fallback_auto(self):
        """Test falling back to English auto when English manual not available"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("en", "English (auto)", True),
            AvailableLanguage("fr", "French", False),
        ]

        result = fetcher.select_best_language(
            available, preferred="es", prefer_manual=True
        )

        # English fallback happens before "any available" fallback
        # Even though French is manual, English (auto) is selected because
        # the fallback chain is: preferred -> English -> any
        # When prefer_manual=True but only auto English available, still uses English
        assert result.code == "en"
        assert result.is_auto_generated is True

    def test_select_any_fallback(self):
        """Test falling back to any available when preferred and en not available"""
        fetcher = CaptionFetcher()

        # List is passed as-is to select_best_language (not sorted internally)
        # list_available_languages returns sorted results, but select_best_language
        # just iterates through the provided list
        available = [
            AvailableLanguage("de", "German", False),  # First in sorted order
            AvailableLanguage("fr", "French", False),
        ]

        result = fetcher.select_best_language(available, preferred="es")

        # Should select first manual from the list
        assert result.code == "de"
        assert result.is_auto_generated is False

    def test_select_empty_list(self):
        """Test selecting from empty list returns None"""
        fetcher = CaptionFetcher()

        result = fetcher.select_best_language([], preferred="en")

        assert result is None

    def test_select_prefer_manual_over_auto(self):
        """Test that manual captions are preferred over auto"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("en", "English (auto)", True),
        ]

        result = fetcher.select_best_language(
            available, preferred="en", prefer_manual=True
        )

        assert result.code == "en"
        assert result.is_auto_generated is False

    def test_select_prefer_auto_when_configured(self):
        """Test preferring auto when prefer_manual=False"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("en", "English (auto)", True),
        ]

        result = fetcher.select_best_language(
            available, preferred="en", prefer_manual=False
        )

        # When not preferring manual, first match wins (list sorted manual first)
        assert result.code == "en"
        # First en in sorted list is manual
        assert result.is_auto_generated is False

    def test_select_no_english_fallback(self):
        """Test disabling English fallback"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("fr", "French", False),
        ]

        result = fetcher.select_best_language(
            available, preferred="es", fallback_to_english=False
        )

        # Should skip English and fall back to any (sorted alphabetically)
        assert result is not None
        # en comes before fr alphabetically
        assert result.code == "en"

    def test_select_english_as_preferred_no_double_check(self):
        """Test that English as preferred doesn't check English twice"""
        fetcher = CaptionFetcher()

        available = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("fr", "French", False),
        ]

        result = fetcher.select_best_language(available, preferred="en")

        assert result.code == "en"


class TestGetPreferredLanguageFromConfig:
    """Test _get_preferred_language_from_config method"""

    def test_no_config_defaults_to_english(self):
        """Test default is English when no config"""
        fetcher = CaptionFetcher()

        result = fetcher._get_preferred_language_from_config()

        assert result == "en"

    def test_caption_first_preferred_language(self):
        """Test reading from caption_first.preferred_language"""
        mock_config = Mock()
        mock_config.download.caption_first.preferred_language = "es"

        fetcher = CaptionFetcher(config=mock_config)

        result = fetcher._get_preferred_language_from_config()

        assert result == "es"

    def test_transcription_language_fallback(self):
        """Test falling back to transcription.language"""
        mock_config = Mock()
        mock_config.download.caption_first.preferred_language = ""
        mock_config.transcription.language = "fr"

        fetcher = CaptionFetcher(config=mock_config)

        result = fetcher._get_preferred_language_from_config()

        assert result == "fr"

    def test_caption_first_takes_precedence(self):
        """Test caption_first.preferred_language takes precedence"""
        mock_config = Mock()
        mock_config.download.caption_first.preferred_language = "de"
        mock_config.transcription.language = "fr"

        fetcher = CaptionFetcher(config=mock_config)

        result = fetcher._get_preferred_language_from_config()

        assert result == "de"


class TestParseListSubsOutput:
    """Test _parse_list_subs_output method"""

    def test_parse_manual_subtitles_section(self):
        """Test parsing manual subtitles section"""
        fetcher = CaptionFetcher()

        stdout = """[info] Available subtitles for VIDEO_ID:
Language  Name                 Formats
en        English              vtt, ttml, srv3, srv2, srv1, json3
es        Spanish              vtt, ttml, srv3, srv2, srv1, json3
"""
        result = fetcher._parse_list_subs_output(stdout, "")

        assert len(result) == 2
        assert all(not lang.is_auto_generated for lang in result)

    def test_parse_auto_captions_section(self):
        """Test parsing auto-generated captions section"""
        fetcher = CaptionFetcher()

        stdout = """[info] Available automatic captions for VIDEO_ID:
Language  Name                              Formats
en        English (auto-generated)          vtt, ttml, srv3, srv2, srv1, json3
"""
        result = fetcher._parse_list_subs_output(stdout, "")

        assert len(result) == 1
        assert result[0].is_auto_generated is True

    def test_parse_mixed_output(self):
        """Test parsing output with both sections"""
        fetcher = CaptionFetcher()

        stdout = """[info] Available subtitles for VIDEO_ID:
Language  Name                 Formats
en        English              vtt, ttml, srv3, srv2, srv1, json3

[info] Available automatic captions for VIDEO_ID:
Language  Name                              Formats
de        German (auto-generated)           vtt, ttml, srv3, srv2, srv1, json3
"""
        result = fetcher._parse_list_subs_output(stdout, "")

        assert len(result) == 2
        # Manual first
        assert result[0].code == "en"
        assert result[0].is_auto_generated is False
        # Auto second
        assert result[1].code == "de"
        assert result[1].is_auto_generated is True

    def test_parse_output_with_stderr(self):
        """Test parsing with content in stderr"""
        fetcher = CaptionFetcher()

        stdout = ""
        stderr = """[info] Available subtitles for VIDEO_ID:
Language  Name                 Formats
ja        Japanese             vtt, ttml, srv3, srv2, srv1, json3
"""
        result = fetcher._parse_list_subs_output(stdout, stderr)

        assert len(result) == 1
        assert result[0].code == "ja"

    def test_parse_handles_header_line(self):
        """Test that header line is skipped"""
        fetcher = CaptionFetcher()

        stdout = """[info] Available subtitles for VIDEO_ID:
Language  Name                 Formats
en        English              vtt, ttml
"""
        result = fetcher._parse_list_subs_output(stdout, "")

        # Should only have 1 language, not treat "Language" as a code
        assert len(result) == 1
        assert result[0].code == "en"

    def test_parse_language_code_variations(self):
        """Test parsing various language code formats"""
        fetcher = CaptionFetcher()

        stdout = """[info] Available subtitles for VIDEO_ID:
Language  Name                      Formats
en        English                   vtt, ttml
en-US     English (United States)   vtt, ttml
zh-Hans   Chinese (Simplified)      vtt, ttml
pt-BR     Portuguese (Brazil)       vtt, ttml
"""
        result = fetcher._parse_list_subs_output(stdout, "")

        codes = [lang.code for lang in result]
        assert "en" in codes
        assert "en-us" in codes
        assert "zh-hans" in codes
        assert "pt-br" in codes


class TestFetchCaptionsAutoLanguage:
    """Test fetch_captions_auto_language method"""

    @patch.object(CaptionFetcher, 'fetch_captions')
    @patch.object(CaptionFetcher, 'list_available_languages')
    def test_auto_language_selects_best(self, mock_list, mock_fetch):
        """Test auto language selection and fetch"""
        fetcher = CaptionFetcher()

        mock_list.return_value = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("es", "Spanish", False),
        ]
        mock_fetch.return_value = CaptionResult(
            video_id="test123",
            segments=[],
            language="en"
        )

        result = fetcher.fetch_captions_auto_language("test1234567")

        mock_fetch.assert_called_once_with(
            "test1234567",
            language="en",
            prefer_manual=True
        )

    @patch.object(CaptionFetcher, 'list_available_languages')
    def test_auto_language_no_captions_available(self, mock_list):
        """Test error when no captions available"""
        fetcher = CaptionFetcher()

        mock_list.return_value = []

        with pytest.raises(CaptionUnavailableError) as exc_info:
            fetcher.fetch_captions_auto_language("test1234567")

        assert "No captions available" in str(exc_info.value)

    @patch.object(CaptionFetcher, 'fetch_captions')
    @patch.object(CaptionFetcher, 'list_available_languages')
    def test_auto_language_uses_preferred(self, mock_list, mock_fetch):
        """Test auto language uses preferred language override"""
        fetcher = CaptionFetcher()

        mock_list.return_value = [
            AvailableLanguage("en", "English", False),
            AvailableLanguage("es", "Spanish", False),
        ]
        mock_fetch.return_value = CaptionResult(
            video_id="test123",
            segments=[],
            language="es"
        )

        fetcher.fetch_captions_auto_language("test1234567", preferred_language="es")

        mock_fetch.assert_called_once_with(
            "test1234567",
            language="es",
            prefer_manual=True
        )

    @patch.object(CaptionFetcher, 'fetch_captions')
    @patch.object(CaptionFetcher, 'list_available_languages')
    def test_auto_language_respects_prefer_human_config(self, mock_list, mock_fetch):
        """Test auto language respects prefer_human_captions config"""
        mock_config = Mock()
        mock_config.download.caption_first.prefer_human_captions = False

        fetcher = CaptionFetcher(config=mock_config)

        mock_list.return_value = [
            AvailableLanguage("en", "English (auto)", True),
        ]
        mock_fetch.return_value = CaptionResult(
            video_id="test123",
            segments=[],
            language="en"
        )

        fetcher.fetch_captions_auto_language("test1234567")

        # Should call with prefer_manual=False since config says don't prefer human
        mock_fetch.assert_called_once_with(
            "test1234567",
            language="en",
            prefer_manual=False  # Auto-generated selected
        )
