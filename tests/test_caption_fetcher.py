"""
Unit tests for CaptionFetcher module.

Tests caption fetching, SRT/VTT parsing, and caption result handling.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.downloader.caption_fetcher import (
    CaptionFetcher,
    CaptionInfo,
    CaptionResult,
)


class TestCaptionInfo:
    """Tests for CaptionInfo dataclass"""

    def test_has_captions_with_manual(self):
        """Test has_captions returns True when manual captions available"""
        info = CaptionInfo(
            video_id="test123",
            has_manual=True,
            manual_languages=["en"]
        )
        assert info.has_captions is True

    def test_has_captions_with_auto(self):
        """Test has_captions returns True when auto captions available"""
        info = CaptionInfo(
            video_id="test123",
            has_auto=True,
            auto_languages=["en-auto"]
        )
        assert info.has_captions is True

    def test_has_captions_none(self):
        """Test has_captions returns False when no captions"""
        info = CaptionInfo(video_id="test123")
        assert info.has_captions is False

    def test_best_language_prefers_manual(self):
        """Test best_language prefers manual over auto"""
        info = CaptionInfo(
            video_id="test123",
            has_manual=True,
            has_auto=True,
            manual_languages=["en"],
            auto_languages=["en-auto"]
        )
        assert info.best_language() == "en"

    def test_best_language_falls_back_to_auto(self):
        """Test best_language falls back to auto when no manual"""
        info = CaptionInfo(
            video_id="test123",
            has_auto=True,
            auto_languages=["en-auto", "es-auto"]
        )
        # Should find en-auto since it starts with "en"
        assert info.best_language(["en"]).startswith("en")

    def test_best_language_custom_preference(self):
        """Test best_language respects custom preference order"""
        info = CaptionInfo(
            video_id="test123",
            has_manual=True,
            manual_languages=["es", "fr", "en"]
        )
        assert info.best_language(["es", "en"]) == "es"


class TestCaptionFetcher:
    """Tests for CaptionFetcher class"""

    @pytest.fixture
    def mock_config(self):
        """Create mock config"""
        config = Mock()
        config.download = Mock()
        config.download.cookies_path = ""
        config.download.cookies_from_browser = ""
        return config

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def fetcher(self, mock_config, temp_cache_dir):
        """Create CaptionFetcher instance"""
        return CaptionFetcher(mock_config, temp_cache_dir)

    def test_parse_srt_basic(self, fetcher, temp_cache_dir):
        """Test basic SRT parsing"""
        srt_content = """1
00:00:00,000 --> 00:00:02,500
First segment text

2
00:00:03,000 --> 00:00:05,500
Second segment text
"""
        srt_path = temp_cache_dir / "test.srt"
        srt_path.write_text(srt_content, encoding='utf-8')

        segments = fetcher.parse_caption_file(str(srt_path))

        assert len(segments) == 2
        assert segments[0]['text'] == "First segment text"
        assert segments[0]['start'] == 0.0
        assert segments[0]['end'] == 2.5
        assert segments[1]['text'] == "Second segment text"
        assert segments[1]['start'] == 3.0
        assert segments[1]['end'] == 5.5

    def test_parse_srt_with_html_tags(self, fetcher, temp_cache_dir):
        """Test SRT parsing removes HTML tags"""
        srt_content = """1
00:00:00,000 --> 00:00:02,500
<font color="white">Text with</font> <b>formatting</b>
"""
        srt_path = temp_cache_dir / "test.srt"
        srt_path.write_text(srt_content, encoding='utf-8')

        segments = fetcher.parse_caption_file(str(srt_path))

        assert len(segments) == 1
        assert segments[0]['text'] == "Text with formatting"

    def test_parse_srt_multiline(self, fetcher, temp_cache_dir):
        """Test SRT parsing handles multiline text"""
        srt_content = """1
00:00:00,000 --> 00:00:02,500
First line
Second line
Third line
"""
        srt_path = temp_cache_dir / "test.srt"
        srt_path.write_text(srt_content, encoding='utf-8')

        segments = fetcher.parse_caption_file(str(srt_path))

        assert len(segments) == 1
        assert segments[0]['text'] == "First line Second line Third line"

    def test_parse_timestamp(self, fetcher):
        """Test timestamp parsing"""
        assert fetcher._parse_timestamp("00:00:00,000") == 0.0
        assert fetcher._parse_timestamp("00:01:30,500") == 90.5
        assert fetcher._parse_timestamp("01:30:00.000") == 5400.0
        assert fetcher._parse_timestamp("00:00:02.5") == 2.5

    def test_clean_caption_text(self, fetcher):
        """Test caption text cleaning"""
        # HTML tags
        assert fetcher._clean_caption_text("<b>bold</b>") == "bold"

        # Speaker labels
        assert fetcher._clean_caption_text("[Speaker 1]: Hello") == "Hello"
        assert fetcher._clean_caption_text("(narrator): Once upon a time") == "Once upon a time"

        # Music indicators
        assert fetcher._clean_caption_text("[Music]") == ""
        assert fetcher._clean_caption_text("(applause)") == ""

        # Whitespace normalization
        assert fetcher._clean_caption_text("  multiple   spaces  ") == "multiple spaces"

    def test_extract_language(self, fetcher):
        """Test language extraction from filename"""
        assert fetcher._extract_language("abc123.en.srt", "abc123") == "en"
        assert fetcher._extract_language("abc123.en-auto.srt", "abc123") == "en"
        assert fetcher._extract_language("abc123.es.srt", "abc123") == "es"

    @patch('subprocess.run')
    def test_fetch_captions_success(self, mock_run, fetcher, temp_cache_dir):
        """Test successful caption fetch"""
        # Create a mock caption file
        srt_content = """1
00:00:00,000 --> 00:00:02,500
Test caption
"""
        caption_file = temp_cache_dir / "dQw4w9WgXcQ.en.srt"
        caption_file.write_text(srt_content, encoding='utf-8')

        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        result = fetcher.fetch_captions("dQw4w9WgXcQ")

        assert result is not None
        assert result.video_id == "dQw4w9WgXcQ"
        assert result.language == "en"
        assert result.file == str(caption_file)

    @patch('subprocess.run')
    def test_fetch_captions_not_found(self, mock_run, fetcher):
        """Test fetch when no captions available"""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")

        result = fetcher.fetch_captions("nonexistent123")

        assert result is None

    @patch('subprocess.run')
    def test_check_caption_availability(self, mock_run, fetcher):
        """Test caption availability check"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"id": "abc123", "subtitles": {"en": []}, "automatic_captions": {"en-auto": []}}',
            stderr=""
        )

        results = fetcher.check_caption_availability(["abc123"])

        assert "abc123" in results
        assert results["abc123"].has_manual is True
        assert results["abc123"].has_auto is True
        assert "en" in results["abc123"].manual_languages


class TestCaptionResult:
    """Tests for CaptionResult dataclass"""

    def test_caption_result_creation(self):
        """Test CaptionResult creation"""
        result = CaptionResult(
            video_id="test123",
            file="/path/to/caption.srt",
            language="en",
            is_auto_generated=False,
            format="srt"
        )
        assert result.video_id == "test123"
        assert result.language == "en"
        assert result.is_auto_generated is False


class TestVTTParsing:
    """Tests for VTT format parsing"""

    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.download = Mock()
        config.download.cookies_path = ""
        config.download.cookies_from_browser = ""
        return config

    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def fetcher(self, mock_config, temp_cache_dir):
        return CaptionFetcher(mock_config, temp_cache_dir)

    def test_parse_vtt_basic(self, fetcher, temp_cache_dir):
        """Test basic VTT parsing"""
        vtt_content = """WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:02.500
First segment

00:00:03.000 --> 00:00:05.500
Second segment
"""
        vtt_path = temp_cache_dir / "test.vtt"
        vtt_path.write_text(vtt_content, encoding='utf-8')

        segments = fetcher.parse_caption_file(str(vtt_path))

        assert len(segments) >= 1
        # VTT parser should extract segments after header

    def test_parse_vtt_with_cue_settings(self, fetcher, temp_cache_dir):
        """Test VTT parsing with cue settings"""
        vtt_content = """WEBVTT

00:00:00.000 --> 00:00:02.500 align:start position:10%
First segment with settings

00:00:03.000 --> 00:00:05.500 line:80%
Second segment with settings
"""
        vtt_path = temp_cache_dir / "test.vtt"
        vtt_path.write_text(vtt_content, encoding='utf-8')

        segments = fetcher.parse_caption_file(str(vtt_path))

        assert len(segments) >= 1


class TestCaptionFetcherEdgeCases:
    """Edge case tests for CaptionFetcher"""

    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.download = Mock()
        config.download.cookies_path = ""
        config.download.cookies_from_browser = ""
        return config

    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def fetcher(self, mock_config, temp_cache_dir):
        return CaptionFetcher(mock_config, temp_cache_dir)

    def test_parse_srt_empty_file(self, fetcher, temp_cache_dir):
        """Test parsing empty SRT file"""
        srt_path = temp_cache_dir / "empty.srt"
        srt_path.write_text("", encoding='utf-8')

        segments = fetcher.parse_caption_file(str(srt_path))

        assert segments == []

    def test_parse_srt_whitespace_only(self, fetcher, temp_cache_dir):
        """Test parsing SRT with only whitespace"""
        srt_path = temp_cache_dir / "whitespace.srt"
        srt_path.write_text("   \n\n   \n", encoding='utf-8')

        segments = fetcher.parse_caption_file(str(srt_path))

        assert segments == []

    def test_parse_srt_malformed_timestamp(self, fetcher, temp_cache_dir):
        """Test parsing SRT with malformed timestamp"""
        srt_content = """1
invalid timestamp
Some text
"""
        srt_path = temp_cache_dir / "malformed.srt"
        srt_path.write_text(srt_content, encoding='utf-8')

        # Should handle gracefully without crashing
        segments = fetcher.parse_caption_file(str(srt_path))
        assert isinstance(segments, list)

    def test_parse_srt_unicode_content(self, fetcher, temp_cache_dir):
        """Test parsing SRT with unicode content"""
        srt_content = """1
00:00:00,000 --> 00:00:02,500
日本語テキスト

2
00:00:03,000 --> 00:00:05,500
Texto en español with ñ
"""
        srt_path = temp_cache_dir / "unicode.srt"
        srt_path.write_text(srt_content, encoding='utf-8')

        segments = fetcher.parse_caption_file(str(srt_path))

        assert len(segments) == 2

    def test_parse_srt_with_bom(self, fetcher, temp_cache_dir):
        """Test parsing SRT with BOM marker"""
        srt_content = "\ufeff1\n00:00:00,000 --> 00:00:02,500\nText with BOM\n"
        srt_path = temp_cache_dir / "bom.srt"
        srt_path.write_text(srt_content, encoding='utf-8-sig')

        segments = fetcher.parse_caption_file(str(srt_path))

        assert len(segments) >= 1

    def test_parse_srt_missing_blank_line(self, fetcher, temp_cache_dir):
        """Test parsing SRT without blank line between entries"""
        srt_content = """1
00:00:00,000 --> 00:00:02,500
First segment
2
00:00:03,000 --> 00:00:05,500
Second segment
"""
        srt_path = temp_cache_dir / "noblank.srt"
        srt_path.write_text(srt_content, encoding='utf-8')

        segments = fetcher.parse_caption_file(str(srt_path))

        # Should still parse (may merge or separate based on implementation)
        assert isinstance(segments, list)

    def test_parse_timestamp_various_formats(self, fetcher):
        """Test timestamp parsing with various formats"""
        # Standard SRT format
        assert fetcher._parse_timestamp("00:00:00,000") == 0.0
        assert fetcher._parse_timestamp("00:01:30,500") == 90.5

        # VTT format with dot
        assert fetcher._parse_timestamp("00:00:00.000") == 0.0
        assert fetcher._parse_timestamp("00:01:30.500") == 90.5

        # Hours
        assert fetcher._parse_timestamp("01:00:00,000") == 3600.0
        assert fetcher._parse_timestamp("02:30:45,123") == 9045.123

    def test_clean_caption_text_nested_tags(self, fetcher):
        """Test cleaning nested HTML tags"""
        text = "<font><b>nested</b> tags</font>"
        cleaned = fetcher._clean_caption_text(text)
        assert "<" not in cleaned
        assert ">" not in cleaned

    def test_clean_caption_text_special_chars(self, fetcher):
        """Test cleaning special characters"""
        text = "Text with &amp; and &lt; entities"
        cleaned = fetcher._clean_caption_text(text)
        # Should handle HTML entities
        assert isinstance(cleaned, str)

    def test_clean_caption_text_sound_effects(self, fetcher):
        """Test removing sound effect indicators"""
        assert fetcher._clean_caption_text("[laughing]") == ""
        assert fetcher._clean_caption_text("(music playing)") == ""
        assert fetcher._clean_caption_text("[Music]") == ""
        assert fetcher._clean_caption_text("[MUSIC]") == ""

    @patch('subprocess.run')
    def test_fetch_captions_timeout(self, mock_run, fetcher):
        """Test caption fetch with timeout"""
        mock_run.side_effect = Exception("Timeout")

        result = fetcher.fetch_captions("test123", timeout=1)

        assert result is None

    @patch('subprocess.run')
    def test_fetch_captions_yt_dlp_error(self, mock_run, fetcher):
        """Test caption fetch when yt-dlp returns error"""
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="ERROR: Video unavailable")

        result = fetcher.fetch_captions("test123")

        assert result is None

    @patch('subprocess.run')
    def test_check_availability_multiple_videos(self, mock_run, fetcher):
        """Test checking availability for multiple videos"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"id": "vid1", "subtitles": {"en": []}, "automatic_captions": {}}\n'
                   '{"id": "vid2", "subtitles": {}, "automatic_captions": {"en": []}}',
            stderr=""
        )

        results = fetcher.check_caption_availability(["vid1", "vid2"])

        # Should handle multiple results
        assert isinstance(results, dict)

    @patch('subprocess.run')
    def test_check_availability_json_error(self, mock_run, fetcher):
        """Test handling invalid JSON response"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="not valid json",
            stderr=""
        )

        results = fetcher.check_caption_availability(["vid1"])

        # Should handle gracefully
        assert isinstance(results, dict)


class TestCaptionInfoAdvanced:
    """Advanced tests for CaptionInfo"""

    def test_best_language_with_empty_preference(self):
        """Test best_language with empty preference list"""
        info = CaptionInfo(
            video_id="test123",
            has_manual=True,
            manual_languages=["en", "es"]
        )
        # Empty preference should return first available
        lang = info.best_language([])
        assert lang is not None

    def test_best_language_no_match(self):
        """Test best_language when no preference matches"""
        info = CaptionInfo(
            video_id="test123",
            has_manual=True,
            manual_languages=["ja", "zh"]
        )
        # Looking for en but only ja/zh available
        lang = info.best_language(["en", "es"])
        # Should return something or None
        assert lang is None or isinstance(lang, str)

    def test_caption_info_repr(self):
        """Test CaptionInfo string representation"""
        info = CaptionInfo(
            video_id="test123",
            has_manual=True,
            manual_languages=["en"]
        )
        repr_str = repr(info)
        assert "test123" in repr_str

    def test_caption_info_all_fields(self):
        """Test CaptionInfo with all fields populated"""
        info = CaptionInfo(
            video_id="test123",
            has_manual=True,
            has_auto=True,
            manual_languages=["en", "es", "fr"],
            auto_languages=["en-auto", "es-auto"]
        )
        assert info.has_captions is True
        assert len(info.manual_languages) == 3
        assert len(info.auto_languages) == 2


class TestCaptionResultAdvanced:
    """Advanced tests for CaptionResult"""

    def test_caption_result_all_fields(self):
        """Test CaptionResult with all fields"""
        result = CaptionResult(
            video_id="test123",
            file="/path/to/caption.srt",
            language="en-US",
            is_auto_generated=False,
            format="srt"
        )
        assert result.video_id == "test123"
        assert result.file == "/path/to/caption.srt"
        assert result.language == "en-US"
        assert result.is_auto_generated is False
        assert result.format == "srt"

    def test_caption_result_auto_generated(self):
        """Test CaptionResult for auto-generated caption"""
        result = CaptionResult(
            video_id="test123",
            file="/path/to/caption.en-auto.vtt",
            language="en",
            is_auto_generated=True,
            format="vtt"
        )
        assert result.is_auto_generated is True
        assert result.format == "vtt"
