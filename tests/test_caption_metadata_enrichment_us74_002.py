"""
Unit tests for US-74-002: Extract video metadata during caption fetch.

Verifies that fetch_captions/fetch_subtitle_with_format populates
video_description, video_chapters, and video_tags on CaptionResult
when info_dict is available via --write-info-json.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.caption.models import CaptionResult, CaptionSegment
from src.caption_fetcher import (
    CaptionFetcher,
    extract_video_metadata_from_info_dict,
)


def _make_info_json(video_id: str, description: str = "Test description",
                    chapters: list = None, tags: list = None) -> dict:
    """Build a mock yt-dlp info_dict."""
    return {
        "id": video_id,
        "title": "Test Video",
        "description": description,
        "chapters": chapters or [
            {"title": "Intro", "start_time": 0.0, "end_time": 30.0},
            {"title": "Main", "start_time": 30.0, "end_time": 120.0},
        ],
        "tags": tags or ["test", "python", "video"],
    }


@pytest.mark.fast
class TestFetchSubtitleWithFormatMetadataEnrichment:
    """Test that _fetch_subtitle_with_format enriches CaptionResult from info.json."""

    def _make_fetcher(self):
        """Create a CaptionFetcher with minimal config for testing."""
        fetcher = CaptionFetcher.__new__(CaptionFetcher)
        fetcher.config = None
        fetcher._timeout = 30
        fetcher._preferred_formats = ['vtt']
        fetcher._format_timeout_policy = MagicMock()
        fetcher._format_timeout_policy.get_timeout.return_value = 30.0
        fetcher._active_metrics = None
        fetcher._active_retry_budget = None
        return fetcher

    @patch('src.caption_fetcher.subprocess.run')
    def test_metadata_enriched_when_info_json_present(self, mock_run):
        """CaptionResult has metadata populated when .info.json file exists."""
        video_id = "dQw4w9WgXcQ"
        info_dict = _make_info_json(video_id, description="A great video about testing")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Write subtitle file
            vtt_content = "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nHello world\n"
            sub_file = temp_path / f"{video_id}.en.vtt"
            sub_file.write_text(vtt_content, encoding='utf-8')

            # Write info.json (simulates --write-info-json output)
            info_file = temp_path / f"{video_id}.info.json"
            info_file.write_text(json.dumps(info_dict), encoding='utf-8')

            # Mock subprocess to succeed (returncode=0, empty output)
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            fetcher = self._make_fetcher()
            fetcher._add_bypass_args_to_cmd = MagicMock()
            fetcher._get_cookies_args = MagicMock(return_value=[])

            result = fetcher._fetch_subtitle_with_format(
                video_url=f"https://www.youtube.com/watch?v={video_id}",
                video_id=video_id,
                temp_dir=temp_path,
                language="en",
                auto_generated=False,
                subtitle_format="vtt",
            )

            assert result is not None
            assert result.video_description == "A great video about testing"
            assert len(result.video_chapters) == 2
            assert result.video_chapters[0]["title"] == "Intro"
            assert result.video_chapters[1]["title"] == "Main"
            assert result.video_tags == ["test", "python", "video"]

    @patch('src.caption_fetcher.subprocess.run')
    def test_metadata_defaults_when_no_info_json(self, mock_run):
        """CaptionResult has empty defaults when no .info.json file exists."""
        video_id = "dQw4w9WgXcQ"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Write subtitle file only — no info.json
            vtt_content = "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nHello world\n"
            sub_file = temp_path / f"{video_id}.en.vtt"
            sub_file.write_text(vtt_content, encoding='utf-8')

            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            fetcher = self._make_fetcher()
            fetcher._add_bypass_args_to_cmd = MagicMock()
            fetcher._get_cookies_args = MagicMock(return_value=[])

            result = fetcher._fetch_subtitle_with_format(
                video_url=f"https://www.youtube.com/watch?v={video_id}",
                video_id=video_id,
                temp_dir=temp_path,
                language="en",
                auto_generated=False,
                subtitle_format="vtt",
            )

            assert result is not None
            assert result.video_description == ""
            assert result.video_chapters == []
            assert result.video_tags == []

    @patch('src.caption_fetcher.subprocess.run')
    def test_metadata_graceful_on_corrupted_info_json(self, mock_run):
        """CaptionResult has empty defaults when info.json is corrupted."""
        video_id = "dQw4w9WgXcQ"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Write subtitle file
            vtt_content = "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nHello world\n"
            sub_file = temp_path / f"{video_id}.en.vtt"
            sub_file.write_text(vtt_content, encoding='utf-8')

            # Write corrupted info.json
            info_file = temp_path / f"{video_id}.info.json"
            info_file.write_text("NOT VALID JSON {{{", encoding='utf-8')

            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            fetcher = self._make_fetcher()
            fetcher._add_bypass_args_to_cmd = MagicMock()
            fetcher._get_cookies_args = MagicMock(return_value=[])

            result = fetcher._fetch_subtitle_with_format(
                video_url=f"https://www.youtube.com/watch?v={video_id}",
                video_id=video_id,
                temp_dir=temp_path,
                language="en",
                auto_generated=False,
                subtitle_format="vtt",
            )

            assert result is not None
            # Should gracefully fall back to defaults
            assert result.video_description == ""
            assert result.video_chapters == []
            assert result.video_tags == []

    @patch('src.caption_fetcher.subprocess.run')
    def test_description_truncation_from_info_json(self, mock_run):
        """Description is truncated to configurable max length (default 500)."""
        video_id = "dQw4w9WgXcQ"
        long_desc = "A" * 1000  # 1000 chars, should be truncated to 500
        info_dict = _make_info_json(video_id, description=long_desc)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            vtt_content = "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nHello world\n"
            sub_file = temp_path / f"{video_id}.en.vtt"
            sub_file.write_text(vtt_content, encoding='utf-8')

            info_file = temp_path / f"{video_id}.info.json"
            info_file.write_text(json.dumps(info_dict), encoding='utf-8')

            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            fetcher = self._make_fetcher()
            fetcher._add_bypass_args_to_cmd = MagicMock()
            fetcher._get_cookies_args = MagicMock(return_value=[])

            result = fetcher._fetch_subtitle_with_format(
                video_url=f"https://www.youtube.com/watch?v={video_id}",
                video_id=video_id,
                temp_dir=temp_path,
                language="en",
                auto_generated=False,
                subtitle_format="vtt",
            )

            assert result is not None
            assert len(result.video_description) == 500

    @patch('src.caption_fetcher.subprocess.run')
    def test_partial_metadata_fields(self, mock_run):
        """When info_dict has some fields missing, others still populate."""
        video_id = "dQw4w9WgXcQ"
        # Only description, no chapters or tags
        info_dict = {"description": "Only description here"}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            vtt_content = "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nHello world\n"
            sub_file = temp_path / f"{video_id}.en.vtt"
            sub_file.write_text(vtt_content, encoding='utf-8')

            info_file = temp_path / f"{video_id}.info.json"
            info_file.write_text(json.dumps(info_dict), encoding='utf-8')

            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            fetcher = self._make_fetcher()
            fetcher._add_bypass_args_to_cmd = MagicMock()
            fetcher._get_cookies_args = MagicMock(return_value=[])

            result = fetcher._fetch_subtitle_with_format(
                video_url=f"https://www.youtube.com/watch?v={video_id}",
                video_id=video_id,
                temp_dir=temp_path,
                language="en",
                auto_generated=False,
                subtitle_format="vtt",
            )

            assert result is not None
            assert result.video_description == "Only description here"
            assert result.video_chapters == []
            assert result.video_tags == []

    @patch('src.caption_fetcher.subprocess.run')
    def test_write_info_json_flag_in_command(self, mock_run):
        """Verify --write-info-json is included in the yt-dlp command."""
        video_id = "dQw4w9WgXcQ"

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # No subtitle file means result will be None, but we can still check the command
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            fetcher = self._make_fetcher()
            fetcher._add_bypass_args_to_cmd = MagicMock()
            fetcher._get_cookies_args = MagicMock(return_value=[])

            fetcher._fetch_subtitle_with_format(
                video_url=f"https://www.youtube.com/watch?v={video_id}",
                video_id=video_id,
                temp_dir=temp_path,
                language="en",
                auto_generated=False,
                subtitle_format="vtt",
            )

            # Check that --write-info-json was in the command
            cmd_args = mock_run.call_args[0][0]
            assert '--write-info-json' in cmd_args


@pytest.mark.fast
class TestExtractVideoMetadataFromInfoDict:
    """Test extract_video_metadata_from_info_dict with edge cases for US-74-002."""

    def test_all_fields_present(self):
        """All metadata fields extracted from complete info_dict."""
        info_dict = _make_info_json("test123")
        desc, chapters, tags, source = extract_video_metadata_from_info_dict(info_dict)
        assert desc == "Test description"
        assert len(chapters) == 2
        assert tags == ["test", "python", "video"]

    def test_partially_present(self):
        """Only some fields present, others default to empty."""
        info_dict = {"tags": ["only_tags"]}
        desc, chapters, tags, source = extract_video_metadata_from_info_dict(info_dict)
        assert desc == ""
        assert chapters == []
        assert tags == ["only_tags"]

    def test_fully_absent(self):
        """No metadata fields at all — all default to empty."""
        desc, chapters, tags, source = extract_video_metadata_from_info_dict({})
        assert desc == ""
        assert chapters == []
        assert tags == []

    def test_none_info_dict(self):
        """None info_dict handled gracefully."""
        desc, chapters, tags, source = extract_video_metadata_from_info_dict(None)
        assert desc == ""
        assert chapters == []
        assert tags == []

    def test_chapters_as_list_of_dicts(self):
        """Chapters extracted with correct keys."""
        info_dict = {
            "chapters": [
                {"title": "Ch1", "start_time": 0.0, "end_time": 10.0},
                {"title": "Ch2", "start_time": 10.0, "end_time": 20.0},
                {"title": "Ch3", "start_time": 20.0, "end_time": 30.0},
            ]
        }
        _, chapters, _, _ = extract_video_metadata_from_info_dict(info_dict)
        assert len(chapters) == 3
        assert chapters[0] == {"title": "Ch1", "start_time": 0.0, "end_time": 10.0}
        assert chapters[2] == {"title": "Ch3", "start_time": 20.0, "end_time": 30.0}

    def test_tags_as_list_of_strings(self):
        """Tags extracted as list of strings from info_dict['tags']."""
        info_dict = {"tags": ["alpha", "beta", "gamma"]}
        _, _, tags, _ = extract_video_metadata_from_info_dict(info_dict)
        assert tags == ["alpha", "beta", "gamma"]

    def test_none_values_in_fields(self):
        """None values in description/chapters/tags handled gracefully."""
        info_dict = {"description": None, "chapters": None, "tags": None}
        desc, chapters, tags, source = extract_video_metadata_from_info_dict(info_dict)
        assert desc == ""
        assert chapters == []
        assert tags == []


@pytest.mark.fast
class TestExtractVideoMetadataExtractionSource:
    """Test extraction_source return value (US-135-008)."""

    def test_metadata_source_when_chapters_exist(self):
        """Returns 'metadata' when metadata chapters exist."""
        info_dict = {
            "chapters": [
                {"title": "Ch1", "start_time": 0.0, "end_time": 10.0},
            ]
        }
        _, _, _, source = extract_video_metadata_from_info_dict(info_dict)
        assert source == "metadata"

    def test_description_source_when_no_metadata(self):
        """Returns 'description' when using description chapters."""
        info_dict = {
            "description": "0:00 Chapter 1\n2:00 Chapter 2"
        }
        _, _, _, source = extract_video_metadata_from_info_dict(info_dict)
        assert source == "description"

    def test_none_source_when_disabled_and_no_metadata(self):
        """Returns 'none' when disabled and no metadata chapters."""
        info_dict = {"description": "0:00 Chapter"}
        _, _, _, source = extract_video_metadata_from_info_dict(
            info_dict, chapter_extraction_fallback='disabled'
        )
        assert source == "none"

    def test_always_merge(self):
        """Returns 'merged' when using both metadata and description."""
        info_dict = {
            "chapters": [
                {"title": "MetaCh1", "start_time": 0.0, "end_time": 10.0},
            ],
            "description": "0:00 DescCh1\n5:00 DescCh2"
        }
        _, chapters, _, source = extract_video_metadata_from_info_dict(
            info_dict, chapter_extraction_fallback='always'
        )
        assert source == "merged"
        # Should have both chapters (metadata takes precedence for same time)
        assert len(chapters) >= 2
