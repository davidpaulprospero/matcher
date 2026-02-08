"""
Unit tests for US-70-002: Video metadata fields on CaptionResult and state dataclasses.

Tests backward-compatible serialization/deserialization with and without new fields.
"""

import pytest
from dataclasses import asdict

from src.caption.models import CaptionResult, CaptionSegment
from src.state import VideoSearchResult, DownloadedVideo


@pytest.mark.fast
class TestCaptionResultMetadataFields:
    """Test new video metadata fields on CaptionResult."""

    def test_default_values(self):
        """New fields default to empty when not provided."""
        result = CaptionResult(video_id="abc123")
        assert result.video_description == ""
        assert result.video_chapters == []
        assert result.video_tags == []

    def test_with_metadata(self):
        """Fields can be set with actual metadata."""
        chapters = [
            {"title": "Intro", "start_time": 0.0, "end_time": 30.0},
            {"title": "Main", "start_time": 30.0, "end_time": 120.0},
        ]
        result = CaptionResult(
            video_id="abc123",
            video_description="A video about testing",
            video_chapters=chapters,
            video_tags=["testing", "python", "code"],
        )
        assert result.video_description == "A video about testing"
        assert len(result.video_chapters) == 2
        assert result.video_chapters[0]["title"] == "Intro"
        assert result.video_tags == ["testing", "python", "code"]

    def test_none_safety_post_init(self):
        """__post_init__ converts None to safe defaults (Rule 2/6)."""
        result = CaptionResult(
            video_id="abc123",
            video_description=None,
            video_chapters=None,
            video_tags=None,
        )
        assert result.video_description == ""
        assert result.video_chapters == []
        assert result.video_tags == []

    def test_to_dict_includes_new_fields(self):
        """to_dict() serializes the new metadata fields."""
        result = CaptionResult(
            video_id="abc123",
            video_description="Test desc",
            video_chapters=[{"title": "Ch1", "start_time": 0.0, "end_time": 10.0}],
            video_tags=["tag1", "tag2"],
        )
        d = result.to_dict()
        assert d["video_description"] == "Test desc"
        assert d["video_chapters"] == [{"title": "Ch1", "start_time": 0.0, "end_time": 10.0}]
        assert d["video_tags"] == ["tag1", "tag2"]

    def test_to_dict_empty_defaults(self):
        """to_dict() serializes empty defaults correctly."""
        result = CaptionResult(video_id="abc123")
        d = result.to_dict()
        assert d["video_description"] == ""
        assert d["video_chapters"] == []
        assert d["video_tags"] == []

    def test_round_trip_with_metadata(self):
        """CaptionResult -> to_dict -> reconstruct preserves metadata."""
        original = CaptionResult(
            video_id="abc123",
            segments=[CaptionSegment(index=0, start_time=0.0, end_time=5.0, text="Hello")],
            language="en",
            is_auto_generated=False,
            format_source="vtt",
            video_description="Round trip test",
            video_chapters=[{"title": "Part 1", "start_time": 0.0, "end_time": 60.0}],
            video_tags=["test", "roundtrip"],
        )
        d = original.to_dict()

        # Reconstruct (simulating what deserialization would do)
        reconstructed = CaptionResult(
            video_id=d["video_id"],
            language=d["language"],
            is_auto_generated=d["is_auto_generated"],
            format_source=d["format_source"],
            video_description=d.get("video_description", ""),
            video_chapters=d.get("video_chapters", []),
            video_tags=d.get("video_tags", []),
        )
        assert reconstructed.video_description == original.video_description
        assert reconstructed.video_chapters == original.video_chapters
        assert reconstructed.video_tags == original.video_tags

    def test_round_trip_without_metadata(self):
        """Simulates loading cached data that lacks new fields (backward compat)."""
        # Old serialized data without new fields
        old_dict = {
            "video_id": "old123",
            "language": "en",
            "is_auto_generated": True,
            "format_source": "srv3",
        }

        # Reconstruct using .get() with defaults — simulates backward compat
        result = CaptionResult(
            video_id=old_dict["video_id"],
            language=old_dict["language"],
            is_auto_generated=old_dict["is_auto_generated"],
            format_source=old_dict["format_source"],
            video_description=old_dict.get("video_description", ""),
            video_chapters=old_dict.get("video_chapters", []),
            video_tags=old_dict.get("video_tags", []),
        )
        assert result.video_description == ""
        assert result.video_chapters == []
        assert result.video_tags == []

    def test_asdict_includes_new_fields(self):
        """dataclasses.asdict includes the new fields."""
        result = CaptionResult(
            video_id="abc123",
            video_description="asdict test",
            video_chapters=[{"title": "A"}],
            video_tags=["x"],
        )
        d = asdict(result)
        assert d["video_description"] == "asdict test"
        assert d["video_chapters"] == [{"title": "A"}]
        assert d["video_tags"] == ["x"]


@pytest.mark.fast
class TestVideoSearchResultDescription:
    """Test new description field on VideoSearchResult."""

    def test_default_empty(self):
        result = VideoSearchResult(video_id="v1")
        assert result.description == ""

    def test_with_description(self):
        result = VideoSearchResult(video_id="v1", description="A search result")
        assert result.description == "A search result"

    def test_asdict_round_trip(self):
        original = VideoSearchResult(video_id="v1", title="Test", description="Desc")
        d = asdict(original)
        assert d["description"] == "Desc"
        reconstructed = VideoSearchResult(**d)
        assert reconstructed.description == original.description


@pytest.mark.fast
class TestDownloadedVideoDescription:
    """Test new description field on DownloadedVideo."""

    def test_default_empty(self):
        vid = DownloadedVideo(file="test.mp4")
        assert vid.description == ""

    def test_with_description(self):
        vid = DownloadedVideo(file="test.mp4", description="Downloaded video desc")
        assert vid.description == "Downloaded video desc"

    def test_asdict_round_trip(self):
        original = DownloadedVideo(file="test.mp4", title="Test", description="Desc")
        d = asdict(original)
        assert d["description"] == "Desc"
        reconstructed = DownloadedVideo(**d)
        assert reconstructed.description == original.description

    def test_backward_compat_no_description(self):
        """Old checkpoint data without description field loads fine."""
        old_data = {"file": "test.mp4", "title": "Old Video"}
        vid = DownloadedVideo(**old_data)
        assert vid.description == ""
