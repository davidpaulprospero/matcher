"""Tests for US-70-003: CaptionResult metadata fields (video_description, video_chapters, video_tags)."""
import pytest
from src.caption.models import CaptionResult, CaptionSegment


class TestCaptionResultMetadataFields:
    """Test new metadata fields on CaptionResult."""

    def test_default_values(self):
        """New fields default to empty values, preserving backward compatibility."""
        result = CaptionResult(video_id="abc123")
        assert result.video_description == ""
        assert result.video_chapters == []
        assert result.video_tags == []

    def test_populated_fields(self):
        """Fields can be populated at construction time."""
        chapters = [
            {"title": "Intro", "start_time": 0.0, "end_time": 30.0},
            {"title": "Main", "start_time": 30.0, "end_time": 120.0},
        ]
        tags = ["tutorial", "python", "programming"]
        result = CaptionResult(
            video_id="abc123",
            video_description="A great tutorial about Python",
            video_chapters=chapters,
            video_tags=tags,
        )
        assert result.video_description == "A great tutorial about Python"
        assert len(result.video_chapters) == 2
        assert result.video_chapters[0]["title"] == "Intro"
        assert result.video_tags == ["tutorial", "python", "programming"]

    def test_to_dict_includes_metadata(self):
        """to_dict serializes the new metadata fields."""
        result = CaptionResult(
            video_id="abc123",
            video_description="Test description",
            video_chapters=[{"title": "Ch1", "start_time": 0.0, "end_time": 60.0}],
            video_tags=["tag1", "tag2"],
        )
        d = result.to_dict()
        assert d["video_description"] == "Test description"
        assert d["video_chapters"] == [{"title": "Ch1", "start_time": 0.0, "end_time": 60.0}]
        assert d["video_tags"] == ["tag1", "tag2"]

    def test_to_dict_empty_defaults(self):
        """to_dict includes empty defaults for new fields."""
        result = CaptionResult(video_id="abc123")
        d = result.to_dict()
        assert d["video_description"] == ""
        assert d["video_chapters"] == []
        assert d["video_tags"] == []

    def test_round_trip_through_dict(self):
        """New fields survive a to_dict -> reconstruct round-trip."""
        segments = [
            CaptionSegment(index=0, start_time=0.0, end_time=5.0, text="Hello world"),
        ]
        chapters = [
            {"title": "Intro", "start_time": 0.0, "end_time": 30.0},
            {"title": "Body", "start_time": 30.0, "end_time": 120.0},
        ]
        tags = ["science", "education"]
        original = CaptionResult(
            video_id="vid_001",
            segments=segments,
            language="en",
            is_auto_generated=True,
            format_source="json3",
            video_description="Science video about space",
            video_chapters=chapters,
            video_tags=tags,
        )

        d = original.to_dict()

        # Reconstruct — simulating cache load where new fields may or may not exist
        reconstructed = CaptionResult(
            video_id=d["video_id"],
            language=d.get("language", ""),
            is_auto_generated=d.get("is_auto_generated", False),
            format_source=d.get("format_source", ""),
            video_description=d.get("video_description", ""),
            video_chapters=d.get("video_chapters", []),
            video_tags=d.get("video_tags", []),
        )

        assert reconstructed.video_description == original.video_description
        assert reconstructed.video_chapters == original.video_chapters
        assert reconstructed.video_tags == original.video_tags

    def test_backward_compat_missing_fields(self):
        """Old cached dicts without new fields still work via .get() defaults."""
        old_cache_data = {
            "video_id": "old_vid",
            "language": "en",
            "is_auto_generated": False,
            "format_source": "vtt",
            # No video_description, video_chapters, video_tags
        }
        result = CaptionResult(
            video_id=old_cache_data["video_id"],
            language=old_cache_data.get("language", ""),
            is_auto_generated=old_cache_data.get("is_auto_generated", False),
            format_source=old_cache_data.get("format_source", ""),
            video_description=old_cache_data.get("video_description", ""),
            video_chapters=old_cache_data.get("video_chapters", []),
            video_tags=old_cache_data.get("video_tags", []),
        )
        assert result.video_description == ""
        assert result.video_chapters == []
        assert result.video_tags == []
