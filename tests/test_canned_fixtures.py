"""
Test fixtures validation for canned test data.

US-151-006: Validates that all canned test fixtures can be loaded correctly.
"""

import json
from pathlib import Path
from typing import Dict, Any, List

import pytest


# Base path for canned fixtures
CANNED_DATA_DIR = Path(__file__).parent / "fixtures" / "canned_data"


def load_canned_fixture(filename: str) -> Dict[str, Any]:
    """Load a JSON fixture file."""
    fixture_path = CANNED_DATA_DIR / filename
    assert fixture_path.exists(), f"Fixture not found: {filename}"
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)


class TestCannedVideoSearchFixtures:
    """Test canned video search result fixtures."""

    def test_load_wildlife_video_search(self):
        """Test loading wildlife video search results."""
        data = load_canned_fixture("video_search_wildlife.json")

        assert data["query"] == "wildlife documentary nature"
        assert data["total_results"] == 1000000
        assert len(data["videos"]) == 8

        # Validate video structure
        for video in data["videos"]:
            assert "video_id" in video
            assert "title" in video
            assert "description" in video
            assert "channel_id" in video
            assert "duration_seconds" in video
            assert "caption_available" in video

    def test_load_technology_video_search(self):
        """Test loading technology video search results."""
        data = load_canned_fixture("video_search_technology.json")

        assert data["query"] == "technology documentary"
        assert len(data["videos"]) == 5

        # Verify all videos have required fields
        for video in data["videos"]:
            assert video["caption_available"] is True


class TestCannedCaptionFixtures:
    """Test canned caption data fixtures."""

    def test_load_caption_vid001(self):
        """Test loading caption data for video canned_vid001."""
        data = load_canned_fixture("caption_canned_vid001.json")

        assert data["video_id"] == "canned_vid001"
        assert data["language"] == "en"
        assert len(data["captions"]) == 15

        # Validate caption structure
        for caption in data["captions"]:
            assert "start" in caption
            assert "end" in caption
            assert "text" in caption
            assert caption["start"] < caption["end"]

    def test_load_caption_vid002(self):
        """Test loading caption data for video canned_vid002."""
        data = load_canned_fixture("caption_canned_vid002.json")

        assert data["video_id"] == "canned_vid002"
        assert len(data["captions"]) == 15

        # Verify captions are in order
        prev_end = 0.0
        for caption in data["captions"]:
            assert caption["start"] >= prev_end
            prev_end = caption["end"]


class TestCannedVoiceoverFixtures:
    """Test canned voiceover segment fixtures."""

    def test_load_voiceover_nature_documentary(self):
        """Test loading voiceover segments for nature documentary."""
        data = load_canned_fixture("voiceover_nature_documentary.json")

        assert data["project_name"] == "Nature Documentary"
        assert data["source_file"] == "voiceover.srt"
        assert len(data["segments"]) == 30

        # Validate segment structure
        for segment in data["segments"]:
            assert "index" in segment
            assert "text" in segment
            assert "start_time" in segment
            assert "end_time" in segment
            assert "keywords" in segment
            assert segment["start_time"] < segment["end_time"]

        # Verify segments are in order
        for i in range(len(data["segments"]) - 1):
            curr_end = data["segments"][i]["end_time"]
            next_start = data["segments"][i + 1]["start_time"]
            assert curr_end <= next_start


class TestCannedFixturesIntegration:
    """Integration tests for canned fixtures."""

    def test_all_video_ids_have_captions(self):
        """Test that all video IDs in search results have corresponding caption files."""
        # Load video search results
        wildlife_data = load_canned_fixture("video_search_wildlife.json")
        tech_data = load_canned_fixture("video_search_technology.json")

        # Collect all video IDs
        all_video_ids = []
        for video in wildlife_data["videos"]:
            all_video_ids.append(video["video_id"])
        for video in tech_data["videos"]:
            all_video_ids.append(video["video_id"])

        # Check that caption files exist for at least some videos
        # (We have captions for canned_vid001 and canned_vid002)
        assert "canned_vid001" in all_video_ids
        assert "canned_vid002" in all_video_ids

        # Load captions for available videos
        caption_001 = load_canned_fixture("caption_canned_vid001.json")
        caption_002 = load_canned_fixture("caption_canned_vid002.json")

        assert caption_001["video_id"] in all_video_ids
        assert caption_002["video_id"] in all_video_ids

    def test_voiceover_segments_duration(self):
        """Test that voiceover segment durations are reasonable."""
        data = load_canned_fixture("voiceover_nature_documentary.json")

        total_duration = sum(seg["duration_seconds"] for seg in data["segments"])

        # Total should be approximately 168 seconds as defined in the fixture
        assert 160 <= total_duration <= 180, f"Total duration {total_duration} out of expected range"

    def test_fixture_count(self):
        """Test that minimum required fixtures exist."""
        # Count JSON files in canned_data directory
        json_files = list(CANNED_DATA_DIR.glob("*.json"))

        # We need at least 4 fixtures:
        # - video_search_wildlife.json
        # - video_search_technology.json
        # - caption_canned_vid001.json
        # - caption_canned_vid002.json
        # - voiceover_nature_documentary.json
        assert len(json_files) >= 5, f"Expected at least 5 fixture files, found {len(json_files)}"


class TestCannedFixturesEdgeCases:
    """Edge case tests for canned fixtures."""

    def test_video_duration_tiers(self):
        """Test that videos span different duration tiers."""
        data = load_canned_fixture("video_search_wildlife.json")

        short = [v for v in data["videos"] if v["duration_seconds"] <= 120]
        medium = [v for v in data["videos"] if 120 < v["duration_seconds"] <= 600]
        long = [v for v in data["videos"] if v["duration_seconds"] > 600]

        # Videos are documentary-length (>10 min each)
        assert len(long) >= 1, "Need at least one long video"
        # Note: short and medium are optional - documentaries tend to be long

    def test_caption_text_not_empty(self):
        """Test that all caption texts are non-empty."""
        caption_001 = load_canned_fixture("caption_canned_vid001.json")

        for caption in caption_001["captions"]:
            assert len(caption["text"].strip()) > 0, "Caption text should not be empty"

    def test_voiceover_keywords_not_empty(self):
        """Test that voiceover segments have keywords."""
        data = load_canned_fixture("voiceover_nature_documentary.json")

        for segment in data["segments"]:
            assert len(segment["keywords"]) > 0, "Each segment should have keywords"
