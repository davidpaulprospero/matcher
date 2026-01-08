"""
Unit tests for state dataclasses.

Tests all pipeline state dataclasses with correct field names.
"""

import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.state import (
    VoiceoverSegment,
    TranscriptSegment,
    DownloadedVideo,
    AudioDownload,
    Match,
    EntityImage,
    EntityVideo,
    PipelineState
)


class TestVoiceoverSegment:
    """Test VoiceoverSegment dataclass."""

    def test_create_voiceover_segment(self):
        """Test creating voiceover segment with correct field names."""
        segment = VoiceoverSegment(
            index=0,
            start=0.0,
            end=5.0,
            text="Test voiceover text"
        )

        assert segment.index == 0
        assert segment.start == 0.0
        assert segment.end == 5.0
        assert segment.text == "Test voiceover text"

    def test_voiceover_duration_auto_calculated(self):
        """Test duration is auto-calculated in post_init."""
        segment = VoiceoverSegment(
            index=0,
            start=10.0,
            end=25.0,
            text="Test"
        )

        assert segment.duration == 15.0

    def test_voiceover_custom_duration(self):
        """Test providing custom duration."""
        segment = VoiceoverSegment(
            index=0,
            start=0.0,
            end=5.0,
            text="Test",
            duration=5.5  # Custom duration
        )

        assert segment.duration == 5.5


class TestTranscriptSegment:
    """Test TranscriptSegment dataclass."""

    def test_create_transcript_segment(self):
        """Test creating transcript segment."""
        segment = TranscriptSegment(
            index=0,
            start_time=0.0,
            end_time=10.0,
            text="Transcribed text"
        )

        assert segment.index == 0
        assert segment.start_time == 0.0
        assert segment.end_time == 10.0
        assert segment.text == "Transcribed text"

    def test_transcript_with_source_file(self):
        """Test transcript with source file."""
        segment = TranscriptSegment(
            index=0,
            start_time=0.0,
            end_time=5.0,
            text="Test",
            source_file="/path/to/video.mp4"
        )

        assert segment.source_file == "/path/to/video.mp4"

    def test_transcript_broll_flag(self):
        """Test B-roll flag."""
        segment = TranscriptSegment(
            index=0,
            start_time=0.0,
            end_time=5.0,
            text="Silent video",
            is_broll=True
        )

        assert segment.is_broll is True

    def test_transcript_description_source(self):
        """Test description source field."""
        segment = TranscriptSegment(
            index=0,
            start_time=0.0,
            end_time=5.0,
            text="Vision-generated description",
            description_source="vision"
        )

        assert segment.description_source == "vision"


class TestDownloadedVideo:
    """Test DownloadedVideo dataclass."""

    def test_create_downloaded_video(self):
        """Test creating downloaded video with correct field names."""
        video = DownloadedVideo(
            file="/path/to/video.mp4",
            url="https://youtube.com/watch?v=abc123",
            title="Test Video",
            duration=120.5
        )

        assert video.file == "/path/to/video.mp4"
        assert video.url == "https://youtube.com/watch?v=abc123"
        assert video.title == "Test Video"
        assert video.duration == 120.5

    def test_video_with_metadata(self):
        """Test video with complete metadata."""
        video = DownloadedVideo(
            file="/path/to/video.mp4",
            channel="Test Channel",
            upload_date="2024-01-01",
            duration_tier="short",
            keyword="test keyword",
            source="download"
        )

        assert video.channel == "Test Channel"
        assert video.upload_date == "2024-01-01"
        assert video.duration_tier == "short"
        assert video.keyword == "test keyword"
        assert video.source == "download"

    def test_video_face_score(self):
        """Test video face score field."""
        video = DownloadedVideo(
            file="/path/to/video.mp4",
            face_score=0.8
        )

        assert video.face_score == 0.8


class TestAudioDownload:
    """Test AudioDownload dataclass."""

    def test_create_audio_download(self):
        """Test creating audio download with correct field names."""
        audio = AudioDownload(
            file="/path/to/audio.mp3",
            video_id="abc123",
            url="https://youtube.com/watch?v=abc123",
            title="Test Audio"
        )

        assert audio.file == "/path/to/audio.mp3"
        assert audio.video_id == "abc123"
        assert audio.url == "https://youtube.com/watch?v=abc123"
        assert audio.title == "Test Audio"

    def test_audio_with_duration_and_keyword(self):
        """Test audio with duration and keyword."""
        audio = AudioDownload(
            file="/path/to/audio.mp3",
            video_id="abc123",
            duration=180.0,
            keyword="test keyword"
        )

        assert audio.duration == 180.0
        assert audio.keyword == "test keyword"


class TestMatch:
    """Test Match dataclass."""

    def test_create_match(self):
        """Test creating match with correct field names."""
        match = Match(
            segment_index=0,
            video_file="/path/to/video.mp4",
            video_start=10.0,
            video_end=20.0,
            confidence=0.85
        )

        assert match.segment_index == 0
        assert match.video_file == "/path/to/video.mp4"
        assert match.video_start == 10.0
        assert match.video_end == 20.0
        assert match.confidence == 0.85

    def test_match_with_strategy_and_reason(self):
        """Test match with strategy and reason."""
        match = Match(
            segment_index=0,
            video_file="/path/to/video.mp4",
            video_start=0.0,
            video_end=5.0,
            confidence=0.9,
            strategy="primary",
            reason="High similarity score"
        )

        assert match.strategy == "primary"
        assert match.reason == "High similarity score"

    def test_match_face_score(self):
        """Test match face score."""
        match = Match(
            segment_index=0,
            video_file="/path/to/video.mp4",
            video_start=0.0,
            video_end=5.0,
            confidence=0.85,
            face_score=0.7
        )

        assert match.face_score == 0.7


class TestEntityImage:
    """Test EntityImage dataclass."""

    def test_create_entity_image(self):
        """Test creating entity image with correct field names."""
        image = EntityImage(
            entity="Eiffel Tower",
            file="/path/to/eiffel.jpg",
            source_url="https://example.com/image.jpg"
        )

        assert image.entity == "Eiffel Tower"
        assert image.file == "/path/to/eiffel.jpg"
        assert image.source_url == "https://example.com/image.jpg"

    def test_entity_image_dimensions(self):
        """Test entity image with dimensions."""
        image = EntityImage(
            entity="Test Entity",
            file="/path/to/image.jpg",
            width=1920,
            height=1080
        )

        assert image.width == 1920
        assert image.height == 1080


class TestEntityVideo:
    """Test EntityVideo dataclass."""

    def test_create_entity_video(self):
        """Test creating entity video with correct field names."""
        video = EntityVideo(
            entity="Ocean Waves",
            file="/path/to/ocean.mp4",
            source="pexels",
            duration=15.0
        )

        assert video.entity == "Ocean Waves"
        assert video.file == "/path/to/ocean.mp4"
        assert video.source == "pexels"
        assert video.duration == 15.0

    def test_entity_video_sources(self):
        """Test different entity video sources."""
        for source in ["pexels", "pixabay"]:
            video = EntityVideo(
                entity="Test",
                file=f"/path/{source}.mp4",
                source=source
            )

            assert video.source == source


class TestPipelineState:
    """Test PipelineState manager."""

    def test_create_pipeline_state(self):
        """Test creating pipeline state."""
        state = PipelineState()

        assert state is not None
        assert state.voiceover_segments == []
        assert state.downloaded_videos == []
        assert state.matches == []

    def test_pipeline_state_input_fields(self):
        """Test input state fields."""
        state = PipelineState()

        state.voiceover_path = "/path/to/voiceover.srt"
        state.keywords = ["keyword1", "keyword2"]
        state.topic_context = "Technology"

        assert state.voiceover_path == "/path/to/voiceover.srt"
        assert len(state.keywords) == 2
        assert state.topic_context == "Technology"

    def test_pipeline_state_download_fields(self):
        """Test download state fields."""
        state = PipelineState()

        video = DownloadedVideo(file="/path/video.mp4")
        state.downloaded_videos.append(video)

        assert len(state.downloaded_videos) == 1
        assert state.downloaded_videos[0].file == "/path/video.mp4"

    def test_pipeline_state_matching_fields(self):
        """Test matching state fields."""
        state = PipelineState()

        match = Match(
            segment_index=0,
            video_file="/path/video.mp4",
            video_start=0.0,
            video_end=5.0,
            confidence=0.9
        )
        state.matches.append(match)

        assert len(state.matches) == 1
        assert state.matches[0].segment_index == 0

    def test_pipeline_state_helper_methods(self):
        """Test pipeline state helper methods."""
        state = PipelineState()

        # Add data
        state.voiceover_segments.append(
            VoiceoverSegment(index=0, start=0.0, end=5.0, text="Test")
        )
        state.downloaded_videos.append(
            DownloadedVideo(file="/path/video.mp4")
        )
        state.matches.append(
            Match(segment_index=0, video_file="/path/video.mp4",
                  video_start=0.0, video_end=5.0, confidence=0.9)
        )

        assert state.get_segment_count() == 1
        assert state.get_video_count() == 1
        assert state.get_match_count() == 1

    def test_pipeline_state_clear_downloads(self):
        """Test clearing download state."""
        state = PipelineState()

        # Add downloads
        state.downloaded_videos.append(
            DownloadedVideo(file="/path/video.mp4")
        )
        state.downloaded_audio.append(
            AudioDownload(file="/path/audio.mp3", video_id="abc123")
        )

        # Clear
        state.clear_downloads()

        assert len(state.downloaded_videos) == 0
        assert len(state.downloaded_audio) == 0

    def test_pipeline_state_entity_media(self):
        """Test entity media state."""
        state = PipelineState()

        image = EntityImage(entity="Test", file="/path/image.jpg")
        video = EntityVideo(entity="Test", file="/path/video.mp4")

        state.entity_images["Test"] = image
        state.entity_videos["Test"] = video

        assert "Test" in state.entity_images
        assert "Test" in state.entity_videos

    def test_pipeline_state_runtime_fields(self):
        """Test runtime state fields."""
        state = PipelineState()

        state.face_preference = "more"
        state.stage_timings["DOWNLOAD"] = 45.2

        assert state.face_preference == "more"
        assert state.stage_timings["DOWNLOAD"] == 45.2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
