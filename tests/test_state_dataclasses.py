"""
Unit tests for state dataclasses.

Tests all pipeline state dataclasses with correct field names.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock
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


class TestTranscriptSegmentToDict:
    """Test TranscriptSegment to_dict method."""

    def test_to_dict_basic(self):
        """Test basic to_dict conversion."""
        segment = TranscriptSegment(
            index=0,
            start_time=0.0,
            end_time=5.0,
            text="Test text"
        )

        # Note: to_dict uses asdict from dataclasses
        from dataclasses import asdict
        result = asdict(segment)

        assert isinstance(result, dict)
        assert result['index'] == 0
        assert result['start_time'] == 0.0
        assert result['end_time'] == 5.0
        assert result['text'] == "Test text"
        assert result['source_file'] == ""
        assert result['is_broll'] is False
        assert result['description_source'] == ""

    def test_to_dict_with_all_fields(self):
        """Test to_dict with all fields populated."""
        segment = TranscriptSegment(
            index=5,
            start_time=10.5,
            end_time=25.5,
            text="Vision-generated description",
            source_file="/path/to/video.mp4",
            is_broll=True,
            description_source="vision"
        )

        from dataclasses import asdict
        result = asdict(segment)

        assert result['index'] == 5
        assert result['start_time'] == 10.5
        assert result['end_time'] == 25.5
        assert result['text'] == "Vision-generated description"
        assert result['source_file'] == "/path/to/video.mp4"
        assert result['is_broll'] is True
        assert result['description_source'] == "vision"


class TestPipelineStateClearMatches:
    """Test clear_matches method."""

    def test_clear_matches_empty(self):
        """Test clearing matches when already empty."""
        state = PipelineState()
        assert state.matches == []
        assert state.alternatives == {}

        state.clear_matches()

        assert state.matches == []
        assert state.alternatives == {}

    def test_clear_matches_with_data(self):
        """Test clearing matches when data exists."""
        state = PipelineState()

        # Add matches
        match1 = Match(
            segment_index=0,
            video_file="/path/video1.mp4",
            video_start=0.0,
            video_end=5.0,
            confidence=0.9
        )
        match2 = Match(
            segment_index=1,
            video_file="/path/video2.mp4",
            video_start=5.0,
            video_end=10.0,
            confidence=0.85
        )
        state.matches = [match1, match2]

        # Add alternatives
        alt_match = Match(
            segment_index=0,
            video_file="/path/video3.mp4",
            video_start=10.0,
            video_end=15.0,
            confidence=0.75
        )
        state.alternatives = {0: [alt_match]}

        assert len(state.matches) == 2
        assert len(state.alternatives) == 1

        # Clear
        state.clear_matches()

        assert state.matches == []
        assert state.alternatives == {}


class TestPipelineStateFromLegacy:
    """Test from_legacy_pipeline class method."""

    def test_from_legacy_basic(self):
        """Test creating PipelineState from legacy pipeline object."""
        # Create mock legacy pipeline
        legacy = Mock()
        legacy.keywords = ["beach", "ocean", "sunset"]
        legacy.topic_context = "Travel Photography"
        legacy.extracted_entities = [{"name": "Eiffel Tower", "type": "landmark"}]
        legacy.failed_keywords = ["mountain"]
        legacy.face_preference = "more"
        legacy.stage_timings = {"DOWNLOAD": 45.2, "TRANSCRIBE": 120.5}
        legacy.voiceover_segments = []
        legacy.downloaded_videos = []
        legacy.transcripts = {}
        legacy.embeddings = []
        legacy.text_metadata = []
        legacy.embedding_index = None
        legacy.matches = []
        legacy.entity_images = {}
        legacy.entity_videos = {}

        state = PipelineState.from_legacy_pipeline(legacy)

        assert state.keywords == ["beach", "ocean", "sunset"]
        assert state.topic_context == "Travel Photography"
        assert len(state.extracted_entities) == 1
        assert state.failed_keywords == ["mountain"]
        assert state.face_preference == "more"
        assert state.stage_timings["DOWNLOAD"] == 45.2

    def test_from_legacy_voiceover_segments_dict(self):
        """Test converting voiceover segments from dicts."""
        legacy = Mock()
        legacy.keywords = []
        legacy.topic_context = ""
        legacy.extracted_entities = []
        legacy.failed_keywords = []
        legacy.face_preference = "neutral"
        legacy.stage_timings = {}
        legacy.voiceover_segments = [
            {"index": 0, "start": 0.0, "end": 5.0, "text": "First segment"},
            {"index": 1, "start": 5.0, "end": 10.0, "text": "Second segment"}
        ]
        legacy.downloaded_videos = []
        legacy.transcripts = {}
        legacy.embeddings = []
        legacy.text_metadata = []
        legacy.embedding_index = None
        legacy.matches = []
        legacy.entity_images = {}
        legacy.entity_videos = {}

        state = PipelineState.from_legacy_pipeline(legacy)

        assert len(state.voiceover_segments) == 2
        assert state.voiceover_segments[0].index == 0
        assert state.voiceover_segments[0].start == 0.0
        assert state.voiceover_segments[0].text == "First segment"
        assert state.voiceover_segments[1].index == 1
        assert state.voiceover_segments[1].end == 10.0

    def test_from_legacy_voiceover_segments_objects(self):
        """Test converting voiceover segments when already objects."""
        legacy = Mock()
        legacy.keywords = []
        legacy.topic_context = ""
        legacy.extracted_entities = []
        legacy.failed_keywords = []
        legacy.face_preference = "neutral"
        legacy.stage_timings = {}

        # Pre-created VoiceoverSegment objects
        seg = VoiceoverSegment(index=0, start=0.0, end=5.0, text="Test")
        legacy.voiceover_segments = [seg]

        legacy.downloaded_videos = []
        legacy.transcripts = {}
        legacy.embeddings = []
        legacy.text_metadata = []
        legacy.embedding_index = None
        legacy.matches = []
        legacy.entity_images = {}
        legacy.entity_videos = {}

        state = PipelineState.from_legacy_pipeline(legacy)

        assert len(state.voiceover_segments) == 1
        assert state.voiceover_segments[0] is seg

    def test_from_legacy_downloaded_videos_dict(self):
        """Test converting downloaded videos from dicts."""
        legacy = Mock()
        legacy.keywords = []
        legacy.topic_context = ""
        legacy.extracted_entities = []
        legacy.failed_keywords = []
        legacy.face_preference = "neutral"
        legacy.stage_timings = {}
        legacy.voiceover_segments = []
        legacy.downloaded_videos = [
            {"file": "video1.mp4", "url": "https://example.com/1", "title": "Beach Video",
             "channel": "TravelCh", "duration": 120.0, "duration_tier": "medium",
             "keyword": "beach", "source": "download"},
            {"path": "video2.mp4", "tier": "short"}  # Alt field names
        ]
        legacy.transcripts = {}
        legacy.embeddings = []
        legacy.text_metadata = []
        legacy.embedding_index = None
        legacy.matches = []
        legacy.entity_images = {}
        legacy.entity_videos = {}

        state = PipelineState.from_legacy_pipeline(legacy)

        assert len(state.downloaded_videos) == 2
        assert state.downloaded_videos[0].file == "video1.mp4"
        assert state.downloaded_videos[0].channel == "TravelCh"
        assert state.downloaded_videos[1].file == "video2.mp4"  # 'path' -> 'file'
        assert state.downloaded_videos[1].duration_tier == "short"  # 'tier' -> 'duration_tier'

    def test_from_legacy_downloaded_videos_objects(self):
        """Test converting downloaded videos when already objects."""
        legacy = Mock()
        legacy.keywords = []
        legacy.topic_context = ""
        legacy.extracted_entities = []
        legacy.failed_keywords = []
        legacy.face_preference = "neutral"
        legacy.stage_timings = {}
        legacy.voiceover_segments = []

        video = DownloadedVideo(file="video.mp4", url="https://example.com", source="download")
        legacy.downloaded_videos = [video]

        legacy.transcripts = {}
        legacy.embeddings = []
        legacy.text_metadata = []
        legacy.embedding_index = None
        legacy.matches = []
        legacy.entity_images = {}
        legacy.entity_videos = {}

        state = PipelineState.from_legacy_pipeline(legacy)

        assert len(state.downloaded_videos) == 1
        assert state.downloaded_videos[0] is video

    def test_from_legacy_matches_dict(self):
        """Test converting matches from dicts."""
        legacy = Mock()
        legacy.keywords = []
        legacy.topic_context = ""
        legacy.extracted_entities = []
        legacy.failed_keywords = []
        legacy.face_preference = "neutral"
        legacy.stage_timings = {}
        legacy.voiceover_segments = []
        legacy.downloaded_videos = []
        legacy.transcripts = {}
        legacy.embeddings = []
        legacy.text_metadata = []
        legacy.embedding_index = None
        legacy.matches = [
            {"segment_index": 0, "video_file": "video.mp4", "video_start": 10.0,
             "video_end": 15.0, "confidence": 0.9, "strategy": "primary", "reason": "Good match"},
            {"vo_index": 1, "file": "video2.mp4", "start": 20.0, "end": 25.0, "confidence": 0.8}  # Alt field names
        ]
        legacy.entity_images = {}
        legacy.entity_videos = {}

        state = PipelineState.from_legacy_pipeline(legacy)

        assert len(state.matches) == 2
        assert state.matches[0].segment_index == 0
        assert state.matches[0].video_file == "video.mp4"
        assert state.matches[0].strategy == "primary"
        assert state.matches[1].segment_index == 1  # 'vo_index' -> 'segment_index'
        assert state.matches[1].video_file == "video2.mp4"  # 'file' -> 'video_file'

    def test_from_legacy_matches_objects(self):
        """Test converting matches when already objects."""
        legacy = Mock()
        legacy.keywords = []
        legacy.topic_context = ""
        legacy.extracted_entities = []
        legacy.failed_keywords = []
        legacy.face_preference = "neutral"
        legacy.stage_timings = {}
        legacy.voiceover_segments = []
        legacy.downloaded_videos = []
        legacy.transcripts = {}
        legacy.embeddings = []
        legacy.text_metadata = []
        legacy.embedding_index = None

        match_obj = Match(segment_index=0, video_file="video.mp4",
                         video_start=0.0, video_end=5.0, confidence=0.9)
        legacy.matches = [match_obj]

        legacy.entity_images = {}
        legacy.entity_videos = {}

        state = PipelineState.from_legacy_pipeline(legacy)

        assert len(state.matches) == 1
        assert state.matches[0] is match_obj

    def test_from_legacy_entity_media(self):
        """Test copying entity media dicts."""
        legacy = Mock()
        legacy.keywords = []
        legacy.topic_context = ""
        legacy.extracted_entities = []
        legacy.failed_keywords = []
        legacy.face_preference = "neutral"
        legacy.stage_timings = {}
        legacy.voiceover_segments = []
        legacy.downloaded_videos = []
        legacy.transcripts = {"video1": [{"text": "test"}]}
        legacy.embeddings = [[0.1, 0.2, 0.3]]
        legacy.text_metadata = [{"source": "video1"}]
        legacy.embedding_index = Mock()
        legacy.matches = []

        img = EntityImage(entity="Tower", file="tower.jpg")
        vid = EntityVideo(entity="Tower", file="tower.mp4", source="pexels")
        legacy.entity_images = {"Tower": img}
        legacy.entity_videos = {"Tower": vid}

        state = PipelineState.from_legacy_pipeline(legacy)

        assert "Tower" in state.entity_images
        assert "Tower" in state.entity_videos
        assert state.entity_images["Tower"] is img
        assert state.entity_videos["Tower"] is vid
        assert state.transcripts == {"video1": [{"text": "test"}]}
        assert len(state.embeddings) == 1
        assert state.embedding_index is legacy.embedding_index

    def test_from_legacy_missing_attributes(self):
        """Test handling missing attributes gracefully."""
        # Create minimal mock with getattr defaults
        legacy = Mock(spec=[])  # Empty spec means no attributes
        # Most attributes will return Mock objects when accessed
        # The from_legacy_pipeline should use getattr with defaults

        state = PipelineState.from_legacy_pipeline(legacy)

        # Should create valid state with defaults
        assert state is not None
        assert isinstance(state.keywords, list)
        assert isinstance(state.voiceover_segments, list)


class TestPipelineStateToCheckpointDict:
    """Test to_checkpoint_dict method."""

    def test_to_checkpoint_dict_empty(self):
        """Test checkpoint dict from empty state."""
        state = PipelineState()

        result = state.to_checkpoint_dict()

        assert isinstance(result, dict)
        assert result['voiceover_path'] == ""
        assert result['keywords'] == []
        assert result['topic_context'] == ""
        assert result['segment_count'] == 0
        assert result['video_count'] == 0
        assert result['match_count'] == 0
        assert result['stage_timings'] == {}

    def test_to_checkpoint_dict_with_data(self):
        """Test checkpoint dict with data."""
        state = PipelineState()
        state.voiceover_path = "/path/to/voiceover.srt"
        state.keywords = ["beach", "ocean"]
        state.topic_context = "Travel"
        state.voiceover_segments = [
            VoiceoverSegment(index=0, start=0.0, end=5.0, text="Test")
        ]
        state.downloaded_videos = [
            DownloadedVideo(file="video.mp4")
        ]
        state.matches = [
            Match(segment_index=0, video_file="video.mp4",
                  video_start=0.0, video_end=5.0, confidence=0.9)
        ]
        state.stage_timings = {"DOWNLOAD": 45.2}

        result = state.to_checkpoint_dict()

        assert result['voiceover_path'] == "/path/to/voiceover.srt"
        assert result['keywords'] == ["beach", "ocean"]
        assert result['topic_context'] == "Travel"
        assert result['segment_count'] == 1
        assert result['video_count'] == 1
        assert result['match_count'] == 1
        assert result['stage_timings'] == {"DOWNLOAD": 45.2}


# ============================================================================
# Coverage Tests - Lines 18-20, 50
# ============================================================================

class TestTranscriptSegmentToDict:
    """Test TranscriptSegment.to_dict() method (line 50)."""

    def test_to_dict_returns_dict(self):
        """Test to_dict converts segment to dictionary."""
        from dataclasses import asdict
        segment = TranscriptSegment(
            index=5,
            start_time=10.5,
            end_time=25.3,
            text="Test transcript text",
            source_file="video.mp4",
            is_broll=True,
            description_source="vision"
        )

        result = segment.to_dict()

        assert isinstance(result, dict)
        assert result['index'] == 5
        assert result['start_time'] == 10.5
        assert result['end_time'] == 25.3
        assert result['text'] == "Test transcript text"
        assert result['source_file'] == "video.mp4"
        assert result['is_broll'] is True
        assert result['description_source'] == "vision"

    def test_to_dict_with_defaults(self):
        """Test to_dict with default field values."""
        segment = TranscriptSegment(
            index=0,
            start_time=0.0,
            end_time=5.0,
            text="Default test"
        )

        result = segment.to_dict()

        assert result['is_broll'] is False
        assert result['description_source'] == ""
        assert result['source_file'] == ""


class TestNumpyImportFallback:
    """Test numpy import fallback (lines 18-20)."""

    def test_has_numpy_flag_exists(self):
        """Test HAS_NUMPY flag is set correctly when numpy is available."""
        # This tests the import path when numpy IS available
        from src import state
        # If numpy is installed, HAS_NUMPY should be True
        import importlib.util
        numpy_available = importlib.util.find_spec("numpy") is not None

        if numpy_available:
            assert state.HAS_NUMPY is True
            assert state.np is not None
        else:
            # If numpy not installed, test the fallback path
            assert state.HAS_NUMPY is False
            assert state.np is None

    def test_state_works_without_numpy_dependency(self):
        """Test state module doesn't require numpy for basic operations."""
        # The state module should work even if numpy operations aren't used
        segment = TranscriptSegment(
            index=0,
            start_time=0.0,
            end_time=5.0,
            text="No numpy needed"
        )
        assert segment.text == "No numpy needed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
