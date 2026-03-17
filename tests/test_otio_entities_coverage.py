"""
Tests for src/otio/entities.py edge cases.

Targets:
- Lines 103-104: Exception handling in semantic matching
- Lines 267, 274-275: Image file not found handling
- Line 297: Video duration fallback when ffprobe fails
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import sys


@pytest.fixture(autouse=True)
def mock_opentimelineio():
    """Mock opentimelineio module for this test file only."""
    original_otio = sys.modules.get('opentimelineio')

    mock_otio = MagicMock()
    mock_otio.schema = MagicMock()
    mock_otio.opentime = MagicMock()
    sys.modules['opentimelineio'] = mock_otio

    yield mock_otio

    # Restore original state
    if original_otio is not None:
        sys.modules['opentimelineio'] = original_otio
    else:
        # Remove the mock if opentimelineio wasn't originally loaded
        if 'opentimelineio' in sys.modules:
            del sys.modules['opentimelineio']


@dataclass
class MockVideoSegment:
    """Mock video segment for testing."""
    source_file: str
    start_time: float
    end_time: float
    text: str = ""


@dataclass
class MockVoiceoverSegment:
    """Mock voiceover segment for testing."""
    start_time: float
    end_time: float
    text: str


@dataclass
class MockMatch:
    """Mock match for testing."""
    video_segment: MockVideoSegment
    voiceover_segment: MockVoiceoverSegment
    confidence: float = 0.85


@dataclass
class MockMatchResult:
    """Mock match result for testing."""
    primary_match: MockMatch


@dataclass
class MockEntityResult:
    """Mock entity result with images or videos."""
    images: List[str] = field(default_factory=list)
    videos: List[str] = field(default_factory=list)
    entity_type: str = "PERSON"
    query: str = ""


@dataclass
class MockImageSearchConfig:
    """Mock image search config."""
    enable_sticky_matching: bool = False
    semantic_match_threshold: float = 0.15


@dataclass
class MockConfig:
    """Mock config for testing."""
    image_search: MockImageSearchConfig = field(default_factory=MockImageSearchConfig)


class TestFindBestEntityMatch:
    """Test _find_best_entity_match function."""

    @pytest.mark.fast
    def test_exact_match_found(self):
        """Test exact match when entity name in voiceover."""
        from src.otio.entities import _find_best_entity_match

        entity_data = {
            "John Smith": MockEntityResult(images=["/path/to/image.jpg"], query="John Smith"),
        }

        entity_name, match_type = _find_best_entity_match(
            "we met john smith today",
            entity_data
        )

        assert entity_name == "John Smith"
        assert match_type == "exact"

    @pytest.mark.fast
    def test_semantic_match_found(self):
        """Test semantic match via word overlap."""
        from src.otio.entities import _find_best_entity_match

        entity_data = {
            "New York City": MockEntityResult(
                images=["/path/to/image.jpg"],
                query="New York skyline",
                entity_type="LOCATION"
            ),
        }

        # "New" and "York" overlap with query
        entity_name, match_type = _find_best_entity_match(
            "the new york skyline is beautiful",
            entity_data,
            semantic_threshold=0.1
        )

        assert entity_name == "New York City"
        assert match_type == "semantic"

    @pytest.mark.fast
    def test_semantic_match_exception_handling(self):
        """Test exception handling in semantic matching (lines 103-104)."""
        from src.otio.entities import _find_best_entity_match

        # Create entity that will cause exception in semantic matching
        class BadEntityResult:
            @property
            def images(self):
                raise ValueError("Broken property")

            @property
            def query(self):
                raise ValueError("Broken property")

        entity_data = {
            "Broken Entity": BadEntityResult(),
        }

        with patch('src.otio.entities.logger') as mock_logger:
            entity_name, match_type = _find_best_entity_match(
                "some text",
                entity_data
            )

            assert entity_name is None
            assert match_type == "none"

    @pytest.mark.fast
    def test_sticky_fallback(self):
        """Test sticky entity fallback."""
        from src.otio.entities import _find_best_entity_match

        entity_data = {
            "Previous Entity": MockEntityResult(images=["/path/to/image.jpg"]),
        }

        entity_name, match_type = _find_best_entity_match(
            "completely unrelated text",
            entity_data,
            last_matched_entity="Previous Entity",
            enable_sticky=True
        )

        assert entity_name == "Previous Entity"
        assert match_type == "sticky"

    @pytest.mark.fast
    def test_no_match_found(self):
        """Test when no entity matches."""
        from src.otio.entities import _find_best_entity_match

        entity_data = {
            "Random Entity": MockEntityResult(images=["/path/to/image.jpg"]),
        }

        entity_name, match_type = _find_best_entity_match(
            "completely different unrelated words",
            entity_data,
            enable_sticky=False
        )

        assert entity_name is None
        assert match_type == "none"

    @pytest.mark.fast
    def test_entity_with_no_assets(self):
        """Test entity with no images or videos is skipped."""
        from src.otio.entities import _find_best_entity_match

        entity_data = {
            "Empty Entity": MockEntityResult(images=[], videos=[]),
        }

        entity_name, match_type = _find_best_entity_match(
            "empty entity test",
            entity_data
        )

        assert entity_name is None


class TestGetVideoDurationFrames:
    """Test _get_video_duration_frames function."""

    @pytest.mark.integration
    def test_ffprobe_success(self):
        """Test successful ffprobe duration extraction."""
        from src.otio.entities import _get_video_duration_frames

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="10.5\n"
            )

            frames = _get_video_duration_frames("/path/to/video.mp4", 24.0)

            assert frames == 252  # 10.5 * 24

    @pytest.mark.integration
    def test_ffprobe_failure_returns_none(self):
        """Test ffprobe failure returns None (line 297 fallback trigger)."""
        from src.otio.entities import _get_video_duration_frames

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout=""
            )

            frames = _get_video_duration_frames("/path/to/video.mp4", 24.0)

            assert frames is None

    @pytest.mark.integration
    def test_ffprobe_exception_returns_none(self):
        """Test ffprobe exception returns None."""
        from src.otio.entities import _get_video_duration_frames

        with patch('subprocess.run', side_effect=Exception("ffprobe not found")):
            frames = _get_video_duration_frames("/path/to/video.mp4", 24.0)

            assert frames is None

    @pytest.mark.integration
    def test_ffprobe_empty_output(self):
        """Test ffprobe with empty output."""
        from src.otio.entities import _get_video_duration_frames

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=""
            )

            frames = _get_video_duration_frames("/path/to/video.mp4", 24.0)

            assert frames is None


class TestAddEntityMediaToTrack:
    """Test add_entity_media_to_track function."""

    @pytest.mark.fast
    def test_image_file_not_found_warning(self, tmp_path):
        """Test warning when image file doesn't exist (lines 273-275)."""
        from src.otio.entities import add_entity_media_to_track

        # Create mock track
        mock_track = MagicMock()
        mock_track.append = MagicMock()

        # Create match
        vo_seg = MockVoiceoverSegment(start_time=0, end_time=5, text="john smith test")
        vid_seg = MockVideoSegment(source_file="/path/to/video.mp4", start_time=0, end_time=10)
        match = MockMatch(video_segment=vid_seg, voiceover_segment=vo_seg)
        match_result = MockMatchResult(primary_match=match)

        # Entity with non-existent image
        entity_data = {
            "John Smith": MockEntityResult(
                images=[str(tmp_path / "nonexistent.jpg")],
                query="John Smith"
            ),
        }

        config = MockConfig()

        with patch('src.otio.entities.logger') as mock_logger:
            add_entity_media_to_track(
                mock_track,
                entity_data,
                [match_result],
                24.0,
                config,
                "images"
            )

            # Should have logged warning about missing file
            mock_logger.warning.assert_called()

    @pytest.mark.fast
    def test_video_duration_fallback(self, tmp_path):
        """Test video duration fallback when ffprobe fails (line 297)."""
        from src.otio.entities import add_entity_media_to_track

        # Create a real video file (just for path existence)
        video_file = tmp_path / "test_video.mp4"
        video_file.touch()

        mock_track = MagicMock()
        mock_track.append = MagicMock()

        vo_seg = MockVoiceoverSegment(start_time=0, end_time=5, text="john smith test")
        vid_seg = MockVideoSegment(source_file="/path/to/video.mp4", start_time=0, end_time=10)
        match = MockMatch(video_segment=vid_seg, voiceover_segment=vo_seg)
        match_result = MockMatchResult(primary_match=match)

        entity_data = {
            "John Smith": MockEntityResult(
                videos=[str(video_file)],
                query="John Smith"
            ),
        }

        config = MockConfig()

        # Mock ffprobe to fail
        with patch('src.otio.entities._get_video_duration_frames', return_value=None):
            add_entity_media_to_track(
                mock_track,
                entity_data,
                [match_result],
                24.0,
                config,
                "videos"
            )

            # Should have added clip even with fallback duration
            mock_track.append.assert_called()

    @pytest.mark.fast
    def test_no_entity_match_adds_gap(self):
        """Test that no entity match adds gap."""
        from src.otio.entities import add_entity_media_to_track

        mock_track = MagicMock()
        mock_track.append = MagicMock()

        vo_seg = MockVoiceoverSegment(start_time=0, end_time=5, text="unrelated text")
        vid_seg = MockVideoSegment(source_file="/path/to/video.mp4", start_time=0, end_time=10)
        match = MockMatch(video_segment=vid_seg, voiceover_segment=vo_seg)
        match_result = MockMatchResult(primary_match=match)

        entity_data = {
            "Random Entity": MockEntityResult(
                images=["/path/to/image.jpg"],
                query="completely different"
            ),
        }

        config = MockConfig()

        add_entity_media_to_track(
            mock_track,
            entity_data,
            [match_result],
            24.0,
            config,
            "images"
        )

        # Should have added gap
        mock_track.append.assert_called()

    @pytest.mark.fast
    def test_deduplication_of_media(self, tmp_path):
        """Test that duplicate media files are deduplicated."""
        from src.otio.entities import add_entity_media_to_track

        # Create real files
        image1 = tmp_path / "image1.jpg"
        image1.touch()

        mock_track = MagicMock()
        mock_track.append = MagicMock()

        vo_seg = MockVoiceoverSegment(start_time=0, end_time=5, text="john smith test")
        vid_seg = MockVideoSegment(source_file="/path/to/video.mp4", start_time=0, end_time=10)
        match = MockMatch(video_segment=vid_seg, voiceover_segment=vo_seg)
        match_result = MockMatchResult(primary_match=match)

        # Entity with duplicate images (same filename)
        entity_data = {
            "John Smith": MockEntityResult(
                images=[str(image1), str(image1), str(image1)],
                query="John Smith"
            ),
        }

        config = MockConfig()

        add_entity_media_to_track(
            mock_track,
            entity_data,
            [match_result],
            24.0,
            config,
            "images"
        )

        # Should only add one clip (duplicates removed)
        assert mock_track.append.call_count == 1


class TestBackwardCompatibility:
    """Test backward compatibility wrapper functions."""

    @pytest.mark.fast
    def test_add_entity_images_wrapper(self):
        """Test _add_entity_images_to_track wrapper."""
        from src.otio.entities import _add_entity_images_to_track

        mock_track = MagicMock()
        config = MockConfig()

        with patch('src.otio.entities.add_entity_media_to_track') as mock_add:
            _add_entity_images_to_track(mock_track, {}, [], 24.0, config)

            mock_add.assert_called_once_with(
                mock_track, {}, [], 24.0, config, "images", 1.0
            )

    @pytest.mark.fast
    def test_add_entity_videos_wrapper(self):
        """Test _add_entity_videos_to_track wrapper."""
        from src.otio.entities import _add_entity_videos_to_track

        mock_track = MagicMock()
        config = MockConfig()

        with patch('src.otio.entities.add_entity_media_to_track') as mock_add:
            _add_entity_videos_to_track(mock_track, {}, [], 24.0, config)

            mock_add.assert_called_once_with(
                mock_track, {}, [], 24.0, config, "videos", 1.0
            )
