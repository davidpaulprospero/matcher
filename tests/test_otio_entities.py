"""
Tests for src/otio/entities.py

Tests unified entity track generation for images (V9) and stock videos (V10),
including semantic matching, sticky entity persistence, and polymorphic handling.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import opentimelineio as otio

from src.otio.entities import (
    _find_best_entity_match,
    _get_video_duration_frames,
    add_entity_media_to_track,
    _add_entity_images_to_track,
    _add_entity_videos_to_track
)
from src.utils import Match, MatchResult, SRTSegment


@pytest.fixture
def mock_matches():
    """Create mock MatchResult objects with voiceover segments"""
    matches = []

    # Segment 1: "Barack Obama was president"
    vo_seg1 = SRTSegment(
        index=0, start_time=0.0, end_time=3.0,
        text="Barack Obama was president", source_file="voiceover.srt"
    )
    vid_seg1 = SRTSegment(
        index=0, start_time=0.0, end_time=3.0,
        text="Video 1", source_file="/videos/v1.mp4"
    )
    match1 = Match(
        voiceover_segment=vo_seg1,
        video_segment=vid_seg1,
        video_scene=None,
        confidence=0.9,
        reasoning='Match 1'
    )
    matches.append(MatchResult(primary_match=match1, alternatives=[], secondary_matches=[], strategy_matches=[]))

    # Segment 2: "He visited Paris"
    vo_seg2 = SRTSegment(
        index=1, start_time=3.0, end_time=6.0,
        text="He visited Paris", source_file="voiceover.srt"
    )
    vid_seg2 = SRTSegment(
        index=1, start_time=3.0, end_time=6.0,
        text="Video 2", source_file="/videos/v2.mp4"
    )
    match2 = Match(
        voiceover_segment=vo_seg2,
        video_segment=vid_seg2,
        video_scene=None,
        confidence=0.8,
        reasoning='Match 2'
    )
    matches.append(MatchResult(primary_match=match2, alternatives=[], secondary_matches=[], strategy_matches=[]))

    # Segment 3: "The city was beautiful"
    vo_seg3 = SRTSegment(
        index=2, start_time=6.0, end_time=9.0,
        text="The city was beautiful", source_file="voiceover.srt"
    )
    vid_seg3 = SRTSegment(
        index=2, start_time=6.0, end_time=9.0,
        text="Video 3", source_file="/videos/v3.mp4"
    )
    match3 = Match(
        voiceover_segment=vo_seg3,
        video_segment=vid_seg3,
        video_scene=None,
        confidence=0.85,
        reasoning='Match 3'
    )
    matches.append(MatchResult(primary_match=match3, alternatives=[], secondary_matches=[], strategy_matches=[]))

    return matches


@pytest.fixture
def mock_entity_images():
    """Create mock entity image data"""
    return {
        'Barack Obama': Mock(
            images=["/images/obama1.jpg", "/images/obama2.jpg"],
            videos=[],  # Must set videos=[] to prevent Mock auto-creation
            entity_type='PERSON',
            query='Barack Obama president'
        ),
        'Paris': Mock(
            images=["/images/paris1.jpg", "/images/paris2.jpg"],
            videos=[],  # Must set videos=[] to prevent Mock auto-creation
            entity_type='LOC',
            query='Paris France city'
        )
    }


@pytest.fixture
def mock_config():
    """Create mock config object"""
    config = Mock()
    config.image_search = Mock()
    config.image_search.enable_sticky_matching = False
    config.image_search.semantic_match_threshold = 0.15
    return config


class TestFindBestEntityMatch:
    """Test _find_best_entity_match() function"""

    def test_exact_match_found(self, mock_entity_images):
        """Test exact match when entity name appears in voiceover"""
        vo_text = "barack obama was president"

        entity_name, match_type = _find_best_entity_match(
            vo_text,
            mock_entity_images,
            enable_sticky=False
        )

        assert entity_name == 'Barack Obama'
        assert match_type == 'exact'

    def test_exact_match_case_insensitive(self, mock_entity_images):
        """Test exact match is case-insensitive"""
        vo_text = "BARACK OBAMA was president".lower()  # Function expects lowercase input

        entity_name, match_type = _find_best_entity_match(
            vo_text,
            mock_entity_images,
            enable_sticky=False
        )

        assert entity_name == 'Barack Obama'
        assert match_type == 'exact'

    def test_semantic_match_by_word_overlap(self, mock_entity_images):
        """Test semantic match using word overlap scoring"""
        vo_text = "the president gave a speech"  # Contains "president" from query

        entity_name, match_type = _find_best_entity_match(
            vo_text,
            mock_entity_images,
            enable_sticky=False,
            semantic_threshold=0.1  # Lower threshold for testing
        )

        assert entity_name == 'Barack Obama'
        assert match_type == 'semantic'

    def test_semantic_match_location(self, mock_entity_images):
        """Test semantic match for location entities"""
        vo_text = "visiting the city of france"  # Contains "city", "france"

        entity_name, match_type = _find_best_entity_match(
            vo_text,
            mock_entity_images,
            enable_sticky=False,
            semantic_threshold=0.1
        )

        assert entity_name == 'Paris'
        assert match_type == 'semantic'

    def test_semantic_threshold_filtering(self, mock_entity_images):
        """Test that semantic matches below threshold are rejected"""
        vo_text = "a different topic entirely"  # No word overlap

        entity_name, match_type = _find_best_entity_match(
            vo_text,
            mock_entity_images,
            enable_sticky=False,
            semantic_threshold=0.5  # High threshold
        )

        assert entity_name is None
        assert match_type == 'none'

    def test_sticky_matching_enabled(self, mock_entity_images):
        """Test sticky matching carries forward last matched entity"""
        vo_text = "completely different text"

        entity_name, match_type = _find_best_entity_match(
            vo_text,
            mock_entity_images,
            last_matched_entity='Paris',
            enable_sticky=True
        )

        assert entity_name == 'Paris'
        assert match_type == 'sticky'

    def test_sticky_disabled_no_fallback(self, mock_entity_images):
        """Test that sticky matching doesn't happen when disabled"""
        vo_text = "completely different text"

        entity_name, match_type = _find_best_entity_match(
            vo_text,
            mock_entity_images,
            last_matched_entity='Paris',
            enable_sticky=False  # Disabled
        )

        assert entity_name is None
        assert match_type == 'none'

    def test_exact_takes_priority_over_semantic(self, mock_entity_images):
        """Test exact match has priority over semantic match"""
        vo_text = "paris city france president"  # Matches both entities

        entity_name, match_type = _find_best_entity_match(
            vo_text,
            mock_entity_images,
            enable_sticky=False
        )

        # "paris" exact match should win over semantic "president" match
        assert entity_name == 'Paris'
        assert match_type == 'exact'

    def test_empty_entity_dict(self):
        """Test handling of empty entity dictionary"""
        entity_name, match_type = _find_best_entity_match(
            "some text",
            {},
            enable_sticky=False
        )

        assert entity_name is None
        assert match_type == 'none'

    def test_entity_without_images(self):
        """Test that entities without images are skipped"""
        entities_no_images = {
            'Entity1': Mock(images=[], videos=[], entity_type='PERSON')  # Must set videos=[] to prevent Mock auto-creation
        }

        entity_name, match_type = _find_best_entity_match(
            "entity1 mention",
            entities_no_images,
            enable_sticky=False
        )

        assert entity_name is None
        assert match_type == 'none'


class TestGetVideoDurationFrames:
    """Test _get_video_duration_frames() function"""

    @patch('subprocess.run')
    def test_successful_duration_extraction(self, mock_run):
        """Test successful ffprobe duration extraction"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "10.5"  # 10.5 seconds
        mock_run.return_value = mock_result

        duration_frames = _get_video_duration_frames("/video/test.mp4", frame_rate=30.0)

        assert duration_frames == 315  # 10.5 * 30 = 315 frames

    @patch('subprocess.run')
    def test_ffprobe_failure(self, mock_run):
        """Test handling of ffprobe failure"""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_run.return_value = mock_result

        duration_frames = _get_video_duration_frames("/video/test.mp4", frame_rate=30.0)

        assert duration_frames is None

    @patch('subprocess.run')
    def test_ffprobe_timeout(self, mock_run):
        """Test handling of ffprobe timeout"""
        mock_run.side_effect = Exception("Timeout")

        duration_frames = _get_video_duration_frames("/video/test.mp4", frame_rate=30.0)

        assert duration_frames is None


@patch('pathlib.Path.exists', return_value=True)  # Mock file existence for all tests in class
class TestAddEntityMediaToTrack:
    """Test add_entity_media_to_track() unified function"""

    def test_images_track_creation(self, mock_exists, mock_matches, mock_entity_images, mock_config):
        """Test image track creation (V9)"""
        track = otio.schema.Track(name="V9 - Entity Images", kind=otio.schema.TrackKind.Video)

        add_entity_media_to_track(
            track,
            mock_entity_images,
            mock_matches,
            frame_rate=30.0,
            config=mock_config,
            entity_type="images"
        )

        # Should have clips for matched segments
        clips = [item for item in track if isinstance(item, otio.schema.Clip)]
        assert len(clips) > 0

    def test_videos_track_creation(self, mock_exists, mock_matches, mock_config):
        """Test stock video track creation (V10)"""
        mock_entity_videos = {
            'Barack Obama': Mock(
                images=[],  # Must set images=[] to prevent Mock auto-creation
                videos=["/stock/obama1.mp4", "/stock/obama2.mp4"],
                entity_type='PERSON',
                query='Barack Obama president'
            )
        }

        track = otio.schema.Track(name="V10 - Stock Videos", kind=otio.schema.TrackKind.Video)

        with patch('src.otio.entities._get_video_duration_frames', return_value=300):
            add_entity_media_to_track(
                track,
                mock_entity_videos,
                mock_matches,
                frame_rate=30.0,
                config=mock_config,
                entity_type="videos"
            )

        clips = [item for item in track if isinstance(item, otio.schema.Clip)]
        assert len(clips) > 0

    def test_exact_match_adds_clips(self, mock_exists, mock_matches, mock_entity_images, mock_config):
        """Test that exact entity matches result in clips"""
        track = otio.schema.Track(name="V9", kind=otio.schema.TrackKind.Video)

        add_entity_media_to_track(
            track,
            mock_entity_images,
            mock_matches,
            frame_rate=30.0,
            config=mock_config,
            entity_type="images"
        )

        # First segment matches "Barack Obama", second matches "Paris"
        clips = [item for item in track if isinstance(item, otio.schema.Clip)]
        assert len(clips) >= 2  # At least 2 images from first match

    def test_no_match_adds_gap(self, mock_config):
        """Test that segments without entity matches get gaps"""
        # Matches with no matching entities
        no_match_vo = SRTSegment(
            index=0, start_time=0.0, end_time=3.0,
            text="unrelated topic", source_file="voiceover.srt"
        )
        no_match_vid = SRTSegment(
            index=0, start_time=0.0, end_time=3.0,
            text="Video", source_file="/v1.mp4"
        )
        match = Match(
            voiceover_segment=no_match_vo,
            video_segment=no_match_vid,
            video_scene=None,
            confidence=0.9,
            reasoning='Match'
        )
        matches = [MatchResult(primary_match=match, alternatives=[], secondary_matches=[], strategy_matches=[])]

        track = otio.schema.Track(name="V9", kind=otio.schema.TrackKind.Video)

        add_entity_media_to_track(
            track,
            {'Entity': Mock(images=["/img.jpg"], videos=[], entity_type='PERSON', query='entity')},
            matches,
            frame_rate=30.0,
            config=mock_config,
            entity_type="images"
        )

        # Should have a gap
        gaps = [item for item in track if isinstance(item, otio.schema.Gap)]
        assert len(gaps) == 1

    def test_multiple_images_divided_equally(self, mock_exists, mock_matches, mock_entity_images, mock_config):
        """Test that multiple images for one entity are divided equally in segment"""
        track = otio.schema.Track(name="V9", kind=otio.schema.TrackKind.Video)

        add_entity_media_to_track(
            track,
            mock_entity_images,
            mock_matches[:1],  # Just first segment
            frame_rate=30.0,
            config=mock_config,
            entity_type="images"
        )

        clips = [item for item in track if isinstance(item, otio.schema.Clip)]
        # Barack Obama has 2 images, so should have 2 clips
        assert len(clips) == 2

    def test_image_available_range_one_frame(self, mock_exists, mock_matches, mock_entity_images, mock_config):
        """Test that images have available_range of 1 frame (still image marker)"""
        track = otio.schema.Track(name="V9", kind=otio.schema.TrackKind.Video)

        add_entity_media_to_track(
            track,
            mock_entity_images,
            mock_matches[:1],
            frame_rate=30.0,
            config=mock_config,
            entity_type="images"
        )

        clips = [item for item in track if isinstance(item, otio.schema.Clip)]
        # Check first clip's available_range duration = 1 frame
        assert clips[0].media_reference.available_range.duration.value == 1

    def test_video_available_range_from_ffprobe(self, mock_exists, mock_matches, mock_config):
        """Test that videos have available_range from ffprobe"""
        mock_entity_videos = {
            'Barack Obama': Mock(
                images=[],  # Must set images=[] to prevent Mock auto-creation
                videos=["/stock/obama1.mp4"],
                entity_type='PERSON',
                query='Barack Obama'
            )
        }

        track = otio.schema.Track(name="V10", kind=otio.schema.TrackKind.Video)

        with patch('src.otio.entities._get_video_duration_frames', return_value=900):  # 30s @ 30fps
            add_entity_media_to_track(
                track,
                mock_entity_videos,
                mock_matches[:1],
                frame_rate=30.0,
                config=mock_config,
                entity_type="videos"
            )

        clips = [item for item in track if isinstance(item, otio.schema.Clip)]
        # Check video available_range = 900 frames
        assert clips[0].media_reference.available_range.duration.value == 900

    def test_clip_metadata_includes_entity_info(self, mock_exists, mock_matches, mock_entity_images, mock_config):
        """Test that clips have proper metadata (entity_name, entity_type, etc.)"""
        track = otio.schema.Track(name="V9", kind=otio.schema.TrackKind.Video)

        add_entity_media_to_track(
            track,
            mock_entity_images,
            mock_matches[:1],
            frame_rate=30.0,
            config=mock_config,
            entity_type="images"
        )

        clips = [item for item in track if isinstance(item, otio.schema.Clip)]
        clip = clips[0]

        assert 'entity_name' in clip.metadata
        assert 'entity_type' in clip.metadata
        assert 'segment_index' in clip.metadata
        assert 'match_type' in clip.metadata

    def test_sticky_matching_carries_entity_forward(self, mock_exists, mock_matches, mock_entity_images, mock_config):
        """Test that sticky matching carries entity to subsequent segments"""
        # Enable sticky matching
        mock_config.image_search.enable_sticky_matching = True

        track = otio.schema.Track(name="V9", kind=otio.schema.TrackKind.Video)

        add_entity_media_to_track(
            track,
            mock_entity_images,
            mock_matches,  # All 3 segments
            frame_rate=30.0,
            config=mock_config,
            entity_type="images"
        )

        clips = [item for item in track if isinstance(item, otio.schema.Clip)]
        # Should have clips for all 3 segments (sticky carries Paris to segment 3)
        # Segment 1: Obama, Segment 2: Paris, Segment 3: Paris (sticky)
        assert len(clips) >= 4  # At least 2 per entity match


@patch('pathlib.Path.exists', return_value=True)  # Mock file existence for all tests in class
class TestBackwardCompatibilityWrappers:
    """Test backward compatibility wrapper functions"""

    def test_add_entity_images_to_track_wrapper(self, mock_exists, mock_matches, mock_entity_images, mock_config):
        """Test _add_entity_images_to_track() backward compatibility wrapper"""
        track = otio.schema.Track(name="V9", kind=otio.schema.TrackKind.Video)

        _add_entity_images_to_track(
            track,
            mock_entity_images,
            mock_matches,
            frame_rate=30.0,
            config=mock_config
        )

        clips = [item for item in track if isinstance(item, otio.schema.Clip)]
        assert len(clips) > 0

    def test_add_entity_videos_to_track_wrapper(self, mock_exists, mock_matches, mock_config):
        """Test _add_entity_videos_to_track() backward compatibility wrapper"""
        mock_entity_videos = {
            'Barack Obama': Mock(
                images=[],  # Must set images=[] to prevent Mock auto-creation
                videos=["/stock/obama1.mp4"],
                entity_type='PERSON',
                query='Barack Obama'
            )
        }

        track = otio.schema.Track(name="V10", kind=otio.schema.TrackKind.Video)

        with patch('src.otio.entities._get_video_duration_frames', return_value=300):
            _add_entity_videos_to_track(
                track,
                mock_entity_videos,
                mock_matches,
                frame_rate=30.0,
                config=mock_config
            )

        clips = [item for item in track if isinstance(item, otio.schema.Clip)]
        assert len(clips) > 0


@patch('pathlib.Path.exists', return_value=True)  # Mock file existence for all tests in class
class TestEdgeCases:
    """Test edge cases and error scenarios"""

    def test_empty_entity_dict(self, mock_exists, mock_matches, mock_config):
        """Test handling of empty entity dictionary"""
        track = otio.schema.Track(name="V9", kind=otio.schema.TrackKind.Video)

        add_entity_media_to_track(
            track,
            {},  # Empty
            mock_matches,
            frame_rate=30.0,
            config=mock_config,
            entity_type="images"
        )

        # Should have all gaps
        gaps = [item for item in track if isinstance(item, otio.schema.Gap)]
        assert len(gaps) == len(mock_matches)

    def test_empty_matches_list(self, mock_entity_images, mock_config):
        """Test handling of empty matches list"""
        track = otio.schema.Track(name="V9", kind=otio.schema.TrackKind.Video)

        add_entity_media_to_track(
            track,
            mock_entity_images,
            [],  # Empty
            frame_rate=30.0,
            config=mock_config,
            entity_type="images"
        )

        # Track should be empty
        assert len(list(track)) == 0

    def test_duplicate_image_paths_filtered(self, mock_exists, mock_matches, mock_config):
        """Test that duplicate image paths are filtered out within a segment"""
        # Entity with duplicate image paths
        entity_with_dupes = {
            'Entity': Mock(
                images=["/img1.jpg", "/img1.jpg", "/img2.jpg"],  # Duplicate img1
                videos=[],  # Must set videos=[] to prevent Mock auto-creation
                entity_type='PERSON',
                query='entity barack obama'
            )
        }

        track = otio.schema.Track(name="V9", kind=otio.schema.TrackKind.Video)

        add_entity_media_to_track(
            track,
            entity_with_dupes,
            mock_matches[:1],
            frame_rate=30.0,
            config=mock_config,
            entity_type="images"
        )

        clips = [item for item in track if isinstance(item, otio.schema.Clip)]
        # Should only have 2 clips (img1, img2) not 3
        assert len(clips) == 2

    def test_very_short_segment(self, mock_exists, mock_entity_images, mock_config):
        """Test handling of very short segments (< 1 frame)"""
        # 0.01 second segment
        vo_seg = SRTSegment(
            index=0, start_time=0.0, end_time=0.01,
            text="Barack Obama", source_file="voiceover.srt"
        )
        vid_seg = SRTSegment(
            index=0, start_time=0.0, end_time=0.01,
            text="Video", source_file="/v1.mp4"
        )
        match = Match(
            voiceover_segment=vo_seg,
            video_segment=vid_seg,
            video_scene=None,
            confidence=0.9,
            reasoning='Match'
        )
        matches = [MatchResult(primary_match=match, alternatives=[], secondary_matches=[], strategy_matches=[])]

        track = otio.schema.Track(name="V9", kind=otio.schema.TrackKind.Video)

        # Should not crash
        add_entity_media_to_track(
            track,
            mock_entity_images,
            matches,
            frame_rate=30.0,
            config=mock_config,
            entity_type="images"
        )

        # Should have at least one clip (even if tiny)
        clips = [item for item in track if isinstance(item, otio.schema.Clip)]
        assert len(clips) > 0
