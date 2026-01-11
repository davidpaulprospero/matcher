"""
Tests for src/otio/xml_export.py

Tests DaVinci Resolve XML generation with media bins and timeline sequences.
"""

import pytest
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock, patch

from src.otio.xml_export import generate_resolve_xml_with_bins, _write_media_xml_part
from src.utils import Match, MatchResult, SRTSegment


@pytest.fixture
def mock_matches():
    """Create mock MatchResult objects with realistic segments"""
    vo_seg1 = SRTSegment(
        index=0, start_time=0.0, end_time=3.0,
        text="First voiceover segment", source_file="voiceover.srt"
    )
    vid_seg1 = SRTSegment(
        index=0, start_time=10.0, end_time=13.0,
        text="Video clip 1", source_file="/videos/clip1.mp4"
    )
    match1 = Match(
        voiceover_segment=vo_seg1,
        video_segment=vid_seg1,
        video_scene=None,
        confidence=0.9,
        reasoning='High confidence match'
    )

    vo_seg2 = SRTSegment(
        index=1, start_time=3.0, end_time=6.0,
        text="Second voiceover segment", source_file="voiceover.srt"
    )
    vid_seg2 = SRTSegment(
        index=1, start_time=20.0, end_time=23.0,
        text="Video clip 2", source_file="/videos/clip2.mp4"
    )
    match2 = Match(
        voiceover_segment=vo_seg2,
        video_segment=vid_seg2,
        video_scene=None,
        confidence=0.8,
        reasoning='Good match'
    )

    return [
        MatchResult(primary_match=match1, alternatives=[], secondary_matches=[], strategy_matches=[]),
        MatchResult(primary_match=match2, alternatives=[], secondary_matches=[], strategy_matches=[])
    ]


@pytest.fixture
def temp_output_path(tmp_path):
    """Create temporary output path for XML generation"""
    return str(tmp_path / "test_output.xml")


class TestGenerateResolveXML:
    """Test generate_resolve_xml_with_bins() function"""

    def test_generates_valid_xml(self, mock_matches, temp_output_path):
        """Test that generated XML is valid and parseable"""
        paths = generate_resolve_xml_with_bins(
            mock_matches,
            temp_output_path,
            frame_rate=30.0
        )

        assert len(paths) >= 1
        project_xml = paths[0]
        assert Path(project_xml).exists()

        # Parse XML to verify structure
        tree = ET.parse(project_xml)
        root = tree.getroot()
        assert root.tag == 'xmeml'
        assert root.get('version') == '4'

    def test_xml_contains_bin_and_timeline(self, mock_matches, temp_output_path):
        """Test that XML contains both media bin and timeline sequence"""
        paths = generate_resolve_xml_with_bins(
            mock_matches,
            temp_output_path,
            frame_rate=30.0
        )

        tree = ET.parse(paths[0])
        root = tree.getroot()

        # Check for bin
        bins = root.findall('.//bin')
        assert len(bins) >= 1

        # Check for sequence
        sequences = root.findall('.//sequence')
        assert len(sequences) >= 1

    def test_xml_includes_all_media_files(self, mock_matches, temp_output_path):
        """Test that all media files are included in the bin"""
        paths = generate_resolve_xml_with_bins(
            mock_matches,
            temp_output_path,
            frame_rate=30.0
        )

        tree = ET.parse(paths[0])
        root = tree.getroot()

        # Find all clips in bin
        clips = root.findall('.//bin/children/clip')
        assert len(clips) >= 2  # At least the 2 video clips

    def test_xml_timeline_has_clips(self, mock_matches, temp_output_path):
        """Test that timeline sequence has clips on V1 track"""
        paths = generate_resolve_xml_with_bins(
            mock_matches,
            temp_output_path,
            frame_rate=30.0
        )

        tree = ET.parse(paths[0])
        root = tree.getroot()

        # Find clips in timeline track
        timeline_clips = root.findall('.//sequence/media/video/track/clipitem')
        assert len(timeline_clips) == 2  # 2 matches

    def test_xml_with_voiceover(self, mock_matches, temp_output_path):
        """Test XML generation with voiceover audio track"""
        paths = generate_resolve_xml_with_bins(
            mock_matches,
            temp_output_path,
            voiceover_path="/audio/voiceover.wav",
            frame_rate=30.0
        )

        tree = ET.parse(paths[0])
        root = tree.getroot()

        # Check for voiceover in bin
        voiceover_clips = [clip for clip in root.findall('.//bin/children/clip')
                          if 'voiceover' in (clip.find('name').text or '').lower()]
        assert len(voiceover_clips) >= 1

        # Check for voiceover in timeline audio track
        audio_clips = root.findall('.//sequence/media/audio/track/clipitem')
        assert len(audio_clips) >= 1

    def test_xml_with_entity_images(self, mock_matches, temp_output_path):
        """Test XML generation with entity images"""
        # Create mock entity images with string paths (not Mock objects)
        entity_images = {
            'Entity1': Mock(
                images=["/images/entity1_img1.jpg", "/images/entity1_img2.jpg"],
                entity_type='PERSON'
            )
        }

        # Mock _validate_entity_images to return the same structure
        with patch('src.otio.xml_export._validate_entity_images') as mock_validate:
            mock_validate.return_value = entity_images

            paths = generate_resolve_xml_with_bins(
                mock_matches,
                temp_output_path,
                entity_images=entity_images,
                frame_rate=30.0
            )

        tree = ET.parse(paths[0])
        root = tree.getroot()

        # Check that entity images are in bin
        image_clips = [clip for clip in root.findall('.//bin/children/clip')
                      if clip.find('.//pathurl') is not None and
                      '.jpg' in (clip.find('.//pathurl').text or '')]
        assert len(image_clips) >= 1

    def test_xml_with_entity_videos(self, mock_matches, temp_output_path):
        """Test XML generation with stock videos"""
        # Create mock entity videos with string paths (not Mock objects)
        entity_videos = {
            'Entity1': Mock(
                videos=["/stock/video1.mp4", "/stock/video2.mp4"],
                entity_type='ORG'
            )
        }

        paths = generate_resolve_xml_with_bins(
            mock_matches,
            temp_output_path,
            entity_videos=entity_videos,
            frame_rate=30.0
        )

        tree = ET.parse(paths[0])
        root = tree.getroot()

        # Check that stock videos are in bin
        stock_clips = [clip for clip in root.findall('.//bin/children/clip')
                      if clip.find('.//pathurl') is not None and
                      'stock' in (clip.find('.//pathurl').text or '')]
        assert len(stock_clips) >= 1

    def test_xml_with_alternatives(self, mock_matches, temp_output_path):
        """Test XML generation includes alternative matches in bin"""
        # Add alternatives to first match
        alt_seg = SRTSegment(
            index=0, start_time=5.0, end_time=8.0,
            text="Alternative clip", source_file="/videos/alt1.mp4"
        )
        alt_match = Match(
            voiceover_segment=mock_matches[0].primary_match.voiceover_segment,
            video_segment=alt_seg,
            video_scene=None,
            confidence=0.75,
            reasoning='Alternative match'
        )
        mock_matches[0].alternatives.append(alt_match)

        paths = generate_resolve_xml_with_bins(
            mock_matches,
            temp_output_path,
            frame_rate=30.0
        )

        tree = ET.parse(paths[0])
        root = tree.getroot()

        # Check that alternative is in bin
        clips = root.findall('.//bin/children/clip')
        assert len(clips) >= 3  # Primary + Alternative + Second primary

    def test_xml_with_secondary_matches(self, mock_matches, temp_output_path):
        """Test XML generation includes secondary diversity matches"""
        sec_seg = SRTSegment(
            index=0, start_time=15.0, end_time=18.0,
            text="Secondary clip", source_file="/videos/secondary1.mp4"
        )
        sec_match = Match(
            voiceover_segment=mock_matches[0].primary_match.voiceover_segment,
            video_segment=sec_seg,
            video_scene=None,
            confidence=0.7,
            reasoning='Secondary diversity match'
        )
        mock_matches[0].secondary_matches.append(sec_match)

        paths = generate_resolve_xml_with_bins(
            mock_matches,
            temp_output_path,
            frame_rate=30.0
        )

        tree = ET.parse(paths[0])
        root = tree.getroot()

        # Check that secondary is in bin
        clips = root.findall('.//bin/children/clip')
        assert len(clips) >= 3  # Primary + Secondary + Second primary

    def test_xml_with_strategy_matches(self, mock_matches, temp_output_path):
        """Test XML generation includes strategy matches (embedding_diversity, broll_only)"""
        strat_seg = SRTSegment(
            index=0, start_time=25.0, end_time=28.0,
            text="B-roll clip", source_file="/videos/broll1.mp4"
        )
        strat_match = Mock(
            video_segment=strat_seg,
            confidence=0.65,
            reasoning='B-roll strategy',
            strategy='broll_only'
        )
        mock_matches[0].strategy_matches.append(strat_match)

        paths = generate_resolve_xml_with_bins(
            mock_matches,
            temp_output_path,
            frame_rate=30.0
        )

        tree = ET.parse(paths[0])
        root = tree.getroot()

        # Check that strategy match is in bin
        clips = root.findall('.//bin/children/clip')
        assert len(clips) >= 3  # Primary + Strategy + Second primary


class TestXMLStructure:
    """Test XML structure and format compliance"""

    def test_xml_has_proper_doctype(self, mock_matches, temp_output_path):
        """Test XML has proper DOCTYPE declaration"""
        paths = generate_resolve_xml_with_bins(
            mock_matches,
            temp_output_path,
            frame_rate=30.0
        )

        with open(paths[0], 'r', encoding='utf-8') as f:
            content = f.read()

        assert '<?xml version="1.0" encoding="UTF-8"?>' in content
        assert '<!DOCTYPE xmeml>' in content

    def test_xml_has_project_structure(self, mock_matches, temp_output_path):
        """Test XML has proper project structure"""
        paths = generate_resolve_xml_with_bins(
            mock_matches,
            temp_output_path,
            frame_rate=30.0
        )

        tree = ET.parse(paths[0])
        root = tree.getroot()

        project = root.find('project')
        assert project is not None
        assert project.find('name') is not None
        assert project.find('children') is not None

    def test_xml_clips_have_unique_names(self, mock_matches, temp_output_path):
        """Test that clips have unique names (folder_filename format)"""
        paths = generate_resolve_xml_with_bins(
            mock_matches,
            temp_output_path,
            frame_rate=30.0
        )

        tree = ET.parse(paths[0])
        root = tree.getroot()

        clip_names = [clip.find('name').text for clip in root.findall('.//bin/children/clip')]
        # Check that names include folder prefix (e.g., "videos_clip1.mp4")
        assert all('_' in name for name in clip_names)

    def test_xml_timeline_clips_have_speed_adjustment(self, mock_matches, temp_output_path):
        """Test that timeline clips have speed adjustment filter when needed"""
        # Modify match to have mismatched source/target duration
        mock_matches[0].primary_match.video_segment.end_time = 16.0  # 6s source vs 3s target

        paths = generate_resolve_xml_with_bins(
            mock_matches,
            temp_output_path,
            frame_rate=30.0
        )

        tree = ET.parse(paths[0])
        root = tree.getroot()

        # Find clips with filters (speed adjustments)
        filters = root.findall('.//sequence/media/video/track/clipitem/filter')
        # First clip should have speed filter (6s -> 3s = 200% speed)
        assert len(filters) >= 1

    def test_xml_timeline_has_timecode(self, mock_matches, temp_output_path):
        """Test that timeline has proper timecode start"""
        paths = generate_resolve_xml_with_bins(
            mock_matches,
            temp_output_path,
            frame_rate=30.0
        )

        tree = ET.parse(paths[0])
        root = tree.getroot()

        timecode = root.find('.//sequence/timecode')
        assert timecode is not None
        assert timecode.find('string') is not None
        assert timecode.find('string').text == '01:00:00:00'  # Default timeline start


class TestXMLSplitting:
    """Test XML file splitting functionality"""

    def test_generates_multiple_parts(self, mock_matches, temp_output_path):
        """Test that num_parts parameter generates multiple XML files"""
        paths = generate_resolve_xml_with_bins(
            mock_matches,
            temp_output_path,
            frame_rate=30.0,
            num_parts=2
        )

        assert len(paths) >= 2  # Project + at least 1 media part

    def test_media_parts_are_valid_xml(self, mock_matches, temp_output_path):
        """Test that media-only XML parts are valid"""
        paths = generate_resolve_xml_with_bins(
            mock_matches,
            temp_output_path,
            frame_rate=30.0,
            num_parts=2
        )

        # Skip project XML, check media parts
        for media_xml in paths[1:]:
            tree = ET.parse(media_xml)
            root = tree.getroot()
            assert root.tag == 'xmeml'

    def test_filename_conflicts_separated(self, mock_matches, temp_output_path):
        """Test that files with same name in different folders get separate XMLs"""
        # Create matches with same filename but different folders
        vo_seg = mock_matches[0].primary_match.voiceover_segment

        vid_seg_a = SRTSegment(
            index=0, start_time=10.0, end_time=13.0,
            text="Clip", source_file="/folder_a/clip.mp4"
        )
        vid_seg_b = SRTSegment(
            index=1, start_time=20.0, end_time=23.0,
            text="Clip", source_file="/folder_b/clip.mp4"
        )

        match_a = Match(
            voiceover_segment=vo_seg,
            video_segment=vid_seg_a,
            video_scene=None,
            confidence=0.9,
            reasoning='Match A'
        )
        match_b = Match(
            voiceover_segment=vo_seg,
            video_segment=vid_seg_b,
            video_scene=None,
            confidence=0.9,
            reasoning='Match B'
        )

        conflicts = [
            MatchResult(primary_match=match_a, alternatives=[], secondary_matches=[], strategy_matches=[]),
            MatchResult(primary_match=match_b, alternatives=[], secondary_matches=[], strategy_matches=[])
        ]

        paths = generate_resolve_xml_with_bins(
            conflicts,
            temp_output_path,
            frame_rate=30.0,
            num_parts=2
        )

        # Should generate separate conflict XMLs
        assert len(paths) >= 3  # Project + unique files + conflicts


class TestWriteMediaXMLPart:
    """Test _write_media_xml_part() helper function"""

    def test_writes_valid_xml(self, tmp_path):
        """Test that media part XML is valid"""
        files_subset = {
            '/videos/test.mp4': {
                'file_id': 'file-1',
                'uuid': 'uuid-1234',
                'duration_frames': 900
            }
        }

        output_path = str(tmp_path / "media_part.xml")
        generated_paths = []

        _write_media_xml_part(
            files_subset,
            part_idx=1,
            output_path=output_path,
            fps_int=30,
            generated_paths=generated_paths,
            logger=Mock()
        )

        assert Path(output_path).exists()
        tree = ET.parse(output_path)
        root = tree.getroot()
        assert root.tag == 'xmeml'

    def test_includes_bin_with_clips(self, tmp_path):
        """Test that media part has bin with clips"""
        files_subset = {
            '/videos/test.mp4': {
                'file_id': 'file-1',
                'uuid': 'uuid-1234',
                'duration_frames': 900
            }
        }

        output_path = str(tmp_path / "media_part.xml")
        generated_paths = []

        _write_media_xml_part(
            files_subset,
            part_idx=1,
            output_path=output_path,
            fps_int=30,
            generated_paths=generated_paths,
            logger=Mock()
        )

        tree = ET.parse(output_path)
        root = tree.getroot()

        bin_node = root.find('bin')
        assert bin_node is not None
        clips = bin_node.findall('.//clip')
        assert len(clips) == 1

    def test_custom_bin_name(self, tmp_path):
        """Test that custom bin name is used"""
        files_subset = {
            '/videos/test.mp4': {
                'file_id': 'file-1',
                'uuid': 'uuid-1234',
                'duration_frames': 900
            }
        }

        output_path = str(tmp_path / "media_part.xml")
        generated_paths = []

        _write_media_xml_part(
            files_subset,
            part_idx=1,
            output_path=output_path,
            fps_int=30,
            generated_paths=generated_paths,
            logger=Mock(),
            bin_name_override="Custom Bin Name"
        )

        tree = ET.parse(output_path)
        root = tree.getroot()

        bin_name = root.find('.//bin/name')
        assert bin_name.text == "Custom Bin Name"
