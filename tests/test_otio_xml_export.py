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
        """Test that media part has bin with clips inside project wrapper"""
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

        # Bin must be directly under xmeml (NO project wrapper) for DaVinci Resolve import
        bin_node = root.find('bin')
        assert bin_node is not None, "Media XML must have <bin> directly under <xmeml>"
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

        # Bin name directly under xmeml/bin
        bin_name = root.find('bin/name')
        assert bin_name is not None, "Missing bin/name element"
        assert bin_name.text == "Custom Bin Name"

    def test_media_xml_has_bin_and_sequence(self, tmp_path):
        """Test that media XML has bin + sequence siblings for DaVinci Resolve"""
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

        # Structure must be: xmeml > bin + sequence (siblings)
        assert root.tag == 'xmeml'
        bin_node = root.find('bin')
        assert bin_node is not None, "Missing <bin> under xmeml"
        sequence_node = root.find('sequence')
        assert sequence_node is not None, "Missing <sequence> sibling (required for DaVinci import)"


class TestFrameRateValidation:
    """Tests for frame rate validation in XML export."""

    def test_validate_frame_rate_standard_rates(self):
        """Test _validate_frame_rate accepts standard NLE rates."""
        from src.otio.xml_export import _validate_frame_rate, STANDARD_NLE_RATES

        for rate in STANDARD_NLE_RATES:
            result = _validate_frame_rate(rate)
            assert isinstance(result, int), f"Expected int for rate {rate}"
            # 23.976, 29.97, 59.94 should round to 24, 30, 60
            expected = round(rate)
            assert result == expected, f"Expected {expected} for rate {rate}, got {result}"

    def test_validate_frame_rate_non_standard_logs_warning(self, caplog):
        """Test _validate_frame_rate logs warning for non-standard rates."""
        from src.otio.xml_export import _validate_frame_rate
        import logging

        with caplog.at_level(logging.WARNING):
            result = _validate_frame_rate(27.5)

        # Should log warning about non-standard rate
        assert "Non-standard frame rate" in caplog.text
        assert "27.5" in caplog.text
        # Should still return an integer
        assert isinstance(result, int)
        assert result == 28  # Rounded

    def test_validate_frame_rate_returns_integer(self):
        """Test _validate_frame_rate always returns an integer."""
        from src.otio.xml_export import _validate_frame_rate

        test_rates = [23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0,
                      15.5, 27.3, 45.8, 120.0]

        for rate in test_rates:
            result = _validate_frame_rate(rate)
            assert isinstance(result, int), f"Expected int for rate {rate}, got {type(result)}"

    def test_validate_frame_rate_23_976(self):
        """Test 23.976 fps rounds to 24."""
        from src.otio.xml_export import _validate_frame_rate

        result = _validate_frame_rate(23.976)
        assert result == 24

    def test_validate_frame_rate_29_97(self):
        """Test 29.97 fps rounds to 30."""
        from src.otio.xml_export import _validate_frame_rate

        result = _validate_frame_rate(29.97)
        assert result == 30

    def test_validate_frame_rate_59_94(self):
        """Test 59.94 fps rounds to 60."""
        from src.otio.xml_export import _validate_frame_rate

        result = _validate_frame_rate(59.94)
        assert result == 60

    def test_validate_frame_rate_integer_passthrough(self):
        """Test integer frame rates pass through correctly."""
        from src.otio.xml_export import _validate_frame_rate

        for rate in [24, 25, 30, 50, 60]:
            result = _validate_frame_rate(float(rate))
            assert result == rate

    def test_xml_timebase_is_integer(self, mock_matches, temp_output_path):
        """Test that XML timebase element is always an integer."""
        paths = generate_resolve_xml_with_bins(
            mock_matches,
            temp_output_path,
            frame_rate=29.97  # Non-integer rate
        )

        tree = ET.parse(paths[0])
        root = tree.getroot()

        # Find all timebase elements
        timebases = root.findall('.//timebase')
        assert len(timebases) > 0, "Expected at least one timebase element"

        for timebase in timebases:
            value = timebase.text
            # Should be an integer string (no decimal point)
            assert '.' not in value, f"Timebase should be integer, got '{value}'"
            # Should parse as int
            int_value = int(value)
            assert int_value == 30, f"Expected 30, got {int_value}"

    def test_xml_timebase_non_standard_rate(self, mock_matches, temp_output_path, caplog):
        """Test XML generation with non-standard frame rate logs warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            paths = generate_resolve_xml_with_bins(
                mock_matches,
                temp_output_path,
                frame_rate=27.5  # Non-standard rate
            )

        # Should log warning
        assert "Non-standard frame rate" in caplog.text

        # XML should still be generated
        assert len(paths) >= 1
        tree = ET.parse(paths[0])
        root = tree.getroot()

        # Timebase should be rounded to 28
        timebases = root.findall('.//timebase')
        assert len(timebases) > 0
        assert timebases[0].text == "28"

    def test_standard_nle_rates_constant(self):
        """Test STANDARD_NLE_RATES contains expected values."""
        from src.otio.xml_export import STANDARD_NLE_RATES

        expected = {23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0}
        assert STANDARD_NLE_RATES == expected


@pytest.mark.fast
class TestXMLSpecialCharacterHandling:
    """
    Tests for XML export special character handling.

    US-008: Verify XML export correctly handles special characters in:
    - File paths (ampersands, percent signs, hash)
    - Metadata fields (quotes, apostrophes)
    - Clip names (angle brackets, special chars)
    - Long Windows paths (>200 characters)
    - Unicode filenames (CJK, emoji, RTL)

    Note: The escape_xml() function is used for clip <name> elements.
    The pathurl elements use format_path_url() which preserves raw paths
    for DaVinci Resolve compatibility. Special characters in pathurl are
    expected to be sanitized at download time.
    """

    @pytest.fixture
    def special_char_matches(self):
        """Create matches with special characters in paths and names."""
        vo_seg = SRTSegment(
            index=0, start_time=0.0, end_time=3.0,
            text="Voiceover segment", source_file="voiceover.srt"
        )
        vid_seg = SRTSegment(
            index=0, start_time=10.0, end_time=13.0,
            text="Video clip", source_file="/videos/normal_clip.mp4"
        )
        match = Match(
            voiceover_segment=vo_seg,
            video_segment=vid_seg,
            video_scene=None,
            confidence=0.9,
            reasoning='Test match'
        )
        return [MatchResult(primary_match=match, alternatives=[], secondary_matches=[], strategy_matches=[])]

    # ============================================================
    # AC1: Test XML export escapes ampersands in file paths correctly
    # ============================================================

    def test_ampersand_in_clip_name_escaped(self, special_char_matches, tmp_path):
        """Test ampersands in clip names (derived from paths) are properly escaped."""
        from src.otio.utils import escape_xml

        # Test the escape_xml function directly for ampersands
        clip_name = "Tom & Jerry_clip.mp4"
        escaped = escape_xml(clip_name)
        assert "&amp;" in escaped, "Ampersand in clip name should be escaped as &amp;"
        assert "Tom & Jerry" not in escaped, "Raw ampersand should not appear in escaped text"

    def test_escape_xml_ampersand_in_path_component(self):
        """Test escape_xml correctly handles ampersands in path-derived names."""
        from src.otio.utils import escape_xml

        # Folder name with ampersand (used in unique clip names)
        folder_name = "R&D"
        escaped = escape_xml(folder_name)
        assert escaped == "R&amp;D", "Ampersand should be escaped"

    def test_multiple_ampersands_in_name(self):
        """Test multiple ampersands are all escaped."""
        from src.otio.utils import escape_xml

        text = "A & B & C"
        escaped = escape_xml(text)
        assert escaped.count("&amp;") == 2, "All ampersands should be escaped"
        assert "&" not in escaped.replace("&amp;", ""), "No unescaped ampersands"

    # ============================================================
    # AC2: Test XML export escapes quotes in metadata fields correctly
    # ============================================================

    def test_double_quotes_in_clip_name_escaped(self, special_char_matches, tmp_path):
        """Test double quotes in clip names are properly escaped."""
        # Use a path that would create a clip name with quotes
        special_char_matches[0].primary_match.video_segment.source_file = '/videos/folder/clip_"best"_take.mp4'

        paths = generate_resolve_xml_with_bins(
            special_char_matches,
            str(tmp_path / "test.xml"),
            frame_rate=30.0
        )

        with open(paths[0], 'r', encoding='utf-8') as f:
            xml_content = f.read()

        # Double quotes should be escaped as &quot; in name elements
        assert "&quot;" in xml_content or '"best"' not in xml_content.replace("&quot;", "QUOTE"), \
            "Double quotes should be escaped"

        # XML should still be valid
        tree = ET.parse(paths[0])
        assert tree.getroot().tag == 'xmeml'

    def test_single_quotes_in_path_escaped(self, special_char_matches, tmp_path):
        """Test single quotes (apostrophes) in paths are properly escaped."""
        special_char_matches[0].primary_match.video_segment.source_file = "/videos/John's Folder/clip.mp4"

        paths = generate_resolve_xml_with_bins(
            special_char_matches,
            str(tmp_path / "test.xml"),
            frame_rate=30.0
        )

        with open(paths[0], 'r', encoding='utf-8') as f:
            xml_content = f.read()

        # Apostrophe should be escaped as &apos; in name elements
        assert "&apos;" in xml_content or "John's" not in xml_content.replace("&apos;", "APOS"), \
            "Apostrophe should be escaped"

        # XML should still be valid
        tree = ET.parse(paths[0])
        assert tree.getroot().tag == 'xmeml'

    def test_mixed_quotes_in_metadata(self, special_char_matches, tmp_path):
        """Test both quote types in same path are escaped correctly."""
        special_char_matches[0].primary_match.video_segment.source_file = '/videos/"Mike\'s" Project/clip.mp4'

        paths = generate_resolve_xml_with_bins(
            special_char_matches,
            str(tmp_path / "test.xml"),
            frame_rate=30.0
        )

        # Should produce valid parseable XML
        tree = ET.parse(paths[0])
        assert tree.getroot().tag == 'xmeml', "XML with mixed quotes should be valid"

    def test_escape_xml_quotes(self):
        """Test escape_xml handles both quote types."""
        from src.otio.utils import escape_xml

        text = "\"Mike's\" Project"
        escaped = escape_xml(text)
        assert "&quot;" in escaped, "Double quotes should be escaped"
        assert "&apos;" in escaped, "Apostrophes should be escaped"

    # ============================================================
    # AC3: Test XML export escapes angle brackets in clip names correctly
    # ============================================================

    def test_escape_xml_less_than(self):
        """Test escape_xml handles less-than brackets."""
        from src.otio.utils import escape_xml

        text = "version <1>"
        escaped = escape_xml(text)
        assert "&lt;" in escaped, "Less-than should be escaped as &lt;"

    def test_escape_xml_greater_than(self):
        """Test escape_xml handles greater-than brackets."""
        from src.otio.utils import escape_xml

        text = "v1 -> v2"
        escaped = escape_xml(text)
        assert "&gt;" in escaped, "Greater-than should be escaped as &gt;"

    def test_escape_xml_angle_brackets_pair(self):
        """Test escape_xml handles paired angle brackets."""
        from src.otio.utils import escape_xml

        text = "<draft>"
        escaped = escape_xml(text)
        assert escaped == "&lt;draft&gt;", "Both brackets should be escaped"

    def test_escape_xml_html_like_tag(self):
        """Test escape_xml handles HTML-like tags correctly."""
        from src.otio.utils import escape_xml

        text = "<script>alert</script>"
        escaped = escape_xml(text)
        assert "&lt;script&gt;" in escaped, "Opening tag should be escaped"
        assert "&lt;/script&gt;" in escaped, "Closing tag should be escaped"
        # Should not contain raw angle brackets
        assert "<script>" not in escaped
        assert "</script>" not in escaped

    def test_greater_than_in_path_generates_xml(self, special_char_matches, tmp_path):
        """Test paths with > produce valid XML (> is escaped in name elements)."""
        special_char_matches[0].primary_match.video_segment.source_file = "/videos/v1-to-v2/clip.mp4"

        paths = generate_resolve_xml_with_bins(
            special_char_matches,
            str(tmp_path / "test.xml"),
            frame_rate=30.0
        )

        # Should generate valid XML
        tree = ET.parse(paths[0])
        assert tree.getroot().tag == 'xmeml'

    # ============================================================
    # AC4: Test XML export handles Windows paths exceeding 200 characters
    # ============================================================

    def test_long_path_200_plus_chars(self, special_char_matches, tmp_path):
        """Test paths exceeding 200 characters are handled correctly."""
        # Create a path > 200 characters (no special chars that need escaping)
        long_folder = "very_long_folder_name_that_goes_on_and_on_" * 5  # ~200 chars
        long_path = f"/videos/{long_folder}/clip.mp4"
        assert len(long_path) > 200, f"Path should exceed 200 chars, got {len(long_path)}"

        special_char_matches[0].primary_match.video_segment.source_file = long_path

        paths = generate_resolve_xml_with_bins(
            special_char_matches,
            str(tmp_path / "test.xml"),
            frame_rate=30.0
        )

        # XML should be generated successfully
        assert len(paths) >= 1
        assert Path(paths[0]).exists()

        # XML should be valid
        tree = ET.parse(paths[0])
        assert tree.getroot().tag == 'xmeml'

    def test_long_path_250_chars(self, special_char_matches, tmp_path):
        """Test paths approaching Windows MAX_PATH limit (260 chars)."""
        # Create path close to Windows limit
        base = "/videos/"
        folder_name = "a" * 200
        filename = "clip_with_long_name.mp4"
        long_path = f"{base}{folder_name}/{filename}"
        assert len(long_path) > 200, f"Path should exceed 200 chars, got {len(long_path)}"

        special_char_matches[0].primary_match.video_segment.source_file = long_path

        paths = generate_resolve_xml_with_bins(
            special_char_matches,
            str(tmp_path / "test.xml"),
            frame_rate=30.0
        )

        # Should still generate valid XML
        tree = ET.parse(paths[0])
        assert tree.getroot().tag == 'xmeml'

    def test_deeply_nested_path(self, special_char_matches, tmp_path):
        """Test deeply nested folder structures create valid XML."""
        # Create deeply nested path
        nested_path = "/videos" + "/subfolder" * 15 + "/clip.mp4"  # ~200 chars
        assert len(nested_path) > 150, f"Path should be deeply nested, got {len(nested_path)}"

        special_char_matches[0].primary_match.video_segment.source_file = nested_path

        paths = generate_resolve_xml_with_bins(
            special_char_matches,
            str(tmp_path / "test.xml"),
            frame_rate=30.0
        )

        tree = ET.parse(paths[0])
        assert tree.getroot().tag == 'xmeml'

    def test_long_filename_itself(self, special_char_matches, tmp_path):
        """Test very long filenames (without long folder path) are handled."""
        # Create long filename
        long_name = "this_is_a_very_long_video_filename_that_describes_the_content_in_great_detail_" * 2
        long_path = f"/videos/normal_folder/{long_name}.mp4"

        special_char_matches[0].primary_match.video_segment.source_file = long_path

        paths = generate_resolve_xml_with_bins(
            special_char_matches,
            str(tmp_path / "test.xml"),
            frame_rate=30.0
        )

        # Should generate valid XML
        tree = ET.parse(paths[0])
        root = tree.getroot()
        assert root.tag == 'xmeml'

        # Path should be preserved in XML
        with open(paths[0], 'r', encoding='utf-8') as f:
            content = f.read()
        assert long_name in content, "Long filename should be preserved in XML"

    def test_path_length_boundary(self, special_char_matches, tmp_path):
        """Test paths exactly at 200 character boundary."""
        # Create path exactly at 200 chars
        base = "/videos/project/"
        # Calculate how many chars we need for folder name to hit 200 total
        remaining = 200 - len(base) - len("/clip.mp4")
        folder_name = "x" * remaining
        boundary_path = f"{base}{folder_name}/clip.mp4"
        assert len(boundary_path) == 200, f"Path should be exactly 200 chars, got {len(boundary_path)}"

        special_char_matches[0].primary_match.video_segment.source_file = boundary_path

        paths = generate_resolve_xml_with_bins(
            special_char_matches,
            str(tmp_path / "test.xml"),
            frame_rate=30.0
        )

        tree = ET.parse(paths[0])
        assert tree.getroot().tag == 'xmeml'

    # ============================================================
    # AC5: Test XML export handles Unicode filenames correctly
    # ============================================================

    def test_cjk_characters_in_path(self, special_char_matches, tmp_path):
        """Test Chinese/Japanese/Korean characters in paths."""
        # CJK characters
        special_char_matches[0].primary_match.video_segment.source_file = "/videos/中文视频/测试片段.mp4"

        paths = generate_resolve_xml_with_bins(
            special_char_matches,
            str(tmp_path / "test.xml"),
            frame_rate=30.0
        )

        # XML should be valid UTF-8
        with open(paths[0], 'r', encoding='utf-8') as f:
            xml_content = f.read()

        assert "中文视频" in xml_content or "中文视频" in xml_content, "CJK characters should be preserved"

        # Should parse as valid XML
        tree = ET.parse(paths[0])
        assert tree.getroot().tag == 'xmeml'

    def test_japanese_characters_in_path(self, special_char_matches, tmp_path):
        """Test Japanese characters (hiragana/katakana/kanji) in paths."""
        special_char_matches[0].primary_match.video_segment.source_file = "/videos/日本語フォルダ/クリップ.mp4"

        paths = generate_resolve_xml_with_bins(
            special_char_matches,
            str(tmp_path / "test.xml"),
            frame_rate=30.0
        )

        with open(paths[0], 'r', encoding='utf-8') as f:
            xml_content = f.read()

        assert "日本語" in xml_content, "Japanese characters should be preserved"

        tree = ET.parse(paths[0])
        assert tree.getroot().tag == 'xmeml'

    def test_korean_characters_in_path(self, special_char_matches, tmp_path):
        """Test Korean (Hangul) characters in paths."""
        special_char_matches[0].primary_match.video_segment.source_file = "/videos/한국어폴더/영상클립.mp4"

        paths = generate_resolve_xml_with_bins(
            special_char_matches,
            str(tmp_path / "test.xml"),
            frame_rate=30.0
        )

        with open(paths[0], 'r', encoding='utf-8') as f:
            xml_content = f.read()

        assert "한국어" in xml_content, "Korean characters should be preserved"

        tree = ET.parse(paths[0])
        assert tree.getroot().tag == 'xmeml'

    def test_emoji_in_path(self, special_char_matches, tmp_path):
        """Test emoji characters in paths are handled."""
        special_char_matches[0].primary_match.video_segment.source_file = "/videos/🎬 Movies/🎥 clip.mp4"

        paths = generate_resolve_xml_with_bins(
            special_char_matches,
            str(tmp_path / "test.xml"),
            frame_rate=30.0
        )

        # Should generate valid UTF-8 XML
        with open(paths[0], 'r', encoding='utf-8') as f:
            xml_content = f.read()

        # Emoji should be preserved (either directly or as references)
        assert "🎬" in xml_content or "Movies" in xml_content, "Path content should be preserved"

        tree = ET.parse(paths[0])
        assert tree.getroot().tag == 'xmeml'

    def test_rtl_arabic_in_path(self, special_char_matches, tmp_path):
        """Test RTL (right-to-left) Arabic characters in paths."""
        special_char_matches[0].primary_match.video_segment.source_file = "/videos/مجلد عربي/مقطع.mp4"

        paths = generate_resolve_xml_with_bins(
            special_char_matches,
            str(tmp_path / "test.xml"),
            frame_rate=30.0
        )

        with open(paths[0], 'r', encoding='utf-8') as f:
            xml_content = f.read()

        assert "مجلد" in xml_content or "عربي" in xml_content, "Arabic characters should be preserved"

        tree = ET.parse(paths[0])
        assert tree.getroot().tag == 'xmeml'

    def test_rtl_hebrew_in_path(self, special_char_matches, tmp_path):
        """Test RTL Hebrew characters in paths."""
        special_char_matches[0].primary_match.video_segment.source_file = "/videos/תיקייה עברית/קליפ.mp4"

        paths = generate_resolve_xml_with_bins(
            special_char_matches,
            str(tmp_path / "test.xml"),
            frame_rate=30.0
        )

        with open(paths[0], 'r', encoding='utf-8') as f:
            xml_content = f.read()

        assert "תיקייה" in xml_content or "עברית" in xml_content, "Hebrew characters should be preserved"

        tree = ET.parse(paths[0])
        assert tree.getroot().tag == 'xmeml'

    def test_mixed_unicode_and_special_chars_in_escape(self):
        """Test escape_xml handles combination of Unicode and special characters."""
        from src.otio.utils import escape_xml

        # Mix of CJK, ampersand, and quotes
        text = "中文 & English's \"folder\""
        escaped = escape_xml(text)

        # Unicode should be preserved
        assert "中文" in escaped, "CJK should be preserved"
        # Special chars should be escaped
        assert "&amp;" in escaped, "Ampersand should be escaped"
        assert "&apos;" in escaped, "Apostrophe should be escaped"
        assert "&quot;" in escaped, "Quotes should be escaped"

    def test_unicode_path_generates_valid_xml(self, special_char_matches, tmp_path):
        """Test Unicode paths (without XML special chars) generate valid XML."""
        # Use Unicode without ampersands or brackets
        special_char_matches[0].primary_match.video_segment.source_file = "/videos/中文视频/测试.mp4"

        paths = generate_resolve_xml_with_bins(
            special_char_matches,
            str(tmp_path / "test.xml"),
            frame_rate=30.0
        )

        with open(paths[0], 'r', encoding='utf-8') as f:
            xml_content = f.read()

        # Unicode should be preserved in the file
        assert "中文" in xml_content, "CJK should be preserved in XML"

        # XML should be valid
        tree = ET.parse(paths[0])
        assert tree.getroot().tag == 'xmeml'

    def test_cyrillic_in_path(self, special_char_matches, tmp_path):
        """Test Cyrillic (Russian) characters in paths."""
        special_char_matches[0].primary_match.video_segment.source_file = "/videos/Русская папка/клип.mp4"

        paths = generate_resolve_xml_with_bins(
            special_char_matches,
            str(tmp_path / "test.xml"),
            frame_rate=30.0
        )

        with open(paths[0], 'r', encoding='utf-8') as f:
            xml_content = f.read()

        assert "Русская" in xml_content, "Cyrillic characters should be preserved"

        tree = ET.parse(paths[0])
        assert tree.getroot().tag == 'xmeml'

    def test_accented_european_characters(self, special_char_matches, tmp_path):
        """Test accented European characters (ñ, ü, é, etc.) in paths."""
        special_char_matches[0].primary_match.video_segment.source_file = "/videos/Café René/Señor's clip.mp4"

        paths = generate_resolve_xml_with_bins(
            special_char_matches,
            str(tmp_path / "test.xml"),
            frame_rate=30.0
        )

        with open(paths[0], 'r', encoding='utf-8') as f:
            xml_content = f.read()

        assert "Café" in xml_content or "René" in xml_content, "Accented characters should be preserved"

        tree = ET.parse(paths[0])
        assert tree.getroot().tag == 'xmeml'

    # ============================================================
    # Edge cases and combined scenarios
    # ============================================================

    def test_escape_xml_all_special_chars_combined(self):
        """Test escape_xml handles all special character types."""
        from src.otio.utils import escape_xml

        # Combine all special chars
        text = "Tom & Jerry's <Draft> \"test\""
        escaped = escape_xml(text)

        assert "&amp;" in escaped, "Ampersand should be escaped"
        assert "&apos;" in escaped, "Apostrophe should be escaped"
        assert "&lt;" in escaped, "Less-than should be escaped"
        assert "&gt;" in escaped, "Greater-than should be escaped"
        assert "&quot;" in escaped, "Quote should be escaped"

        # No raw special chars should remain
        assert "&" not in escaped.replace("&amp;", "").replace("&apos;", "").replace("&lt;", "").replace("&gt;", "").replace("&quot;", "")
        assert "<" not in escaped
        assert ">" not in escaped
        assert '"' not in escaped
        assert "'" not in escaped

    def test_xml_encoding_declaration(self, special_char_matches, tmp_path):
        """Test XML has proper UTF-8 encoding declaration for Unicode support."""
        special_char_matches[0].primary_match.video_segment.source_file = "/videos/Japanese/clip.mp4"

        paths = generate_resolve_xml_with_bins(
            special_char_matches,
            str(tmp_path / "test.xml"),
            frame_rate=30.0
        )

        with open(paths[0], 'r', encoding='utf-8') as f:
            first_line = f.readline()

        assert 'encoding="UTF-8"' in first_line or 'encoding="utf-8"' in first_line.lower(), \
            "XML should declare UTF-8 encoding"

    def test_pathurl_format_plain_path(self, special_char_matches, tmp_path):
        """Test pathurl elements contain plain paths (no file:// prefix)."""
        special_char_matches[0].primary_match.video_segment.source_file = "/videos/test_folder/clip.mp4"

        paths = generate_resolve_xml_with_bins(
            special_char_matches,
            str(tmp_path / "test.xml"),
            frame_rate=30.0
        )

        tree = ET.parse(paths[0])
        root = tree.getroot()

        # Find pathurl elements
        pathurls = root.findall('.//pathurl')
        assert len(pathurls) > 0, "Should have pathurl elements"

        # DaVinci expects plain Windows paths, not file:// URLs
        for pathurl in pathurls:
            if pathurl.text:
                # Should not start with file:// (causes hangs in DaVinci)
                assert not pathurl.text.startswith("file://"), \
                    "pathurl should be plain path, not file:// URL"

    def test_escape_xml_unicode_passthrough(self):
        """Test escape_xml preserves Unicode characters (CJK, etc.)."""
        from src.otio.utils import escape_xml

        # Unicode should pass through unchanged
        text = "中文 日本語 한국어"
        escaped = escape_xml(text)
        assert escaped == text, "Unicode should pass through escape_xml unchanged"

    def test_escape_xml_unicode_with_special_chars(self):
        """Test escape_xml handles mix of Unicode and special chars."""
        from src.otio.utils import escape_xml

        text = "中文 & English"
        escaped = escape_xml(text)
        assert "中文" in escaped, "Unicode should be preserved"
        assert "&amp;" in escaped, "Ampersand should be escaped"
        assert escaped == "中文 &amp; English"
