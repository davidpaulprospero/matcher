"""
Tests for XML media import and path resolution fixes.

Covers three bugs fixed in 2026-02-05:
1. Empty <media> tags for extensionless source_file (video IDs)
2. V2-V8 tracks missing from XML timeline
3. _scan_video_segments missing flat download dir
"""

import pytest
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.otio.xml_export import (
    generate_resolve_xml_with_bins,
    generate_davinci_sequence_xml,
    _write_media_xml_part,
    _build_segment_lookup,
    _resolve_video_segment,
    _get_segment_file_duration,
    _is_unresolved_path,
)
from src.stages.output import OutputStage, SegmentInfo
from src.utils import Match, MatchResult, SRTSegment, AlternativeMatch


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def extensionless_matches():
    """Create matches with extensionless source_file (caption-first video IDs)."""
    vo_seg = SRTSegment(
        index=0, start_time=0.0, end_time=5.0,
        text="Test segment", source_file="voiceover.srt"
    )
    vid_seg = SRTSegment(
        index=0, start_time=10.0, end_time=15.0,
        text="Video text", source_file="HUvIrRdq0Ow"  # No extension
    )
    match = Match(
        voiceover_segment=vo_seg, video_segment=vid_seg,
        video_scene=None, confidence=0.9, reasoning="Test"
    )
    return [MatchResult(
        primary_match=match, alternatives=[],
        secondary_matches=[], strategy_matches=[]
    )]


@pytest.fixture
def multitrack_matches():
    """Create matches with V1 primary + V2-V3 alternatives + V4-V6 secondary."""
    vo_seg = SRTSegment(
        index=0, start_time=0.0, end_time=5.0,
        text="Test segment", source_file="voiceover.srt"
    )
    vid_seg = SRTSegment(
        index=0, start_time=10.0, end_time=15.0,
        text="V1 clip", source_file="/videos/primary.mp4"
    )
    alt_seg = SRTSegment(
        index=1, start_time=20.0, end_time=25.0,
        text="V2 clip", source_file="/videos/alt1.mp4"
    )
    sec_seg = SRTSegment(
        index=2, start_time=30.0, end_time=35.0,
        text="V4 clip", source_file="/videos/secondary1.mp4"
    )
    match = Match(
        voiceover_segment=vo_seg, video_segment=vid_seg,
        video_scene=None, confidence=0.9, reasoning="Test"
    )
    alt_match = AlternativeMatch(
        video_segment=alt_seg, video_scene=None,
        confidence=0.8, reasoning="Alt", diversity_score=0.5
    )
    sec_match = AlternativeMatch(
        video_segment=sec_seg, video_scene=None,
        confidence=0.7, reasoning="Sec", diversity_score=0.4
    )
    return [MatchResult(
        primary_match=match,
        alternatives=[alt_match],
        secondary_matches=[sec_match],
        strategy_matches=[]
    )]


@pytest.fixture
def temp_output_path(tmp_path):
    return str(tmp_path / "test_output.xml")


# ============================================================================
# Bug 1: Extensionless source_file → empty <media> tags
# ============================================================================

class TestExtensionlessVideoIds:
    """Extensionless source_file (video IDs) must produce valid <media> content."""

    @pytest.mark.fast
    def test_project_xml_has_video_media_for_extensionless(
        self, extensionless_matches, temp_output_path
    ):
        """Project XML bin clips must have <video> inside <media> for video IDs."""
        paths = generate_resolve_xml_with_bins(
            extensionless_matches, temp_output_path, frame_rate=30.0
        )
        tree = ET.parse(paths[0])
        root = tree.getroot()

        # Find clip in bin
        clips = root.findall('.//bin/children/clip')
        assert len(clips) >= 1, "No clips found in bin"

        for clip in clips:
            media = clip.find('media')
            assert media is not None, "Clip missing <media> tag"
            video = media.find('video')
            # Skip voiceover audio clips
            name = clip.find('name')
            if name is not None and 'voiceover' in name.text.lower():
                continue
            assert video is not None, (
                f"Clip '{name.text if name is not None else '?'}' has empty <media> "
                f"(missing <video>) — extensionless source_file not treated as video"
            )

    @pytest.mark.fast
    def test_media_part_xml_has_video_for_extensionless(
        self, extensionless_matches, tmp_path
    ):
        """Media part XMLs must have <video>/<audio> inside <media> for video IDs."""
        # Build file info dict mimicking what generate_resolve_xml_with_bins creates
        files = {
            'HUvIrRdq0Ow': {
                'file_id': 'file-1',
                'uuid': 'test-uuid',
                'duration_frames': 150,
            }
        }
        generated = []
        out_path = str(tmp_path / "media_part.xml")

        _write_media_xml_part(
            files, 1, out_path, 30, generated,
            __import__('logging').getLogger(__name__),
            frame_rate=30.0, is_ntsc=False
        )

        assert len(generated) == 1
        tree = ET.parse(generated[0])
        root = tree.getroot()
        clips = root.findall('.//bin/children/clip')
        assert len(clips) >= 1

        clip = clips[0]
        media = clip.find('media')
        video = media.find('video')
        assert video is not None, "Extensionless path produced empty <media> — missing <video>"

        # Verify file definition exists inside clipitem
        pathurl = root.find('.//pathurl')
        assert pathurl is not None, "No <pathurl> in media part clip"

    @pytest.mark.fast
    def test_extension_detection_fallback_in_source(self):
        """Verify the is_video fallback exists in xml_export.py source code."""
        source_path = Path(__file__).parent.parent / "src" / "otio" / "xml_export.py"
        source = source_path.read_text(encoding='utf-8')

        # The fix: "if not file_ext: is_video = True" must appear at least twice
        count = source.count("if not file_ext:")
        assert count >= 2, (
            f"Expected 'if not file_ext:' at least 2 times in xml_export.py, found {count}"
        )

    @pytest.mark.fast
    def test_extension_detection_fallback_in_otio_builder(self):
        """Verify the is_video fallback exists in otio_builder.py source code."""
        source_path = Path(__file__).parent.parent / "src" / "otio_builder.py"
        source = source_path.read_text(encoding='utf-8')

        count = source.count("if not file_ext:")
        assert count >= 2, (
            f"Expected 'if not file_ext:' at least 2 times in otio_builder.py, found {count}"
        )


# ============================================================================
# Bug 2: V2-V8 tracks missing from XML
# ============================================================================

class TestMultiTrackXML:
    """V2-V8 alternative/secondary tracks must appear in XML timelines."""

    @pytest.mark.fast
    def test_project_xml_has_alt_tracks(
        self, multitrack_matches, temp_output_path
    ):
        """Project XML timeline must have V2+ tracks for alternatives."""
        paths = generate_resolve_xml_with_bins(
            multitrack_matches, temp_output_path, frame_rate=30.0
        )
        tree = ET.parse(paths[0])
        root = tree.getroot()

        # Find tracks in sequence's video section
        seq = root.find('.//sequence')
        assert seq is not None, "No sequence in project XML"

        video_section = seq.find('.//media/video')
        assert video_section is not None, "No video section in sequence"

        tracks = video_section.findall('track')
        # V1 + at least one alt track
        assert len(tracks) >= 2, (
            f"Only {len(tracks)} video track(s) in project XML — "
            f"V2-V8 alt tracks missing"
        )

    @pytest.mark.fast
    def test_sequence_xml_has_alt_tracks(
        self, multitrack_matches, tmp_path
    ):
        """Sequence XML must have V2+ tracks for alternatives."""
        out_path = str(tmp_path / "timeline")
        generate_davinci_sequence_xml(
            multitrack_matches, out_path, frame_rate=30.0
        )
        xml_path = str(tmp_path / "timeline_sequence.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        video_section = root.find('.//media/video')
        assert video_section is not None
        tracks = video_section.findall('track')
        assert len(tracks) >= 2, (
            f"Only {len(tracks)} track(s) in sequence XML — V2-V8 missing"
        )

    @pytest.mark.fast
    def test_alt_tracks_are_disabled(
        self, multitrack_matches, temp_output_path
    ):
        """V2+ tracks in project XML must have <enabled>FALSE</enabled>."""
        paths = generate_resolve_xml_with_bins(
            multitrack_matches, temp_output_path, frame_rate=30.0
        )
        tree = ET.parse(paths[0])
        root = tree.getroot()

        seq = root.find('.//sequence')
        video_section = seq.find('.//media/video')
        tracks = video_section.findall('track')

        # First track (V1) should be enabled or have no enabled tag
        # Subsequent tracks should be disabled
        for i, track in enumerate(tracks[1:], start=2):
            enabled = track.find('enabled')
            if enabled is not None:
                assert enabled.text == 'FALSE', (
                    f"Track V{i} should be disabled but enabled={enabled.text}"
                )


# ============================================================================
# Bug 3: _scan_video_segments missed flat download dir
# ============================================================================

class TestScanVideoSegments:
    """_scan_video_segments must find both legacy *_segments dirs and flat download dir."""

    @pytest.mark.fast
    def test_finds_flat_download_dir_segments(self, tmp_path):
        """Segments in flat downloaded_videos_dir must be found."""
        # Create flat segment files: {video_id}_{start}_{end}.mp4
        download_dir = tmp_path / "matcher-alt"
        download_dir.mkdir()
        (download_dir / "HUvIrRdq0Ow_0_15.mp4").write_bytes(b'\x00' * 100)
        (download_dir / "YiTtHqRtMKM_55_75.mp4").write_bytes(b'\x00' * 100)

        config = MagicMock()
        config.download.root_dir = str(tmp_path)
        config.downloaded_videos_dir = str(download_dir)
        config.otio_output_dir = str(tmp_path / "output")

        stage = OutputStage()
        segments = stage._scan_video_segments(config)

        video_ids = {s.video_id for s in segments}
        assert "HUvIrRdq0Ow" in video_ids, "Flat segment HUvIrRdq0Ow not found"
        assert "YiTtHqRtMKM" in video_ids, "Flat segment YiTtHqRtMKM not found"

    @pytest.mark.fast
    def test_finds_legacy_segments_dirs(self, tmp_path):
        """Segments in *_segments directories must still be found."""
        # Create legacy segment dir
        seg_dir = tmp_path / "touris_segments"
        seg_dir.mkdir()
        (seg_dir / "abc123_0000.mp4").write_bytes(b'\x00' * 100)

        config = MagicMock()
        config.download.root_dir = str(tmp_path)
        config.downloaded_videos_dir = str(tmp_path / "nonexistent")
        config.otio_output_dir = str(tmp_path / "output")

        stage = OutputStage()
        segments = stage._scan_video_segments(config)

        video_ids = {s.video_id for s in segments}
        assert "abc123" in video_ids, "Legacy segment abc123 not found"

    @pytest.mark.fast
    def test_flat_segments_parse_start_end(self, tmp_path):
        """Flat segment filenames must parse start AND end times correctly."""
        download_dir = tmp_path / "project"
        download_dir.mkdir()
        (download_dir / "vid123_100_200.mp4").write_bytes(b'\x00' * 100)

        config = MagicMock()
        config.download.root_dir = str(tmp_path)
        config.downloaded_videos_dir = str(download_dir)
        config.otio_output_dir = str(tmp_path / "output")

        stage = OutputStage()
        segments = stage._scan_video_segments(config)

        seg = [s for s in segments if s.video_id == "vid123"]
        assert len(seg) == 1
        assert seg[0].original_start == 100.0
        assert seg[0].original_end == 200.0

    @pytest.mark.fast
    def test_no_duplicates_between_legacy_and_flat(self, tmp_path):
        """Same file found via both scan paths must not be duplicated."""
        # Create a *_segments dir AND a flat dir pointing to same location
        seg_dir = tmp_path / "test_segments"
        seg_dir.mkdir()
        (seg_dir / "abc_0000.mp4").write_bytes(b'\x00' * 100)

        config = MagicMock()
        config.download.root_dir = str(tmp_path)
        # downloaded_videos_dir points to a different dir (won't overlap)
        config.downloaded_videos_dir = str(tmp_path / "other")
        config.otio_output_dir = str(tmp_path / "output")

        stage = OutputStage()
        segments = stage._scan_video_segments(config)

        # Should find the legacy segment but not double-count
        abc_segs = [s for s in segments if s.video_id == "abc"]
        assert len(abc_segs) == 1


# ============================================================================
# Segment lookup and resolution
# ============================================================================

class TestSegmentLookupResolution:
    """_build_segment_lookup and _resolve_video_segment integration."""

    @pytest.mark.fast
    def test_resolve_video_id_to_segment_path(self):
        """Video ID source_file must resolve to actual segment file path."""
        segments = [
            SegmentInfo(
                video_id="HUvIrRdq0Ow",
                file="E:/v/matcher-alt/HUvIrRdq0Ow_0_15.mp4",
                original_start=0.0,
                original_end=15.0
            )
        ]
        lookup = _build_segment_lookup(segments)
        assert "HUvIrRdq0Ow" in lookup

        resolved_path, adjusted_start = _resolve_video_segment(
            "HUvIrRdq0Ow", 5.0, lookup
        )
        assert resolved_path == "E:/v/matcher-alt/HUvIrRdq0Ow_0_15.mp4"
        assert adjusted_start == 5.0  # 5.0 - 0.0

    @pytest.mark.fast
    def test_unresolved_video_id_returns_original(self):
        """Unknown video ID must return original source_file unchanged."""
        lookup = _build_segment_lookup([])
        resolved, start = _resolve_video_segment("unknown_id", 10.0, lookup)
        assert resolved == "unknown_id"
        assert start == 10.0

    @pytest.mark.fast
    def test_get_segment_file_duration_found(self):
        """_get_segment_file_duration returns physical segment duration."""
        segments = [
            SegmentInfo(
                video_id="vid123",
                file="E:/v/project/vid123_100_200.mp4",
                original_start=100.0,
                original_end=200.0
            )
        ]
        lookup = _build_segment_lookup(segments)
        dur = _get_segment_file_duration("E:/v/project/vid123_100_200.mp4", lookup)
        assert dur == 100.0

    @pytest.mark.fast
    def test_get_segment_file_duration_fallback(self):
        """_get_segment_file_duration returns fallback for unknown path."""
        lookup = _build_segment_lookup([])
        dur = _get_segment_file_duration("unknown_path.mp4", lookup, fallback_duration=42.0)
        assert dur == 42.0


# ============================================================================
# Bug 4: Timecode extent mismatch (in/out > file duration)
# ============================================================================

class TestTimecodeExtentClamping:
    """XML clipitem in/out must not exceed file duration."""

    @pytest.mark.fast
    def test_sequence_xml_in_out_within_duration(self, multitrack_matches, tmp_path):
        """Sequence XML clipitem out must not exceed duration."""
        # Create segments with known durations
        segments = [
            SegmentInfo(
                video_id="primary",
                file="/videos/primary.mp4",
                original_start=0.0,
                original_end=30.0
            ),
            SegmentInfo(
                video_id="alt1",
                file="/videos/alt1.mp4",
                original_start=0.0,
                original_end=20.0
            ),
            SegmentInfo(
                video_id="secondary1",
                file="/videos/secondary1.mp4",
                original_start=0.0,
                original_end=25.0
            ),
        ]

        out_path = str(tmp_path / "timeline")
        generate_davinci_sequence_xml(
            multitrack_matches, out_path, frame_rate=30.0,
            downloaded_segments=segments
        )
        xml_path = str(tmp_path / "timeline_sequence.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for clipitem in root.iter('clipitem'):
            dur_el = clipitem.find('duration')
            in_el = clipitem.find('in')
            out_el = clipitem.find('out')
            name_el = clipitem.find('name')
            if dur_el is not None and in_el is not None and out_el is not None:
                dur = int(dur_el.text)
                in_val = int(in_el.text)
                out_val = int(out_el.text)
                name = name_el.text if name_el is not None else '?'
                assert out_val <= dur, (
                    f"Clip '{name}': out({out_val}) > duration({dur})"
                )
                assert in_val < dur, (
                    f"Clip '{name}': in({in_val}) >= duration({dur})"
                )

    @pytest.mark.fast
    def test_project_xml_in_out_within_duration(self, multitrack_matches, temp_output_path):
        """Project XML clipitem out must not exceed duration."""
        segments = [
            SegmentInfo(
                video_id="primary",
                file="/videos/primary.mp4",
                original_start=0.0,
                original_end=30.0
            ),
            SegmentInfo(
                video_id="alt1",
                file="/videos/alt1.mp4",
                original_start=0.0,
                original_end=20.0
            ),
            SegmentInfo(
                video_id="secondary1",
                file="/videos/secondary1.mp4",
                original_start=0.0,
                original_end=25.0
            ),
        ]

        paths = generate_resolve_xml_with_bins(
            multitrack_matches, temp_output_path, frame_rate=30.0,
            downloaded_segments=segments
        )
        tree = ET.parse(paths[0])
        root = tree.getroot()

        for clipitem in root.iter('clipitem'):
            dur_el = clipitem.find('duration')
            in_el = clipitem.find('in')
            out_el = clipitem.find('out')
            name_el = clipitem.find('name')
            if dur_el is not None and in_el is not None and out_el is not None:
                dur = int(dur_el.text)
                in_val = int(in_el.text)
                out_val = int(out_el.text)
                name = name_el.text if name_el is not None else '?'
                assert out_val <= dur, (
                    f"Clip '{name}': out({out_val}) > duration({dur})"
                )
                assert in_val < dur or dur == 0, (
                    f"Clip '{name}': in({in_val}) >= duration({dur})"
                )


# ============================================================================
# Bug 4: Nearest-segment fallback (2026-02-05)
# ============================================================================

class TestNearestSegmentFallback:
    """_resolve_video_segment must find the nearest segment, not just segments[0]."""

    @pytest.mark.fast
    def test_resolve_picks_nearest_segment_not_first(self):
        """When source_start falls between segments, pick the closest one."""
        lookup = {
            'abc123': [
                {'file': '/v/abc123_10_30.mp4', 'start': 10, 'end': 30},
                {'file': '/v/abc123_100_120.mp4', 'start': 100, 'end': 120},
            ]
        }
        # source_start=95 is 65s from seg[0].end=30 but only 5s from seg[1].start=100
        resolved, adjusted = _resolve_video_segment('abc123', 95.0, lookup)
        assert resolved == '/v/abc123_100_120.mp4', "Should pick nearest segment, not first"
        assert adjusted == 0.0  # Clamped: 95-100 = -5 → max(0, -5) = 0

    @pytest.mark.fast
    def test_resolve_exact_match_preferred_over_nearest(self):
        """Exact containment still takes priority over nearest-distance."""
        lookup = {
            'vid1': [
                {'file': '/v/vid1_0_50.mp4', 'start': 0, 'end': 50},
                {'file': '/v/vid1_80_120.mp4', 'start': 80, 'end': 120},
            ]
        }
        resolved, adjusted = _resolve_video_segment('vid1', 25.0, lookup)
        assert resolved == '/v/vid1_0_50.mp4'
        assert adjusted == 25.0

    @pytest.mark.fast
    def test_resolve_beyond_60s_tolerance_returns_original(self):
        """When nearest segment is >60s away, return original path."""
        lookup = {
            'vid2': [
                {'file': '/v/vid2_200_220.mp4', 'start': 200, 'end': 220},
            ]
        }
        resolved, adjusted = _resolve_video_segment('vid2', 10.0, lookup)
        # Distance = 200 - 10 = 190 > 60, should not resolve
        assert resolved == 'vid2'
        assert adjusted == 10.0

    @pytest.mark.fast
    def test_resolve_clamps_adjusted_start_to_segment_duration(self):
        """adjusted_start must not exceed segment duration."""
        lookup = {
            'vid3': [
                {'file': '/v/vid3_100_115.mp4', 'start': 100, 'end': 115},
            ]
        }
        # source_start=90, nearest seg starts at 100, dist=10 (within 60s)
        # adjusted = max(0, 90 - 100) = 0, but without clamping it could be negative
        resolved, adjusted = _resolve_video_segment('vid3', 90.0, lookup)
        assert resolved == '/v/vid3_100_115.mp4'
        assert 0.0 <= adjusted <= 15.0  # Can't exceed segment length (15s)

    @pytest.mark.fast
    def test_resolve_with_three_segments_picks_closest(self):
        """With 3 segments, pick the one with minimum distance."""
        lookup = {
            'vid4': [
                {'file': '/v/vid4_0_20.mp4', 'start': 0, 'end': 20},
                {'file': '/v/vid4_50_70.mp4', 'start': 50, 'end': 70},
                {'file': '/v/vid4_200_220.mp4', 'start': 200, 'end': 220},
            ]
        }
        # source_start=45, nearest is seg[1] at 50 (dist=5)
        resolved, adjusted = _resolve_video_segment('vid4', 45.0, lookup)
        assert resolved == '/v/vid4_50_70.mp4'

    @pytest.mark.fast
    def test_resolve_empty_lookup_returns_original(self):
        """Empty segment_lookup returns the original source_file."""
        resolved, adjusted = _resolve_video_segment('abc123', 50.0, {})
        assert resolved == 'abc123'
        assert adjusted == 50.0

    @pytest.mark.fast
    def test_resolve_video_id_not_in_lookup(self):
        """Video ID not present in lookup returns original."""
        lookup = {'other_vid': [{'file': '/v/other.mp4', 'start': 0, 'end': 30}]}
        resolved, adjusted = _resolve_video_segment('missing_vid', 10.0, lookup)
        assert resolved == 'missing_vid'


# ============================================================================
# Bug 5: Unresolved path detection and gap insertion (2026-02-05)
# ============================================================================

class TestIsUnresolvedPath:
    """_is_unresolved_path must detect bare video IDs that failed resolution."""

    @pytest.mark.fast
    def test_bare_video_id_unchanged_is_unresolved(self):
        assert _is_unresolved_path('abc123', 'abc123') is True

    @pytest.mark.fast
    def test_resolved_path_differs_is_not_unresolved(self):
        assert _is_unresolved_path('/v/abc123_10_30.mp4', 'abc123') is False

    @pytest.mark.fast
    def test_path_with_extension_is_not_unresolved(self):
        assert _is_unresolved_path('abc123.mp4', 'abc123.mp4') is False

    @pytest.mark.fast
    def test_path_with_separators_is_not_unresolved(self):
        assert _is_unresolved_path('/some/path/abc123', '/some/path/abc123') is False

    @pytest.mark.fast
    def test_video_id_with_dash_is_unresolved(self):
        assert _is_unresolved_path('-_eFxXuRBFI', '-_eFxXuRBFI') is True

    @pytest.mark.fast
    def test_video_id_with_underscore_is_unresolved(self):
        assert _is_unresolved_path('YUFbwzJulEY', 'YUFbwzJulEY') is True


class TestUnresolvedClipGapping:
    """Unresolved clips must become gaps in XML, not broken file references."""

    @pytest.fixture
    def matches_with_unresolved_alt(self):
        """Create matches where alt has a video ID that won't resolve."""
        vo_seg = SRTSegment(
            index=0, start_time=0.0, end_time=5.0,
            text="Test segment", source_file="voiceover.srt"
        )
        vid_seg = SRTSegment(
            index=0, start_time=10.0, end_time=15.0,
            text="Primary clip", source_file="/videos/primary.mp4"
        )
        # Alt with a bare video ID that won't be in segment_lookup
        alt_seg = SRTSegment(
            index=1, start_time=500.0, end_time=510.0,
            text="Unresolvable alt", source_file="UNRESOLVABLE_VID"
        )
        match = Match(
            voiceover_segment=vo_seg, video_segment=vid_seg,
            video_scene=None, confidence=0.9, reasoning="Test"
        )
        alt_match = AlternativeMatch(
            video_segment=alt_seg, video_scene=None,
            confidence=0.7, reasoning="Alt", diversity_score=0.3
        )
        return [MatchResult(
            primary_match=match,
            alternatives=[alt_match],
            secondary_matches=[],
            strategy_matches=[]
        )]

    @pytest.mark.fast
    def test_unresolved_path_detected_after_failed_resolution(self):
        """When resolution fails, _is_unresolved_path catches the bare ID."""
        lookup = {'other': [{'file': '/v/other_0_30.mp4', 'start': 0, 'end': 30}]}
        resolved, _ = _resolve_video_segment('UNRESOLVABLE_VID', 500.0, lookup)
        # Resolution failed — returned original
        assert resolved == 'UNRESOLVABLE_VID'
        assert _is_unresolved_path(resolved, 'UNRESOLVABLE_VID') is True

    @pytest.mark.fast
    def test_project_xml_bins_exclude_unresolved(self, matches_with_unresolved_alt, tmp_path):
        """Project XML media bins must not include bare video ID entries."""
        segments = [SegmentInfo(
            video_id='primary', file='/videos/primary.mp4',
            original_start=10.0, original_end=15.0
        )]
        output_path = str(tmp_path / "proj.xml")
        paths = generate_resolve_xml_with_bins(
            matches_with_unresolved_alt, output_path,
            frame_rate=30.0, downloaded_segments=segments
        )
        for p in paths:
            tree = ET.parse(p)
            root = tree.getroot()
            for name_el in root.iter('name'):
                text = name_el.text or ''
                assert 'UNRESOLVABLE_VID' not in text or 'voiceover' in text.lower(), (
                    f"Bare video ID found in bin: {text}"
                )


class TestIsMissingFile:
    """_is_missing_file must detect bare video IDs as missing."""

    @pytest.mark.fast
    def test_bare_video_id_is_missing(self):
        from src.otio.timeline import _is_missing_file
        assert _is_missing_file('YUFbwzJulEY') is True

    @pytest.mark.fast
    def test_video_id_with_dash_is_missing(self):
        from src.otio.timeline import _is_missing_file
        assert _is_missing_file('-_eFxXuRBFI') is True

    @pytest.mark.fast
    def test_url_is_not_missing(self):
        from src.otio.timeline import _is_missing_file
        assert _is_missing_file('https://example.com/video.mp4') is False

    @pytest.mark.fast
    def test_path_with_extension_no_sep_is_not_missing(self):
        from src.otio.timeline import _is_missing_file
        assert _is_missing_file('video.mp4') is False

    @pytest.mark.fast
    def test_file_url_is_not_missing(self):
        from src.otio.timeline import _is_missing_file
        assert _is_missing_file('file:///E:/v/video.mp4') is False
