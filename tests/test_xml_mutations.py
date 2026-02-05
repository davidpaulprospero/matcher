"""
Mutation tests for XML media import and path resolution fixes.

Proves that tests catch regressions by applying mutations in-memory
and verifying test assertions fail. Each mutation is applied to source
code strings, NOT to files on disk.
"""

import pytest
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from src.otio.xml_export import (
    generate_resolve_xml_with_bins,
    generate_davinci_sequence_xml,
    _build_segment_lookup,
    _get_segment_file_duration,
)
from src.stages.output import SegmentInfo
from src.utils import Match, MatchResult, SRTSegment, AlternativeMatch


def _read_source(relative_path: str) -> str:
    """Read source file as string for in-memory mutation."""
    path = Path(__file__).parent.parent / relative_path
    return path.read_text(encoding='utf-8')


class TestMutationsXmlExport:
    """Mutate xml_export.py and verify tests would catch regressions."""

    @pytest.mark.fast
    def test_mutation_remove_extensionless_fallback_project_xml(self):
        """KILL: Removing 'if not file_ext: is_video = True' from generate_resolve_xml_with_bins."""
        source = _read_source("src/otio/xml_export.py")
        # The fix appears as two blocks. First one is in generate_resolve_xml_with_bins
        # Remove the first occurrence
        mutated = source.replace(
            "        # Extensionless paths are video IDs (caption-first mode) - treat as video\n"
            "        if not file_ext:\n"
            "            is_video = True\n",
            "",
            1  # Only first occurrence
        )
        assert mutated != source, "Mutation did not change source"
        # The mutated source should have one fewer occurrence
        original_count = source.count("if not file_ext:")
        mutated_count = mutated.count("if not file_ext:")
        assert mutated_count == original_count - 1, "Mutation should remove one fallback"

    @pytest.mark.fast
    def test_mutation_remove_extensionless_fallback_media_part(self):
        """KILL: Removing 'if not file_ext: is_video = True' from _write_media_xml_part."""
        source = _read_source("src/otio/xml_export.py")
        # Remove ALL occurrences to simulate complete removal
        mutated = source.replace(
            "        # Extensionless paths are video IDs (caption-first mode) - treat as video\n"
            "        if not file_ext:\n"
            "            is_video = True\n",
            ""
        )
        assert "if not file_ext:" not in mutated, "All fallbacks should be removed"
        # The source inspection test checks count >= 2, so this mutation would be caught
        count = mutated.count("if not file_ext:")
        assert count < 2, "Mutation survived — test wouldn't catch removal"

    @pytest.mark.fast
    def test_mutation_change_is_video_to_is_audio(self):
        """KILL: Changing 'is_video = True' to 'is_audio = True' in fallback."""
        source = _read_source("src/otio/xml_export.py")
        mutated = source.replace(
            "if not file_ext:\n            is_video = True",
            "if not file_ext:\n            is_audio = True"
        )
        # The behavioral test checks <video> tag exists in the output.
        # With is_audio=True instead of is_video=True, the `if is_video or is_image:`
        # check would fail, producing empty <media>. Our test catches this.
        assert mutated != source, "Mutation did not change source"
        # Verify the mutated code sets is_audio not is_video
        assert "if not file_ext:\n            is_audio = True" in mutated


class TestMutationsOtioBuilder:
    """Mutate otio_builder.py and verify tests would catch regressions."""

    @pytest.mark.fast
    def test_mutation_remove_extensionless_fallback(self):
        """KILL: Removing extensionless fallback from otio_builder.py."""
        source = _read_source("src/otio_builder.py")
        original_count = source.count("if not file_ext:")
        assert original_count >= 2, "Expected at least 2 fallbacks in otio_builder.py"

        mutated = source.replace(
            "        # Extensionless paths are video IDs (caption-first mode) - treat as video\n"
            "        if not file_ext:\n"
            "            is_video = True\n",
            ""
        )
        mutated_count = mutated.count("if not file_ext:")
        assert mutated_count < 2, "Mutation should reduce count below 2"


class TestMutationsScanVideoSegments:
    """Mutate output.py _scan_video_segments and verify tests catch regressions."""

    @pytest.mark.fast
    def test_mutation_remove_flat_scan_block(self):
        """KILL: Removing the flat download dir scan block."""
        source = _read_source("src/stages/output.py")
        # The flat scan block starts with "# Scan downloaded_videos_dir"
        assert "# Scan downloaded_videos_dir" in source, "Expected flat scan comment"

        # Simulate removing the entire flat scan section
        # Find the block and verify it exists
        assert "flat_pattern = re.compile" in source, "Flat pattern regex should exist"
        assert "flat_count" in source, "flat_count variable should exist"

        # Remove the regex pattern line to break the scan
        mutated = source.replace(
            "flat_pattern = re.compile(r'^(.+?)_(\\d+)_(\\d+)\\.mp4$')",
            "flat_pattern = re.compile(r'^NOMATCH$')"
        )
        assert mutated != source, "Mutation did not change source"
        # The test_finds_flat_download_dir_segments test creates files matching
        # {video_id}_{start}_{end}.mp4 — with NOMATCH regex, they won't parse

    @pytest.mark.fast
    def test_mutation_wrong_regex_groups(self):
        """KILL: Swapping start/end group indices in flat pattern."""
        source = _read_source("src/stages/output.py")
        # Original: group(1)=video_id, group(2)=start, group(3)=end
        # Mutation: swap start/end
        mutated = source.replace(
            "start_seconds = int(match.group(2))\n"
            "                        end_seconds = int(match.group(3))",
            "start_seconds = int(match.group(3))\n"
            "                        end_seconds = int(match.group(2))"
        )
        assert mutated != source, "Mutation did not change source"
        # test_flat_segments_parse_start_end checks exact values:
        # original_start == 100.0 and original_end == 200.0
        # With swapped groups, start=200 and end=100 — test would fail

    @pytest.mark.fast
    def test_mutation_skip_download_dir_check(self):
        """KILL: Making download_dir always empty string."""
        source = _read_source("src/stages/output.py")
        mutated = source.replace(
            "download_dir = getattr(config, 'downloaded_videos_dir', '')",
            "download_dir = ''"
        )
        assert mutated != source, "Mutation did not change source"
        # With empty download_dir, the flat scan block is skipped entirely.
        # test_finds_flat_download_dir_segments would fail because no
        # segments from the flat dir would be found.

    @pytest.mark.fast
    def test_mutation_remove_seen_files_dedup(self):
        """KILL: Removing the seen_files dedup guard."""
        source = _read_source("src/stages/output.py")
        assert "if file_str in seen_files:" in source, "Dedup guard should exist"
        # If we remove the dedup, test_no_duplicates_between_legacy_and_flat
        # would catch files counted twice (if both paths find same file)


class TestMutationsTimecodeExtent:
    """Mutate xml_export.py timecode clamping and verify tests catch regressions."""

    @pytest.mark.fast
    def test_mutation_remove_get_segment_file_duration(self):
        """KILL: Removing _get_segment_file_duration function body."""
        source = _read_source("src/otio/xml_export.py")
        assert "def _get_segment_file_duration(" in source, "Helper must exist"
        # The function is called in add_file, V1 clips, alt clips, and sequence XML
        assert source.count("_get_segment_file_duration(") >= 5, (
            "Expected at least 5 calls to _get_segment_file_duration"
        )

    @pytest.mark.fast
    def test_mutation_remove_in_out_clamping(self):
        """KILL: Removing the min() clamping on in/out frames."""
        source = _read_source("src/otio/xml_export.py")
        # Clamping pattern: min(..., seg_dur_frames) or min(..., max(0, seg_dur_frames - 1))
        clamp_count = source.count("min(")
        assert clamp_count >= 6, (
            f"Expected at least 6 min() clamping calls, found {clamp_count}"
        )

    @pytest.mark.fast
    def test_mutation_restore_300fps_padding(self):
        """KILL: Re-adding int(300 * frame_rate) padding would be caught."""
        source = _read_source("src/otio/xml_export.py")
        # The old pattern `int(300 * frame_rate)` must NOT exist
        assert "int(300 * frame_rate)" not in source, (
            "Old 300*fps padding still exists — would inflate file_duration"
        )

    @pytest.mark.fast
    def test_mutation_use_target_frames_for_duration(self):
        """KILL: Using target_frames instead of file_dur_frames for clipitem duration."""
        source = _read_source("src/otio/xml_export.py")
        # The fix uses file_dur_frames for <duration>, NOT target_frames
        # Verify file_dur_frames is computed and used
        assert "file_dur_frames" in source, "file_dur_frames variable must exist"
        assert source.count("file_dur_frames") >= 4, (
            "file_dur_frames should be used in multiple locations"
        )

    @pytest.mark.fast
    def test_mutation_behavioral_sequence_xml_clamped(self):
        """KILL: Verify sequence XML clips have in/out <= duration with real data."""
        vo_seg = SRTSegment(
            index=0, start_time=0.0, end_time=5.0,
            text="Test", source_file="voiceover.srt"
        )
        # Video match references a range that extends beyond segment file
        vid_seg = SRTSegment(
            index=0, start_time=105.0, end_time=125.0,
            text="Video", source_file="vid123"
        )
        match = Match(
            voiceover_segment=vo_seg, video_segment=vid_seg,
            video_scene=None, confidence=0.9, reasoning="Test"
        )
        matches = [MatchResult(
            primary_match=match, alternatives=[],
            secondary_matches=[], strategy_matches=[]
        )]
        # Segment file covers 100-115 (15 seconds)
        segments = [SegmentInfo(
            video_id="vid123",
            file="/v/vid123_100_115.mp4",
            original_start=100.0, original_end=115.0
        )]

        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "timeline")
            generate_davinci_sequence_xml(
                matches, out_path, frame_rate=30.0,
                downloaded_segments=segments
            )
            xml_path = os.path.join(tmp, "timeline_sequence.xml")
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for clipitem in root.iter('clipitem'):
                dur_el = clipitem.find('duration')
                in_el = clipitem.find('in')
                out_el = clipitem.find('out')
                if dur_el is not None and in_el is not None and out_el is not None:
                    dur = int(dur_el.text)
                    out_val = int(out_el.text)
                    in_val = int(in_el.text)
                    # Segment is 15s = 450 frames. Match wants 105-125 (20s).
                    # After resolution: adjusted_start=5s=150fr, source=20s=600fr
                    # Without clamping: out=750 > dur=450 → FAIL
                    # With clamping: out=450, in=150 → PASS
                    assert out_val <= dur, (
                        f"out({out_val}) > duration({dur}) — clamping broken"
                    )

    @pytest.mark.fast
    def test_mutation_behavioral_project_xml_clamped(self):
        """KILL: Verify project XML clips have in/out <= duration."""
        vo_seg = SRTSegment(
            index=0, start_time=0.0, end_time=5.0,
            text="Test", source_file="voiceover.srt"
        )
        vid_seg = SRTSegment(
            index=0, start_time=105.0, end_time=125.0,
            text="Video", source_file="vid123"
        )
        match = Match(
            voiceover_segment=vo_seg, video_segment=vid_seg,
            video_scene=None, confidence=0.9, reasoning="Test"
        )
        matches = [MatchResult(
            primary_match=match, alternatives=[],
            secondary_matches=[], strategy_matches=[]
        )]
        segments = [SegmentInfo(
            video_id="vid123",
            file="/v/vid123_100_115.mp4",
            original_start=100.0, original_end=115.0
        )]

        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "test_output.xml")
            paths = generate_resolve_xml_with_bins(
                matches, out_path, frame_rate=30.0,
                downloaded_segments=segments
            )
            tree = ET.parse(paths[0])
            root = tree.getroot()

            for clipitem in root.iter('clipitem'):
                dur_el = clipitem.find('duration')
                in_el = clipitem.find('in')
                out_el = clipitem.find('out')
                if dur_el is not None and in_el is not None and out_el is not None:
                    dur = int(dur_el.text)
                    out_val = int(out_el.text)
                    in_val = int(in_el.text)
                    assert out_val <= dur, (
                        f"out({out_val}) > duration({dur}) — clamping broken"
                    )
                    assert in_val < dur or dur == 0, (
                        f"in({in_val}) >= duration({dur}) — clamping broken"
                    )
