"""
Unit tests for US-73-011: Propagate video chapter titles to OTIO clip metadata.

Verifies:
- Chapter metadata appears on OTIO clips when video segment has chapter_title
- Clips without chapter_title have no 'chapter' key in metadata
- Chapter metadata survives OTIO serialization/deserialization round-trip
- XML export includes chapter title as <comment> element
"""

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import opentimelineio as otio
import pytest


@dataclass
class MockVoiceoverSegment:
    start: float
    end: float
    text: str = "Test voiceover text"
    start_time: float = None
    end_time: float = None

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = self.start
        if self.end_time is None:
            self.end_time = self.end


@dataclass
class MockVideoSegment:
    source_file: str
    start_time: float
    end_time: float
    text: str = "Test video text"
    chapter_title: str = ''
    chapter_index: Optional[int] = None


@dataclass
class MockMatch:
    file: str
    start: float
    end: float
    confidence: float = 0.8
    speed: float = 1.0
    reasoning: str = "Test reasoning"
    strategy: str = "embedding"
    is_keyword_match: bool = False
    is_visual_match: bool = False
    keyword: str = ""
    embedding_similarity: float = 0.85
    clip_reuse_count: int = 0
    voiceover_segment: MockVoiceoverSegment = None
    video_segment: MockVideoSegment = None

    def __post_init__(self):
        if self.voiceover_segment is None:
            self.voiceover_segment = MockVoiceoverSegment(self.start, self.end)
        if self.video_segment is None:
            self.video_segment = MockVideoSegment(self.file, self.start, self.end)


@dataclass
class MockMatchResult:
    primary: MockMatch
    alternatives: List[MockMatch] = field(default_factory=list)
    secondaries: List[MockMatch] = field(default_factory=list)
    strategies: Dict[str, MockMatch] = field(default_factory=dict)
    primary_match: MockMatch = None
    alternative_matches: List[MockMatch] = None
    secondary_matches: List[MockMatch] = None
    strategy_matches: List[MockMatch] = None

    def __post_init__(self):
        if self.primary_match is None:
            self.primary_match = self.primary
        if self.alternative_matches is None:
            self.alternative_matches = self.alternatives
        if self.secondary_matches is None:
            self.secondary_matches = self.secondaries
        if self.strategy_matches is None:
            self.strategy_matches = list(self.strategies.values()) if self.strategies else []


@dataclass
class MockOutputConfig:
    include_alternatives: bool = False
    num_alternatives: int = 0
    include_strategy_tracks: bool = False
    strategy_tracks: List[str] = field(default_factory=list)
    include_entity_images: bool = False
    include_entity_videos: bool = False
    voiceover_offset: float = 0.0
    min_gap_threshold: float = 0.0
    time_scale_factor: float = 1.0
    gap_mode: str = 'scale'


@dataclass
class MockConfig:
    output: MockOutputConfig = field(default_factory=MockOutputConfig)


def _make_match(source_file, vo_start, vo_end, vid_start, vid_end, chapter_title=''):
    """Helper to create a match with optional chapter_title on the video segment."""
    vo_seg = MockVoiceoverSegment(start=vo_start, end=vo_end)
    vid_seg = MockVideoSegment(
        source_file=source_file,
        start_time=vid_start,
        end_time=vid_end,
        chapter_title=chapter_title,
    )
    return MockMatch(
        file=source_file,
        start=vo_start,
        end=vo_end,
        voiceover_segment=vo_seg,
        video_segment=vid_seg,
    )


class TestChapterMetadataInOTIO:
    """Test chapter metadata propagation to OTIO clips."""

    @pytest.mark.fast
    def test_chapter_metadata_present_on_clip(self, tmp_path):
        """When video segment has chapter_title, the OTIO clip metadata includes 'chapter'."""
        from src.otio.timeline import create_timeline

        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b'\x00' * 100)

        match = _make_match(
            source_file=str(video_file),
            vo_start=0.0, vo_end=5.0,
            vid_start=10.0, vid_end=15.0,
            chapter_title="Introduction",
        )
        match_result = MockMatchResult(primary=match)
        config = MockConfig()

        timeline = create_timeline(
            matches=[match_result],
            config=config,
            frame_rate=30.0,
        )

        # Find the V1 primary clip
        v1_track = timeline.tracks[0]
        clips = [item for item in v1_track if isinstance(item, otio.schema.Clip)]
        assert len(clips) >= 1, "Expected at least one clip on V1"

        clip = clips[0]
        assert 'chapter' in clip.metadata, "Expected 'chapter' key in clip metadata"
        assert clip.metadata['chapter'] == "Introduction"

    @pytest.mark.fast
    def test_no_chapter_metadata_when_empty(self, tmp_path):
        """When video segment has no chapter_title, no 'chapter' key is added to metadata."""
        from src.otio.timeline import create_timeline

        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b'\x00' * 100)

        match = _make_match(
            source_file=str(video_file),
            vo_start=0.0, vo_end=5.0,
            vid_start=10.0, vid_end=15.0,
            chapter_title='',  # Empty = no chapter
        )
        match_result = MockMatchResult(primary=match)
        config = MockConfig()

        timeline = create_timeline(
            matches=[match_result],
            config=config,
            frame_rate=30.0,
        )

        v1_track = timeline.tracks[0]
        clips = [item for item in v1_track if isinstance(item, otio.schema.Clip)]
        assert len(clips) >= 1

        clip = clips[0]
        assert 'chapter' not in clip.metadata, "Empty chapter_title should not add 'chapter' to metadata"

    @pytest.mark.fast
    def test_chapter_metadata_survives_roundtrip(self, tmp_path):
        """Chapter metadata survives OTIO serialization and deserialization."""
        from src.otio.timeline import create_timeline

        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b'\x00' * 100)

        match = _make_match(
            source_file=str(video_file),
            vo_start=0.0, vo_end=5.0,
            vid_start=10.0, vid_end=15.0,
            chapter_title="Chapter 3: Advanced Topics",
        )
        match_result = MockMatchResult(primary=match)
        config = MockConfig()

        timeline = create_timeline(
            matches=[match_result],
            config=config,
            frame_rate=30.0,
        )

        # Serialize to OTIO JSON
        otio_path = str(tmp_path / "test_timeline.otio")
        otio.adapters.write_to_file(timeline, otio_path)

        # Deserialize
        loaded_timeline = otio.adapters.read_from_file(otio_path)

        # Verify chapter metadata survived
        v1_track = loaded_timeline.tracks[0]
        clips = [item for item in v1_track if isinstance(item, otio.schema.Clip)]
        assert len(clips) >= 1

        clip = clips[0]
        assert 'chapter' in clip.metadata, "chapter metadata should survive OTIO round-trip"
        assert clip.metadata['chapter'] == "Chapter 3: Advanced Topics"


class TestChapterMetadataInXML:
    """Test chapter title in XML export as <comment> element."""

    @pytest.mark.fast
    def test_xml_includes_chapter_comment(self, tmp_path):
        """XML export includes chapter title as <comment> on clips with chapter metadata."""
        from src.otio.xml_export import generate_resolve_xml_with_bins

        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b'\x00' * 100)

        match = _make_match(
            source_file=str(video_file),
            vo_start=0.0, vo_end=5.0,
            vid_start=10.0, vid_end=15.0,
            chapter_title="Getting Started",
        )
        match_result = MockMatchResult(primary=match)

        output_path = str(tmp_path / "output.xml")
        paths = generate_resolve_xml_with_bins(
            matches=[match_result],
            output_path=output_path,
            frame_rate=30.0,
            num_parts=1,
        )

        # Read the project XML
        project_xml = Path(paths[0]).read_text(encoding='utf-8')
        assert '<comment>Getting Started</comment>' in project_xml

    @pytest.mark.fast
    def test_xml_no_comment_without_chapter(self, tmp_path):
        """XML export does not include <comment> for clips without chapter metadata."""
        from src.otio.xml_export import generate_resolve_xml_with_bins

        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b'\x00' * 100)

        match = _make_match(
            source_file=str(video_file),
            vo_start=0.0, vo_end=5.0,
            vid_start=10.0, vid_end=15.0,
            chapter_title='',
        )
        match_result = MockMatchResult(primary=match)

        output_path = str(tmp_path / "output.xml")
        paths = generate_resolve_xml_with_bins(
            matches=[match_result],
            output_path=output_path,
            frame_rate=30.0,
            num_parts=1,
        )

        project_xml = Path(paths[0]).read_text(encoding='utf-8')
        assert '<comment>' not in project_xml

    @pytest.mark.fast
    def test_sequence_xml_includes_chapter_comment(self, tmp_path):
        """Sequence XML export includes chapter title as <comment> on clips."""
        from src.otio.xml_export import generate_davinci_sequence_xml

        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b'\x00' * 100)

        match = _make_match(
            source_file=str(video_file),
            vo_start=0.0, vo_end=5.0,
            vid_start=10.0, vid_end=15.0,
            chapter_title="Conclusion",
        )
        match_result = MockMatchResult(primary=match)

        output_path = str(tmp_path / "output.xml")
        result_path = generate_davinci_sequence_xml(
            matches=[match_result],
            output_path=output_path,
            frame_rate=30.0,
        )

        sequence_xml = Path(result_path).read_text(encoding='utf-8')
        assert '<comment>Conclusion</comment>' in sequence_xml

    @pytest.mark.fast
    def test_xml_escapes_special_chars_in_chapter(self, tmp_path):
        """XML export properly escapes special characters in chapter titles."""
        from src.otio.xml_export import generate_resolve_xml_with_bins

        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(b'\x00' * 100)

        match = _make_match(
            source_file=str(video_file),
            vo_start=0.0, vo_end=5.0,
            vid_start=10.0, vid_end=15.0,
            chapter_title="Q&A: What's Next?",
        )
        match_result = MockMatchResult(primary=match)

        output_path = str(tmp_path / "output.xml")
        paths = generate_resolve_xml_with_bins(
            matches=[match_result],
            output_path=output_path,
            frame_rate=30.0,
            num_parts=1,
        )

        project_xml = Path(paths[0]).read_text(encoding='utf-8')
        # Should have escaped & and '
        assert '<comment>' in project_xml
        assert '&amp;' in project_xml
