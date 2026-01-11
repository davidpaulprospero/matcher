"""
Tests for segment utility classes and functions in src/utils.py

Tests dataclasses and functions for:
- Chapter - Chapter/topic sections
- SRTSegment - SRT segment representation
- parse_srt_file() - Parse SRT files
- write_srt_file() - Write SRT files
"""

import pytest
import sys
from pathlib import Path
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import Chapter, SRTSegment, parse_srt_file, write_srt_file


class TestChapter:
    """Test Chapter dataclass"""

    def test_chapter_initialization(self):
        """Test basic initialization"""
        chapter = Chapter(
            chapter_id=1,
            start_segment_idx=0,
            end_segment_idx=10,
            title="Introduction",
            topics=["overview", "background"]
        )

        assert chapter.chapter_id == 1
        assert chapter.start_segment_idx == 0
        assert chapter.end_segment_idx == 10
        assert chapter.title == "Introduction"
        assert chapter.topics == ["overview", "background"]

    def test_chapter_minimal_initialization(self):
        """Test with minimal required fields"""
        chapter = Chapter(
            chapter_id=1,
            start_segment_idx=5,
            end_segment_idx=15
        )

        assert chapter.chapter_id == 1
        assert chapter.start_segment_idx == 5
        assert chapter.end_segment_idx == 15
        assert chapter.title == ""
        assert chapter.topics == []

    def test_chapter_contains_segment(self):
        """Test contains_segment() method"""
        chapter = Chapter(
            chapter_id=1,
            start_segment_idx=10,
            end_segment_idx=20
        )

        # Within range
        assert chapter.contains_segment(10) is True
        assert chapter.contains_segment(15) is True
        assert chapter.contains_segment(20) is True

        # Outside range
        assert chapter.contains_segment(9) is False
        assert chapter.contains_segment(21) is False

    def test_chapter_contains_segment_single_segment(self):
        """Test contains_segment() with single-segment chapter"""
        chapter = Chapter(
            chapter_id=1,
            start_segment_idx=5,
            end_segment_idx=5
        )

        assert chapter.contains_segment(5) is True
        assert chapter.contains_segment(4) is False
        assert chapter.contains_segment(6) is False

    def test_chapter_to_dict(self):
        """Test to_dict() method"""
        chapter = Chapter(
            chapter_id=1,
            start_segment_idx=0,
            end_segment_idx=10,
            title="Chapter 1",
            topics=["topic1", "topic2"]
        )

        result = chapter.to_dict()

        assert result['chapter_id'] == 1
        assert result['start_segment_idx'] == 0
        assert result['end_segment_idx'] == 10
        assert result['title'] == "Chapter 1"
        assert result['topics'] == ["topic1", "topic2"]

    def test_chapter_multiple_topics(self):
        """Test with many topics"""
        topics = [f"topic{i}" for i in range(10)]
        chapter = Chapter(
            chapter_id=1,
            start_segment_idx=0,
            end_segment_idx=50,
            topics=topics
        )

        assert len(chapter.topics) == 10
        assert chapter.topics[0] == "topic0"
        assert chapter.topics[-1] == "topic9"


class TestSRTSegment:
    """Test SRTSegment dataclass"""

    def test_srt_segment_basic_initialization(self):
        """Test basic initialization"""
        segment = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="Hello world"
        )

        assert segment.index == 1
        assert segment.start_time == 0.0
        assert segment.end_time == 5.0
        assert segment.text == "Hello world"
        assert segment.source_file == ""
        assert segment.keywords == []
        assert segment.entities == []
        assert segment.topic_id is None
        assert segment.topics == []
        assert segment.is_broll is False

    def test_srt_segment_with_all_fields(self):
        """Test with all fields populated"""
        segment = SRTSegment(
            index=1,
            start_time=10.5,
            end_time=15.25,
            text="This is a test segment",
            source_file="video.mp4",
            keywords=["test", "segment"],
            entities=[{"text": "entity1", "type": "PERSON"}],
            topic_id=5,
            topics=["topic1", "topic2"],
            is_broll=True
        )

        assert segment.index == 1
        assert segment.start_time == 10.5
        assert segment.end_time == 15.25
        assert segment.text == "This is a test segment"
        assert segment.source_file == "video.mp4"
        assert segment.keywords == ["test", "segment"]
        assert len(segment.entities) == 1
        assert segment.topic_id == 5
        assert segment.topics == ["topic1", "topic2"]
        assert segment.is_broll is True

    def test_srt_segment_to_dict(self):
        """Test to_dict() method"""
        segment = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="Test text"
        )

        result = segment.to_dict()

        assert result['index'] == 1
        assert result['start_time'] == 0.0
        assert result['end_time'] == 5.0
        assert result['text'] == "Test text"
        assert result['source_file'] == ""
        assert result['keywords'] == []
        assert result['entities'] == []
        assert result['topic_id'] is None
        assert result['topics'] == []
        assert result['is_broll'] is False

    def test_srt_segment_to_dict_with_all_fields(self):
        """Test to_dict() with all fields"""
        segment = SRTSegment(
            index=1,
            start_time=10.5,
            end_time=15.25,
            text="Full segment",
            source_file="video.mp4",
            keywords=["key1", "key2"],
            entities=[{"text": "entity1"}],
            topic_id=3,
            topics=["topic1"],
            is_broll=True
        )

        result = segment.to_dict()

        assert result['index'] == 1
        assert result['start_time'] == 10.5
        assert result['end_time'] == 15.25
        assert result['text'] == "Full segment"
        assert result['source_file'] == "video.mp4"
        assert result['keywords'] == ["key1", "key2"]
        assert result['entities'] == [{"text": "entity1"}]
        assert result['topic_id'] == 3
        assert result['topics'] == ["topic1"]
        assert result['is_broll'] is True

    def test_srt_segment_duration_calculation(self):
        """Test duration can be calculated from times"""
        segment = SRTSegment(
            index=1,
            start_time=10.0,
            end_time=15.5,
            text="Test"
        )

        duration = segment.end_time - segment.start_time
        assert duration == 5.5

    def test_srt_segment_empty_text(self):
        """Test with empty text"""
        segment = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text=""
        )

        assert segment.text == ""


class TestParseSrtFile:
    """Test parse_srt_file() function"""

    def test_parse_srt_file_basic(self, tmp_path):
        """Test parsing basic SRT file"""
        srt_content = """1
00:00:00,000 --> 00:00:05,000
First segment

2
00:00:05,000 --> 00:00:10,000
Second segment

3
00:00:10,000 --> 00:00:15,000
Third segment
"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content, encoding='utf-8')

        segments = parse_srt_file(str(srt_file))

        assert len(segments) == 3
        assert segments[0].index == 1
        assert segments[0].text == "First segment"
        assert segments[1].index == 2
        assert segments[1].text == "Second segment"
        assert segments[2].index == 3
        assert segments[2].text == "Third segment"

    def test_parse_srt_file_timestamps(self, tmp_path):
        """Test timestamp parsing"""
        srt_content = """1
00:00:10,500 --> 00:00:15,750
Test segment
"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content, encoding='utf-8')

        segments = parse_srt_file(str(srt_file))

        assert len(segments) == 1
        assert segments[0].start_time == 10.5
        assert segments[0].end_time == 15.75

    def test_parse_srt_file_multiline_text(self, tmp_path):
        """Test parsing segment with multiline text"""
        srt_content = """1
00:00:00,000 --> 00:00:05,000
First line
Second line
Third line
"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content, encoding='utf-8')

        segments = parse_srt_file(str(srt_file))

        assert len(segments) == 1
        assert "First line" in segments[0].text
        assert "Second line" in segments[0].text
        assert "Third line" in segments[0].text

    def test_parse_srt_file_nonexistent(self):
        """Test with nonexistent file"""
        segments = parse_srt_file("/nonexistent/file.srt")
        assert segments == []

    def test_parse_srt_file_empty(self, tmp_path):
        """Test with empty file"""
        srt_file = tmp_path / "empty.srt"
        srt_file.write_text("", encoding='utf-8')

        segments = parse_srt_file(str(srt_file))
        assert segments == []

    def test_parse_srt_file_malformed_skips_bad_entries(self, tmp_path):
        """Test that malformed entries are skipped"""
        srt_content = """1
00:00:00,000 --> 00:00:05,000
Valid segment

BAD ENTRY
This should be skipped

2
00:00:05,000 --> 00:00:10,000
Another valid segment
"""
        srt_file = tmp_path / "test.srt"
        srt_file.write_text(srt_content, encoding='utf-8')

        segments = parse_srt_file(str(srt_file))

        # Should only get valid segments
        assert len(segments) == 2
        assert segments[0].index == 1
        assert segments[1].index == 2


class TestWriteSrtFile:
    """Test write_srt_file() function"""

    def test_write_srt_file_basic(self, tmp_path):
        """Test writing basic SRT file"""
        segments = [
            SRTSegment(index=1, start_time=0.0, end_time=5.0, text="First"),
            SRTSegment(index=2, start_time=5.0, end_time=10.0, text="Second"),
        ]

        output_file = tmp_path / "output.srt"
        write_srt_file(segments, str(output_file))

        assert output_file.exists()

        content = output_file.read_text(encoding='utf-8')
        assert "1\n" in content
        assert "First" in content
        assert "2\n" in content
        assert "Second" in content
        assert "-->" in content

    def test_write_srt_file_timestamps(self, tmp_path):
        """Test timestamp formatting"""
        segments = [
            SRTSegment(index=1, start_time=10.5, end_time=15.75, text="Test"),
        ]

        output_file = tmp_path / "output.srt"
        write_srt_file(segments, str(output_file))

        content = output_file.read_text(encoding='utf-8')
        assert "00:00:10,500" in content
        assert "00:00:15,750" in content

    def test_write_srt_file_empty_list(self, tmp_path):
        """Test with empty segment list"""
        output_file = tmp_path / "empty.srt"
        write_srt_file([], str(output_file))

        assert output_file.exists()
        content = output_file.read_text(encoding='utf-8')
        assert content == ""


class TestSrtRoundtrip:
    """Test roundtrip parsing and writing"""

    def test_srt_roundtrip(self, tmp_path):
        """Test parse → write → parse roundtrip"""
        # Create original file
        original_segments = [
            SRTSegment(index=1, start_time=0.0, end_time=5.0, text="First"),
            SRTSegment(index=2, start_time=5.0, end_time=10.0, text="Second"),
            SRTSegment(index=3, start_time=10.0, end_time=15.5, text="Third"),
        ]

        temp_file = tmp_path / "test.srt"
        write_srt_file(original_segments, str(temp_file))

        # Parse it back
        parsed_segments = parse_srt_file(str(temp_file))

        # Should match
        assert len(parsed_segments) == 3
        assert parsed_segments[0].text == "First"
        assert parsed_segments[0].start_time == 0.0
        assert parsed_segments[0].end_time == 5.0
        assert parsed_segments[1].text == "Second"
        assert parsed_segments[2].text == "Third"
        assert parsed_segments[2].end_time == 15.5


class TestSegmentEdgeCases:
    """Test edge cases for segment utilities"""

    def test_chapter_zero_length(self):
        """Test chapter with same start and end"""
        chapter = Chapter(
            chapter_id=1,
            start_segment_idx=10,
            end_segment_idx=10
        )

        assert chapter.contains_segment(10) is True
        assert chapter.contains_segment(9) is False

    def test_chapter_large_range(self):
        """Test chapter with large segment range"""
        chapter = Chapter(
            chapter_id=1,
            start_segment_idx=0,
            end_segment_idx=10000
        )

        assert chapter.contains_segment(5000) is True
        assert chapter.contains_segment(10001) is False

    def test_srt_segment_zero_duration(self):
        """Test segment with zero duration"""
        segment = SRTSegment(
            index=1,
            start_time=5.0,
            end_time=5.0,
            text="Instant"
        )

        assert segment.end_time - segment.start_time == 0.0

    def test_srt_segment_very_long_text(self):
        """Test segment with very long text"""
        long_text = "A" * 10000
        segment = SRTSegment(
            index=1,
            start_time=0.0,
            end_time=10.0,
            text=long_text
        )

        assert len(segment.text) == 10000

    def test_parse_srt_file_unicode_content(self, tmp_path):
        """Test parsing SRT with Unicode characters"""
        srt_content = """1
00:00:00,000 --> 00:00:05,000
Hello 世界 🌍
"""
        srt_file = tmp_path / "unicode.srt"
        srt_file.write_text(srt_content, encoding='utf-8')

        segments = parse_srt_file(str(srt_file))

        assert len(segments) == 1
        assert "世界" in segments[0].text
        assert "🌍" in segments[0].text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
