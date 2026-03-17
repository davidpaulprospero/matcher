"""
Unit tests for the chapter_detector module.

Tests chapter extraction from video metadata (e.g., YouTube chapters).
"""

import pytest
from src.chapter_detector import VideoChapter, extract_chapters_from_metadata
from src.chapter_detector.detector import (
    _normalize_title,
    _parse_timestamp_string,
    _extract_chapters_from_description,
)


@pytest.mark.fast
class TestVideoChapterDataclass:
    """Tests for the VideoChapter dataclass."""

    def test_basic_creation(self):
        """VideoChapter can be created with basic fields."""
        chapter = VideoChapter(
            title="Introduction",
            start_time=0.0,
            end_time=60.0
        )
        assert chapter.title == "Introduction"
        assert chapter.start_time == 0.0
        assert chapter.end_time == 60.0

    def test_duration_calculation(self):
        """Duration is calculated from start and end times."""
        chapter = VideoChapter(
            title="Chapter 1",
            start_time=30.0,
            end_time=90.0
        )
        assert chapter.duration == 60.0

    def test_duration_calculation_zero_for_same_times(self):
        """Duration is zero when start equals end."""
        chapter = VideoChapter(
            title="Empty",
            start_time=50.0,
            end_time=50.0
        )
        assert chapter.duration == 0.0

    def test_to_dict(self):
        """VideoChapter serializes to dict correctly."""
        chapter = VideoChapter(
            title="Test",
            start_time=10.0,
            end_time=20.0
        )
        d = chapter.to_dict()
        assert d['title'] == "Test"
        assert d['start_time'] == 10.0
        assert d['duration'] == 10.0

    def test_from_dict(self):
        """VideoChapter deserializes from dict correctly."""
        data = {
            'title': 'From Dict',
            'start_time': 5.0,
            'end_time': 15.0,
            'duration': 10.0
        }
        chapter = VideoChapter.from_dict(data)
        assert chapter.title == 'From Dict'
        assert chapter.start_time == 5.0
        assert chapter.duration == 10.0


@pytest.mark.fast
class TestExtractChaptersFromMetadata:
    """Tests for chapter extraction from video metadata."""

    def test_empty_metadata_returns_empty_list(self):
        """Empty metadata returns empty list."""
        result = extract_chapters_from_metadata({})
        assert result == []

    def test_none_metadata_returns_empty_list(self):
        """None metadata returns empty list."""
        result = extract_chapters_from_metadata(None)
        assert result == []

    def test_no_chapters_field_returns_empty_list(self):
        """Metadata without chapters returns empty list."""
        metadata = {
            'title': 'Some Video',
            'description': 'No timestamps here'
        }
        result = extract_chapters_from_metadata(metadata)
        assert result == []

    def test_structured_youtube_chapters(self):
        """Extract chapters from YouTube structured metadata."""
        metadata = {
            'chapters': [
                {'title': 'Introduction', 'start_time': 0},
                {'title': 'Main Content', 'start_time': 30},
                {'title': 'Conclusion', 'start_time': 120}
            ]
        }
        result = extract_chapters_from_metadata(metadata, video_duration=180)

        assert len(result) == 3
        assert result[0].title == 'Introduction'
        assert result[0].start_time == 0
        assert result[0].end_time == 30
        assert result[0].duration == 30

        assert result[1].title == 'Main Content'
        assert result[1].start_time == 30
        assert result[1].end_time == 120

        assert result[2].title == 'Conclusion'
        assert result[2].start_time == 120
        assert result[2].end_time == 180

    def test_chapters_with_end_time(self):
        """Extract chapters that include end_time field."""
        metadata = {
            'chapters': [
                {'title': 'Part 1', 'start_time': 0, 'end_time': 60},
                {'title': 'Part 2', 'start_time': 60, 'end_time': 120}
            ]
        }
        result = extract_chapters_from_metadata(metadata)

        assert len(result) == 2
        assert result[0].duration == 60
        assert result[1].duration == 60


@pytest.mark.fast
class TestTitleParsing:
    """Tests for chapter title parsing and normalization."""

    def test_title_is_normalized(self):
        """Titles are stripped and normalized."""
        assert _normalize_title('  Test Title  ') == 'Test Title'

    def test_multiple_spaces_collapsed(self):
        """Multiple spaces are collapsed to single space."""
        assert _normalize_title('Title   with   spaces') == 'Title with spaces'

    def test_numbered_prefix_removed(self):
        """Numbered prefixes are removed."""
        assert _normalize_title('1. Introduction') == 'Introduction'
        assert _normalize_title('2) Chapter Two') == 'Chapter Two'

    def test_bullet_prefix_removed(self):
        """Bullet prefixes are removed."""
        assert _normalize_title('• Bullet Point') == 'Bullet Point'
        assert _normalize_title('- Dash Item') == 'Dash Item'

    def test_quotes_stripped(self):
        """Surrounding quotes are stripped."""
        assert _normalize_title('"Quoted Title"') == 'Quoted Title'
        assert _normalize_title("'Single Quoted'") == 'Single Quoted'

    def test_empty_title_returns_empty(self):
        """Empty string returns empty string."""
        assert _normalize_title('') == ''
        assert _normalize_title('   ') == ''


@pytest.mark.fast
class TestVideosWithoutChapters:
    """Tests for videos that don't have chapter data."""

    def test_empty_chapters_list(self):
        """Empty chapters list returns empty result."""
        metadata = {'chapters': []}
        result = extract_chapters_from_metadata(metadata)
        assert result == []

    def test_chapters_not_a_list(self):
        """Non-list chapters field is handled gracefully."""
        metadata = {'chapters': 'not a list'}
        result = extract_chapters_from_metadata(metadata)
        assert result == []

    def test_chapters_is_none(self):
        """None chapters field returns empty list."""
        metadata = {'chapters': None}
        result = extract_chapters_from_metadata(metadata)
        assert result == []


@pytest.mark.fast
class TestMalformedChapterData:
    """Tests for handling malformed chapter data gracefully."""

    def test_chapter_missing_title(self):
        """Chapters without title are skipped."""
        metadata = {
            'chapters': [
                {'start_time': 0},  # No title
                {'title': 'Valid', 'start_time': 30}
            ]
        }
        result = extract_chapters_from_metadata(metadata)
        assert len(result) == 1
        assert result[0].title == 'Valid'

    def test_chapter_missing_start_time(self):
        """Chapters without start_time are skipped."""
        metadata = {
            'chapters': [
                {'title': 'No Time'},  # No start_time
                {'title': 'Valid', 'start_time': 30}
            ]
        }
        result = extract_chapters_from_metadata(metadata)
        assert len(result) == 1
        assert result[0].title == 'Valid'

    def test_non_dict_chapters_skipped(self):
        """Non-dict items in chapters list are skipped."""
        metadata = {
            'chapters': [
                None,
                'invalid',
                123,
                {'title': 'Valid', 'start_time': 0}
            ]
        }
        result = extract_chapters_from_metadata(metadata)
        assert len(result) == 1
        assert result[0].title == 'Valid'

    def test_invalid_timestamp_skipped(self):
        """Invalid timestamps are handled gracefully."""
        metadata = {
            'chapters': [
                {'title': 'Bad Time', 'start_time': 'not-a-number'},
                {'title': 'Good Time', 'start_time': 60}
            ]
        }
        result = extract_chapters_from_metadata(metadata)
        assert len(result) == 1
        assert result[0].title == 'Good Time'


@pytest.mark.fast
class TestChapterDurationCalculation:
    """Tests for accurate chapter duration calculation."""

    def test_duration_from_explicit_end_time(self):
        """Duration calculated from explicit end_time."""
        metadata = {
            'chapters': [
                {'title': 'Ch1', 'start_time': 0, 'end_time': 45.5}
            ]
        }
        result = extract_chapters_from_metadata(metadata)
        assert result[0].duration == 45.5

    def test_duration_from_next_chapter_start(self):
        """Duration calculated from next chapter's start."""
        metadata = {
            'chapters': [
                {'title': 'First', 'start_time': 0},
                {'title': 'Second', 'start_time': 100}
            ]
        }
        result = extract_chapters_from_metadata(metadata)
        assert result[0].duration == 100

    def test_last_chapter_duration_from_video_length(self):
        """Last chapter duration uses video_duration."""
        metadata = {
            'chapters': [
                {'title': 'Intro', 'start_time': 0},
                {'title': 'Outro', 'start_time': 500}
            ]
        }
        result = extract_chapters_from_metadata(metadata, video_duration=600)
        assert result[1].end_time == 600
        assert result[1].duration == 100

    def test_negative_duration_prevented(self):
        """Negative durations are clamped to zero."""
        chapter = VideoChapter(
            title='Weird',
            start_time=100.0,
            end_time=50.0  # End before start
        )
        assert chapter.duration == 0.0


@pytest.mark.fast
class TestTimestampParsing:
    """Tests for timestamp string parsing."""

    def test_parse_mm_ss(self):
        """Parse MM:SS format."""
        assert _parse_timestamp_string('1:30') == 90.0
        assert _parse_timestamp_string('10:00') == 600.0

    def test_parse_hh_mm_ss(self):
        """Parse HH:MM:SS format."""
        assert _parse_timestamp_string('1:00:00') == 3600.0
        assert _parse_timestamp_string('1:30:30') == 5430.0

    def test_parse_with_decimal_seconds(self):
        """Parse timestamps with decimal seconds."""
        assert _parse_timestamp_string('1:30.5') == 90.5

    def test_parse_invalid_returns_none(self):
        """Invalid timestamps return None."""
        assert _parse_timestamp_string('invalid') is None
        assert _parse_timestamp_string('') is None
        assert _parse_timestamp_string(None) is None


@pytest.mark.fast
class TestDescriptionChapterExtraction:
    """Tests for extracting chapters from video descriptions."""

    def test_basic_description_chapters(self):
        """Extract chapters from timestamped description."""
        description = """
0:00 Introduction
1:30 Main Topic
5:00 Conclusion
        """
        result = _extract_chapters_from_description(description, video_duration=360)

        assert len(result) == 3
        assert result[0].title == 'Introduction'
        assert result[0].start_time == 0
        assert result[1].title == 'Main Topic'
        assert result[1].start_time == 90

    def test_description_with_dashes(self):
        """Parse description with dashes after timestamps."""
        description = """
0:00 - Intro
2:30 - Part One
        """
        result = _extract_chapters_from_description(description)
        assert len(result) == 2
        assert result[0].title == 'Intro'

    def test_description_with_brackets(self):
        """Parse description with bracketed timestamps."""
        description = """
[0:00] Opening
[3:45] Chapter 1
        """
        result = _extract_chapters_from_description(description)
        assert len(result) == 2
        assert result[0].title == 'Opening'

    def test_empty_description_returns_empty(self):
        """Empty description returns empty list."""
        assert _extract_chapters_from_description('') == []
        assert _extract_chapters_from_description(None) == []


@pytest.mark.fast
class TestAlternativeFieldNames:
    """Tests for handling alternative field names in metadata."""

    def test_name_field_instead_of_title(self):
        """Accept 'name' field for chapter title."""
        metadata = {
            'chapters': [
                {'name': 'Using Name', 'start_time': 0}
            ]
        }
        result = extract_chapters_from_metadata(metadata)
        assert len(result) == 1
        assert result[0].title == 'Using Name'

    def test_start_field_instead_of_start_time(self):
        """Accept 'start' field for start time."""
        metadata = {
            'chapters': [
                {'title': 'Alt Start', 'start': 45}
            ]
        }
        result = extract_chapters_from_metadata(metadata)
        assert len(result) == 1
        assert result[0].start_time == 45


@pytest.mark.fast
class TestEdgeCases:
    """Tests for edge cases in chapter detection."""

    def test_single_chapter(self):
        """Handle video with only one chapter."""
        metadata = {
            'chapters': [
                {'title': 'Full Video', 'start_time': 0}
            ]
        }
        result = extract_chapters_from_metadata(metadata, video_duration=300)
        assert len(result) == 1
        assert result[0].duration == 300

    def test_very_long_title_preserved(self):
        """Long titles are preserved."""
        long_title = "A" * 500
        metadata = {
            'chapters': [
                {'title': long_title, 'start_time': 0}
            ]
        }
        result = extract_chapters_from_metadata(metadata)
        assert result[0].title == long_title

    def test_unicode_titles(self):
        """Unicode characters in titles are preserved."""
        metadata = {
            'chapters': [
                {'title': '日本語タイトル', 'start_time': 0},
                {'title': 'Émoji 🎬 Chapter', 'start_time': 30}
            ]
        }
        result = extract_chapters_from_metadata(metadata)
        assert len(result) == 2
        assert result[0].title == '日本語タイトル'
        assert result[1].title == 'Émoji 🎬 Chapter'
