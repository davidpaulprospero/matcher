"""Tests for chapter detection models."""

import pytest
from src.chapter_detection.models import (
    ChapterCandidate,
    ChapterConfidence,
    ValidationResult,
    DetectionResult,
)


class TestChapterCandidate:
    """Test ChapterCandidate dataclass."""

    @pytest.mark.fast
    def test_create_with_defaults(self):
        """Test creating with default values."""
        chapter = ChapterCandidate()
        assert chapter.chapter_id == 0
        assert chapter.start_segment_idx == 0
        assert chapter.end_segment_idx == 0
        assert chapter.confidence == 0.8
        assert chapter.detection_strategy == "topic"

    @pytest.mark.fast
    def test_create_with_values(self):
        """Test creating with specified values."""
        chapter = ChapterCandidate(
            chapter_id=1,
            start_segment_idx=5,
            end_segment_idx=10,
            title="Test Chapter",
            location_name="Paris",
            confidence=0.9,
        )
        assert chapter.chapter_id == 1
        assert chapter.start_segment_idx == 5
        assert chapter.end_segment_idx == 10
        assert chapter.title == "Test Chapter"
        assert chapter.location_name == "Paris"
        assert chapter.confidence == 0.9

    @pytest.mark.fast
    def test_segment_range_property(self):
        """Test segment_range property."""
        chapter = ChapterCandidate(start_segment_idx=3, end_segment_idx=7)
        assert chapter.segment_range == (3, 7)

    @pytest.mark.fast
    def test_segment_count_property(self):
        """Test segment_count property."""
        chapter = ChapterCandidate(start_segment_idx=3, end_segment_idx=7)
        assert chapter.segment_count == 5  # 3, 4, 5, 6, 7

    @pytest.mark.fast
    def test_to_dict(self):
        """Test conversion to dict."""
        chapter = ChapterCandidate(
            chapter_id=1,
            title="Test",
            confidence=0.85,
        )
        d = chapter.to_dict()
        assert d['chapter_id'] == 1
        assert d['title'] == "Test"
        assert d['confidence'] == 0.85

    @pytest.mark.fast
    def test_from_dict_basic(self):
        """Test creating from dict."""
        data = {
            'chapter_id': 2,
            'start_segment_idx': 5,
            'end_segment_idx': 10,
            'title': 'From Dict',
            'confidence': 0.75,
        }
        chapter = ChapterCandidate.from_dict(data)
        assert chapter.chapter_id == 2
        assert chapter.start_segment_idx == 5
        assert chapter.end_segment_idx == 10
        assert chapter.title == 'From Dict'
        assert chapter.confidence == 0.75

    @pytest.mark.fast
    def test_from_dict_backward_compat(self):
        """Test from_dict handles missing new fields (backward compatibility)."""
        # Old checkpoint format without new fields
        data = {
            'chapter_id': 0,
            'start_segment_idx': 0,
            'end_segment_idx': 5,
            'location_name': 'Paris',
            'location_type': 'city',
            'title': 'Paris Tour',
        }
        chapter = ChapterCandidate.from_dict(data)

        # New fields should have defaults
        assert chapter.confidence == 0.8  # Default
        assert chapter.detection_strategy == 'legacy'  # Default
        assert chapter.boundary_reasoning == ''  # Default

    @pytest.mark.fast
    def test_from_location_chapter_dict(self):
        """Test creating from LocationChapter dict format."""
        lc_dict = {
            'chapter_id': 0,
            'start_segment_idx': 0,
            'end_segment_idx': 5,
            'location_name': 'Paris',
            'location_type': 'city',
            'visual_keywords': ['Eiffel Tower'],
            'context_keywords': ['romantic'],
            'title': 'Paris Tour',
            'topics': ['paris', 'france'],
            'location_data': {'name': 'Paris', 'country_code': 'FR'},
        }
        chapter = ChapterCandidate.from_location_chapter(lc_dict)
        assert chapter.location_name == 'Paris'
        assert chapter.visual_keywords == ['Eiffel Tower']
        assert chapter.location_data == {'name': 'Paris', 'country_code': 'FR'}


class TestChapterConfidence:
    """Test ChapterConfidence dataclass."""

    @pytest.mark.fast
    def test_defaults(self):
        """Test default confidence values."""
        conf = ChapterConfidence()
        assert conf.overall == 0.8
        assert conf.boundary_confidence == 0.8

    @pytest.mark.fast
    def test_to_dict(self):
        """Test conversion to dict."""
        conf = ChapterConfidence(overall=0.9, gap_score=0.3)
        d = conf.to_dict()
        assert d['overall'] == 0.9
        assert d['gap_score'] == 0.3


class TestDetectionResult:
    """Test DetectionResult dataclass."""

    @pytest.mark.fast
    def test_empty_result(self):
        """Test empty detection result."""
        result = DetectionResult(chapters=[], total_segments=0)
        assert len(result.chapters) == 0
        assert result.total_segments == 0
        assert result.fallback_used is False

    @pytest.mark.fast
    def test_with_chapters(self):
        """Test result with chapters."""
        chapters = [
            ChapterCandidate(chapter_id=0, title="Ch1"),
            ChapterCandidate(chapter_id=1, title="Ch2"),
        ]
        result = DetectionResult(
            chapters=chapters,
            total_segments=10,
            content_type='travel',
            detection_passes_run=['initial', 'validation'],
        )
        assert len(result.chapters) == 2
        assert result.content_type == 'travel'
        assert 'initial' in result.detection_passes_run

    @pytest.mark.fast
    def test_to_dict(self):
        """Test conversion to dict."""
        chapters = [ChapterCandidate(chapter_id=0, title="Test")]
        result = DetectionResult(chapters=chapters, total_segments=5)
        d = result.to_dict()
        assert len(d['chapters']) == 1
        assert d['total_segments'] == 5
