"""
Tests for Coverage Analyzer (High Matches Mode)

Tests the coverage analysis functionality that identifies weak segments
and tracks matching progress.
"""

import pytest
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

from src.matching.coverage_analyzer import (
    WeakSegment,
    CoverageReport,
    analyze_coverage,
    get_improvement_delta,
)


class TestWeakSegment:
    """Tests for WeakSegment dataclass"""

    def test_basic_creation(self):
        """Test creating a WeakSegment with required fields"""
        ws = WeakSegment(
            segment_id="S001",
            segment_index=1,
            text="Test segment text",
            current_confidence=0.5,
        )

        assert ws.segment_id == "S001"
        assert ws.segment_index == 1
        assert ws.text == "Test segment text"
        assert ws.current_confidence == 0.5
        assert ws.current_match is None
        assert ws.suggested_keywords == []

    def test_with_optional_fields(self):
        """Test creating WeakSegment with all optional fields"""
        ws = WeakSegment(
            segment_id="S002",
            segment_index=2,
            text="Another segment",
            current_confidence=0.3,
            current_match="video.mp4",
            suggested_keywords=["ocean", "waves"],
        )

        assert ws.current_match == "video.mp4"
        assert ws.suggested_keywords == ["ocean", "waves"]

    def test_to_dict(self):
        """Test WeakSegment serialization to dict"""
        ws = WeakSegment(
            segment_id="S003",
            segment_index=3,
            text="Segment for dict test",
            current_confidence=0.6,
            current_match="match.mp4",
            suggested_keywords=["test"],
        )

        d = ws.to_dict()

        assert d["segment_id"] == "S003"
        assert d["segment_index"] == 3
        assert d["text"] == "Segment for dict test"
        assert d["current_confidence"] == 0.6
        assert d["current_match"] == "match.mp4"
        assert d["suggested_keywords"] == ["test"]


class TestCoverageReport:
    """Tests for CoverageReport dataclass"""

    def test_basic_creation(self):
        """Test creating a CoverageReport with required fields"""
        report = CoverageReport(
            total_segments=100,
            high_confidence=60,
            medium_confidence=25,
            low_confidence=15,
            coverage_ratio=0.60,
        )

        assert report.total_segments == 100
        assert report.high_confidence == 60
        assert report.medium_confidence == 25
        assert report.low_confidence == 15
        assert report.coverage_ratio == 0.60
        assert report.target_confidence == 0.90  # Default

    def test_with_weak_segments(self):
        """Test CoverageReport with weak segments list"""
        weak = [
            WeakSegment("S001", 1, "text", 0.5),
            WeakSegment("S002", 2, "text", 0.3),
        ]
        report = CoverageReport(
            total_segments=10,
            high_confidence=8,
            medium_confidence=1,
            low_confidence=1,
            coverage_ratio=0.80,
            weak_segments=weak,
        )

        assert len(report.weak_segments) == 2

    def test_to_dict(self):
        """Test CoverageReport serialization"""
        weak = [WeakSegment("S001", 1, "text", 0.5)]
        report = CoverageReport(
            total_segments=50,
            high_confidence=40,
            medium_confidence=5,
            low_confidence=5,
            coverage_ratio=0.80,
            weak_segments=weak,
            target_confidence=0.85,
        )

        d = report.to_dict()

        assert d["total_segments"] == 50
        assert d["high_confidence"] == 40
        assert d["coverage_ratio"] == 0.80
        assert d["target_confidence"] == 0.85
        assert d["weak_segment_count"] == 1

    def test_summary(self):
        """Test human-readable summary generation"""
        report = CoverageReport(
            total_segments=100,
            high_confidence=85,
            medium_confidence=10,
            low_confidence=5,
            coverage_ratio=0.85,
            target_confidence=0.90,
        )

        summary = report.summary()

        assert "85.0%" in summary
        assert "85/100" in summary
        assert "High: 85" in summary
        assert "Medium: 10" in summary
        assert "Low: 5" in summary


class TestAnalyzeCoverage:
    """Tests for analyze_coverage function"""

    @pytest.fixture
    def simple_segment(self):
        """Create a simple voiceover segment mock"""
        @dataclass
        class SimpleSegment:
            index: int
            text: str
        return SimpleSegment

    @pytest.fixture
    def simple_match(self):
        """Create a simple match mock"""
        @dataclass
        class SimpleMatch:
            segment_index: int
            confidence: float
            video_file: str = ""
        return SimpleMatch

    def test_empty_segments(self):
        """Test with no voiceover segments"""
        report = analyze_coverage(matches=[], voiceover_segments=[])

        assert report.total_segments == 0
        assert report.high_confidence == 0
        assert report.coverage_ratio == 0.0

    def test_all_high_confidence(self, simple_segment, simple_match):
        """Test when all segments have high confidence matches"""
        segments = [simple_segment(i, f"Text {i}") for i in range(5)]
        matches = [simple_match(i, 0.95) for i in range(5)]

        report = analyze_coverage(matches, segments, target_confidence=0.90)

        assert report.total_segments == 5
        assert report.high_confidence == 5
        assert report.medium_confidence == 0
        assert report.low_confidence == 0
        assert report.coverage_ratio == 1.0
        assert len(report.weak_segments) == 0

    def test_all_low_confidence(self, simple_segment, simple_match):
        """Test when all segments have low confidence matches"""
        segments = [simple_segment(i, f"Text {i}") for i in range(5)]
        matches = [simple_match(i, 0.3) for i in range(5)]

        report = analyze_coverage(matches, segments, target_confidence=0.90)

        assert report.high_confidence == 0
        assert report.low_confidence == 5
        assert report.coverage_ratio == 0.0
        assert len(report.weak_segments) == 5

    def test_mixed_confidence(self, simple_segment, simple_match):
        """Test mixed confidence distribution"""
        segments = [simple_segment(i, f"Text {i}") for i in range(10)]
        # 5 high (0.92), 3 medium (0.75), 2 low (0.5)
        matches = [
            simple_match(0, 0.92), simple_match(1, 0.95), simple_match(2, 0.91),
            simple_match(3, 0.93), simple_match(4, 0.90),
            simple_match(5, 0.75), simple_match(6, 0.78), simple_match(7, 0.72),
            simple_match(8, 0.50), simple_match(9, 0.40),
        ]

        report = analyze_coverage(matches, segments, target_confidence=0.90)

        assert report.high_confidence == 5
        assert report.medium_confidence == 3
        assert report.low_confidence == 2
        assert report.coverage_ratio == 0.5
        assert len(report.weak_segments) == 5  # medium + low

    def test_weak_segments_sorted_by_confidence(self, simple_segment, simple_match):
        """Test that weak segments are sorted by confidence (lowest first)"""
        segments = [simple_segment(i, f"Text {i}") for i in range(5)]
        matches = [
            simple_match(0, 0.50),
            simple_match(1, 0.30),
            simple_match(2, 0.70),
            simple_match(3, 0.20),
            simple_match(4, 0.60),
        ]

        report = analyze_coverage(matches, segments, target_confidence=0.90)

        # Check weak segments are sorted by confidence ascending
        confs = [ws.current_confidence for ws in report.weak_segments]
        assert confs == sorted(confs)
        assert confs[0] == 0.20  # Lowest first

    def test_dict_match_format(self, simple_segment):
        """Test with dict-format matches"""
        segments = [simple_segment(i, f"Text {i}") for i in range(3)]
        matches = [
            {"segment_index": 0, "confidence": 0.95, "video_file": "v1.mp4"},
            {"segment_index": 1, "confidence": 0.50, "video_file": "v2.mp4"},
            {"segment_index": 2, "confidence": 0.75, "video_file": "v3.mp4"},
        ]

        report = analyze_coverage(matches, segments, target_confidence=0.90)

        assert report.high_confidence == 1
        assert report.medium_confidence == 1
        assert report.low_confidence == 1

    def test_dict_segment_format(self, simple_match):
        """Test with dict-format voiceover segments"""
        segments = [
            {"index": 0, "text": "First segment"},
            {"index": 1, "text": "Second segment"},
        ]
        matches = [simple_match(0, 0.85), simple_match(1, 0.95)]

        report = analyze_coverage(matches, segments, target_confidence=0.90)

        assert report.total_segments == 2
        assert report.high_confidence == 1

    def test_custom_medium_threshold(self, simple_segment, simple_match):
        """Test with custom medium confidence threshold"""
        segments = [simple_segment(i, f"Text {i}") for i in range(4)]
        matches = [
            simple_match(0, 0.95),  # High
            simple_match(1, 0.80),  # Medium with custom threshold
            simple_match(2, 0.60),  # Low with custom threshold
            simple_match(3, 0.55),  # Low
        ]

        report = analyze_coverage(
            matches, segments,
            target_confidence=0.90,
            medium_threshold=0.75,
        )

        assert report.high_confidence == 1
        assert report.medium_confidence == 1
        assert report.low_confidence == 2

    def test_text_truncation_in_weak_segments(self, simple_match):
        """Test that weak segment text is truncated to 200 chars"""
        long_text = "A" * 300

        @dataclass
        class LongSegment:
            index: int
            text: str

        segments = [LongSegment(0, long_text)]
        matches = [simple_match(0, 0.50)]

        report = analyze_coverage(matches, segments)

        assert len(report.weak_segments) == 1
        assert len(report.weak_segments[0].text) == 200

    def test_missing_match_treated_as_zero_confidence(self, simple_segment):
        """Test segments without matches are treated as zero confidence"""
        segments = [simple_segment(i, f"Text {i}") for i in range(3)]
        matches = []  # No matches at all

        report = analyze_coverage(matches, segments)

        assert report.low_confidence == 3
        assert len(report.weak_segments) == 3

    def test_match_result_wrapper_format(self, simple_segment):
        """Test with MatchResultWrapper format (nested primary_match)"""
        segments = [simple_segment(0, "Test")]

        # Simulate MatchResultWrapper structure
        match = Mock()
        match.primary_match = Mock()
        match.primary_match.confidence = 0.85
        match.primary_match.voiceover_segment = Mock()
        match.primary_match.voiceover_segment.index = 0
        match.primary_match.video_segment = Mock()
        match.primary_match.video_segment.source_file = "test.mp4"
        match._simple_match = None

        report = analyze_coverage([match], segments)

        assert report.total_segments == 1
        assert report.medium_confidence == 1  # 0.85 is medium


class TestGetImprovementDelta:
    """Tests for get_improvement_delta function"""

    def test_positive_improvement(self):
        """Test detecting positive improvement"""
        delta, is_improving = get_improvement_delta(0.70, 0.85)

        assert delta == pytest.approx(0.15)
        assert is_improving is True

    def test_no_improvement(self):
        """Test when coverage stays the same"""
        delta, is_improving = get_improvement_delta(0.70, 0.70)

        assert delta == pytest.approx(0.0)
        assert is_improving is False

    def test_negative_improvement(self):
        """Test when coverage decreases"""
        delta, is_improving = get_improvement_delta(0.70, 0.65)

        assert delta == pytest.approx(-0.05)
        assert is_improving is False

    def test_minimal_improvement_not_counted(self):
        """Test improvement below threshold is not counted"""
        delta, is_improving = get_improvement_delta(0.70, 0.71, min_improvement=0.02)

        assert delta == pytest.approx(0.01)
        assert is_improving is False  # 1% < 2% threshold

    def test_exactly_at_threshold(self):
        """Test improvement exactly at threshold"""
        delta, is_improving = get_improvement_delta(0.70, 0.72, min_improvement=0.02)

        assert delta == pytest.approx(0.02)
        assert is_improving is True  # 2% >= 2% threshold

    def test_custom_min_improvement(self):
        """Test with custom minimum improvement threshold"""
        # With strict threshold
        delta, is_improving = get_improvement_delta(0.70, 0.75, min_improvement=0.10)
        assert is_improving is False  # 5% < 10%

        # With lenient threshold
        delta, is_improving = get_improvement_delta(0.70, 0.75, min_improvement=0.01)
        assert is_improving is True  # 5% >= 1%
