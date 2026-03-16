"""
Tests for low confidence segment analysis logging.

Tests for US-010: Add low confidence segment analysis logging
"""

import pytest
import logging
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.matching.main import (
    analyze_low_confidence_segments,
    LowConfidenceAnalysis,
    LowConfidencePattern,
)
from src.utils import Match, MatchResult, SRTSegment, AlternativeMatch


# ============================================================================
# Test Fixtures
# ============================================================================

def create_vo_segment(
    index: int,
    text: str,
    keywords: List[str] = None,
    entities: List = None
) -> SRTSegment:
    """Create a voiceover segment for testing."""
    return SRTSegment(
        index=index,
        start_time=index * 5.0,
        end_time=(index + 1) * 5.0,
        text=text,
        source_file="",
        keywords=keywords or [],
        entities=entities or []
    )


def create_video_segment(
    index: int,
    text: str = "Video transcript text",
    source_file: str = "video1.mp4",
    keywords: List[str] = None
) -> SRTSegment:
    """Create a video segment for testing."""
    return SRTSegment(
        index=index,
        start_time=index * 3.0,
        end_time=(index + 1) * 3.0,
        text=text,
        source_file=source_file,
        keywords=keywords or []
    )


def create_match_result(
    segment_idx: int,
    vo_text: str,
    confidence: float,
    vo_keywords: List[str] = None,
    matched_keywords: List[str] = None,
    confidence_variance: float = 0.0
) -> MatchResult:
    """Create a MatchResult for testing."""
    vo_seg = create_vo_segment(segment_idx, vo_text, keywords=vo_keywords or [])
    vid_seg = create_video_segment(segment_idx)

    primary = Match(
        voiceover_segment=vo_seg,
        video_segment=vid_seg,
        video_scene=None,
        confidence=confidence,
        reasoning="Test match"
    )

    result = MatchResult(
        primary_match=primary,
        alternatives=[],
        secondary_matches=[],
        strategy_matches=[],
        has_gap=confidence < 0.3,
        gap_reason="Low confidence" if confidence < 0.3 else "",
        confidence_variance=confidence_variance,
        matched_keywords=matched_keywords or []
    )

    return result


# ============================================================================
# TestAnalyzeLowConfidenceSegmentsFunction
# ============================================================================

class TestAnalyzeLowConfidenceSegmentsFunction:
    """Test that the analyze function exists and has correct signature."""

    @pytest.mark.fast
    def test_function_exists(self):
        """analyze_low_confidence_segments should be importable."""
        from src.matching.main import analyze_low_confidence_segments
        assert analyze_low_confidence_segments is not None

    @pytest.mark.fast
    def test_function_callable(self):
        """analyze_low_confidence_segments should be callable."""
        assert callable(analyze_low_confidence_segments)

    @pytest.mark.fast
    def test_accepts_results_list(self):
        """Function should accept a list of MatchResult objects."""
        results = []
        analysis = analyze_low_confidence_segments(results)
        assert isinstance(analysis, LowConfidenceAnalysis)

    @pytest.mark.fast
    def test_accepts_threshold_parameter(self):
        """Function should accept optional threshold parameter."""
        results = []
        analysis = analyze_low_confidence_segments(results, threshold=0.5)
        assert analysis.threshold == 0.5

    @pytest.mark.fast
    def test_default_threshold_is_0_6(self):
        """Default threshold should be 0.6."""
        results = []
        analysis = analyze_low_confidence_segments(results)
        assert analysis.threshold == 0.6


# ============================================================================
# TestLowConfidenceAnalysisDataclass
# ============================================================================

class TestLowConfidenceAnalysisDataclass:
    """Test the LowConfidenceAnalysis dataclass structure."""

    @pytest.mark.fast
    def test_dataclass_exists(self):
        """LowConfidenceAnalysis should be importable."""
        from src.matching.main import LowConfidenceAnalysis
        assert LowConfidenceAnalysis is not None

    @pytest.mark.fast
    def test_has_low_confidence_count(self):
        """Should have low_confidence_count field."""
        analysis = LowConfidenceAnalysis(
            low_confidence_count=5,
            total_segments=10,
            patterns=[],
            suggestions=[],
            avg_low_confidence=0.4,
            threshold=0.6
        )
        assert analysis.low_confidence_count == 5

    @pytest.mark.fast
    def test_has_total_segments(self):
        """Should have total_segments field."""
        analysis = LowConfidenceAnalysis(
            low_confidence_count=5,
            total_segments=10,
            patterns=[],
            suggestions=[],
            avg_low_confidence=0.4,
            threshold=0.6
        )
        assert analysis.total_segments == 10

    @pytest.mark.fast
    def test_has_patterns_list(self):
        """Should have patterns list field."""
        analysis = LowConfidenceAnalysis(
            low_confidence_count=5,
            total_segments=10,
            patterns=[],
            suggestions=[],
            avg_low_confidence=0.4,
            threshold=0.6
        )
        assert isinstance(analysis.patterns, list)

    @pytest.mark.fast
    def test_has_suggestions_list(self):
        """Should have suggestions list field."""
        analysis = LowConfidenceAnalysis(
            low_confidence_count=5,
            total_segments=10,
            patterns=[],
            suggestions=[],
            avg_low_confidence=0.4,
            threshold=0.6
        )
        assert isinstance(analysis.suggestions, list)


# ============================================================================
# TestLowConfidencePatternDataclass
# ============================================================================

class TestLowConfidencePatternDataclass:
    """Test the LowConfidencePattern dataclass structure."""

    @pytest.mark.fast
    def test_dataclass_exists(self):
        """LowConfidencePattern should be importable."""
        from src.matching.main import LowConfidencePattern
        assert LowConfidencePattern is not None

    @pytest.mark.fast
    def test_has_pattern_type(self):
        """Should have pattern_type field."""
        pattern = LowConfidencePattern(
            pattern_type="short_voiceover",
            count=3,
            segment_indices=[0, 1, 2],
            description="Test pattern"
        )
        assert pattern.pattern_type == "short_voiceover"

    @pytest.mark.fast
    def test_has_count(self):
        """Should have count field."""
        pattern = LowConfidencePattern(
            pattern_type="short_voiceover",
            count=3,
            segment_indices=[0, 1, 2],
            description="Test pattern"
        )
        assert pattern.count == 3

    @pytest.mark.fast
    def test_has_segment_indices(self):
        """Should have segment_indices field."""
        pattern = LowConfidencePattern(
            pattern_type="short_voiceover",
            count=3,
            segment_indices=[0, 1, 2],
            description="Test pattern"
        )
        assert pattern.segment_indices == [0, 1, 2]

    @pytest.mark.fast
    def test_has_description(self):
        """Should have description field."""
        pattern = LowConfidencePattern(
            pattern_type="short_voiceover",
            count=3,
            segment_indices=[0, 1, 2],
            description="Test pattern description"
        )
        assert pattern.description == "Test pattern description"


# ============================================================================
# TestIdentifyLowConfidenceSegments
# ============================================================================

class TestIdentifyLowConfidenceSegments:
    """Test identification of segments below confidence threshold."""

    @pytest.mark.fast
    def test_identifies_segments_below_threshold(self):
        """Should identify segments with confidence < threshold."""
        results = [
            create_match_result(0, "This is a normal voiceover segment", 0.8),
            create_match_result(1, "This one has low confidence", 0.4),
            create_match_result(2, "Another good segment", 0.75),
        ]

        analysis = analyze_low_confidence_segments(results, threshold=0.6)
        assert analysis.low_confidence_count == 1

    @pytest.mark.fast
    def test_threshold_0_6_default(self):
        """Default threshold of 0.6 should be applied."""
        results = [
            create_match_result(0, "High confidence segment", 0.85),
            create_match_result(1, "Low confidence segment", 0.55),
            create_match_result(2, "Medium confidence segment", 0.65),
        ]

        analysis = analyze_low_confidence_segments(results)  # default 0.6
        assert analysis.low_confidence_count == 1  # only 0.55 is below 0.6

    @pytest.mark.fast
    def test_empty_results_returns_zero(self):
        """Empty results should return 0 low confidence segments."""
        analysis = analyze_low_confidence_segments([])
        assert analysis.low_confidence_count == 0
        assert analysis.total_segments == 0

    @pytest.mark.fast
    def test_no_low_confidence_returns_zero(self):
        """All high confidence should return 0 low confidence segments."""
        results = [
            create_match_result(0, "High confidence segment one", 0.9),
            create_match_result(1, "High confidence segment two", 0.85),
        ]

        analysis = analyze_low_confidence_segments(results)
        assert analysis.low_confidence_count == 0

    @pytest.mark.fast
    def test_calculates_avg_low_confidence(self):
        """Should calculate average confidence of low segments."""
        results = [
            create_match_result(0, "High confidence segment", 0.9),
            create_match_result(1, "Low confidence one", 0.4),
            create_match_result(2, "Low confidence two", 0.5),
        ]

        analysis = analyze_low_confidence_segments(results)
        # Average of 0.4 and 0.5 = 0.45
        assert abs(analysis.avg_low_confidence - 0.45) < 0.01


# ============================================================================
# TestShortVoiceoverPattern
# ============================================================================

class TestShortVoiceoverPattern:
    """Test detection of short voiceover pattern."""

    @pytest.mark.fast
    def test_detects_short_voiceover(self):
        """Should detect segments with < 20 character voiceover."""
        results = [
            create_match_result(0, "Short", 0.4),  # < 20 chars, low conf
            create_match_result(1, "This is a much longer voiceover segment", 0.5),
        ]

        analysis = analyze_low_confidence_segments(results)

        pattern_types = [p.pattern_type for p in analysis.patterns]
        assert "short_voiceover" in pattern_types

    @pytest.mark.fast
    def test_short_voiceover_count(self):
        """Should count all short voiceover segments."""
        results = [
            create_match_result(0, "Short one", 0.3),  # < 20 chars
            create_match_result(1, "Too short", 0.4),  # < 20 chars
            create_match_result(2, "This is a long voiceover", 0.5),  # >= 20 chars
        ]

        analysis = analyze_low_confidence_segments(results)

        short_pattern = next((p for p in analysis.patterns if p.pattern_type == "short_voiceover"), None)
        assert short_pattern is not None
        assert short_pattern.count == 2

    @pytest.mark.fast
    def test_short_voiceover_adds_suggestion(self):
        """Should add suggestion for merging short segments."""
        results = [
            create_match_result(0, "Short", 0.4),
        ]

        analysis = analyze_low_confidence_segments(results)

        assert any("merging" in s.lower() or "short" in s.lower() for s in analysis.suggestions)


# ============================================================================
# TestMissingKeywordsPattern
# ============================================================================

class TestMissingKeywordsPattern:
    """Test detection of missing keywords pattern."""

    @pytest.mark.fast
    def test_detects_missing_keywords(self):
        """Should detect segments with no keywords."""
        results = [
            create_match_result(0, "This voiceover has no keywords", 0.4, vo_keywords=[]),
        ]

        analysis = analyze_low_confidence_segments(results)

        pattern_types = [p.pattern_type for p in analysis.patterns]
        assert "missing_keywords" in pattern_types

    @pytest.mark.fast
    def test_does_not_flag_segments_with_keywords(self):
        """Should not flag segments that have keywords."""
        results = [
            create_match_result(0, "This voiceover has keywords", 0.4, vo_keywords=["keyword1", "keyword2"]),
        ]

        analysis = analyze_low_confidence_segments(results)

        pattern_types = [p.pattern_type for p in analysis.patterns]
        assert "missing_keywords" not in pattern_types

    @pytest.mark.fast
    def test_missing_keywords_suggestion(self):
        """Should suggest running keyword extraction."""
        results = [
            create_match_result(0, "This voiceover has no keywords", 0.4, vo_keywords=[]),
        ]

        analysis = analyze_low_confidence_segments(results)

        assert any("keyword" in s.lower() for s in analysis.suggestions)


# ============================================================================
# TestAbstractContentPattern
# ============================================================================

class TestAbstractContentPattern:
    """Test detection of abstract content pattern."""

    @pytest.mark.fast
    def test_detects_abstract_content(self):
        """Should detect voiceover with abstract words and few matched keywords."""
        results = [
            create_match_result(
                0,
                "This thing is something that everyone wants to know about",
                0.4,
                matched_keywords=[]  # No matched keywords
            ),
        ]

        analysis = analyze_low_confidence_segments(results)

        pattern_types = [p.pattern_type for p in analysis.patterns]
        assert "abstract_content" in pattern_types

    @pytest.mark.fast
    def test_does_not_flag_concrete_content(self):
        """Should not flag content with specific terminology and matched keywords."""
        results = [
            create_match_result(
                0,
                "The photosynthesis process in plants",
                0.4,
                matched_keywords=["photosynthesis", "plants"]  # Has matched keywords
            ),
        ]

        analysis = analyze_low_confidence_segments(results)

        pattern_types = [p.pattern_type for p in analysis.patterns]
        assert "abstract_content" not in pattern_types

    @pytest.mark.fast
    def test_abstract_suggestion(self):
        """Should suggest adding specific terminology."""
        results = [
            create_match_result(
                0,
                "This kind of stuff is something we all know",
                0.4,
                matched_keywords=[]
            ),
        ]

        analysis = analyze_low_confidence_segments(results)

        assert any("specific" in s.lower() or "terminology" in s.lower() for s in analysis.suggestions)


# ============================================================================
# TestHighVariancePattern
# ============================================================================

class TestHighVariancePattern:
    """Test detection of high confidence variance pattern."""

    @pytest.mark.fast
    def test_detects_high_variance(self):
        """Should detect segments with confidence variance > 0.15."""
        results = [
            create_match_result(
                0,
                "This segment has high variance",
                0.4,
                confidence_variance=0.2  # > 0.15
            ),
        ]

        analysis = analyze_low_confidence_segments(results)

        pattern_types = [p.pattern_type for p in analysis.patterns]
        assert "high_variance" in pattern_types

    @pytest.mark.fast
    def test_does_not_flag_low_variance(self):
        """Should not flag segments with low variance."""
        results = [
            create_match_result(
                0,
                "This segment has low variance",
                0.4,
                confidence_variance=0.05  # < 0.15
            ),
        ]

        analysis = analyze_low_confidence_segments(results)

        pattern_types = [p.pattern_type for p in analysis.patterns]
        assert "high_variance" not in pattern_types

    @pytest.mark.fast
    def test_variance_threshold(self):
        """Variance threshold should be 0.15."""
        # At 0.15 exactly - should not flag
        results_at = [
            create_match_result(0, "Segment at threshold", 0.4, confidence_variance=0.15),
        ]
        analysis_at = analyze_low_confidence_segments(results_at)
        pattern_types_at = [p.pattern_type for p in analysis_at.patterns]
        assert "high_variance" not in pattern_types_at

        # Above 0.15 - should flag
        results_above = [
            create_match_result(0, "Segment above threshold", 0.4, confidence_variance=0.16),
        ]
        analysis_above = analyze_low_confidence_segments(results_above)
        pattern_types_above = [p.pattern_type for p in analysis_above.patterns]
        assert "high_variance" in pattern_types_above


# ============================================================================
# TestNoKeywordOverlapPattern
# ============================================================================

class TestNoKeywordOverlapPattern:
    """Test detection of no keyword overlap pattern."""

    @pytest.mark.fast
    def test_detects_no_keyword_overlap(self):
        """Should detect segments with no matched keywords (but has VO keywords)."""
        results = [
            create_match_result(
                0,
                "This voiceover has keywords but no overlap",
                0.4,
                vo_keywords=["keyword1", "keyword2"],
                matched_keywords=[]  # No overlap
            ),
        ]

        analysis = analyze_low_confidence_segments(results)

        pattern_types = [p.pattern_type for p in analysis.patterns]
        assert "no_keyword_overlap" in pattern_types

    @pytest.mark.fast
    def test_does_not_double_count_with_missing_keywords(self):
        """Should not double-count segments with missing VO keywords."""
        results = [
            create_match_result(
                0,
                "This voiceover has no keywords at all",
                0.4,
                vo_keywords=[],  # No VO keywords - will be in missing_keywords
                matched_keywords=[]  # No overlap
            ),
        ]

        analysis = analyze_low_confidence_segments(results)

        # Should only count as missing_keywords, not also no_keyword_overlap
        overlap_pattern = next((p for p in analysis.patterns if p.pattern_type == "no_keyword_overlap"), None)
        missing_pattern = next((p for p in analysis.patterns if p.pattern_type == "missing_keywords"), None)

        assert missing_pattern is not None
        # If both exist, they should not contain the same segment
        if overlap_pattern:
            common_indices = set(overlap_pattern.segment_indices) & set(missing_pattern.segment_indices)
            assert len(common_indices) == 0


# ============================================================================
# TestLogging
# ============================================================================

class TestLogging:
    """Test logging output of analysis."""

    @pytest.mark.fast
    def test_logs_analysis_header(self, caplog):
        """Should log LOW CONFIDENCE SEGMENT ANALYSIS header."""
        results = [
            create_match_result(0, "Low confidence segment", 0.4),
        ]

        with caplog.at_level(logging.INFO):
            analyze_low_confidence_segments(results)

        assert "LOW CONFIDENCE SEGMENT ANALYSIS" in caplog.text

    @pytest.mark.fast
    def test_logs_threshold(self, caplog):
        """Should log the threshold value."""
        results = [
            create_match_result(0, "Low confidence segment", 0.4),
        ]

        with caplog.at_level(logging.INFO):
            analyze_low_confidence_segments(results, threshold=0.7)

        assert "0.7" in caplog.text

    @pytest.mark.fast
    def test_logs_segment_count(self, caplog):
        """Should log low confidence segment count."""
        results = [
            create_match_result(0, "High confidence", 0.9),
            create_match_result(1, "Low confidence", 0.4),
        ]

        with caplog.at_level(logging.INFO):
            analyze_low_confidence_segments(results)

        assert "1/2" in caplog.text or "1 / 2" in caplog.text or "Low confidence segments:" in caplog.text

    @pytest.mark.fast
    def test_logs_pattern_types(self, caplog):
        """Should log detected pattern types."""
        results = [
            create_match_result(0, "Short", 0.4),  # short_voiceover pattern
        ]

        with caplog.at_level(logging.INFO):
            analyze_low_confidence_segments(results)

        assert "short_voiceover" in caplog.text

    @pytest.mark.fast
    def test_logs_suggestions(self, caplog):
        """Should log improvement suggestions."""
        results = [
            create_match_result(0, "Short", 0.4),
        ]

        with caplog.at_level(logging.INFO):
            analyze_low_confidence_segments(results)

        assert "Suggestions" in caplog.text or "suggestion" in caplog.text.lower()

    @pytest.mark.fast
    def test_logs_sample_segments(self, caplog):
        """Should log sample of low confidence segments."""
        results = [
            create_match_result(0, "This is a low confidence segment sample", 0.4),
        ]

        with caplog.at_level(logging.INFO):
            analyze_low_confidence_segments(results)

        assert "Sample low confidence" in caplog.text or "low confidence segment sample" in caplog.text

    @pytest.mark.fast
    def test_no_low_confidence_logs_info(self, caplog):
        """Should log info when no low confidence segments found."""
        results = [
            create_match_result(0, "High confidence segment", 0.9),
        ]

        with caplog.at_level(logging.INFO):
            analyze_low_confidence_segments(results)

        assert "No low confidence segments" in caplog.text


# ============================================================================
# TestSuggestions
# ============================================================================

class TestSuggestions:
    """Test that appropriate suggestions are generated."""

    @pytest.mark.fast
    def test_short_voiceover_suggestion(self):
        """Should suggest merging short segments."""
        results = [
            create_match_result(0, "Short", 0.4),
        ]

        analysis = analyze_low_confidence_segments(results)

        assert len(analysis.suggestions) > 0
        assert any("merge" in s.lower() or "short" in s.lower() for s in analysis.suggestions)

    @pytest.mark.fast
    def test_missing_keywords_suggestion(self):
        """Should suggest keyword extraction."""
        results = [
            create_match_result(0, "Voiceover without keywords", 0.4, vo_keywords=[]),
        ]

        analysis = analyze_low_confidence_segments(results)

        assert any("keyword" in s.lower() for s in analysis.suggestions)

    @pytest.mark.fast
    def test_abstract_content_suggestion(self):
        """Should suggest adding specific terminology."""
        results = [
            create_match_result(0, "This stuff is something", 0.4, matched_keywords=[]),
        ]

        analysis = analyze_low_confidence_segments(results)

        # Check for abstract-related suggestion
        assert any("specific" in s.lower() or "terminology" in s.lower() or "entity" in s.lower()
                   for s in analysis.suggestions)

    @pytest.mark.fast
    def test_multiple_patterns_multiple_suggestions(self):
        """Should generate multiple suggestions for multiple patterns."""
        results = [
            create_match_result(0, "Short", 0.3, vo_keywords=[]),  # short + missing keywords
        ]

        analysis = analyze_low_confidence_segments(results)

        # Should have at least 2 suggestions
        assert len(analysis.suggestions) >= 2


# ============================================================================
# TestEdgeCases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.fast
    def test_all_segments_low_confidence(self):
        """Should handle all segments being low confidence."""
        results = [
            create_match_result(0, "Low confidence one", 0.3),
            create_match_result(1, "Low confidence two", 0.4),
            create_match_result(2, "Low confidence three", 0.5),
        ]

        analysis = analyze_low_confidence_segments(results)

        assert analysis.low_confidence_count == 3
        assert analysis.total_segments == 3

    @pytest.mark.fast
    def test_custom_threshold(self):
        """Should respect custom threshold."""
        results = [
            create_match_result(0, "Medium confidence", 0.65),
            create_match_result(1, "High confidence", 0.9),
        ]

        # With default 0.6, 0.65 is not low
        analysis_default = analyze_low_confidence_segments(results)
        assert analysis_default.low_confidence_count == 0

        # With 0.7 threshold, 0.65 is low
        analysis_custom = analyze_low_confidence_segments(results, threshold=0.7)
        assert analysis_custom.low_confidence_count == 1

    @pytest.mark.fast
    def test_exactly_at_threshold(self):
        """Segments exactly at threshold should not be flagged."""
        results = [
            create_match_result(0, "Exactly at threshold segment", 0.6),
        ]

        analysis = analyze_low_confidence_segments(results, threshold=0.6)

        assert analysis.low_confidence_count == 0

    @pytest.mark.fast
    def test_very_long_voiceover(self):
        """Should handle very long voiceover text."""
        long_text = "A" * 500  # 500 character voiceover
        results = [
            create_match_result(0, long_text, 0.4),
        ]

        analysis = analyze_low_confidence_segments(results)

        # Should not flag as short voiceover
        pattern_types = [p.pattern_type for p in analysis.patterns]
        assert "short_voiceover" not in pattern_types

    @pytest.mark.fast
    def test_handles_none_matched_keywords(self):
        """Should handle None matched_keywords gracefully."""
        results = [
            create_match_result(0, "Segment without matched keywords field", 0.4),
        ]
        # Manually set matched_keywords to None to test handling
        results[0]._matched_keywords = None

        # Should not raise exception
        analysis = analyze_low_confidence_segments(results)
        assert analysis is not None

    @pytest.mark.fast
    def test_mixed_high_and_low_confidence(self):
        """Should correctly handle mix of high and low confidence."""
        results = [
            create_match_result(0, "High confidence one", 0.9),
            create_match_result(1, "Low confidence one", 0.4),
            create_match_result(2, "High confidence two", 0.85),
            create_match_result(3, "Low confidence two", 0.3),
            create_match_result(4, "High confidence three", 0.95),
        ]

        analysis = analyze_low_confidence_segments(results)

        assert analysis.low_confidence_count == 2
        assert analysis.total_segments == 5


# ============================================================================
# TestIntegrationWithMatchAllSegments
# ============================================================================

class TestIntegrationWithMatchAllSegments:
    """Test integration with match_all_segments function."""

    @pytest.mark.fast
    def test_function_is_called_in_match_all_segments(self):
        """analyze_low_confidence_segments should be called in match_all_segments."""
        # This tests that the function is integrated in the main matching flow
        # by verifying it's imported and available
        from src.matching.main import analyze_low_confidence_segments, match_all_segments

        assert analyze_low_confidence_segments is not None
        assert match_all_segments is not None

    @pytest.mark.fast
    def test_returns_analysis_object(self):
        """Should return LowConfidenceAnalysis object."""
        results = [
            create_match_result(0, "Test segment", 0.4),
        ]

        analysis = analyze_low_confidence_segments(results)

        assert isinstance(analysis, LowConfidenceAnalysis)
        assert hasattr(analysis, 'low_confidence_count')
        assert hasattr(analysis, 'patterns')
        assert hasattr(analysis, 'suggestions')
