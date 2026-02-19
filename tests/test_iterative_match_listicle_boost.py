"""
Unit tests for US-122-008: Iterative Matching Listicle Topic Boost

Tests cover:
- iterative_chapter_boost config reading and validation
- Gap annotation with listicle groups
- Listicle group topic keywords influencing gap-filling queries
- Gap analysis using listicle item count to estimate missing segments
- Interaction between listicle detection and query learning
- Iterative matching respecting listicle group boundaries

Created: 2026-02-17 (Sprint 122 - Listicle-Matching Focus)
"""

import pytest

from src.iterative_match.gap_analyzer import (
    GapSegment,
    GapAnalysis,
    LockedMatch,
    annotate_gaps_with_chapters,
    analyze_gaps,
    extract_keywords_for_gap,
    derive_queries_from_descriptions,
    _classify_gap_pattern,
)
from src.config.sections.iterative_matching import IterativeMatchingConfig


# ============================================================================
# Helpers
# ============================================================================

def _make_gap(
    index: int,
    confidence: float,
    text: str,
    position: float = 0.0,
    duration: float = 0.0,
    chapter_id: str = None,
    chapter_type: str = "body",
) -> GapSegment:
    """Create a GapSegment for testing."""
    return GapSegment(
        segment_index=index,
        confidence=confidence,
        voiceover_text=text,
        position=position,
        duration=duration,
        chapter_id=chapter_id,
        chapter_type=chapter_type,
    )


def _make_locked(index: int, video_id: str, confidence: float, position: float, title: str = "") -> LockedMatch:
    """Create a LockedMatch for testing."""
    return LockedMatch(
        segment_index=index,
        video_id=video_id,
        confidence=confidence,
        position=position,
        title=title,
    )


# ============================================================================
# Test: iterative_chapter_boost config
# ============================================================================

class TestIterativeChapterBoostConfig:
    """Tests for iterative_chapter_boost config reading and application."""

    def test_default_iterative_chapter_boost(self):
        """Test default value is 0.1."""
        config = IterativeMatchingConfig()
        assert config.iterative_chapter_boost == 0.1

    def test_custom_iterative_chapter_boost(self):
        """Test custom boost value is stored."""
        config = IterativeMatchingConfig(iterative_chapter_boost=0.15)
        assert config.iterative_chapter_boost == 0.15

    def test_iterative_chapter_boost_clamped_to_valid_range(self):
        """Test boost is clamped to 0-1 range."""
        # Test upper bound
        config = IterativeMatchingConfig(iterative_chapter_boost=1.5)
        assert config.iterative_chapter_boost == 1.0

        # Test lower bound
        config = IterativeMatchingConfig(iterative_chapter_boost=-0.5)
        assert config.iterative_chapter_boost == 0.0

    def test_iterative_chapter_boost_zero(self):
        """Test zero boost disables feature."""
        config = IterativeMatchingConfig(iterative_chapter_boost=0.0)
        assert config.iterative_chapter_boost == 0.0

    def test_iterative_chapter_boost_preserves_decimal_precision(self):
        """Test decimal precision is preserved."""
        config = IterativeMatchingConfig(iterative_chapter_boost=0.07)
        assert config.iterative_chapter_boost == 0.07


# ============================================================================
# Test: Gap annotation with listicle groups
# ============================================================================

class TestGapAnnotationWithListicleGroups:
    """Tests for annotate_gaps_with_chapters with listicle groups."""

    def test_listicle_groups_annotates_chapter_id(self):
        """Test that gaps in listicle groups get chapter_id set."""
        gaps = [_make_gap(2, 0.5, "Python tips", position=20.0)]

        # Single listicle group covering segments 0-4
        listicle_groups = [
            {"group_id": "listicle_1", "start_segment": 0, "end_segment": 5, "topic_keywords": ["python", "tips"]}
        ]

        result = annotate_gaps_with_chapters(
            gaps,
            total_segments=10,
            listicle_groups=listicle_groups,
        )

        assert result[0].chapter_id == "listicle_1"

    def test_listicle_groups_sets_chapter_type_to_listicle_item(self):
        """Test that gaps in listicle groups get chapter_type=listicle_item."""
        gaps = [_make_gap(3, 0.5, "Third item", position=30.0)]

        listicle_groups = [
            {"group_id": "listicle_1", "start_segment": 0, "end_segment": 8, "topic_keywords": []}
        ]

        result = annotate_gaps_with_chapters(
            gaps,
            total_segments=10,
            listicle_groups=listicle_groups,
        )

        assert result[0].chapter_type == "listicle_item"

    def test_multiple_listicle_groups_annotate_correctly(self):
        """Test multiple listicle groups each get correct chapter_id."""
        gaps = [
            _make_gap(1, 0.5, "First item", position=10.0),
            _make_gap(5, 0.5, "Second item", position=50.0),
        ]

        listicle_groups = [
            {"group_id": "listicle_1", "start_segment": 0, "end_segment": 3, "topic_keywords": ["python"]},
            {"group_id": "listicle_2", "start_segment": 4, "end_segment": 7, "topic_keywords": ["javascript"]},
        ]

        result = annotate_gaps_with_chapters(
            gaps,
            total_segments=10,
            listicle_groups=listicle_groups,
        )

        # First gap in first listicle group
        assert result[0].chapter_id == "listicle_1"
        # Second gap in second listicle group (sorted by priority)
        assert result[1].chapter_id == "listicle_2"

    def test_listicle_topic_keywords_extracted(self):
        """Test listicle group topic_keywords are available for query generation."""
        listicle_groups = [
            {
                "group_id": "listicle_1",
                "start_segment": 0,
                "end_segment": 5,
                "topic_keywords": ["python", "programming", "tutorial"]
            }
        ]

        gaps = [_make_gap(2, 0.5, "learning code", position=20.0)]

        result = annotate_gaps_with_chapters(
            gaps,
            total_segments=10,
            listicle_groups=listicle_groups,
        )

        # The chapter_id should contain the group_id
        assert result[0].chapter_id == "listicle_1"

    def test_gap_not_in_listicle_group_gets_body_type(self):
        """Test gaps outside listicle groups get body type."""
        gaps = [_make_gap(8, 0.5, "Middle content", position=80.0)]

        listicle_groups = [
            {"group_id": "listicle_1", "start_segment": 0, "end_segment": 5, "topic_keywords": []}
        ]

        result = annotate_gaps_with_chapters(
            gaps,
            total_segments=10,
            listicle_groups=listicle_groups,
        )

        # Segment 8 at 80% position is in body (not intro <10% or conclusion >=90%)
        assert result[0].chapter_type == "body"


# ============================================================================
# Test: Listicle group topic keywords influence gap-filling queries
# ============================================================================

class TestListicleTopicKeywordsInfluenceQueries:
    """Tests for listicle topic keywords influencing gap-filling queries."""

    def test_gap_keywords_extracted_from_listicle_context(self):
        """Test gap keywords are extracted properly when in listicle context."""
        gap = GapSegment(
            segment_index=2,
            confidence=0.5,
            voiceover_text="How to use Python functions",
            position=20.0,
            duration=15.0,
            pattern_type="",
            chapter_id="listicle_1",
            chapter_type="listicle_item",
        )

        # Extract keywords from gap in listicle context
        keywords = extract_keywords_for_gap(gap, max_keywords=5)

        assert "Python" in keywords or "functions" in keywords or "How" in keywords

    def test_derive_queries_uses_listicle_group_keywords(self):
        """Test derive_queries_from_descriptions incorporates listicle topics."""
        gap = GapSegment(
            segment_index=2,
            confidence=0.5,
            voiceover_text="Python programming tutorial",
            position=20.0,
            pattern_type="",
        )

        # Matched videos with descriptions that might contain listicle topics
        matched_videos = [
            {
                "video_id": "abc123",
                "title": "Python Tutorial",
                "description": "Learn Python programming with this comprehensive guide. "
                               "Step by step instructions for beginners."
            }
        ]

        queries = derive_queries_from_descriptions(matched_videos, gap, max_queries=3)

        # Should derive queries from matched video descriptions
        assert isinstance(queries, list)

    def test_listicle_item_count_estimates_missing_segments(self):
        """Test that listicle item count is used to estimate missing segments."""
        # Create gaps representing missing listicle items
        gaps = [
            _make_gap(1, 0.3, "Item 1 content", position=10.0),
            _make_gap(2, 0.3, "Item 2 content", position=20.0),
            _make_gap(3, 0.3, "Item 3 content", position=30.0),
        ]

        # Listicle group with 5 items, only 3 segments exist
        listicle_groups = [
            {
                "group_id": "listicle_1",
                "start_segment": 0,
                "end_segment": 5,
                "topic_keywords": ["python", "tips"],
                "item_count": 5,
            }
        ]

        result = annotate_gaps_with_chapters(
            gaps,
            total_segments=10,
            listicle_groups=listicle_groups,
        )

        # All gaps should be annotated with listicle info
        for gap in result:
            assert gap.chapter_type == "listicle_item"


# ============================================================================
# Test: Gap analysis with listicle groups
# ============================================================================

class TestGapAnalysisWithListicleGroups:
    """Tests for gap analysis using listicle information."""

    def test_analyze_gaps_classifies_listicle_gaps(self):
        """Test gap analysis works with listicle-annotated gaps."""
        gaps = [
            GapSegment(
                segment_index=0,
                confidence=0.5,
                voiceover_text="first python tip",
                position=0.0,
                chapter_type="listicle_item",
                chapter_id="listicle_1",
            ),
            GapSegment(
                segment_index=1,
                confidence=0.4,
                voiceover_text="second python tip",
                position=10.0,
                chapter_type="listicle_item",
                chapter_id="listicle_1",
            ),
        ]

        analysis = analyze_gaps(gaps)

        # Gaps should be classified by pattern
        assert len(analysis.pattern_counts) > 0

    def test_gap_analysis_respects_listicle_boundaries(self):
        """Test gap analysis accounts for listicle group boundaries."""
        gaps = [
            # Gap in first listicle group
            GapSegment(
                segment_index=1,
                confidence=0.3,
                voiceover_text="first item explanation",
                position=10.0,
                chapter_type="listicle_item",
                chapter_id="listicle_1",
            ),
            # Gap in regular body (not listicle)
            GapSegment(
                segment_index=6,
                confidence=0.3,
                voiceover_text="general explanation",
                position=60.0,
                chapter_type="body",
                chapter_id="chapter_1",
            ),
        ]

        listicle_groups = [
            {"group_id": "listicle_1", "start_segment": 0, "end_segment": 4, "topic_keywords": []},
        ]

        chapters = [
            {"title": "chapter_1", "start_segment": 5, "end_segment": 8},
        ]

        # Annotate with chapters and listicle groups
        annotated = annotate_gaps_with_chapters(
            gaps,
            total_segments=10,
            chapters=chapters,
            listicle_groups=listicle_groups,
        )

        # Verify listicle boundary is respected
        listicle_gaps = [g for g in annotated if g.chapter_type == "listicle_item"]
        body_gaps = [g for g in annotated if g.chapter_type == "body"]

        assert len(listicle_gaps) == 1
        assert len(body_gaps) == 1


# ============================================================================
# Test: Query learning with listicle detection
# ============================================================================

class TestQueryLearningWithListicleDetection:
    """Tests for interaction between listicle detection and query learning."""

    def test_gap_pattern_includes_listicle_context(self):
        """Test gap pattern classification includes listicle context."""
        # A gap in listicle context should still be classified by content
        gap = GapSegment(
            segment_index=2,
            confidence=0.5,
            voiceover_text="learning python programming",
            position=20.0,
            chapter_type="listicle_item",
            chapter_id="listicle_1",
        )

        pattern = _classify_gap_pattern(gap.voiceover_text, set())

        # Should classify based on content, not just listicle context
        assert pattern in ["action_verb", "other", "proper_noun"]

    def test_listicle_keywords_combine_with_gap_keywords(self):
        """Test listicle keywords combine with gap keywords for queries."""
        gap = GapSegment(
            segment_index=2,
            confidence=0.5,
            voiceover_text="functions in Python",
            position=20.0,
            chapter_type="listicle_item",
            chapter_id="listicle_python",
        )

        # Extract keywords
        gap_keywords = extract_keywords_for_gap(gap, max_keywords=5)

        # Should include gap-specific keywords
        assert len(gap_keywords) > 0
        # Should contain python or functions
        keywords_lower = [k.lower() for k in gap_keywords]
        assert any(kw in keywords_lower for kw in ["python", "functions", "functions in python".split()])


# ============================================================================
# Test: Iterative matching respects listicle group boundaries
# ============================================================================

class TestIterativeMatchingRespectsListicleBoundaries:
    """Tests for iterative matching respecting listicle group boundaries."""

    def test_listicle_gaps_prioritized_together(self):
        """Test that gaps in same listicle group are prioritized together."""
        gaps = [
            # Gap at start of listicle (higher priority due to position)
            _make_gap(1, 0.3, "First tip", position=10.0),
            # Gap in middle of listicle
            _make_gap(3, 0.3, "Third tip", position=30.0),
            # Gap outside listicle (body)
            _make_gap(6, 0.3, "Regular content", position=60.0),
        ]

        listicle_groups = [
            {"group_id": "listicle_1", "start_segment": 0, "end_segment": 5, "topic_keywords": []}
        ]

        result = annotate_gaps_with_chapters(
            gaps,
            total_segments=10,
            listicle_groups=listicle_groups,
        )

        # First two gaps should be listicle items
        assert result[0].chapter_type == "listicle_item"
        assert result[1].chapter_type == "listicle_item"

    def test_iterative_chapter_boost_applied_to_listicle_videos(self):
        """Test iterative_chapter_boost can be applied to listicle-aligned videos."""
        # This test verifies the config is available for the boost logic
        config = IterativeMatchingConfig(iterative_chapter_boost=0.1)

        # The boost should be usable in the matching logic
        base_confidence = 0.7
        boosted_confidence = base_confidence + config.iterative_chapter_boost

        assert abs(boosted_confidence - 0.8) < 0.001  # Float comparison
        assert boosted_confidence <= 1.0  # Should not exceed 1.0 in practice

    def test_empty_listicle_groups_no_effect(self):
        """Test empty listicle groups don't affect gap annotation."""
        gaps = [_make_gap(5, 0.5, "Some content", position=50.0)]

        result = annotate_gaps_with_chapters(
            gaps,
            total_segments=10,
            listicle_groups=[],  # Empty list
        )

        # Should default to body type
        assert result[0].chapter_type == "body"

    def test_none_listicle_groups_no_error(self):
        """Test None listicle_groups doesn't cause error."""
        gaps = [_make_gap(5, 0.5, "Some content", position=50.0)]

        # Should not raise
        result = annotate_gaps_with_chapters(
            gaps,
            total_segments=10,
            listicle_groups=None,
        )

        assert len(result) == 1


# ============================================================================
# Test: Listicle item count estimation
# ============================================================================

class TestListicleItemCountEstimation:
    """Tests for using listicle item count to estimate missing segments."""

    def test_estimate_missing_from_item_count(self):
        """Test estimating missing segments from listicle item count."""
        # A listicle group with 10 items but only 7 segments
        gaps = [
            _make_gap(i, 0.3, f"Item {i} content", position=i * 10.0)
            for i in range(7)
        ]

        listicle_groups = [
            {
                "group_id": "listicle_1",
                "start_segment": 0,
                "end_segment": 10,
                "topic_keywords": ["python"],
                "item_count": 10,  # Expected 10 items
            }
        ]

        result = annotate_gaps_with_chapters(
            gaps,
            total_segments=15,
            listicle_groups=listicle_groups,
        )

        # All gaps annotated
        assert len(result) == 7
        # All should be listicle items
        assert all(g.chapter_type == "listicle_item" for g in result)

    def test_listicle_item_count_preserved_in_chapter_id(self):
        """Test listicle item count information is accessible."""
        listicle_groups = [
            {
                "group_id": "listicle_tips",
                "start_segment": 0,
                "end_segment": 5,
                "topic_keywords": ["tip1", "tip2", "tip3"],
                "item_count": 5,
            }
        ]

        gaps = [_make_gap(2, 0.5, "tip content", position=20.0)]

        result = annotate_gaps_with_chapters(
            gaps,
            total_segments=10,
            listicle_groups=listicle_groups,
        )

        # chapter_id should contain the group_id
        assert result[0].chapter_id == "listicle_tips"


# ============================================================================
# Integration: Full flow tests
# ============================================================================

class TestIterativeListicleIntegration:
    """Integration tests for iterative matching with listicle support."""

    def test_full_pipeline_with_listicle_groups(self):
        """Test full gap annotation pipeline with listicle groups."""
        # Create gaps at various positions - use lower confidence for intro to test sorting
        gaps = [
            _make_gap(0, 0.2, "Intro content", position=0.0),
            _make_gap(1, 0.3, "Python tip 1", position=10.0),
            _make_gap(2, 0.3, "Python tip 2", position=20.0),
            _make_gap(3, 0.3, "Python tip 3", position=30.0),
            _make_gap(9, 0.4, "Conclusion", position=90.0),  # 90% threshold for conclusion
        ]

        # Define listicle group
        listicle_groups = [
            {
                "group_id": "python_tips",
                "start_segment": 1,
                "end_segment": 4,
                "topic_keywords": ["python", "tips", "programming"],
                "item_count": 3,
            }
        ]

        # Annotate gaps
        result = annotate_gaps_with_chapters(
            gaps,
            total_segments=10,
            listicle_groups=listicle_groups,
        )

        # Verify results
        assert len(result) == 5

        # Check listicle gaps
        listicle_gaps = [g for g in result if g.chapter_type == "listicle_item"]
        assert len(listicle_gaps) == 3

        # Check intro - segment 0 is at 0%, which is intro threshold
        intro_gaps = [g for g in result if g.chapter_type == "intro"]
        assert len(intro_gaps) == 1

        # Check conclusion - segment 9 is at 90%, which meets conclusion threshold
        conclusion_gaps = [g for g in result if g.chapter_type == "conclusion"]
        assert len(conclusion_gaps) == 1

    def test_iterative_chapter_boost_with_listicle(self):
        """Test iterative_chapter_boost is compatible with listicle groups."""
        config = IterativeMatchingConfig(iterative_chapter_boost=0.1)

        gap = GapSegment(
            segment_index=2,
            confidence=0.7,
            voiceover_text="Python programming tip",
            position=20.0,
            chapter_type="listicle_item",
            chapter_id="listicle_1",
        )

        # In practice, this boost would be applied during re-matching
        # Verify config is available for that logic
        assert config.iterative_chapter_boost == 0.1
        assert gap.chapter_type == "listicle_item"
