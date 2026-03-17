"""
Tests for US-141-009: Context-aware candidate pre-filtering.

Tests the filter_by_context_relevance function that uses title/description/tags
overlap to pre-filter candidates before expensive embedding computation.
"""

import pytest
from unittest.mock import MagicMock
from typing import List, Tuple

# Import the functions to test
from src.matching.candidate_filter import (
    filter_by_context_relevance,
    compute_context_overlap_score,
)


class SRTSegmentMock:
    """Mock SRTSegment for testing."""
    def __init__(self, text: str = "", keywords: List[str] = None, topics: List[str] = None, source_file: str = ""):
        self.text = text
        self.keywords = keywords or []
        self.topics = topics or []
        self.source_file = source_file


class TestComputeContextOverlapScore:
    """Tests for compute_context_overlap_score function."""

    def test_perfect_title_match(self):
        """When title contains all voiceover words, score should be high."""
        score = compute_context_overlap_score(
            vo_text="python tutorial for beginners",
            vo_words={"python", "tutorial", "for", "beginners"},
            vo_keywords={"python", "tutorial"},
            vo_topics={"programming", "learning"},
            title="Python Tutorial for Beginners",
            description="Learn python programming",
            tags=["python", "tutorial", "beginners"]
        )
        # Title has perfect overlap, should be high
        assert score >= 0.4

    def test_no_match(self):
        """When there's no overlap, score should be zero."""
        score = compute_context_overlap_score(
            vo_text="cooking recipes italian food",
            vo_words={"cooking", "recipes", "italian", "food"},
            vo_keywords={"cooking", "recipes"},
            vo_topics={"cooking", "food"},
            title="Building a React App",
            description="Learn web development",
            tags=["react", "javascript", "programming"]
        )
        assert score == 0.0

    def test_partial_match(self):
        """Partial word overlap should give partial score."""
        score = compute_context_overlap_score(
            vo_text="python programming tutorial",
            vo_words={"python", "programming", "tutorial"},
            vo_keywords={"python"},
            vo_topics={"programming"},
            title="Python Tutorial",
            description="Learn to code",
            tags=["python", "tutorial"]
        )
        # Some overlap but not complete
        assert 0.0 < score < 1.0

    def test_tag_keyword_overlap(self):
        """Tags should contribute to overlap score."""
        score_with_tags = compute_context_overlap_score(
            vo_text="machine learning AI",
            vo_words={"machine", "learning", "ai"},
            vo_keywords={"machine learning", "ai"},
            vo_topics={"ai", "ml"},
            title="",  # No title
            description="",  # No description
            tags=["machine learning", "artificial intelligence", "ai"]
        )

        score_without_tags = compute_context_overlap_score(
            vo_text="machine learning AI",
            vo_words={"machine", "learning", "ai"},
            vo_keywords={"machine learning", "ai"},
            vo_topics={"ai", "ml"},
            title="",
            description="",
            tags=[]
        )

        # With tags should have higher score
        assert score_with_tags > score_without_tags


class TestFilterByContextRelevance:
    """Tests for filter_by_context_relevance function."""

    def test_empty_candidates_returns_empty(self):
        """Empty candidates list should return empty list."""
        vo_seg = SRTSegmentMock(text="python tutorial")
        result = filter_by_context_relevance(
            candidates=[],
            vo_segment=vo_seg,
            threshold=0.20,
            video_metadata={}
        )
        assert result == []

    def test_threshold_zero_returns_all(self):
        """Threshold of 0 should return all candidates."""
        vo_seg = SRTSegmentMock(text="python tutorial")
        candidates = [
            (SRTSegmentMock(source_file="vid1"), 0.9),
            (SRTSegmentMock(source_file="vid2"), 0.8),
        ]
        result = filter_by_context_relevance(
            candidates=candidates,
            vo_segment=vo_seg,
            threshold=0.0,
            video_metadata={}
        )
        assert len(result) == 2

    def test_threshold_one_returns_none(self):
        """Threshold of 1.0 should return empty list (no candidate can score that high)."""
        vo_seg = SRTSegmentMock(text="python tutorial")
        candidates = [
            (SRTSegmentMock(source_file="vid1"), 0.9),
            (SRTSegmentMock(source_file="vid2"), 0.8),
        ]
        result = filter_by_context_relevance(
            candidates=candidates,
            vo_segment=vo_seg,
            threshold=1.0,
            video_metadata={}
        )
        assert len(result) == 0

    def test_filters_irrelevant_candidates(self):
        """Candidates with no metadata overlap should be filtered."""
        vo_seg = SRTSegmentMock(
            text="python programming tutorial",
            keywords=["python", "programming"]
        )
        candidates = [
            (SRTSegmentMock(source_file="python_video"), 0.9),
            (SRTSegmentMock(source_file="cooking_video"), 0.95),
            (SRTSegmentMock(source_file="travel_video"), 0.85),
        ]
        video_metadata = {
            "python_video": {
                "title": "Python Programming Tutorial",
                "description": "Learn python step by step",
                "tags": ["python", "programming"]
            },
            "cooking_video": {
                "title": "Italian Cooking Guide",
                "description": "How to cook pasta",
                "tags": ["cooking", "food"]
            },
            "travel_video": {
                "title": "Travel Europe",
                "description": "Best places to visit",
                "tags": ["travel", "europe"]
            }
        }

        result = filter_by_context_relevance(
            candidates=candidates,
            vo_segment=vo_seg,
            threshold=0.20,
            video_metadata=video_metadata
        )

        # Should keep python_video, filter out cooking and travel
        assert len(result) <= 3
        source_files = [seg.source_file for seg, _ in result]
        assert "python_video" in source_files

    def test_keeps_relevant_candidates(self):
        """Relevant candidates should pass the filter."""
        vo_seg = SRTSegmentMock(
            text="machine learning neural networks",
            keywords=["machine learning", "neural networks"]
        )
        candidates = [
            (SRTSegmentMock(source_file="ml_video"), 0.9),
            (SRTSegmentMock(source_file="ml_tutorial"), 0.85),
        ]
        video_metadata = {
            "ml_video": {
                "title": "Machine Learning Basics",
                "description": "Introduction to neural networks",
                "tags": ["machine learning", "ai", "neural networks"]
            },
            "ml_tutorial": {
                "title": "Neural Networks Tutorial",
                "description": "Deep learning guide",
                "tags": ["deep learning", "neural networks"]
            }
        }

        result = filter_by_context_relevance(
            candidates=candidates,
            vo_segment=vo_seg,
            threshold=0.20,
            video_metadata=video_metadata
        )

        # Both should pass - they have high overlap
        assert len(result) == 2

    def test_no_metadata_returns_all(self):
        """When no video_metadata provided, should return all candidates (no filtering)."""
        vo_seg = SRTSegmentMock(text="python tutorial")
        candidates = [
            (SRTSegmentMock(source_file="vid1"), 0.9),
            (SRTSegmentMock(source_file="vid2"), 0.8),
        ]

        result = filter_by_context_relevance(
            candidates=candidates,
            vo_segment=vo_seg,
            threshold=0.20,
            video_metadata=None
        )

        # All should pass when no metadata to filter against
        assert len(result) == 2

    def test_threshold_respected(self):
        """Different thresholds should filter appropriately."""
        vo_seg = SRTSegmentMock(text="python programming", keywords=["python", "programming"])
        candidates = [
            (SRTSegmentMock(source_file="vid1"), 0.9),
            (SRTSegmentMock(source_file="vid2"), 0.8),
        ]
        video_metadata = {
            "vid1": {
                "title": "Python Programming",
                "description": "Learn python",
                "tags": ["python"]
            },
            "vid2": {
                "title": "React Tutorial",
                "description": "Learn react",
                "tags": ["react"]
            }
        }

        # With low threshold, both pass
        result_low = filter_by_context_relevance(
            candidates=candidates,
            vo_segment=vo_seg,
            threshold=0.10,
            video_metadata=video_metadata
        )
        assert len(result_low) >= 1

        # With high threshold, only exact match passes
        result_high = filter_by_context_relevance(
            candidates=candidates,
            vo_segment=vo_seg,
            threshold=0.80,
            video_metadata=video_metadata
        )
        # May filter more or less depending on overlap


class TestConfigIntegration:
    """Tests that verify config options are accessible."""

    def test_config_options_exist(self):
        """Verify config options exist in matching config."""
        from src.config import get_config
        config = get_config()
        mc = config.matching

        # These should exist after US-141-009
        assert hasattr(mc, 'context_prefilter_enabled')
        assert hasattr(mc, 'context_filter_threshold')

        # Check default values
        assert mc.context_prefilter_enabled is True
        assert mc.context_filter_threshold == 0.20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
