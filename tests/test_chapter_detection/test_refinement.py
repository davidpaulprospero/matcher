"""Tests for boundary refinement pass."""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.chapter_detection.passes.refinement import (
    run_boundary_refinement,
    compute_boundary_scores,
    _cosine_similarity,
    _has_transition_phrase,
    _refine_chapter_boundary,
)
from src.chapter_detection.models import ChapterCandidate


class TestComputeBoundaryScores:
    """Test boundary score computation."""

    @pytest.mark.fast
    def test_returns_empty_for_none(self):
        """Test returns empty list for None embeddings."""
        result = compute_boundary_scores(None)
        assert result == []

    @pytest.mark.fast
    def test_returns_empty_for_single_embedding(self):
        """Test returns empty list for single embedding."""
        embeddings = np.array([[1.0, 0.0, 0.0]])
        result = compute_boundary_scores(embeddings)
        assert result == []

    @pytest.mark.fast
    def test_computes_gap_scores(self):
        """Test computes gap scores between adjacent embeddings."""
        # Two identical vectors should have similarity=1, gap=0
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        result = compute_boundary_scores(embeddings)
        assert len(result) == 1
        assert result[0] == pytest.approx(0.0, abs=0.01)

    @pytest.mark.fast
    def test_high_gap_for_different_vectors(self):
        """Test high gap score for very different vectors."""
        # Orthogonal vectors should have similarity=0, gap=1
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        result = compute_boundary_scores(embeddings)
        assert len(result) == 1
        assert result[0] == pytest.approx(1.0, abs=0.01)

    @pytest.mark.fast
    def test_multiple_boundaries(self):
        """Test multiple boundary scores."""
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # Segment 0
            [0.9, 0.1, 0.0],  # Segment 1 - similar to 0
            [0.0, 1.0, 0.0],  # Segment 2 - different
        ])
        result = compute_boundary_scores(embeddings)
        assert len(result) == 2
        # First boundary: 0-1 should have low gap
        # Second boundary: 1-2 should have high gap
        assert result[0] < result[1]


class TestCosineSimilarity:
    """Test cosine similarity computation."""

    @pytest.mark.fast
    def test_identical_vectors(self):
        """Test similarity of identical vectors."""
        a = np.array([1.0, 2.0, 3.0])
        result = _cosine_similarity(a, a)
        assert result == pytest.approx(1.0, abs=0.01)

    @pytest.mark.fast
    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        result = _cosine_similarity(a, b)
        assert result == pytest.approx(0.0, abs=0.01)

    @pytest.mark.fast
    def test_opposite_vectors(self):
        """Test similarity of opposite vectors."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        result = _cosine_similarity(a, b)
        assert result == pytest.approx(-1.0, abs=0.01)

    @pytest.mark.fast
    def test_zero_vector(self):
        """Test similarity with zero vector."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 0.0, 0.0])
        result = _cosine_similarity(a, b)
        assert result == 0.0


class TestHasTransitionPhrase:
    """Test transition phrase detection."""

    @pytest.mark.fast
    def test_detects_now_let(self):
        """Test detects 'now let' phrase."""
        assert _has_transition_phrase("Now let's move to the next topic")

    @pytest.mark.fast
    def test_detects_moving_on(self):
        """Test detects 'moving on' phrase."""
        assert _has_transition_phrase("Moving on to the city center")

    @pytest.mark.fast
    def test_detects_finally(self):
        """Test detects 'finally' phrase."""
        assert _has_transition_phrase("Finally, we reach our destination")

    @pytest.mark.fast
    def test_case_insensitive(self):
        """Test case insensitive matching."""
        assert _has_transition_phrase("FIRST, let me show you")
        assert _has_transition_phrase("In Conclusion, this was amazing")

    @pytest.mark.fast
    def test_no_transition(self):
        """Test returns false for no transition."""
        assert not _has_transition_phrase("The weather is nice today")
        assert not _has_transition_phrase("Look at that beautiful building")


class TestRunBoundaryRefinement:
    """Test full boundary refinement pass."""

    @pytest.mark.fast
    def test_single_chapter_unchanged(self, mock_config):
        """Test single chapter is returned unchanged."""
        chapters = [ChapterCandidate(
            chapter_id=0,
            start_segment_idx=0,
            end_segment_idx=5,
            title="Only Chapter",
        )]
        result = run_boundary_refinement(
            chapters=chapters,
            segments=[{"text": "test"}] * 6,
            config=mock_config,
        )
        assert len(result) == 1
        assert result[0].start_segment_idx == 0
        assert result[0].end_segment_idx == 5

    @pytest.mark.fast
    def test_empty_chapters(self, mock_config):
        """Test empty list returned for empty input."""
        result = run_boundary_refinement(
            chapters=[],
            segments=[],
            config=mock_config,
        )
        assert result == []

    @pytest.mark.fast
    def test_with_embeddings(self, mock_config):
        """Test refinement with provided embeddings."""
        chapters = [
            ChapterCandidate(
                chapter_id=0,
                start_segment_idx=0,
                end_segment_idx=4,
                title="Chapter 1",
            ),
            ChapterCandidate(
                chapter_id=1,
                start_segment_idx=5,
                end_segment_idx=9,
                title="Chapter 2",
            ),
        ]
        segments = [{"text": f"Segment {i}"} for i in range(10)]

        # Create embeddings with clear boundary at segment 5
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # 0-4: similar
            [0.95, 0.05, 0.0],
            [0.9, 0.1, 0.0],
            [0.85, 0.15, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 1.0, 0.0],  # 5-9: different direction
            [0.05, 0.95, 0.0],
            [0.1, 0.9, 0.0],
            [0.15, 0.85, 0.0],
            [0.2, 0.8, 0.0],
        ])

        result = run_boundary_refinement(
            chapters=chapters,
            segments=segments,
            config=mock_config,
            embeddings=embeddings,
        )

        assert len(result) == 2
        # Boundaries should be preserved or refined near the semantic gap


class TestRefineChapterBoundary:
    """Test individual chapter boundary refinement."""

    @pytest.mark.fast
    def test_no_change_without_scores(self):
        """Test chapter unchanged when no boundary scores."""
        chapter = ChapterCandidate(
            chapter_id=0,
            start_segment_idx=0,
            end_segment_idx=5,
            title="Test",
        )
        result = _refine_chapter_boundary(
            chapter=chapter,
            prev_chapter=None,
            next_chapter=None,
            boundary_scores=[],
            segments=[],
        )
        assert result.start_segment_idx == 0
        assert result.end_segment_idx == 5

    @pytest.mark.fast
    def test_refines_to_high_score_boundary(self):
        """Test boundary moves to highest score position."""
        chapter = ChapterCandidate(
            chapter_id=1,
            start_segment_idx=5,
            end_segment_idx=10,
            title="Test",
        )
        prev_chapter = ChapterCandidate(
            chapter_id=0,
            start_segment_idx=0,
            end_segment_idx=4,
            title="Previous",
        )

        # Boundary scores: high gap at position 4 (between seg 4 and 5)
        boundary_scores = [0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
        segments = [{"text": f"Segment {i}"} for i in range(11)]

        result = _refine_chapter_boundary(
            chapter=chapter,
            prev_chapter=prev_chapter,
            next_chapter=None,
            boundary_scores=boundary_scores,
            segments=segments,
            search_window=3,
        )

        # Start should be refined toward the high-gap boundary
        assert result.start_segment_idx >= 4
