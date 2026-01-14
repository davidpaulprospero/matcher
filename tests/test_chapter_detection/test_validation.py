"""Tests for validation pass."""

import pytest
from unittest.mock import Mock, patch

from src.chapter_detection.passes.validation import (
    run_validation,
    _apply_validation_results,
    _apply_merge_suggestions,
    _add_missed_chapters,
    _merge_chapters,
)
from src.chapter_detection.models import ChapterCandidate


class TestRunValidation:
    """Test validation pass."""

    def test_empty_chapters(self, mock_config, mock_llm_client):
        """Test empty list returned for empty input."""
        result = run_validation(
            chapters=[],
            segments=[],
            llm_client=mock_llm_client,
            config=mock_config,
        )
        assert result == []

    def test_validation_disabled(self, mock_llm_client):
        """Test skips validation when disabled in config."""
        config = Mock()
        config.matching.chapter_detection.use_validation_pass = False

        chapters = [ChapterCandidate(chapter_id=0, title="Test")]

        result = run_validation(
            chapters=chapters,
            segments=[{"text": "test"}],
            llm_client=mock_llm_client,
            config=config,
        )

        # Should return original chapters unchanged
        assert len(result) == 1
        assert result[0].title == "Test"
        # LLM should not be called
        mock_llm_client.generate.assert_not_called()

    def test_with_validation_response(self, mock_config, mock_llm_client):
        """Test applies validation response."""
        mock_response = Mock()
        mock_response.parsed_data = {
            "validations": [
                {
                    "chapter_id": 0,
                    "boundary_correct": True,
                    "title_accurate": True,
                    "content_coherent": True,
                }
            ],
            "missed_chapters": [],
            "merge_suggestions": [],
        }
        mock_llm_client.generate.return_value = mock_response

        chapters = [
            ChapterCandidate(
                chapter_id=0,
                start_segment_idx=0,
                end_segment_idx=5,
                title="Test Chapter",
                confidence=0.7,
            )
        ]

        result = run_validation(
            chapters=chapters,
            segments=[{"text": f"Segment {i}"} for i in range(6)],
            llm_client=mock_llm_client,
            config=mock_config,
        )

        assert len(result) == 1
        # Confidence should increase for coherent content
        assert result[0].confidence >= 0.7


class TestApplyValidationResults:
    """Test applying validation results."""

    def test_boundary_correction(self):
        """Test boundary correction is applied."""
        chapters = [
            ChapterCandidate(
                chapter_id=0,
                start_segment_idx=0,
                end_segment_idx=10,
                title="Test",
            )
        ]
        validation_result = {
            "validations": [
                {
                    "chapter_id": 0,
                    "boundary_correct": False,
                    "suggested_start": 2,
                    "suggested_end": 8,
                    "title_accurate": True,
                    "content_coherent": True,
                }
            ],
        }
        segments = [{"text": f"Seg {i}"} for i in range(11)]

        result = _apply_validation_results(chapters, validation_result, segments)

        assert result[0].start_segment_idx == 2
        assert result[0].end_segment_idx == 8
        assert "[Corrected by validation]" in result[0].boundary_reasoning

    def test_title_correction(self):
        """Test title correction is applied."""
        chapters = [
            ChapterCandidate(
                chapter_id=0,
                title="Wrong Title",
            )
        ]
        validation_result = {
            "validations": [
                {
                    "chapter_id": 0,
                    "title_accurate": False,
                    "suggested_title": "Better Title",
                    "content_coherent": True,
                }
            ],
        }

        result = _apply_validation_results(chapters, validation_result, [])

        assert result[0].title == "Better Title"

    def test_confidence_decrease_for_incoherent(self):
        """Test confidence decreases for incoherent content."""
        chapters = [
            ChapterCandidate(
                chapter_id=0,
                confidence=0.8,
            )
        ]
        validation_result = {
            "validations": [
                {
                    "chapter_id": 0,
                    "content_coherent": False,
                }
            ],
        }

        result = _apply_validation_results(chapters, validation_result, [])

        assert result[0].confidence < 0.8


class TestApplyMergeSuggestions:
    """Test merge suggestions."""

    def test_merges_chapters(self):
        """Test chapters are merged."""
        chapters = [
            ChapterCandidate(
                chapter_id=0,
                start_segment_idx=0,
                end_segment_idx=3,
                title="Part 1",
                confidence=0.7,
            ),
            ChapterCandidate(
                chapter_id=1,
                start_segment_idx=4,
                end_segment_idx=7,
                title="Part 2",
                confidence=0.8,
            ),
            ChapterCandidate(
                chapter_id=2,
                start_segment_idx=8,
                end_segment_idx=10,
                title="Separate",
                confidence=0.9,
            ),
        ]
        merge_suggestions = [[0, 1]]  # Merge chapters 0 and 1

        result = _apply_merge_suggestions(chapters, merge_suggestions)

        assert len(result) == 2  # Two chapters: merged + separate
        # Find merged chapter
        merged = [c for c in result if c.start_segment_idx == 0][0]
        assert merged.end_segment_idx == 7
        assert "Part 1" in merged.title
        assert "Part 2" in merged.title

    def test_no_merge_empty_suggestions(self):
        """Test no merge with empty suggestions."""
        chapters = [ChapterCandidate(chapter_id=0)]
        result = _apply_merge_suggestions(chapters, [])
        assert len(result) == 1


class TestMergeChapters:
    """Test chapter merging."""

    def test_merge_two_chapters(self):
        """Test merging two chapters."""
        chapters = [
            ChapterCandidate(
                chapter_id=0,
                start_segment_idx=0,
                end_segment_idx=5,
                title="First",
                topics=["topic1"],
                confidence=0.7,
            ),
            ChapterCandidate(
                chapter_id=1,
                start_segment_idx=6,
                end_segment_idx=10,
                title="Second",
                topics=["topic2"],
                confidence=0.8,
            ),
        ]

        result = _merge_chapters(chapters)

        assert result.start_segment_idx == 0
        assert result.end_segment_idx == 10
        assert "First" in result.title
        assert "Second" in result.title
        assert "topic1" in result.topics
        assert "topic2" in result.topics
        assert result.confidence == pytest.approx(0.75)  # Average

    def test_merge_single_chapter(self):
        """Test merge of single chapter returns it unchanged."""
        chapters = [ChapterCandidate(chapter_id=0, title="Only")]
        result = _merge_chapters(chapters)
        assert result.title == "Only"

    def test_merge_empty(self):
        """Test merge of empty list."""
        result = _merge_chapters([])
        assert result.title == ""


class TestAddMissedChapters:
    """Test adding missed chapters."""

    def test_adds_non_overlapping(self):
        """Test adds chapter in gap."""
        chapters = [
            ChapterCandidate(
                chapter_id=0,
                start_segment_idx=0,
                end_segment_idx=4,
            ),
            ChapterCandidate(
                chapter_id=1,
                start_segment_idx=10,
                end_segment_idx=15,
            ),
        ]
        missed = [
            {
                "start_segment_idx": 5,
                "end_segment_idx": 9,
                "suggested_title": "Missed Section",
            }
        ]
        segments = [{"text": f"Seg {i}"} for i in range(16)]

        result = _add_missed_chapters(chapters, missed, segments)

        assert len(result) == 3
        # Find the new chapter
        new_ch = [c for c in result if c.title == "Missed Section"][0]
        assert new_ch.start_segment_idx == 5
        assert new_ch.end_segment_idx == 9
        assert new_ch.detection_strategy == "validation_added"

    def test_skips_overlapping(self):
        """Test skips chapter that overlaps existing."""
        chapters = [
            ChapterCandidate(
                chapter_id=0,
                start_segment_idx=0,
                end_segment_idx=10,
            ),
        ]
        missed = [
            {
                "start_segment_idx": 5,
                "end_segment_idx": 15,  # Overlaps with existing
                "suggested_title": "Overlapping",
            }
        ]

        result = _add_missed_chapters(chapters, missed, [{"text": ""}] * 16)

        assert len(result) == 1  # No new chapter added
