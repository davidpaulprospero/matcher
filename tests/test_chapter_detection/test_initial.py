"""Tests for initial detection pass."""

import pytest
from unittest.mock import Mock, patch

from src.chapter_detection.passes.initial import (
    run_initial_detection,
    detect_content_type,
    _validate_chapters,
    _dict_to_candidate,
)
from src.chapter_detection.models import ChapterCandidate


class TestRunInitialDetection:
    """Test initial detection pass."""

    @pytest.mark.fast
    def test_empty_segments(self, mock_config, mock_llm_client):
        """Test with empty segments."""
        result = run_initial_detection(
            segments=[],
            llm_client=mock_llm_client,
            config=mock_config,
        )
        assert result == []

    @pytest.mark.fast
    def test_detection_with_mock_llm(
        self, mock_config, mock_llm_client, sample_segments, mock_llm_response_two_chapters
    ):
        """Test detection with mocked LLM response."""
        mock_llm_client.generate.return_value = mock_llm_response_two_chapters

        with patch('src.chapter_detection.passes.initial.create_chunks') as mock_chunks:
            # Simulate single chunk covering all segments
            mock_chunk = Mock()
            mock_chunk.text = "test text"
            mock_chunk.segment_count = len(sample_segments)
            mock_chunk.start_segment_idx = 0
            mock_chunk.end_segment_idx = len(sample_segments) - 1
            mock_chunks.return_value = [mock_chunk]

            result = run_initial_detection(
                segments=sample_segments,
                llm_client=mock_llm_client,
                config=mock_config,
                strategy='topic',
            )

        assert len(result) == 2
        assert isinstance(result[0], ChapterCandidate)
        assert result[0].title == "Paris Tour"
        assert result[1].title == "Tokyo Adventure"

    @pytest.mark.fast
    def test_detection_strategy_topic(self, mock_config, mock_llm_client, sample_segments):
        """Test topic detection strategy."""
        mock_response = Mock()
        mock_response.parsed_data = [
            {"start_segment_idx": 0, "end_segment_idx": 5, "title": "Test", "confidence": "medium"}
        ]
        mock_llm_client.generate.return_value = mock_response

        with patch('src.chapter_detection.passes.initial.create_chunks') as mock_chunks:
            mock_chunk = Mock()
            mock_chunk.text = "test"
            mock_chunk.segment_count = 6
            mock_chunk.start_segment_idx = 0
            mock_chunk.end_segment_idx = 5
            mock_chunks.return_value = [mock_chunk]

            result = run_initial_detection(
                segments=sample_segments,
                llm_client=mock_llm_client,
                config=mock_config,
                strategy='topic',
            )

        assert len(result) >= 1
        assert result[0].detection_strategy == 'topic'


class TestValidateChapters:
    """Test chapter validation."""

    @pytest.mark.fast
    def test_valid_chapters(self):
        """Test validation of valid chapters."""
        chapters = [
            {"start_segment_idx": 0, "end_segment_idx": 4, "title": "Ch1", "confidence": "high"},
            {"start_segment_idx": 5, "end_segment_idx": 9, "title": "Ch2", "confidence": "medium"},
        ]
        result = _validate_chapters(chapters, total_segments=10, min_segments=3)
        assert len(result) == 2
        assert result[0]['chapter_id'] == 0
        assert result[1]['chapter_id'] == 1

    @pytest.mark.fast
    def test_filters_small_chapters(self):
        """Test that small chapters are filtered."""
        chapters = [
            {"start_segment_idx": 0, "end_segment_idx": 1, "title": "TooSmall", "confidence": "high"},
            {"start_segment_idx": 2, "end_segment_idx": 6, "title": "GoodSize", "confidence": "high"},
        ]
        result = _validate_chapters(chapters, total_segments=10, min_segments=3)
        assert len(result) == 1
        assert result[0]['title'] == "GoodSize"

    @pytest.mark.fast
    def test_clamps_indices(self):
        """Test that out-of-range indices are clamped."""
        chapters = [
            {"start_segment_idx": -5, "end_segment_idx": 100, "title": "OutOfRange"},
        ]
        result = _validate_chapters(chapters, total_segments=10, min_segments=1)
        assert len(result) == 1
        assert result[0]['start_segment_idx'] == 0
        assert result[0]['end_segment_idx'] == 9

    @pytest.mark.fast
    def test_handles_non_dict(self):
        """Test handling of non-dict items in list."""
        chapters = [
            {"start_segment_idx": 0, "end_segment_idx": 5, "title": "Valid"},
            "not a dict",
            None,
        ]
        result = _validate_chapters(chapters, total_segments=10, min_segments=1)
        assert len(result) == 1


class TestDictToCandidate:
    """Test dict to ChapterCandidate conversion."""

    @pytest.mark.fast
    def test_basic_conversion(self):
        """Test basic conversion."""
        ch_dict = {
            "chapter_id": 1,
            "start_segment_idx": 5,
            "end_segment_idx": 10,
            "title": "Test Chapter",
            "confidence": "high",
        }
        candidate = _dict_to_candidate(ch_dict, strategy="topic")
        assert candidate.chapter_id == 1
        assert candidate.confidence == 0.9  # "high" maps to 0.9
        assert candidate.detection_strategy == "topic"

    @pytest.mark.fast
    def test_confidence_mapping(self):
        """Test confidence string to float mapping."""
        high = _dict_to_candidate({"confidence": "high"}, "topic")
        medium = _dict_to_candidate({"confidence": "medium"}, "topic")
        low = _dict_to_candidate({"confidence": "low"}, "topic")

        assert high.confidence == 0.9
        assert medium.confidence == 0.7
        assert low.confidence == 0.5

    @pytest.mark.fast
    def test_preserves_location_fields(self):
        """Test that location fields are preserved."""
        ch_dict = {
            "location_name": "Paris",
            "location_type": "city",
            "visual_keywords": ["Eiffel Tower"],
            "context_keywords": ["romantic"],
        }
        candidate = _dict_to_candidate(ch_dict, strategy="location")
        assert candidate.location_name == "Paris"
        assert candidate.location_type == "city"
        assert candidate.visual_keywords == ["Eiffel Tower"]


class TestDetectContentType:
    """Test content type detection."""

    @pytest.mark.fast
    def test_returns_general_on_empty(self, mock_llm_client):
        """Test returns general for empty segments."""
        result = detect_content_type([], mock_llm_client)
        assert result == 'general'

    @pytest.mark.fast
    def test_with_travel_content(self, mock_llm_client):
        """Test detection of travel content."""
        mock_response = Mock()
        mock_response.parsed_data = {
            "content_type": "travel",
            "confidence": "high",
        }
        mock_llm_client.generate.return_value = mock_response

        segments = [
            {"text": "Welcome to Paris, let's explore the city."},
            {"text": "The Eiffel Tower is magnificent."},
        ]
        result = detect_content_type(segments, mock_llm_client)
        assert result == "travel"

    @pytest.mark.fast
    def test_fallback_on_error(self, mock_llm_client):
        """Test fallback to general on LLM error."""
        mock_llm_client.generate.side_effect = Exception("API Error")

        segments = [{"text": "Some text"}]
        result = detect_content_type(segments, mock_llm_client)
        assert result == "general"
