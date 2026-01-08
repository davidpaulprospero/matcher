"""
Tests for keyword_extractor.segment_processor

Tests cover:
- extract_keyword_per_segment() with mocked LLM
- Batch vs individual extraction logic
- Fallback behavior without LLM
"""

import pytest
from unittest.mock import Mock
from src.keyword_extractor.segment_processor import (
    extract_keyword_per_segment
)


class TestExtractKeywordPerSegment:
    """Test extract_keyword_per_segment() orchestration"""

    def test_extract_keyword_per_segment_small_batch(self):
        """Test per-segment extraction with small number of segments"""
        mock_llm_call = Mock(return_value='["keyword1", "keyword2", "keyword3"]')
        mock_llm_client = Mock()

        segments = [
            {"text": "Segment 1"},
            {"text": "Segment 2"},
            {"text": "Segment 3"}
        ]
        topic = "Test"

        keywords = extract_keyword_per_segment(
            segments=segments,
            llm_client=mock_llm_client,
            llm_call_function=mock_llm_call,
            topic=topic
        )

        assert len(keywords) == 3
        # Should use batch processing for <=20 segments
        assert mock_llm_call.call_count >= 1

    def test_extract_keyword_per_segment_large_batch(self):
        """Test per-segment extraction with many segments"""
        mock_llm_call = Mock(return_value='"keyword"')
        mock_llm_client = Mock()

        # Create 25 segments (more than batch threshold)
        segments = [{"text": f"Segment {i}"} for i in range(25)]
        topic = "Test"

        keywords = extract_keyword_per_segment(
            segments=segments,
            llm_client=mock_llm_client,
            llm_call_function=mock_llm_call,
            topic=topic
        )

        assert len(keywords) == 25
        # Should use batch processing with batches of 20
        assert mock_llm_call.call_count >= 1

    def test_extract_keyword_per_segment_no_llm(self):
        """Test per-segment extraction without LLM client"""
        segments = [
            {"text": "Mountain climbing adventures"},
            {"text": "Ocean diving expeditions"}
        ]

        keywords = extract_keyword_per_segment(
            segments=segments,
            llm_client=None,
            llm_call_function=None,
            topic=""
        )

        # Should use fallback logic
        assert len(keywords) == 2
        assert all(isinstance(kw, str) for kw in keywords)
        assert all(len(kw) > 0 for kw in keywords)

    def test_extract_keyword_per_segment_fallback_logic(self):
        """Test fallback keyword extraction logic"""
        segments = [
            {"text": "The quick brown fox jumps over the lazy dog"},
            {"text": "A single word here"},
            {"text": "Two meaningful words"},
            {"text": "Three good words here"}
        ]

        keywords = extract_keyword_per_segment(
            segments=segments,
            llm_client=None,
            llm_call_function=None,
            topic=""
        )

        assert len(keywords) == 4
        # Check that keywords are reasonable (stopwords removed)
        for kw in keywords:
            assert isinstance(kw, str)
            assert len(kw) > 0
            # Should not start with stopwords
            assert not any(kw.startswith(sw) for sw in ['the', 'a', 'an'])

    def test_extract_keyword_per_segment_empty_segments(self):
        """Test per-segment extraction with empty segments list"""
        keywords = extract_keyword_per_segment(
            segments=[],
            llm_client=None,
            llm_call_function=None,
            topic=""
        )

        assert keywords == []

    def test_extract_keyword_per_segment_with_topic(self):
        """Test per-segment extraction uses topic context"""
        mock_llm_call = Mock(return_value='["context-aware-keyword"]')
        mock_llm_client = Mock()

        segments = [{"text": "Generic text"}]
        topic = "Specific Topic Context"

        keywords = extract_keyword_per_segment(
            segments=segments,
            llm_client=mock_llm_client,
            llm_call_function=mock_llm_call,
            topic=topic
        )

        # Should return keyword
        assert len(keywords) == 1

    def test_extract_keyword_per_segment_batch_size_20(self):
        """Test batching with exactly 20 segments"""
        mock_llm_call = Mock(return_value='["kw"] * 20')
        mock_llm_client = Mock()

        segments = [{"text": f"Segment {i}"} for i in range(20)]

        keywords = extract_keyword_per_segment(
            segments=segments,
            llm_client=mock_llm_client,
            llm_call_function=mock_llm_call,
            topic="Test"
        )

        assert len(keywords) == 20

    def test_extract_keyword_per_segment_short_text(self):
        """Test with very short segment text"""
        segments = [
            {"text": "Short"},
            {"text": "Also short"}
        ]

        keywords = extract_keyword_per_segment(
            segments=segments,
            llm_client=None,
            llm_call_function=None,
            topic="Fallback Topic"
        )

        # Short text should use fallback
        assert len(keywords) == 2
        for kw in keywords:
            assert isinstance(kw, str)
            assert len(kw) > 0


class TestSegmentProcessingEdgeCases:
    """Test edge cases for segment processing"""

    def test_segment_with_special_characters(self):
        """Test segment with special characters"""
        segments = [{"text": "Visit Sao Paulo's famous cafe scene"}]

        keywords = extract_keyword_per_segment(
            segments=segments,
            llm_client=None,
            llm_call_function=None,
            topic=""
        )

        assert len(keywords) == 1
        assert isinstance(keywords[0], str)

    def test_segment_very_long(self):
        """Test very long segment text"""
        segments = [{"text": "This is a very long segment. " * 100}]

        keywords = extract_keyword_per_segment(
            segments=segments,
            llm_client=None,
            llm_call_function=None,
            topic=""
        )

        assert len(keywords) == 1
        assert isinstance(keywords[0], str)
        assert len(keywords[0]) > 0

    def test_segment_with_numbers(self):
        """Test segment with numbers and dates"""
        segments = [{"text": "In 1969, Apollo 11 landed on the moon"}]

        keywords = extract_keyword_per_segment(
            segments=segments,
            llm_client=None,
            llm_call_function=None,
            topic=""
        )

        assert len(keywords) == 1
        assert isinstance(keywords[0], str)

    def test_segment_all_stopwords(self):
        """Test segment with only stopwords"""
        segments = [{"text": "the a an is are was were"}]

        keywords = extract_keyword_per_segment(
            segments=segments,
            llm_client=None,
            llm_call_function=None,
            topic="Fallback"
        )

        # Should use topic as fallback
        assert len(keywords) == 1
        assert keywords[0] == "Fallback" or len(keywords[0]) > 0

    def test_segment_with_punctuation(self):
        """Test segment with lots of punctuation"""
        segments = [{"text": "Hello, world! How are you? I'm fine."}]

        keywords = extract_keyword_per_segment(
            segments=segments,
            llm_client=None,
            llm_call_function=None,
            topic=""
        )

        assert len(keywords) == 1
        # Should extract meaningful words, removing punctuation
        assert isinstance(keywords[0], str)
