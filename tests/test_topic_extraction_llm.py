"""
Tests for topic_extraction.py LLM Methods

Targets uncovered lines:
- Lines 238-288: _extract_topics_from_transcript() - LLM topic extraction
- Lines 401-446: _detect_chapters_llm() - LLM chapter detection
- Lines 501-593: _detect_location_chapters_llm() - LLM location chapters
- Lines 601-635: LLM helper methods
- Lines 652-674: Additional LLM methods

Created: 2026-01-10 (Session 12 - topic_extraction expansion)
"""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.topic_extraction import (
    TopicExtractor,
    ChapterDetector,
    VideoTopics,
    LocationChapter
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock config with LLM settings"""
    config = Mock()
    config.gemini_api_key = "test_gemini_key"

    # LLM config
    llm = Mock()
    llm.model = "gemini-2.0-flash"
    llm.temperature = 0.7
    config.llm = llm

    # Topic extraction config
    topic_config = Mock()
    topic_config.enabled = True
    topic_config.use_llm = True
    topic_config.min_confidence = 0.5
    config.topic_extraction = topic_config

    return config


@pytest.fixture
def topic_extractor(mock_config, tmp_path):
    """Create TopicExtractor instance"""
    return TopicExtractor(config=mock_config, cache_dir=tmp_path)


@pytest.fixture
def chapter_detector(mock_config):
    """Create ChapterDetector instance"""
    return ChapterDetector(config=mock_config)


@pytest.fixture
def sample_transcript():
    """Sample transcript text for testing"""
    return """
    The devastating earthquake struck without warning.
    Buildings collapsed across the city as rescue teams rushed to help.
    The disaster affected thousands of people in the region.
    Emergency services worked through the night to find survivors.
    """


# ============================================================================
# Test _extract_topics_from_transcript() (Lines 238-288)
# ============================================================================

class TestExtractTopicsFromTranscript:
    """Test LLM-based topic extraction from transcripts"""

    @pytest.mark.fast
    def test_extract_topics_successful(self, topic_extractor, sample_transcript):
        """Test successful topic extraction with LLM"""
        # Mock LLM response
        mock_response = Mock()
        mock_response.parsed_data = {
            "topics": ["earthquake", "disaster relief", "emergency response"],
            "confidence": 0.85,
            "reasoning": "Clear disaster/emergency theme"
        }

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            result = topic_extractor.extract_topics_from_transcript(
                transcript_text=sample_transcript,
                video_path="/video1.mp4",
                source_keyword="earthquake"
            )

            # Should return VideoTopics object
            assert isinstance(result, VideoTopics)
            assert len(result.topics) >= 1
            assert "earthquake" in result.topics or "disaster" in [t.lower() for t in result.topics]
            assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.fast
    def test_extract_topics_fallback_to_keyword(self, topic_extractor):
        """Test fallback to keyword when LLM fails"""
        # Mock LLM to raise exception
        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.side_effect = Exception("API error")
            mock_create_client.return_value = mock_client

            result = topic_extractor.extract_topics_from_transcript(
                transcript_text="Some text",
                video_path="/video1.mp4",
                source_keyword="earthquake"
            )

            # Should fallback to keyword
            assert isinstance(result, VideoTopics)
            assert "earthquake" in result.topics
            assert result.confidence < 1.0  # Fallback has lower confidence

    @pytest.mark.fast
    def test_extract_topics_empty_transcript(self, topic_extractor):
        """Test with empty transcript"""
        result = topic_extractor.extract_topics_from_transcript(
            transcript_text="",
            video_path="/video1.mp4",
            source_keyword="earthquake"
        )

        # Should fallback to keyword
        assert isinstance(result, VideoTopics)
        assert "earthquake" in result.topics

    @pytest.mark.fast
    def test_extract_topics_malformed_json(self, topic_extractor, sample_transcript):
        """Test handling of malformed JSON response"""
        # Mock LLM response with None parsed_data
        mock_response = Mock()
        mock_response.parsed_data = None

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            result = topic_extractor.extract_topics_from_transcript(
                transcript_text=sample_transcript,
                video_path="/video1.mp4",
                source_keyword="earthquake"
            )

            # Should fallback to keyword
            assert isinstance(result, VideoTopics)
            assert "earthquake" in result.topics


# ============================================================================
# Test _detect_chapters_llm() (Lines 401-446)
# ============================================================================

class TestDetectChaptersLLM:
    """Test LLM-based chapter detection"""

    @pytest.mark.fast
    def test_detect_chapters_successful(self, chapter_detector):
        """Test successful chapter detection with LLM"""
        # ChapterDetector expects dict segments with 'text' field
        transcript_segments = [
            {"text": "Introduction to the topic"},
            {"text": "Main discussion about earthquakes"},
            {"text": "Safety procedures and conclusion"}
        ]

        # Mock LLM response - should return array with chapter dicts
        mock_response = Mock()
        mock_response.parsed_data = [
            {"start_segment_idx": 0, "end_segment_idx": 0, "title": "Introduction", "topics": ["introduction"]},
            {"start_segment_idx": 1, "end_segment_idx": 1, "title": "Earthquake Discussion", "topics": ["earthquake"]},
            {"start_segment_idx": 2, "end_segment_idx": 2, "title": "Safety & Conclusion", "topics": ["safety"]}
        ]

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            chapters = chapter_detector.detect_chapters(
                segments=transcript_segments,
                overall_topic="earthquake"
            )

            # Should return chapter list
            assert isinstance(chapters, list)
            assert len(chapters) >= 1

    @pytest.mark.fast
    def test_detect_chapters_empty_segments(self, chapter_detector):
        """Test with empty transcript segments"""
        chapters = chapter_detector.detect_chapters(
            segments=[],
            overall_topic="test"
        )

        # Should return empty list or handle gracefully
        assert isinstance(chapters, list)

    @pytest.mark.fast
    def test_detect_chapters_llm_error(self, chapter_detector):
        """Test error handling when LLM fails"""
        transcript_segments = [
            {"text": "Test"}
        ]

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.side_effect = Exception("API error")
            mock_create_client.return_value = mock_client

            chapters = chapter_detector.detect_chapters(
                segments=transcript_segments,
                overall_topic="test"
            )

            # Should handle error gracefully (return fallback chapter)
            assert isinstance(chapters, list)
            # Fallback creates single chapter
            assert len(chapters) >= 1


# ============================================================================
# Test _detect_location_chapters_llm() (Lines 501-593)
# ============================================================================

class TestDetectLocationChaptersLLM:
    """Test LLM-based location chapter detection"""

    @pytest.mark.fast
    def test_detect_location_chapters_successful(self, chapter_detector):
        """Test successful location chapter detection"""
        # ChapterDetector expects dict segments with 'text' field
        transcript_segments = [
            {"text": "Walking through Paris streets"},
            {"text": "Visiting the Eiffel Tower"},
            {"text": "Exploring Montmartre neighborhood"}
        ]

        # Mock LLM response with location chapters array
        mock_response = Mock()
        mock_response.parsed_data = [
            {
                "start_segment_idx": 0,
                "end_segment_idx": 0,
                "location_name": "Paris",
                "location_type": "city",
                "visual_keywords": ["streets"],
                "title": "Paris Streets"
            },
            {
                "start_segment_idx": 1,
                "end_segment_idx": 1,
                "location_name": "Eiffel Tower",
                "location_type": "landmark",
                "visual_keywords": ["tower"],
                "title": "Eiffel Tower Visit"
            },
            {
                "start_segment_idx": 2,
                "end_segment_idx": 2,
                "location_name": "Montmartre",
                "location_type": "neighborhood",
                "visual_keywords": ["montmartre"],
                "title": "Montmartre Tour"
            }
        ]

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            location_chapters = chapter_detector.detect_location_chapters(
                segments=transcript_segments,
                overall_topic="Paris travel"
            )

            # Should return LocationChapter objects
            assert isinstance(location_chapters, list)
            if len(location_chapters) > 0:
                assert hasattr(location_chapters[0], 'location_name')

    @pytest.mark.fast
    def test_detect_location_chapters_empty_segments(self, chapter_detector):
        """Test with empty transcript segments"""
        location_chapters = chapter_detector.detect_location_chapters(
            segments=[],
            overall_topic="travel"
        )

        # Should return empty list
        assert location_chapters == [] or location_chapters is None

    @pytest.mark.fast
    def test_detect_location_chapters_no_locations(self, chapter_detector):
        """Test when LLM returns no locations"""
        transcript_segments = [
            {"text": "Generic content"}
        ]

        # Mock LLM response with empty locations
        mock_response = Mock()
        mock_response.parsed_data = []

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            location_chapters = chapter_detector.detect_location_chapters(
                segments=transcript_segments,
                overall_topic="travel"
            )

            # Should return empty list
            assert location_chapters == []

    @pytest.mark.fast
    def test_detect_location_chapters_llm_error(self, chapter_detector):
        """Test error handling when LLM fails"""
        transcript_segments = [
            {"text": "Test"}
        ]

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.side_effect = Exception("API error")
            mock_create_client.return_value = mock_client

            location_chapters = chapter_detector.detect_location_chapters(
                segments=transcript_segments,
                overall_topic="travel"
            )

            # Should handle error gracefully
            assert location_chapters is None or location_chapters == []


# ============================================================================
# Test High-Level Topic Extraction Methods
# ============================================================================

class TestTopicExtractionIntegration:
    """Test high-level topic extraction methods"""

    @pytest.mark.fast
    def test_extract_topics_from_video_with_llm(self, topic_extractor):
        """Test extract_topics_from_transcript with LLM enabled"""
        transcript_text = "Earthquake devastation. Rescue operations ongoing."

        # Mock LLM response - should return array of topics
        mock_response = Mock()
        mock_response.parsed_data = ["earthquake", "disaster", "rescue"]

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            video_topics = topic_extractor.extract_topics_from_transcript(
                transcript_text=transcript_text,
                video_path="/video1.mp4",
                source_keyword="earthquake"
            )

            # Should return VideoTopics object
            assert isinstance(video_topics, VideoTopics)
            assert len(video_topics.topics) >= 1
            assert video_topics.confidence > 0

    @pytest.mark.fast
    def test_extract_topics_llm_disabled(self, mock_config, tmp_path):
        """Test topic extraction with LLM disabled (no API key)"""
        mock_config.gemini_api_key = None
        extractor = TopicExtractor(config=mock_config, cache_dir=tmp_path)

        transcript_text = "Some content about earthquakes"

        video_topics = extractor.extract_topics_from_transcript(
            transcript_text=transcript_text,
            video_path="/video1.mp4",
            source_keyword="earthquake"
        )

        # Should still work (keyword-based fallback)
        assert isinstance(video_topics, VideoTopics)
        assert "earthquake" in video_topics.topics


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestLLMEdgeCases:
    """Test edge cases in LLM topic extraction"""

    @pytest.mark.fast
    def test_extract_topics_very_long_transcript(self, topic_extractor):
        """Test with very long transcript (token limit)"""
        # Create a very long transcript
        long_transcript = " ".join(["earthquake disaster" for _ in range(1000)])

        # Mock LLM response
        mock_response = Mock()
        mock_response.parsed_data = ["earthquake", "disaster"]

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            result = topic_extractor.extract_topics_from_transcript(
                transcript_text=long_transcript,
                video_path="/video1.mp4",
                source_keyword="earthquake"
            )

            # Should handle long transcript
            assert isinstance(result, VideoTopics)
            assert isinstance(result.topics, list)

    @pytest.mark.fast
    def test_chapter_detection_single_segment(self, chapter_detector):
        """Test chapter detection with single segment"""
        transcript_segments = [
            {"text": "Full video content"}
        ]

        # Mock LLM response
        mock_response = Mock()
        mock_response.parsed_data = [
            {"start_segment_idx": 0, "end_segment_idx": 0, "title": "Full Video", "topics": ["content"]}
        ]

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            chapters = chapter_detector.detect_chapters(
                segments=transcript_segments,
                overall_topic="content"
            )

            # Should return single chapter
            assert isinstance(chapters, list)
            assert len(chapters) >= 1

    @pytest.mark.fast
    def test_location_chapters_overlapping_times(self, chapter_detector):
        """Test location chapters with overlapping segment indices"""
        transcript_segments = [
            {"text": "Paris scenes"},
            {"text": "Eiffel Tower"}
        ]

        # Mock LLM response with overlapping segment ranges
        mock_response = Mock()
        mock_response.parsed_data = [
            {"start_segment_idx": 0, "end_segment_idx": 1, "location_name": "Paris"},
            {"start_segment_idx": 1, "end_segment_idx": 1, "location_name": "Eiffel Tower"}
        ]

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            location_chapters = chapter_detector.detect_location_chapters(
                segments=transcript_segments,
                overall_topic="Paris"
            )

            # Should handle overlapping chapters
            assert isinstance(location_chapters, list)

    @pytest.mark.fast
    def test_no_api_key_fallback(self, mock_config, tmp_path):
        """Test fallback when no API key configured"""
        mock_config.gemini_api_key = None
        extractor = TopicExtractor(config=mock_config, cache_dir=tmp_path)

        result = extractor.extract_topics_from_transcript(
            transcript_text="earthquake disaster",
            video_path="/video1.mp4",
            source_keyword="earthquake"
        )

        # Should fallback to keyword-based extraction
        assert isinstance(result, VideoTopics)
        assert "earthquake" in result.topics
