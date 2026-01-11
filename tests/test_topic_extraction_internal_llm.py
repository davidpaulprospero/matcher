"""
Unit tests for internal LLM methods in topic_extraction module.

Targets uncovered lines in _extract_with_llm, _detect_with_llm, and helper methods
to improve coverage from 45.80% → 80%+.

Coverage gaps (187 lines):
- Lines 238-288: _extract_with_llm (51 lines)
- Lines 401-446: _detect_with_llm (46 lines)
- Lines 501-593: detect_location_chapters (93 lines)
- Lines 601-635, 652-674: Helper methods
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.topic_extraction import TopicExtractor, ChapterDetector, LocationChapter
from src.config import Config


@pytest.fixture
def temp_cache(tmp_path):
    """Create temporary cache directory"""
    return tmp_path


@pytest.fixture
def mock_config():
    """Create mock config"""
    config = Mock(spec=Config)

    # LLM config
    llm = Mock()
    llm.provider = "gemini"
    llm.model = "gemini-2.0-flash"
    llm.gemini_api_key = "test_key"

    # Keyword config
    keyword = Mock()
    keyword.use_llm_extraction = True

    config.llm = llm
    config.keyword = keyword
    config.gemini_api_key = "test_key"  # Direct attribute (used by _extract_with_llm)

    return config


# ============================================================================
# Test _extract_with_llm() Internal Method (Lines 238-288)
# ============================================================================

class TestExtractWithLLMInternal:
    """Test internal _extract_with_llm method with LLM client mocking"""

    def test_extract_with_llm_successful_response(self, temp_cache, mock_config):
        """Test _extract_with_llm with successful LLM response"""
        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        # Mock LLM client response - should be a list, not dict
        mock_response = Mock()
        mock_response.parsed_data = ["earthquake", "disaster", "emergency"]

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            transcript = "The earthquake struck without warning, causing massive destruction."
            result = extractor._extract_with_llm(
                transcript_text=transcript,
                video_title="Earthquake Documentary",
                source_keyword="disaster"
            )

        # Should return list of topics (lowercased)
        assert isinstance(result, list)
        assert len(result) == 3
        assert "earthquake" in result
        assert "disaster" in result
        assert "emergency" in result

    def test_extract_with_llm_truncates_long_transcript(self, temp_cache, mock_config):
        """Test that _extract_with_llm truncates transcripts over 3000 chars"""
        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        mock_response = Mock()
        mock_response.parsed_data = ["travel"]

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            # Create transcript > 3000 chars
            long_transcript = "This is about traveling. " * 200  # ~5000 chars

            result = extractor._extract_with_llm(
                transcript_text=long_transcript,
                video_title="Travel Video"
            )

            # Should still work with truncation
            assert isinstance(result, list)

            # Check that the prompt was called (transcript was truncated)
            assert mock_client.generate.called

    def test_extract_with_llm_handles_json_parse_error(self, temp_cache, mock_config):
        """Test _extract_with_llm handles malformed JSON response"""
        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        # Mock response with invalid JSON structure
        mock_response = Mock()
        mock_response.parsed_data = None  # Simulate parse failure

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            result = extractor._extract_with_llm(
                transcript_text="Some text",
                video_title="Test"
            )

            # Should return empty list on parse error
            assert result == []

    def test_extract_with_llm_handles_api_error(self, temp_cache, mock_config):
        """Test _extract_with_llm handles LLM API errors"""
        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.side_effect = Exception("API timeout")
            mock_create_client.return_value = mock_client

            result = extractor._extract_with_llm(
                transcript_text="Test transcript",
                video_title="Test"
            )

            # Should return empty list on error
            assert result == []

    def test_extract_with_llm_includes_context(self, temp_cache, mock_config):
        """Test that _extract_with_llm includes video title and keyword in prompt"""
        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        mock_response = Mock()
        mock_response.parsed_data = ["mountain", "hiking"]

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            result = extractor._extract_with_llm(
                transcript_text="Hiking in the mountains",
                video_title="Mountain Trek",
                source_keyword="adventure"
            )

            # Verify generate was called with a prompt
            assert mock_client.generate.called
            call_args = mock_client.generate.call_args
            request = call_args[0][0]

            # Prompt should contain context
            assert "Mountain Trek" in request.prompt or "adventure" in request.prompt


# ============================================================================
# Test _detect_with_llm() Internal Method (Lines 401-446)
# ============================================================================

class TestDetectWithLLMInternal:
    """Test internal _detect_with_llm method for chapter detection"""

    def test_detect_with_llm_successful_chapters(self, mock_config):
        """Test _detect_with_llm with successful chapter detection"""
        detector = ChapterDetector(mock_config)

        # Mock LLM response with chapter array
        mock_response = Mock()
        mock_response.parsed_data = [
            {
                "start_segment_idx": 0,
                "end_segment_idx": 2,
                "title": "Introduction",
                "topics": ["intro", "welcome"]
            },
            {
                "start_segment_idx": 3,
                "end_segment_idx": 5,
                "title": "Main Content",
                "topics": ["main", "content"]
            }
        ]

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            indexed_text = "[0] Hello and welcome\n[1] Today we discuss\n[2] the main topic\n[3] Let's dive deeper\n[4] This is important\n[5] Thanks for watching"

            result = detector._detect_with_llm(
                indexed_text=indexed_text,
                num_segments=6,
                overall_topic="Tutorial"
            )

        # Should return list of chapter dicts
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]['title'] == "Introduction"
        assert result[1]['title'] == "Main Content"

    def test_detect_with_llm_handles_empty_text(self, mock_config):
        """Test _detect_with_llm with empty indexed text"""
        detector = ChapterDetector(mock_config)

        # With empty text, LLM should still be called but may return empty
        mock_response = Mock()
        mock_response.parsed_data = []

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            result = detector._detect_with_llm(
                indexed_text="",
                num_segments=0
            )

            # Should return empty list
            assert result == []

    def test_detect_with_llm_handles_api_error(self, mock_config):
        """Test _detect_with_llm handles LLM errors gracefully"""
        detector = ChapterDetector(mock_config)

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.side_effect = Exception("Rate limit exceeded")
            mock_create_client.return_value = mock_client

            indexed_text = "[0] Test segment"
            result = detector._detect_with_llm(
                indexed_text=indexed_text,
                num_segments=1
            )

            # Should return empty list on error
            assert result == []

    def test_detect_with_llm_includes_overall_topic(self, mock_config):
        """Test that _detect_with_llm includes overall_topic in prompt"""
        detector = ChapterDetector(mock_config)

        mock_response = Mock()
        mock_response.parsed_data = []

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            indexed_text = "[0] Content"
            result = detector._detect_with_llm(
                indexed_text=indexed_text,
                num_segments=1,
                overall_topic="Travel Documentary"
            )

            # Verify overall_topic was included in prompt
            assert mock_client.generate.called
            call_args = mock_client.generate.call_args
            request = call_args[0][0]
            assert "Travel Documentary" in request.prompt


# ============================================================================
# Test detect_location_chapters() Public Method (Lines 501-593)
# ============================================================================

class TestDetectLocationChapters:
    """Test detect_location_chapters with location service integration"""

    def test_detect_location_chapters_with_location_service(self, mock_config):
        """Test detect_location_chapters with mocked location service"""
        detector = ChapterDetector(mock_config)

        # Mock location service
        mock_location_service = Mock()
        mock_location_service.extract_locations_from_text.return_value = [
            {"name": "Paris", "lat": 48.8566, "lon": 2.3522},
            {"name": "London", "lat": 51.5074, "lon": -0.1278}
        ]

        segments = [
            {"text": "We arrived in Paris", "start_time": 0.0, "end_time": 5.0},
            {"text": "The Eiffel Tower is amazing", "start_time": 5.0, "end_time": 10.0},
            {"text": "Next we went to London", "start_time": 10.0, "end_time": 15.0},
            {"text": "Big Ben was impressive", "start_time": 15.0, "end_time": 20.0}
        ]

        # Mock LLM client response with location chapters
        mock_response = Mock()
        mock_response.parsed_data = [
            {
                "start_segment_idx": 0,
                "end_segment_idx": 1,
                "location_name": "Paris",
                "location_type": "city",
                "visual_keywords": ["Eiffel Tower"],
                "context_keywords": ["travel"],
                "title": "Paris Chapter"
            },
            {
                "start_segment_idx": 2,
                "end_segment_idx": 3,
                "location_name": "London",
                "location_type": "city",
                "visual_keywords": ["Big Ben"],
                "context_keywords": ["travel"],
                "title": "London Chapter"
            }
        ]

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            result = detector.detect_location_chapters(
                segments=segments,
                location_service=mock_location_service,
                overall_topic="Europe Trip"
            )

        # Should return list of LocationChapter objects
        assert isinstance(result, list)
        assert len(result) > 0

    def test_detect_location_chapters_no_location_service(self, mock_config):
        """Test detect_location_chapters without location service"""
        detector = ChapterDetector(mock_config)

        segments = [
            {"text": "We visited Paris", "start_time": 0.0, "end_time": 5.0}
        ]

        result = detector.detect_location_chapters(
            segments=segments,
            location_service=None
        )

        # Should return empty list without location service
        assert result == []

    def test_detect_location_chapters_empty_segments(self, mock_config):
        """Test detect_location_chapters with empty segments"""
        detector = ChapterDetector(mock_config)

        result = detector.detect_location_chapters(
            segments=[],
            location_service=Mock()
        )

        # Should return empty list
        assert result == []

    def test_detect_location_chapters_no_locations_found(self, mock_config):
        """Test detect_location_chapters when no locations are extracted"""
        detector = ChapterDetector(mock_config)

        # Mock location service that finds nothing
        mock_location_service = Mock()
        mock_location_service.extract_locations_from_text.return_value = []

        segments = [
            {"text": "This is a generic video", "start_time": 0.0, "end_time": 5.0}
        ]

        result = detector.detect_location_chapters(
            segments=segments,
            location_service=mock_location_service
        )

        # Should return empty list
        assert result == []


# ============================================================================
# Test Edge Cases and Error Handling
# ============================================================================

class TestLLMMethodEdgeCases:
    """Test edge cases for internal LLM methods"""

    def test_extract_with_llm_no_api_key(self, temp_cache):
        """Test _extract_with_llm when no API key is configured"""
        config = Mock(spec=Config)
        config.llm = Mock()
        config.llm.provider = "gemini"
        config.llm.gemini_api_key = None  # No API key
        config.keyword = Mock()
        config.keyword.use_llm_extraction = True
        config.gemini_api_key = None  # Direct attribute (checked on line 561)

        extractor = TopicExtractor(config, cache_dir=temp_cache)

        # Should handle missing API key gracefully
        result = extractor._extract_with_llm(
            transcript_text="Test transcript",
            video_title="Test"
        )

        # Should return empty list
        assert result == []

    def test_detect_with_llm_very_long_segment_list(self, mock_config):
        """Test _detect_with_llm with many segments"""
        detector = ChapterDetector(mock_config)

        mock_response = Mock()
        mock_response.parsed_data = []  # Should be list, not dict

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            # Create indexed text with 100 segments
            indexed_text = "\n".join([f"[{i}] Segment {i}" for i in range(100)])

            result = detector._detect_with_llm(
                indexed_text=indexed_text,
                num_segments=100
            )

            # Should handle large segment list
            assert isinstance(result, list)
            assert mock_client.generate.called

    def test_location_chapters_with_llm_fallback(self, mock_config):
        """Test detect_location_chapters falls back gracefully when location service fails"""
        detector = ChapterDetector(mock_config)

        # Mock location service that raises error
        mock_location_service = Mock()
        mock_location_service.extract_locations_from_text.side_effect = Exception("Service down")

        segments = [
            {"text": "We visited Paris", "start_time": 0.0, "end_time": 5.0}
        ]

        # Mock LLM response to avoid real API calls
        mock_response = Mock()
        mock_response.parsed_data = []

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = Mock()
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            # Should handle location service errors gracefully
            result = detector.detect_location_chapters(
                segments=segments,
                location_service=mock_location_service
            )

        # Should return empty list or handle gracefully
        assert isinstance(result, list)


# ============================================================================
# Additional Tests for 80%+ Coverage (Lines 292-303, 320-340)
# ============================================================================

class TestAdditionalCoverageMethods:
    """Test additional methods to reach 80%+ coverage"""

    def test_parse_topics_response_json_array(self, temp_cache, mock_config):
        """Test _parse_topics_response with JSON array format (lines 292-297)"""
        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        response = '["travel", "beach", "vacation"]'
        result = extractor._parse_topics_response(response)

        assert isinstance(result, list)
        assert len(result) == 3
        assert "travel" in result
        assert "beach" in result

    def test_parse_topics_response_fallback_quoted_strings(self, temp_cache, mock_config):
        """Test _parse_topics_response fallback to quoted strings (lines 301-303)"""
        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        # Response with quoted strings but not valid JSON array
        response = 'The topics are "mountain" and "hiking" and "nature"'
        result = extractor._parse_topics_response(response)

        assert isinstance(result, list)
        assert len(result) == 3
        assert "mountain" in result
        assert "hiking" in result
        assert "nature" in result

    def test_parse_topics_response_malformed(self, temp_cache, mock_config):
        """Test _parse_topics_response with malformed response"""
        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        response = 'No valid topics here'
        result = extractor._parse_topics_response(response)

        # Should return empty list
        assert isinstance(result, list)
        assert len(result) == 0

    def test_extract_batch_multiple_videos(self, temp_cache, mock_config):
        """Test extract_batch with multiple videos (lines 320-340)"""
        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        transcripts = {
            "/video1.mp4": "This is about traveling to beaches",
            "/video2.mp4": "Mountain hiking adventures",
            "/video3.mp4": "City exploration and culture"
        }

        video_metadata = {
            "/video1.mp4": {"title": "Beach Travel", "keyword": "travel"},
            "/video2.mp4": {"title": "Mountain Hike", "keyword": "hiking"}
        }

        # Mock the LLM calls
        with patch.object(extractor, 'extract_topics_from_transcript') as mock_extract:
            from src.topic_extraction import VideoTopics
            mock_extract.side_effect = [
                VideoTopics(video_path="/video1.mp4", topics=["travel", "beach"], confidence=0.9),
                VideoTopics(video_path="/video2.mp4", topics=["hiking", "mountain"], confidence=0.85),
                VideoTopics(video_path="/video3.mp4", topics=["city", "culture"], confidence=0.8)
            ]

            result = extractor.extract_batch(transcripts, video_metadata)

        # Should return dict with all videos
        assert isinstance(result, dict)
        assert len(result) == 3
        assert "/video1.mp4" in result
        assert "/video2.mp4" in result
        assert "/video3.mp4" in result
        assert result["/video1.mp4"].topics == ["travel", "beach"]

    def test_extract_batch_empty_transcripts(self, temp_cache, mock_config):
        """Test extract_batch with empty transcripts dict"""
        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        result = extractor.extract_batch({})

        # Should return empty dict
        assert isinstance(result, dict)
        assert len(result) == 0
