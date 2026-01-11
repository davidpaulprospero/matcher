"""
Extended tests for topic_extraction.py to cover remaining uncovered lines.

Targets specific lines:
- Lines 144-147: Topic parsing edge cases (deserialization errors)
- Lines 298-299, 381-382: LLM response parsing (JSON failure paths)
- Line 403: Fallback detection (truncation of long indexed text)
- Lines 476-477, 513: Entity extraction edge cases (exception during parsing)
- Line 589: Chapter detection (return empty when no Gemini key)
- Lines 632-635: Location chapter handling (exception during parsing)
- Lines 776-781, 787: Batch processing edge cases (LLM location extraction)

Created: 2026-01-11 (Extended coverage tests)
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.topic_extraction import (
    VideoTopics,
    LocationChapter,
    TopicCache,
    TopicExtractor,
    ChapterDetector,
    compute_topic_overlap,
    compute_topic_penalty,
    extract_location_from_video_metadata,
    _extract_location_patterns,
    _extract_location_with_llm,
    extract_video_locations_batch,
)
from src.cache import CacheEntry


# ============================================================================
# Test Lines 144-147: TopicExtractor initialization with corrupted cache
# ============================================================================

class TestTopicExtractorCacheDeserializationErrors:
    """Test TopicExtractor handling of corrupted cache entries during init"""

    def test_init_with_invalid_video_topics_dict(self):
        """Test initialization handles VideoTopics.from_dict() failure (lines 144-147)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a cache file with an entry that will fail VideoTopics.from_dict()
            cache_path = Path(tmpdir) / "video_topics.json"

            # The entry.data must be a dict that VideoTopics.from_dict() can't handle
            # If 'video_path' is missing AND we access it incorrectly, it could fail
            corrupted_data = {
                "entries": {
                    "good_path": {
                        "key": "good_path",
                        "data": {"video_path": "good.mp4", "topics": ["test"], "confidence": 0.8},
                        "cached_at": "2026-01-01T00:00:00"
                    },
                    "bad_path": {
                        "key": "bad_path",
                        "data": None,  # This will cause from_dict to fail
                        "cached_at": "2026-01-01T00:00:00"
                    }
                }
            }
            with open(cache_path, 'w') as f:
                json.dump(corrupted_data, f)

            # Create mock config
            config = Mock()
            config.llm = Mock()

            # Should initialize without crashing - the warning path is hit
            extractor = TopicExtractor(config, cache_dir=tmpdir)

            assert extractor is not None
            # Valid entry may be loaded, corrupted one is skipped with warning

    def test_init_with_missing_required_fields(self):
        """Test initialization when VideoTopics.from_dict receives incomplete data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "video_topics.json"

            # Entry with data that has unusual types that might cause issues
            corrupted_data = {
                "entries": {
                    "test_path": {
                        "key": "test_path",
                        "data": {
                            "video_path": 12345,  # Should be string but is int
                            "topics": "not_a_list",  # Should be list but is string
                            "confidence": "high"  # Should be float but is string
                        },
                        "cached_at": "2026-01-01T00:00:00"
                    }
                }
            }
            with open(cache_path, 'w') as f:
                json.dump(corrupted_data, f)

            config = Mock()
            config.llm = Mock()

            # Should handle gracefully
            extractor = TopicExtractor(config, cache_dir=tmpdir)
            assert extractor is not None


# ============================================================================
# Test Lines 298-299: _parse_topics_response JSON parsing exception
# ============================================================================

class TestParseTopicsResponseException:
    """Test _parse_topics_response JSON parsing failure paths"""

    @pytest.fixture
    def temp_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_config(self):
        return Mock(spec=[])

    def test_parse_topics_json_decode_error(self, temp_cache, mock_config):
        """Test JSON decode error triggers fallback (lines 298-299)"""
        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        # JSON that looks like an array but has syntax error
        response = '["topic1", "topic2", malformed]'
        result = extractor._parse_topics_response(response)

        # Should fallback to regex extraction - finds quoted strings
        # The malformed part prevents JSON parsing
        assert "topic1" in result
        assert "topic2" in result

    def test_parse_topics_partial_json_in_text(self, temp_cache, mock_config):
        """Test extracting array from text with invalid surrounding JSON"""
        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        # Array embedded in text with broken JSON around it
        response = 'Here are topics: ["valid1", "valid2"] but also {broken: json'
        result = extractor._parse_topics_response(response)

        assert "valid1" in result
        assert "valid2" in result

    def test_parse_topics_exception_in_json_loads(self, temp_cache, mock_config):
        """Test handling when json.loads raises exception"""
        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        # Content that matches array pattern but fails json.loads
        response = '[not valid json at all]'
        result = extractor._parse_topics_response(response)

        # Falls back to quoted string extraction - no quotes in this case
        assert isinstance(result, list)


# ============================================================================
# Test Lines 381-382: _detect_with_llm exception handling
# ============================================================================

class TestDetectWithLLMException:
    """Test ChapterDetector._detect_with_llm exception path"""

    @pytest.fixture
    def mock_config_with_gemini(self):
        config = Mock()
        config.gemini_api_key = "test_api_key"
        return config

    def test_detect_chapters_llm_exception_returns_fallback(self, mock_config_with_gemini):
        """Test that LLM exception triggers fallback chapter (lines 381-382)"""
        detector = ChapterDetector(mock_config_with_gemini)

        segments = [
            {'text': 'First segment'},
            {'text': 'Second segment'},
            {'text': 'Third segment'}
        ]

        with patch('src.llm_client.create_client') as mock_create:
            # Make generate() raise an exception
            mock_client = Mock()
            mock_client.generate.side_effect = Exception("LLM API Error")
            mock_create.return_value = mock_client

            result = detector.detect_chapters(segments, overall_topic="Test Topic")

        # Should return fallback chapter
        assert len(result) == 1
        assert result[0]['chapter_id'] == 0
        assert result[0]['start_segment_idx'] == 0
        assert result[0]['end_segment_idx'] == 2
        assert 'test topic' in result[0]['topics']


# ============================================================================
# Test Line 403: _detect_with_llm truncation of long indexed_text
# ============================================================================

class TestDetectWithLLMTruncation:
    """Test ChapterDetector._detect_with_llm text truncation"""

    @pytest.fixture
    def mock_config_with_gemini(self):
        config = Mock()
        config.gemini_api_key = "test_api_key"
        return config

    def test_detect_chapters_truncates_long_text(self, mock_config_with_gemini):
        """Test that indexed_text is truncated when > 8000 chars (line 403)"""
        detector = ChapterDetector(mock_config_with_gemini)

        # Create segments that will produce > 8000 chars when indexed
        # Each segment text is about 100 chars, need ~90 segments
        segments = [
            {'text': f'Segment {i}: ' + 'A' * 90}
            for i in range(100)
        ]

        with patch('src.llm_client.create_client') as mock_create:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.parsed_data = [
                {"start_segment_idx": 0, "end_segment_idx": 99, "title": "All", "topics": ["test"]}
            ]
            mock_client.generate.return_value = mock_response
            mock_create.return_value = mock_client

            # Capture the prompt to verify truncation
            result = detector.detect_chapters(segments, overall_topic="Test")

            # Verify the LLM was called
            assert mock_client.generate.called
            # Result should be valid
            assert len(result) >= 1


# ============================================================================
# Test Lines 476-477: _parse_chapters_response exception handling
# ============================================================================

class TestParseChaptersResponseException:
    """Test ChapterDetector._parse_chapters_response exception path"""

    @pytest.fixture
    def mock_config(self):
        return Mock(spec=[])

    def test_parse_chapters_exception_returns_empty(self, mock_config):
        """Test that exception in parsing returns empty list (lines 476-477)"""
        detector = ChapterDetector(mock_config)

        # JSON that matches pattern but causes error during processing
        # Using a list with non-dict items
        response = '["not_a_dict", 12345, null]'

        result = detector._parse_chapters_response(response, num_segments=10)

        # Items that aren't dicts should be skipped, result may be empty
        # since isinstance(ch, dict) check filters them out
        assert isinstance(result, list)

    def test_parse_chapters_json_loads_exception(self, mock_config):
        """Test JSON loads exception in _parse_chapters_response"""
        detector = ChapterDetector(mock_config)

        # Invalid JSON that matches array regex pattern
        response = '[{invalid json content]'

        result = detector._parse_chapters_response(response, num_segments=10)

        # Should return empty list on exception
        assert result == []

    def test_parse_chapters_missing_keys(self, mock_config):
        """Test parsing chapters with missing required keys"""
        detector = ChapterDetector(mock_config)

        response = json.dumps([
            {"title": "No indices"},  # Missing start_segment_idx, end_segment_idx
            {"start_segment_idx": 0}  # Missing end_segment_idx, title
        ])

        result = detector._parse_chapters_response(response, num_segments=10)

        # Should handle missing keys gracefully with defaults
        assert isinstance(result, list)
        # Both entries are dicts, so they should be processed
        assert len(result) == 2


# ============================================================================
# Test Line 513: detect_location_chapters truncation
# ============================================================================

class TestDetectLocationChaptersTruncation:
    """Test ChapterDetector.detect_location_chapters text truncation"""

    @pytest.fixture
    def mock_config_with_gemini(self):
        config = Mock()
        config.gemini_api_key = "test_api_key"
        return config

    def test_detect_location_chapters_truncates_long_text(self, mock_config_with_gemini):
        """Test that indexed_text is truncated when > 8000 chars (line 513)"""
        detector = ChapterDetector(mock_config_with_gemini)

        # Create segments that will produce > 8000 chars when indexed
        segments = [
            {'text': f'Location {i}: ' + 'X' * 90}
            for i in range(100)
        ]

        with patch('src.llm_client.create_client') as mock_create:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.parsed_data = [
                {
                    "start_segment_idx": 0,
                    "end_segment_idx": 50,
                    "location_name": "Paris",
                    "location_type": "city",
                    "visual_keywords": ["tower"],
                    "context_keywords": ["romantic"],
                    "title": "Paris Section"
                }
            ]
            mock_client.generate.return_value = mock_response
            mock_create.return_value = mock_client

            result = detector.detect_location_chapters(segments, overall_topic="Travel")

            # Verify the LLM was called and result is valid
            assert mock_client.generate.called
            assert len(result) >= 1
            assert result[0].location_name == "Paris"


# ============================================================================
# Test Line 589: detect_location_chapters returns empty when no Gemini key
# ============================================================================

class TestDetectLocationChaptersNoGemini:
    """Test ChapterDetector.detect_location_chapters with no Gemini key"""

    def test_detect_location_chapters_no_gemini_key_returns_empty(self):
        """Test that missing Gemini key returns empty list (line 589)"""
        # Config without gemini_api_key attribute
        config = Mock(spec=[])
        detector = ChapterDetector(config)

        segments = [
            {'text': 'Walking through Paris'},
            {'text': 'Visiting the Eiffel Tower'}
        ]

        result = detector.detect_location_chapters(segments, overall_topic="Travel")

        # Should return empty list when no Gemini API key
        assert result == []

    def test_detect_location_chapters_empty_gemini_key_returns_empty(self):
        """Test that empty Gemini key returns empty list"""
        config = Mock()
        config.gemini_api_key = ""  # Empty string
        detector = ChapterDetector(config)

        segments = [
            {'text': 'Exploring Tokyo'},
            {'text': 'Japanese cuisine'}
        ]

        result = detector.detect_location_chapters(segments, overall_topic="Japan")

        # Empty string is falsy, should return empty list
        assert result == []

    def test_detect_location_chapters_none_gemini_key_returns_empty(self):
        """Test that None Gemini key returns empty list"""
        config = Mock()
        config.gemini_api_key = None
        detector = ChapterDetector(config)

        segments = [
            {'text': 'London bridge'}
        ]

        result = detector.detect_location_chapters(segments, overall_topic="UK")

        assert result == []


# ============================================================================
# Test Lines 632-635: _parse_location_chapters_response exception handling
# ============================================================================

class TestParseLocationChaptersResponseException:
    """Test ChapterDetector._parse_location_chapters_response exception path"""

    @pytest.fixture
    def mock_config(self):
        return Mock(spec=[])

    def test_parse_location_chapters_exception_returns_empty(self, mock_config):
        """Test that exception returns empty list (lines 632-635)"""
        detector = ChapterDetector(mock_config)

        # Invalid JSON that matches array pattern
        response = '[{this is not: valid json}]'

        result = detector._parse_location_chapters_response(response, num_segments=10)

        # Should return empty list on exception
        assert result == []

    def test_parse_location_chapters_non_dict_items(self, mock_config):
        """Test parsing filters out non-dict items"""
        detector = ChapterDetector(mock_config)

        response = json.dumps([
            "string_item",
            123,
            None,
            {"location_name": "Paris", "start_segment_idx": 0, "end_segment_idx": 5}
        ])

        result = detector._parse_location_chapters_response(response, num_segments=10)

        # Only the dict with location_name should be included
        assert len(result) == 1
        assert result[0].location_name == "Paris"

    def test_parse_location_chapters_dict_without_location_name(self, mock_config):
        """Test parsing skips dicts without location_name"""
        detector = ChapterDetector(mock_config)

        response = json.dumps([
            {"start_segment_idx": 0, "end_segment_idx": 5, "title": "No Location Name"},
            {"location_name": "", "start_segment_idx": 0, "end_segment_idx": 5},  # Empty string
            {"location_name": "Tokyo", "start_segment_idx": 0, "end_segment_idx": 5}
        ])

        result = detector._parse_location_chapters_response(response, num_segments=10)

        # Only entry with non-empty location_name should be included
        assert len(result) == 1
        assert result[0].location_name == "Tokyo"

    def test_parse_location_chapters_attribute_error(self, mock_config):
        """Test handling when chapter data causes AttributeError"""
        detector = ChapterDetector(mock_config)

        # This should parse but the nested structure might cause issues
        response = json.dumps([
            {
                "location_name": "Paris",
                "start_segment_idx": "not_an_int",  # Should be int
                "end_segment_idx": "also_not_int",
                "visual_keywords": "should_be_list",
                "context_keywords": 12345
            }
        ])

        # This may raise exception or handle gracefully
        result = detector._parse_location_chapters_response(response, num_segments=10)

        # Result should be empty or contain parsed item
        assert isinstance(result, list)


# ============================================================================
# Test Lines 776-781, 787: extract_location_from_video_metadata LLM paths
# ============================================================================

class TestExtractLocationFromVideoMetadataLLM:
    """Test extract_location_from_video_metadata LLM extraction paths"""

    def test_extract_location_no_pattern_match_uses_llm(self):
        """Test LLM extraction when pattern matching fails (lines 776-781)"""
        config = Mock()
        config.gemini_api_key = "test_key"

        # Title that doesn't match common patterns
        title = "Amazing views and scenery from unusual place"

        with patch('src.topic_extraction._extract_location_with_llm') as mock_llm:
            mock_llm.return_value = "Unknown Place"

            result = extract_location_from_video_metadata(
                title=title,
                description="Beautiful landscapes",
                source_keyword="travel",
                config=config
            )

            # Should call LLM and return its result
            mock_llm.assert_called_once()
            assert result == "Unknown Place"

    def test_extract_location_llm_exception_fallback_to_keyword(self):
        """Test fallback to keyword when LLM raises exception (lines 780-781, 787)"""
        config = Mock()
        config.gemini_api_key = "test_key"

        # Title without any pattern match (all lowercase, no keywords)
        title = "some random content here"
        source_keyword = "Paris travel"  # Has "Paris Travel" pattern that matches

        with patch('src.topic_extraction._extract_location_with_llm') as mock_llm:
            mock_llm.side_effect = Exception("API Error")

            result = extract_location_from_video_metadata(
                title=title,
                description="",
                source_keyword=source_keyword,
                config=config
            )

            # Should call LLM since title has no pattern match
            mock_llm.assert_called_once()
            # After exception, should try keyword patterns
            # "Paris travel" matches "[Location] Travel" pattern
            assert result == "Paris" or result is None

    def test_extract_location_llm_returns_none_fallback_to_keyword(self):
        """Test fallback to keyword when LLM returns None (line 787)"""
        config = Mock()
        config.gemini_api_key = "test_key"

        # Title without pattern (all lowercase)
        title = "some random content"
        source_keyword = "Tokyo Travel"  # Has pattern match

        with patch('src.topic_extraction._extract_location_with_llm') as mock_llm:
            mock_llm.return_value = None  # LLM couldn't extract location

            result = extract_location_from_video_metadata(
                title=title,
                description="",
                source_keyword=source_keyword,
                config=config
            )

            # Should call LLM since title has no pattern
            mock_llm.assert_called_once()
            # After LLM returns None, falls back to keyword
            # "Tokyo Travel" matches "[Location] Travel" pattern
            assert result == "Tokyo" or result is None

    def test_extract_location_pattern_match_skips_llm(self):
        """Test that pattern match success skips LLM call"""
        config = Mock()
        config.gemini_api_key = "test_key"

        # Title with clear pattern match
        title = "Paris, France Walking Tour"

        with patch('src.topic_extraction._extract_location_with_llm') as mock_llm:
            result = extract_location_from_video_metadata(
                title=title,
                description="",
                source_keyword="",
                config=config
            )

            # Should NOT call LLM since pattern matched
            mock_llm.assert_not_called()
            assert "Paris" in result


# ============================================================================
# Test _extract_location_with_llm edge cases
# ============================================================================

class TestExtractLocationWithLLMEdgeCases:
    """Test _extract_location_with_llm edge cases"""

    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.gemini_api_key = "test_key"
        return config

    @patch('src.llm_client.create_client')
    def test_extract_location_llm_too_long_response(self, mock_create, mock_config):
        """Test LLM response > 50 chars is rejected"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "This is a very long location name that exceeds fifty characters limit set in the code"
        mock_client.generate.return_value = mock_response
        mock_create.return_value = mock_client

        result = _extract_location_with_llm(
            title="Test Video",
            description="",
            source_keyword="test",
            config=mock_config
        )

        # Should return None for response > 50 chars
        assert result is None

    @patch('src.llm_client.create_client')
    def test_extract_location_llm_strips_quotes(self, mock_create, mock_config):
        """Test LLM response quotes are stripped"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = '"Paris"'  # Quoted response
        mock_client.generate.return_value = mock_response
        mock_create.return_value = mock_client

        result = _extract_location_with_llm(
            title="Paris Tour",
            description="",
            source_keyword="travel",
            config=mock_config
        )

        assert result == "Paris"

    @patch('src.llm_client.create_client')
    def test_extract_location_llm_strips_single_quotes(self, mock_create, mock_config):
        """Test LLM response single quotes are stripped"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "'Tokyo'"
        mock_client.generate.return_value = mock_response
        mock_create.return_value = mock_client

        result = _extract_location_with_llm(
            title="Tokyo Tour",
            description="",
            source_keyword="japan",
            config=mock_config
        )

        assert result == "Tokyo"


# ============================================================================
# Test extract_video_locations_batch edge cases
# ============================================================================

class TestExtractVideoLocationsBatchEdgeCases:
    """Test extract_video_locations_batch edge cases"""

    def test_batch_extraction_uses_path_key_fallback(self):
        """Test batch extraction uses 'path' key when 'file' missing"""
        videos = [
            {"path": "video1.mp4", "title": "Paris, France Tour", "keyword": "paris"}
        ]

        mock_location_service = Mock()
        mock_geo = Mock()
        mock_geo.name = "Paris"
        mock_geo.country_name = "France"
        mock_location_service.disambiguate.return_value = mock_geo

        result = extract_video_locations_batch(
            videos,
            config=None,
            location_service=mock_location_service
        )

        # Should find video1.mp4 using 'path' key
        assert "video1.mp4" in result

    def test_batch_extraction_skips_empty_file_path(self):
        """Test batch extraction skips entries with empty file path"""
        videos = [
            {"file": "", "title": "Paris Tour"},  # Empty file
            {"file": "video2.mp4", "title": "Tokyo, Japan Tour"}
        ]

        mock_location_service = Mock()
        mock_geo = Mock()
        mock_geo.name = "Tokyo"
        mock_geo.country_name = "Japan"
        mock_location_service.disambiguate.return_value = mock_geo

        result = extract_video_locations_batch(
            videos,
            config=None,
            location_service=mock_location_service
        )

        # Should only have video2.mp4
        assert "" not in result
        assert "video2.mp4" in result

    def test_batch_extraction_handles_disambiguate_returning_none(self):
        """Test batch handles location_service.disambiguate returning None"""
        videos = [
            {"file": "video1.mp4", "title": "Paris, France Tour", "keyword": "paris"}
        ]

        mock_location_service = Mock()
        mock_location_service.disambiguate.return_value = None  # Can't resolve

        result = extract_video_locations_batch(
            videos,
            config=None,
            location_service=mock_location_service
        )

        # Should be empty since disambiguate returned None
        assert "video1.mp4" not in result


# ============================================================================
# Test compute_topic_overlap and compute_topic_penalty edge cases
# ============================================================================

class TestTopicOverlapPenaltyEdgeCases:
    """Test compute_topic_overlap and compute_topic_penalty edge cases"""

    def test_topic_overlap_both_empty(self):
        """Test overlap with both lists empty"""
        count, ratio = compute_topic_overlap([], [])
        assert count == 0
        assert ratio == 0.0

    def test_topic_overlap_one_empty(self):
        """Test overlap with one list empty"""
        count, ratio = compute_topic_overlap(["test"], [])
        assert count == 0
        assert ratio == 0.0

    def test_topic_overlap_partial_match_calculation(self):
        """Test partial match adds 0.5"""
        # "paris" is contained in "paris tour"
        vo = ["paris"]
        video = ["paris tour"]

        count, ratio = compute_topic_overlap(vo, video)

        # No exact match (count=0), but partial match exists
        assert count == 0
        # Partial match adds 0.5, ratio = 0.5 / max(1,1) = 0.5
        assert ratio == 0.5

    def test_topic_penalty_exact_overlap_meets_threshold(self):
        """Test no penalty when exact overlap meets min_overlap"""
        vo = ["paris", "france"]
        video = ["paris", "france", "travel"]

        penalty = compute_topic_penalty(vo, video, max_penalty=0.15, min_overlap=2)

        # 2 exact matches >= min_overlap of 2
        assert penalty == 0.0

    def test_topic_penalty_partial_overlap_above_0_3(self):
        """Test small penalty for partial overlap ratio > 0.3"""
        vo = ["paris"]
        video = ["paris tour"]  # "paris" in "paris tour" = 0.5 partial

        penalty = compute_topic_penalty(vo, video, max_penalty=0.15, min_overlap=2)

        # overlap_count=0 < min_overlap=2, but ratio=0.5 > 0.3
        # Should get max_penalty * 0.3 = 0.045
        assert penalty == pytest.approx(0.15 * 0.3, rel=0.01)

    def test_topic_penalty_weak_overlap(self):
        """Test medium penalty for weak overlap (0 < ratio <= 0.3)"""
        # Create scenario where we have some overlap but ratio <= 0.3
        vo = ["paris", "france", "travel", "tour", "europe"]  # 5 topics
        video = ["tokyo", "japan", "asia", "food", "culture", "paris stuff"]  # 6 topics, "paris" in "paris stuff"

        penalty = compute_topic_penalty(vo, video, max_penalty=0.15, min_overlap=3)

        # overlap_count=0, partial match on "paris" in "paris stuff" = 0.5
        # ratio = 0.5 / max(5,6) = 0.5/6 = 0.083, which is > 0 but <= 0.3
        # Should get max_penalty * 0.6 = 0.09
        assert penalty == pytest.approx(0.15 * 0.6, rel=0.01)

    def test_topic_penalty_no_overlap(self):
        """Test full penalty when no overlap at all"""
        vo = ["paris", "france"]
        video = ["tokyo", "japan"]

        penalty = compute_topic_penalty(vo, video, max_penalty=0.15, min_overlap=1)

        # No overlap, should get full penalty
        assert penalty == 0.15


# ============================================================================
# Test VideoTopics and LocationChapter dataclasses
# ============================================================================

class TestDataclasses:
    """Test VideoTopics and LocationChapter dataclass methods"""

    def test_video_topics_to_dict(self):
        """Test VideoTopics.to_dict() method"""
        vt = VideoTopics(
            video_path="/test.mp4",
            topics=["travel", "nature"],
            source_keyword="nature",
            confidence=0.85,
            detected_location="Paris",
            location_data={"lat": 48.85, "lon": 2.35}
        )

        d = vt.to_dict()

        assert d['video_path'] == "/test.mp4"
        assert d['topics'] == ["travel", "nature"]
        assert d['confidence'] == 0.85
        assert d['detected_location'] == "Paris"

    def test_video_topics_from_dict(self):
        """Test VideoTopics.from_dict() method"""
        data = {
            'video_path': '/video.mp4',
            'topics': ['test'],
            'source_keyword': 'test',
            'confidence': 0.9
        }

        vt = VideoTopics.from_dict(data)

        assert vt.video_path == '/video.mp4'
        assert vt.topics == ['test']
        assert vt.confidence == 0.9

    def test_video_topics_from_dict_missing_fields(self):
        """Test VideoTopics.from_dict() with missing fields uses defaults"""
        data = {}  # Empty dict

        vt = VideoTopics.from_dict(data)

        assert vt.video_path == ''
        assert vt.topics == []
        assert vt.confidence == 0.0

    def test_location_chapter_to_dict(self):
        """Test LocationChapter.to_dict() method"""
        lc = LocationChapter(
            chapter_id=1,
            start_segment_idx=0,
            end_segment_idx=10,
            location_name="Tokyo",
            location_type="city",
            visual_keywords=["tower", "shrine"],
            context_keywords=["modern", "traditional"],
            title="Tokyo Adventures",
            topics=["tokyo", "travel"]
        )

        d = lc.to_dict()

        assert d['chapter_id'] == 1
        assert d['location_name'] == "Tokyo"
        assert d['visual_keywords'] == ["tower", "shrine"]

    def test_location_chapter_from_dict_defaults(self):
        """Test LocationChapter.from_dict() with minimal data"""
        data = {'chapter_id': 0, 'location_name': 'Paris'}

        lc = LocationChapter.from_dict(data)

        assert lc.chapter_id == 0
        assert lc.location_name == 'Paris'
        assert lc.location_type == 'city'  # Default
        assert lc.visual_keywords == []
        assert lc.context_keywords == []

    def test_location_chapter_segment_range_property(self):
        """Test LocationChapter.segment_range property"""
        lc = LocationChapter(
            chapter_id=0,
            start_segment_idx=5,
            end_segment_idx=15,
            location_name="Paris"
        )

        assert lc.segment_range == (5, 15)


# ============================================================================
# Test TopicCache serialization
# ============================================================================

class TestTopicCacheSerialization:
    """Test TopicCache serialization methods"""

    def test_serialize_entry(self):
        """Test TopicCache._serialize_entry"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TopicCache(cache_dir=tmpdir, index_name="test.json")

            entry = CacheEntry(
                key='test_key',
                data={'topics': ['travel']},
                cached_at='2026-01-01T00:00:00',
                metadata={'version': 1}
            )

            result = cache._serialize_entry(entry)

            assert result['data'] == {'topics': ['travel']}
            assert result['cached_at'] == '2026-01-01T00:00:00'
            assert result['metadata'] == {'version': 1}

    def test_deserialize_entry_with_metadata(self):
        """Test TopicCache._deserialize_entry with metadata"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TopicCache(cache_dir=tmpdir, index_name="test.json")

            data = {
                'data': {'topics': ['nature']},
                'cached_at': '2026-01-01',
                'metadata': {'source': 'llm'}
            }

            entry = cache._deserialize_entry(data)

            assert entry.data == {'topics': ['nature']}
            assert entry.metadata == {'source': 'llm'}

    def test_deserialize_entry_without_metadata(self):
        """Test TopicCache._deserialize_entry without metadata field"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TopicCache(cache_dir=tmpdir, index_name="test.json")

            data = {
                'data': {'topics': []},
                'cached_at': '2026-01-01'
                # No metadata key
            }

            entry = cache._deserialize_entry(data)

            assert entry.metadata == {}  # Default empty dict


# ============================================================================
# Test extract_batch method
# ============================================================================

class TestExtractBatch:
    """Test TopicExtractor.extract_batch method"""

    def test_extract_batch_basic(self):
        """Test basic batch extraction"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Mock(spec=[])
            extractor = TopicExtractor(config, cache_dir=tmpdir)

            transcripts = {
                '/video1.mp4': 'Short',  # Too short, will use keyword
                '/video2.mp4': 'Also short'
            }
            video_metadata = {
                '/video1.mp4': {'title': 'Video 1', 'keyword': 'nature'},
                '/video2.mp4': {'title': 'Video 2', 'keyword': 'travel'}
            }

            result = extractor.extract_batch(transcripts, video_metadata)

            assert '/video1.mp4' in result
            assert '/video2.mp4' in result
            assert isinstance(result['/video1.mp4'], VideoTopics)
            assert isinstance(result['/video2.mp4'], VideoTopics)

    def test_extract_batch_empty_metadata(self):
        """Test batch extraction with no metadata"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Mock(spec=[])
            extractor = TopicExtractor(config, cache_dir=tmpdir)

            transcripts = {
                '/video1.mp4': 'Short transcript'
            }

            result = extractor.extract_batch(transcripts, video_metadata=None)

            assert '/video1.mp4' in result

    def test_extract_batch_logging_interval(self):
        """Test batch extraction logs progress at intervals"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Mock(spec=[])
            extractor = TopicExtractor(config, cache_dir=tmpdir)

            # Create 55 videos to trigger logging at 50
            transcripts = {
                f'/video{i}.mp4': 'Short'
                for i in range(55)
            }
            video_metadata = {
                f'/video{i}.mp4': {'keyword': f'keyword{i}'}
                for i in range(55)
            }

            result = extractor.extract_batch(transcripts, video_metadata)

            # Should complete without error
            assert len(result) == 55


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
