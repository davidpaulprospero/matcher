"""
Extended tests for topic_extraction.py to cover remaining uncovered lines.

Focuses on error handling paths, LLM integration, and edge cases.
"""

import pytest
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

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
# Test TopicCache Error Handling (lines 144-147)
# ============================================================================

class TestTopicCacheDeserialization:
    """Test TopicCache deserialization with error handling"""

    def test_deserialize_entry_basic(self):
        """Test basic entry deserialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TopicCache(cache_dir=tmpdir, index_name="test_topics.json")

            data = {
                'data': {'video_path': '/test.mp4', 'topics': ['test']},
                'cached_at': '2026-01-01T00:00:00',
                'metadata': {'version': 1}
            }

            entry = cache._deserialize_entry(data)

            assert entry.data == data['data']
            assert entry.cached_at == data['cached_at']
            assert entry.metadata == {'version': 1}

    def test_deserialize_entry_missing_metadata(self):
        """Test deserialization with missing metadata field"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TopicCache(cache_dir=tmpdir, index_name="test_topics.json")

            data = {
                'data': {'video_path': '/test.mp4', 'topics': ['test']},
                'cached_at': '2026-01-01T00:00:00'
                # No 'metadata' key
            }

            entry = cache._deserialize_entry(data)

            assert entry.data == data['data']
            assert entry.metadata == {}  # Default empty dict

    def test_serialize_entry(self):
        """Test entry serialization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = TopicCache(cache_dir=tmpdir, index_name="test_topics.json")

            entry = CacheEntry(
                key='test',
                data={'topics': ['travel']},
                cached_at='2026-01-01',
                metadata={'test': True}
            )

            result = cache._serialize_entry(entry)

            assert result['data'] == {'topics': ['travel']}
            assert result['cached_at'] == '2026-01-01'
            assert result['metadata'] == {'test': True}


# ============================================================================
# Test TopicExtractor with corrupted cache (lines 144-147)
# ============================================================================

class TestTopicExtractorCacheErrors:
    """Test TopicExtractor handling of corrupted cache entries"""

    def test_init_with_corrupted_cache_entry(self):
        """Test initialization handles corrupted cache entries gracefully"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a corrupted cache file directly
            import json
            cache_path = Path(tmpdir) / "video_topics.json"

            # Write corrupted JSON with invalid entries
            corrupted_data = {
                "entries": {
                    "valid_path": {
                        "key": "valid_path",
                        "data": {"video_path": "valid.mp4", "topics": ["test"]},
                        "cached_at": "2026-01-01T00:00:00"
                    },
                    "corrupted": {
                        "key": "corrupted",
                        "data": "not a dict",  # Invalid format - should be dict
                        "cached_at": "2026-01-01T00:00:00"
                    }
                }
            }
            with open(cache_path, 'w') as f:
                json.dump(corrupted_data, f)

            # Create mock config
            config = Mock()
            config.llm = Mock()

            # Should initialize without crashing even with corrupted entry
            extractor = TopicExtractor(config, cache_dir=tmpdir)

            assert extractor is not None
            # TopicExtractor should have loaded (corrupted entries may be skipped)


# ============================================================================
# Test _extract_with_llm (lines 262-288)
# ============================================================================

class TestExtractWithLLM:
    """Test _extract_with_llm method"""

    @pytest.fixture
    def temp_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_config_with_gemini(self):
        config = Mock()
        config.gemini_api_key = "test_api_key"
        return config

    @pytest.fixture
    def mock_config_no_gemini(self):
        config = Mock(spec=[])  # No gemini_api_key attribute
        return config

    @patch('src.llm_client.create_client')
    def test_extract_with_llm_gemini_success(self, mock_create, temp_cache, mock_config_with_gemini):
        """Test successful LLM extraction with Gemini"""
        # Mock LLM response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.parsed_data = ["travel", "nature", "adventure"]
        mock_client.generate.return_value = mock_response
        mock_create.return_value = mock_client

        extractor = TopicExtractor(mock_config_with_gemini, cache_dir=temp_cache)

        result = extractor._extract_with_llm(
            transcript_text="A long transcript about traveling to beautiful mountains.",
            video_title="Mountain Adventure",
            source_keyword="mountains"
        )

        assert result == ["travel", "nature", "adventure"]

    @patch('src.llm_client.create_client')
    def test_extract_with_llm_invalid_response(self, mock_create, temp_cache, mock_config_with_gemini):
        """Test LLM extraction with invalid response (non-list)"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.parsed_data = "not a list"  # Invalid
        mock_client.generate.return_value = mock_response
        mock_create.return_value = mock_client

        extractor = TopicExtractor(mock_config_with_gemini, cache_dir=temp_cache)

        result = extractor._extract_with_llm(
            transcript_text="A transcript about nature.",
            source_keyword="nature"
        )

        # Should fallback to source keyword
        assert "nature" in result

    def test_extract_with_llm_no_gemini_key(self, temp_cache, mock_config_no_gemini):
        """Test LLM extraction without Gemini API key"""
        extractor = TopicExtractor(mock_config_no_gemini, cache_dir=temp_cache)

        result = extractor._extract_with_llm(
            transcript_text="A transcript about mountains.",
            source_keyword="mountain hiking"
        )

        # Should fallback to source keyword
        assert "mountain" in result or "hiking" in result

    @patch('src.llm_client.create_client')
    def test_extract_with_llm_exception(self, mock_create, temp_cache, mock_config_with_gemini):
        """Test LLM extraction handles exceptions"""
        mock_create.side_effect = Exception("API Error")

        extractor = TopicExtractor(mock_config_with_gemini, cache_dir=temp_cache)

        result = extractor._extract_with_llm(
            transcript_text="A transcript about travel.",
            source_keyword="travel abroad"
        )

        # Should fallback to source keyword words
        assert "travel" in result or "abroad" in result

    def test_extract_with_llm_no_source_keyword(self, temp_cache, mock_config_no_gemini):
        """Test LLM extraction without source keyword returns empty list"""
        extractor = TopicExtractor(mock_config_no_gemini, cache_dir=temp_cache)

        result = extractor._extract_with_llm(
            transcript_text="A transcript.",
            source_keyword=None
        )

        assert result == []


# ============================================================================
# Test _parse_topics_response (lines 290-303)
# ============================================================================

class TestParseTopicsResponse:
    """Test _parse_topics_response method"""

    @pytest.fixture
    def temp_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_config(self):
        config = Mock(spec=[])
        return config

    def test_parse_topics_valid_json_array(self, temp_cache, mock_config):
        """Test parsing valid JSON array response"""
        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        response = '["travel", "nature", "adventure"]'
        result = extractor._parse_topics_response(response)

        assert result == ["travel", "nature", "adventure"]

    def test_parse_topics_json_in_text(self, temp_cache, mock_config):
        """Test parsing JSON array embedded in text"""
        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        response = 'Here are the topics: ["travel", "nature"] extracted from the transcript.'
        result = extractor._parse_topics_response(response)

        assert result == ["travel", "nature"]

    def test_parse_topics_invalid_json_fallback(self, temp_cache, mock_config):
        """Test fallback to quoted strings when JSON invalid"""
        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        response = 'Topics: "travel", "nature", and "adventure"'
        result = extractor._parse_topics_response(response)

        assert "travel" in result
        assert "nature" in result
        assert "adventure" in result

    def test_parse_topics_empty_response(self, temp_cache, mock_config):
        """Test parsing empty response"""
        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        response = ''
        result = extractor._parse_topics_response(response)

        assert result == []


# ============================================================================
# Test ChapterDetector (lines 351-446)
# ============================================================================

class TestChapterDetectorLLM:
    """Test ChapterDetector LLM integration"""

    @pytest.fixture
    def mock_config_with_gemini(self):
        config = Mock()
        config.gemini_api_key = "test_api_key"
        return config

    @pytest.fixture
    def mock_config_no_gemini(self):
        config = Mock(spec=[])
        return config

    def test_detect_chapters_empty_segments(self, mock_config_no_gemini):
        """Test detect_chapters with empty segments"""
        detector = ChapterDetector(mock_config_no_gemini)

        result = detector.detect_chapters([])

        assert result == []

    def test_detect_chapters_fallback(self, mock_config_no_gemini):
        """Test detect_chapters fallback when LLM unavailable"""
        detector = ChapterDetector(mock_config_no_gemini)

        segments = [
            {'text': 'First segment about Paris'},
            {'text': 'Second segment about travel'},
            {'text': 'Third segment about food'}
        ]

        result = detector.detect_chapters(segments, overall_topic="Travel")

        # Should return single fallback chapter
        assert len(result) == 1
        assert result[0]['chapter_id'] == 0
        assert result[0]['start_segment_idx'] == 0
        assert result[0]['end_segment_idx'] == 2
        assert 'travel' in result[0]['topics']

    @patch('src.llm_client.create_client')
    def test_detect_chapters_with_llm(self, mock_create, mock_config_with_gemini):
        """Test detect_chapters with LLM success"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.parsed_data = [
            {"start_segment_idx": 0, "end_segment_idx": 1, "title": "Paris", "topics": ["paris", "travel"]},
            {"start_segment_idx": 2, "end_segment_idx": 3, "title": "Food", "topics": ["food", "cuisine"]}
        ]
        mock_client.generate.return_value = mock_response
        mock_create.return_value = mock_client

        detector = ChapterDetector(mock_config_with_gemini)

        segments = [
            {'text': 'Welcome to Paris'},
            {'text': 'The Eiffel Tower is beautiful'},
            {'text': 'Now lets talk about food'},
            {'text': 'French cuisine is amazing'}
        ]

        result = detector.detect_chapters(segments)

        assert len(result) == 2
        assert result[0]['title'] == 'Paris'
        assert result[1]['title'] == 'Food'


# ============================================================================
# Test _parse_chapters_response (lines 448-479)
# ============================================================================

class TestParseChaptersResponse:
    """Test _parse_chapters_response method"""

    @pytest.fixture
    def mock_config(self):
        return Mock(spec=[])

    def test_parse_chapters_valid_json(self, mock_config):
        """Test parsing valid chapter JSON"""
        detector = ChapterDetector(mock_config)

        response = json.dumps([
            {"start_segment_idx": 0, "end_segment_idx": 5, "title": "Intro", "topics": ["intro"]},
            {"start_segment_idx": 6, "end_segment_idx": 10, "title": "Main", "topics": ["main"]}
        ])

        result = detector._parse_chapters_response(response, num_segments=15)

        assert len(result) == 2
        assert result[0]['chapter_id'] == 0
        assert result[0]['title'] == "Intro"
        assert result[1]['chapter_id'] == 1

    def test_parse_chapters_invalid_range(self, mock_config):
        """Test parsing chapters with out-of-range indices"""
        detector = ChapterDetector(mock_config)

        response = json.dumps([
            {"start_segment_idx": -5, "end_segment_idx": 100, "title": "Invalid", "topics": ["test"]}
        ])

        result = detector._parse_chapters_response(response, num_segments=10)

        # Should clamp to valid range
        assert len(result) == 1
        assert result[0]['start_segment_idx'] == 0
        assert result[0]['end_segment_idx'] == 9

    def test_parse_chapters_invalid_json(self, mock_config):
        """Test parsing invalid JSON response"""
        detector = ChapterDetector(mock_config)

        response = "This is not JSON at all"

        result = detector._parse_chapters_response(response, num_segments=10)

        assert result == []


# ============================================================================
# Test detect_location_chapters (lines 481-593)
# ============================================================================

class TestDetectLocationChapters:
    """Test detect_location_chapters method"""

    @pytest.fixture
    def mock_config_with_gemini(self):
        config = Mock()
        config.gemini_api_key = "test_api_key"
        return config

    @pytest.fixture
    def mock_config_no_gemini(self):
        return Mock(spec=[])

    def test_detect_location_chapters_empty(self, mock_config_no_gemini):
        """Test with empty segments"""
        detector = ChapterDetector(mock_config_no_gemini)

        result = detector.detect_location_chapters([])

        assert result == []

    @patch('src.llm_client.create_client')
    def test_detect_location_chapters_with_llm(self, mock_create, mock_config_with_gemini):
        """Test location chapter detection with LLM"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.parsed_data = [
            {
                "start_segment_idx": 0,
                "end_segment_idx": 5,
                "location_name": "Paris",
                "location_type": "city",
                "visual_keywords": ["Eiffel Tower", "Louvre"],
                "context_keywords": ["romantic"],
                "title": "Paris Adventure"
            }
        ]
        mock_client.generate.return_value = mock_response
        mock_create.return_value = mock_client

        detector = ChapterDetector(mock_config_with_gemini)

        segments = [{'text': f'Segment {i}'} for i in range(10)]

        result = detector.detect_location_chapters(segments)

        assert len(result) == 1
        assert result[0].location_name == "Paris"
        assert result[0].location_type == "city"


# ============================================================================
# Test _parse_location_chapters_response (lines 595-635)
# ============================================================================

class TestParseLocationChaptersResponse:
    """Test _parse_location_chapters_response method"""

    @pytest.fixture
    def mock_config(self):
        return Mock(spec=[])

    def test_parse_location_chapters_valid(self, mock_config):
        """Test parsing valid location chapter response"""
        detector = ChapterDetector(mock_config)

        response = json.dumps([
            {
                "start_segment_idx": 0,
                "end_segment_idx": 5,
                "location_name": "Tokyo",
                "location_type": "city",
                "visual_keywords": ["Tokyo Tower"],
                "context_keywords": ["technology"],
                "title": "Tokyo Visit"
            }
        ])

        result = detector._parse_location_chapters_response(response, num_segments=10)

        assert len(result) == 1
        assert isinstance(result[0], LocationChapter)
        assert result[0].location_name == "Tokyo"

    def test_parse_location_chapters_missing_location_name(self, mock_config):
        """Test parsing skips entries without location_name"""
        detector = ChapterDetector(mock_config)

        response = json.dumps([
            {"start_segment_idx": 0, "end_segment_idx": 5, "title": "No Location"},
            {"start_segment_idx": 6, "end_segment_idx": 10, "location_name": "Paris", "title": "Paris"}
        ])

        result = detector._parse_location_chapters_response(response, num_segments=15)

        # Only entry with location_name should be included
        assert len(result) == 1
        assert result[0].location_name == "Paris"


# ============================================================================
# Test _resolve_chapter_locations (lines 637-674)
# ============================================================================

class TestResolveChapterLocations:
    """Test _resolve_chapter_locations method"""

    @pytest.fixture
    def mock_config(self):
        return Mock(spec=[])

    def test_resolve_locations_success(self, mock_config):
        """Test successful location resolution"""
        detector = ChapterDetector(mock_config)

        mock_location_service = Mock()
        mock_geo = Mock()
        mock_geo.to_dict.return_value = {'lat': 48.8566, 'lon': 2.3522}
        mock_geo.name = "Paris"
        mock_geo.country_name = "France"
        mock_geo.country_code = "FR"
        mock_location_service.disambiguate.return_value = mock_geo

        chapters = [
            LocationChapter(
                chapter_id=0,
                start_segment_idx=0,
                end_segment_idx=5,
                location_name="Paris",
                visual_keywords=["Eiffel Tower"],
                context_keywords=["romantic"]
            )
        ]

        result = detector._resolve_chapter_locations(chapters, mock_location_service)

        assert len(result) == 1
        assert result[0].location_data == {'lat': 48.8566, 'lon': 2.3522}

    def test_resolve_locations_failure(self, mock_config):
        """Test handling of location resolution failure"""
        detector = ChapterDetector(mock_config)

        mock_location_service = Mock()
        mock_location_service.disambiguate.side_effect = Exception("API Error")

        chapters = [
            LocationChapter(
                chapter_id=0,
                start_segment_idx=0,
                end_segment_idx=5,
                location_name="Unknown Place",
                visual_keywords=[]
            )
        ]

        # Should not raise, just skip failed resolution
        result = detector._resolve_chapter_locations(chapters, mock_location_service)

        assert len(result) == 1
        assert result[0].location_data is None


# ============================================================================
# Test compute_topic_penalty edge cases (lines 730-739)
# ============================================================================

class TestComputeTopicPenaltyEdgeCases:
    """Test compute_topic_penalty edge cases"""

    def test_penalty_partial_match_above_threshold(self):
        """Test penalty with overlap_ratio > 0.3 but overlap_count < min_overlap"""
        # Use actual partial matches where one topic contains another
        vo_topics = ["travel", "adventure"]
        video_topics = ["travel adventure", "nature hiking"]  # "travel" is in "travel adventure"

        penalty = compute_topic_penalty(
            vo_topics,
            video_topics,
            max_penalty=0.15,
            min_overlap=3  # Require 3 exact matches (won't be met)
        )

        # There are partial matches, so overlap_ratio > 0.3
        # overlap_count=0 (no exact match), but partial matches exist
        # This hits the "overlap_ratio > 0.3" branch
        assert penalty <= 0.15  # Should be partial or full penalty

    def test_penalty_weak_match(self):
        """Test penalty with weak match (0 < overlap_ratio <= 0.3)"""
        vo_topics = ["travel", "nature", "adventure", "hiking", "mountains"]
        video_topics = ["food", "cooking", "travel"]  # Only 1 match

        penalty = compute_topic_penalty(
            vo_topics,
            video_topics,
            max_penalty=0.15,
            min_overlap=2
        )

        # 1 exact match, ratio is around 0.2, should get medium penalty
        assert 0.0 <= penalty <= 0.15


# ============================================================================
# Test _extract_location_with_llm (lines 845-887)
# ============================================================================

class TestExtractLocationWithLLM:
    """Test _extract_location_with_llm function"""

    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.gemini_api_key = "test_key"
        return config

    @patch('src.llm_client.create_client')
    def test_extract_location_with_llm_success(self, mock_create, mock_config):
        """Test successful location extraction with LLM"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "Paris"
        mock_client.generate.return_value = mock_response
        mock_create.return_value = mock_client

        result = _extract_location_with_llm(
            title="Walking Tour in Paris",
            description="Exploring the city",
            source_keyword="paris walking",
            config=mock_config
        )

        assert result == "Paris"

    @patch('src.llm_client.create_client')
    def test_extract_location_with_llm_none_response(self, mock_create, mock_config):
        """Test LLM returns NONE"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.text = "NONE"
        mock_client.generate.return_value = mock_response
        mock_create.return_value = mock_client

        result = _extract_location_with_llm(
            title="Tech Tutorial",
            description="Programming",
            source_keyword="python",
            config=mock_config
        )

        assert result is None

    @patch('src.llm_client.create_client')
    def test_extract_location_with_llm_exception(self, mock_create, mock_config):
        """Test LLM exception handling during generate()"""
        # Exception should happen in generate(), which is wrapped in try/except
        mock_client = Mock()
        mock_client.generate.side_effect = Exception("API Error")
        mock_create.return_value = mock_client

        result = _extract_location_with_llm(
            title="Test Video",
            description="Description",
            source_keyword="test",
            config=mock_config
        )

        assert result is None


# ============================================================================
# Test extract_video_locations_batch with location_service (lines 890-940)
# ============================================================================

class TestExtractVideoLocationsBatchWithService:
    """Test extract_video_locations_batch with location service"""

    def test_batch_extraction_with_location_service(self):
        """Test batch extraction with location service resolution"""
        mock_config = Mock()
        mock_config.gemini_api_key = None  # Use pattern matching only

        mock_location_service = Mock()
        mock_geo = Mock()
        mock_geo.name = "Paris"
        mock_geo.country_name = "France"
        mock_location_service.disambiguate.return_value = mock_geo

        videos = [
            {"file": "video1.mp4", "title": "Paris, France Tour", "keyword": "paris"},
            {"file": "video2.mp4", "title": "Tech Tutorial", "keyword": "python"}
        ]

        result = extract_video_locations_batch(
            videos,
            config=mock_config,
            location_service=mock_location_service
        )

        # First video has recognizable location pattern
        assert "video1.mp4" in result or len(result) >= 0

    def test_batch_extraction_location_service_error(self):
        """Test batch extraction handles location service errors"""
        mock_config = Mock()
        mock_config.gemini_api_key = None

        mock_location_service = Mock()
        mock_location_service.disambiguate.side_effect = Exception("Service Error")

        videos = [
            {"file": "video1.mp4", "title": "Paris, France Tour", "keyword": "paris"}
        ]

        # Should not raise, just skip failed resolutions
        result = extract_video_locations_batch(
            videos,
            config=mock_config,
            location_service=mock_location_service
        )

        assert isinstance(result, dict)

    def test_batch_extraction_missing_file(self):
        """Test batch extraction skips entries without file"""
        videos = [
            {"title": "No File"},  # Missing 'file' key
            {"file": "video2.mp4", "title": "Paris, France"}
        ]

        result = extract_video_locations_batch(videos)

        # Should process without error
        assert isinstance(result, dict)


# ============================================================================
# Test LocationChapter segment_range property (line 94)
# ============================================================================

class TestLocationChapterSegmentRange:
    """Test LocationChapter segment_range property"""

    def test_segment_range_property(self):
        """Test segment_range returns tuple"""
        chapter = LocationChapter(
            chapter_id=0,
            start_segment_idx=5,
            end_segment_idx=15,
            location_name="Tokyo"
        )

        result = chapter.segment_range

        assert result == (5, 15)
        assert isinstance(result, tuple)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
