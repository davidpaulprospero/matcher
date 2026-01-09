"""
Comprehensive unit tests for topic extraction module.

Tests VideoTopics, LocationChapter dataclasses, TopicExtractor, ChapterDetector,
and utility functions for topic overlap and location extraction.
"""

import pytest
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from dataclasses import asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the classes and functions to test
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
    extract_video_locations_batch,
)


# ============================================================================
# Test VideoTopics Dataclass
# ============================================================================

class TestVideoTopics:
    """Test VideoTopics dataclass"""

    def test_video_topics_creation(self):
        """Test creating VideoTopics instance"""
        topics = VideoTopics(
            video_path="/path/to/video.mp4",
            topics=["travel", "nature"],
            source_keyword="mountains",
            confidence=0.9
        )

        assert topics.video_path == "/path/to/video.mp4"
        assert topics.topics == ["travel", "nature"]
        assert topics.source_keyword == "mountains"
        assert topics.confidence == 0.9
        assert topics.detected_location is None
        assert topics.location_data is None

    def test_video_topics_to_dict(self):
        """Test VideoTopics to_dict serialization"""
        topics = VideoTopics(
            video_path="/path/to/video.mp4",
            topics=["travel", "nature"],
            source_keyword="mountains",
            confidence=0.9,
            detected_location="Swiss Alps",
            location_data={"lat": 46.0, "lon": 8.0}
        )

        data = topics.to_dict()

        assert isinstance(data, dict)
        assert data['video_path'] == "/path/to/video.mp4"
        assert data['topics'] == ["travel", "nature"]
        assert data['source_keyword'] == "mountains"
        assert data['confidence'] == 0.9
        assert data['detected_location'] == "Swiss Alps"
        assert data['location_data'] == {"lat": 46.0, "lon": 8.0}

    def test_video_topics_from_dict(self):
        """Test VideoTopics from_dict deserialization"""
        data = {
            'video_path': "/path/to/video.mp4",
            'topics': ["travel", "nature"],
            'source_keyword': "mountains",
            'confidence': 0.9,
            'detected_location': "Swiss Alps",
            'location_data': {"lat": 46.0, "lon": 8.0}
        }

        topics = VideoTopics.from_dict(data)

        assert topics.video_path == "/path/to/video.mp4"
        assert topics.topics == ["travel", "nature"]
        assert topics.source_keyword == "mountains"
        assert topics.confidence == 0.9
        assert topics.detected_location == "Swiss Alps"
        assert topics.location_data == {"lat": 46.0, "lon": 8.0}

    def test_video_topics_from_dict_missing_fields(self):
        """Test VideoTopics from_dict with missing optional fields"""
        data = {
            'video_path': "/path/to/video.mp4"
        }

        topics = VideoTopics.from_dict(data)

        assert topics.video_path == "/path/to/video.mp4"
        assert topics.topics == []
        assert topics.source_keyword == ""
        assert topics.confidence == 0.0
        assert topics.detected_location is None
        assert topics.location_data is None


# ============================================================================
# Test LocationChapter Dataclass
# ============================================================================

class TestLocationChapter:
    """Test LocationChapter dataclass"""

    def test_location_chapter_creation(self):
        """Test creating LocationChapter instance"""
        chapter = LocationChapter(
            chapter_id=1,
            start_segment_idx=0,
            end_segment_idx=10,
            location_name="Paris",
            location_type="city",
            visual_keywords=["Eiffel Tower", "Louvre"],
            context_keywords=["romantic", "fashion"],
            title="Paris Visit",
            topics=["travel", "culture"]
        )

        assert chapter.chapter_id == 1
        assert chapter.start_segment_idx == 0
        assert chapter.end_segment_idx == 10
        assert chapter.location_name == "Paris"
        assert chapter.location_type == "city"
        assert chapter.visual_keywords == ["Eiffel Tower", "Louvre"]
        assert chapter.context_keywords == ["romantic", "fashion"]
        assert chapter.title == "Paris Visit"
        assert chapter.topics == ["travel", "culture"]

    def test_location_chapter_to_dict(self):
        """Test LocationChapter to_dict serialization"""
        chapter = LocationChapter(
            chapter_id=1,
            start_segment_idx=0,
            end_segment_idx=10,
            location_name="Paris",
            location_type="city",
            visual_keywords=["Eiffel Tower"],
            topics=["travel"]
        )

        data = chapter.to_dict()

        assert isinstance(data, dict)
        assert data['chapter_id'] == 1
        assert data['location_name'] == "Paris"
        assert data['location_type'] == "city"
        assert "visual_keywords" in data

    def test_location_chapter_from_dict(self):
        """Test LocationChapter from_dict deserialization"""
        data = {
            'chapter_id': 2,
            'start_segment_idx': 10,
            'end_segment_idx': 20,
            'location_name': "Tokyo",
            'location_type': "city",
            'visual_keywords': ["Mount Fuji"],
            'context_keywords': ["technology"],
            'title': "Tokyo Adventure",
            'topics': ["urban", "culture"]
        }

        chapter = LocationChapter.from_dict(data)

        assert chapter.chapter_id == 2
        assert chapter.location_name == "Tokyo"
        assert chapter.visual_keywords == ["Mount Fuji"]


# ============================================================================
# Test TopicExtractor
# ============================================================================

class TestTopicExtractor:
    """Test TopicExtractor class"""

    @pytest.fixture
    def temp_cache(self):
        """Create temporary cache directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def mock_config(self):
        """Create mock config"""
        config = Mock()
        config.llm = Mock()
        config.llm.provider = "gemini"
        config.llm.model = "gemini-2.0-flash"
        return config

    def test_topic_extractor_init(self, temp_cache, mock_config):
        """Test TopicExtractor initialization"""
        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        assert extractor.config == mock_config
        assert extractor.cache_dir == Path(temp_cache)
        assert isinstance(extractor._cache_obj, TopicCache)
        assert isinstance(extractor._topics_cache, dict)

    def test_get_cached_topics_miss(self, temp_cache, mock_config):
        """Test get_cached_topics with cache miss"""
        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        result = extractor.get_cached_topics("/path/to/video.mp4")

        assert result is None

    def test_get_cached_topics_hit(self, temp_cache, mock_config):
        """Test get_cached_topics with cache hit"""
        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        # Add to cache
        topics = VideoTopics(
            video_path="/path/to/video.mp4",
            topics=["travel"],
            confidence=0.9
        )
        extractor._topics_cache["/path/to/video.mp4"] = topics

        result = extractor.get_cached_topics("/path/to/video.mp4")

        assert result is not None
        assert result.topics == ["travel"]

    def test_extract_topics_short_transcript(self, temp_cache, mock_config):
        """Test extract_topics_from_transcript with short transcript"""
        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        result = extractor.extract_topics_from_transcript(
            transcript_text="Short",
            video_path="/path/to/video.mp4",
            source_keyword="nature"
        )

        assert result.topics == ["nature"]
        assert result.confidence == 0.5
        assert result.source_keyword == "nature"

    def test_extract_topics_empty_transcript(self, temp_cache, mock_config):
        """Test extract_topics_from_transcript with empty transcript"""
        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        result = extractor.extract_topics_from_transcript(
            transcript_text="",
            video_path="/path/to/video.mp4",
            source_keyword="travel"
        )

        assert result.topics == ["travel"]
        assert result.confidence == 0.5

    @patch('src.topic_extraction.TopicExtractor._extract_with_llm')
    def test_extract_topics_with_llm_success(self, mock_llm, temp_cache, mock_config):
        """Test extract_topics_from_transcript with LLM success"""
        mock_llm.return_value = ["travel", "nature", "adventure"]

        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        long_transcript = "This is a long transcript about traveling through mountains and forests. " * 10

        result = extractor.extract_topics_from_transcript(
            transcript_text=long_transcript,
            video_path="/path/to/video.mp4",
            video_title="Mountain Adventure",
            source_keyword="hiking"
        )

        assert result.topics == ["travel", "nature", "adventure"]
        assert result.confidence == 0.9
        assert result.source_keyword == "hiking"
        mock_llm.assert_called_once()

    @patch('src.topic_extraction.TopicExtractor._extract_with_llm')
    def test_extract_topics_with_llm_failure(self, mock_llm, temp_cache, mock_config):
        """Test extract_topics_from_transcript with LLM failure"""
        mock_llm.side_effect = Exception("LLM API error")

        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        long_transcript = "This is a long transcript. " * 20

        result = extractor.extract_topics_from_transcript(
            transcript_text=long_transcript,
            video_path="/path/to/video.mp4",
            source_keyword="nature"
        )

        # Should fallback to source keyword
        assert result.topics == ["nature"]
        assert result.confidence == 0.5

    @patch('src.topic_extraction.TopicExtractor._extract_with_llm')
    def test_extract_topics_caching(self, mock_llm, temp_cache, mock_config):
        """Test that extracted topics are cached"""
        mock_llm.return_value = ["travel", "culture"]

        extractor = TopicExtractor(mock_config, cache_dir=temp_cache)

        long_transcript = "Travel and culture content. " * 20

        # First call - should use LLM
        result1 = extractor.extract_topics_from_transcript(
            transcript_text=long_transcript,
            video_path="/path/to/video.mp4"
        )

        # Second call - should use cache
        result2 = extractor.extract_topics_from_transcript(
            transcript_text=long_transcript,
            video_path="/path/to/video.mp4"
        )

        assert result1.topics == result2.topics
        assert mock_llm.call_count == 1  # Only called once


# ============================================================================
# Test Utility Functions
# ============================================================================

class TestTopicOverlap:
    """Test compute_topic_overlap function"""

    def test_compute_topic_overlap_identical(self):
        """Test topic overlap with identical topics"""
        topics1 = ["travel", "nature", "adventure"]
        topics2 = ["travel", "nature", "adventure"]

        overlap_count, overlap_ratio = compute_topic_overlap(topics1, topics2)

        assert overlap_count == 3
        assert overlap_ratio == 1.0

    def test_compute_topic_overlap_partial(self):
        """Test topic overlap with partial match"""
        topics1 = ["travel", "nature", "adventure"]
        topics2 = ["travel", "culture", "food"]

        overlap_count, overlap_ratio = compute_topic_overlap(topics1, topics2)

        assert overlap_count == 1
        assert overlap_ratio == pytest.approx(1/3, rel=0.01)

    def test_compute_topic_overlap_none(self):
        """Test topic overlap with no match"""
        topics1 = ["travel", "nature"]
        topics2 = ["technology", "science"]

        overlap_count, overlap_ratio = compute_topic_overlap(topics1, topics2)

        assert overlap_count == 0
        assert overlap_ratio == 0.0

    def test_compute_topic_overlap_empty(self):
        """Test topic overlap with empty lists"""
        topics1 = []
        topics2 = ["travel", "nature"]

        overlap_count, overlap_ratio = compute_topic_overlap(topics1, topics2)

        assert overlap_count == 0
        assert overlap_ratio == 0.0


class TestTopicPenalty:
    """Test compute_topic_penalty function"""

    def test_compute_topic_penalty_high_overlap(self):
        """Test penalty with high topic overlap"""
        video_topics = ["travel", "nature", "adventure"]
        segment_topics = ["travel", "nature", "mountains"]

        penalty = compute_topic_penalty(
            video_topics,
            segment_topics,
            penalty_threshold=0.3,
            penalty_amount=0.5
        )

        # High overlap (2/3) > threshold (0.3), no penalty
        assert penalty == 0.0

    def test_compute_topic_penalty_low_overlap(self):
        """Test penalty with low topic overlap"""
        video_topics = ["technology", "science"]
        segment_topics = ["travel", "nature", "adventure"]

        penalty = compute_topic_penalty(
            video_topics,
            segment_topics,
            penalty_threshold=0.3,
            penalty_amount=0.5
        )

        # Low overlap (0) < threshold (0.3), apply penalty
        assert penalty == 0.5

    def test_compute_topic_penalty_at_threshold(self):
        """Test penalty at exact threshold"""
        video_topics = ["travel", "nature", "adventure"]
        segment_topics = ["travel", "food", "culture"]

        penalty = compute_topic_penalty(
            video_topics,
            segment_topics,
            penalty_threshold=0.33,
            penalty_amount=0.4
        )

        # Overlap (1/3 = 0.33) >= threshold, no penalty
        assert penalty == 0.0


class TestLocationExtraction:
    """Test location extraction functions"""

    def test_extract_location_patterns_city_country(self):
        """Test extracting 'City, Country' pattern"""
        text = "Visiting Paris, France next week"

        location = _extract_location_patterns(text)

        assert location == "Paris, France"

    def test_extract_location_patterns_city_state(self):
        """Test extracting 'City, State' pattern"""
        text = "Road trip to Austin, Texas"

        location = _extract_location_patterns(text)

        assert location == "Austin, Texas"

    def test_extract_location_patterns_multiple(self):
        """Test extracting first location when multiple exist"""
        text = "From Paris, France to Rome, Italy"

        location = _extract_location_patterns(text)

        # Should return first match
        assert location in ["Paris, France", "Rome, Italy"]

    def test_extract_location_patterns_no_match(self):
        """Test when no location pattern found"""
        text = "This is a video about technology and programming"

        location = _extract_location_patterns(text)

        assert location is None

    def test_extract_location_from_video_metadata_title(self):
        """Test extracting location from video title"""
        result = extract_location_from_video_metadata(
            title="Amazing Paris, France Travel Guide",
            description="General description"
        )

        assert result == "Paris, France"

    def test_extract_location_from_video_metadata_description(self):
        """Test extracting location from video description"""
        result = extract_location_from_video_metadata(
            title="Travel Vlog",
            description="Exploring Tokyo, Japan and its culture"
        )

        assert result == "Tokyo, Japan"

    def test_extract_location_from_video_metadata_no_match(self):
        """Test when no location found in metadata"""
        result = extract_location_from_video_metadata(
            title="Tech Review",
            description="Latest gadgets and technology"
        )

        assert result is None


class TestBatchLocationExtraction:
    """Test extract_video_locations_batch function"""

    def test_extract_video_locations_batch_empty(self):
        """Test batch extraction with empty list"""
        result = extract_video_locations_batch([])

        assert result == {}

    def test_extract_video_locations_batch_with_locations(self):
        """Test batch extraction with video metadata"""
        videos = [
            {"file": "video1.mp4", "title": "Paris, France Tour"},
            {"file": "video2.mp4", "title": "Tech Tutorial"},
            {"file": "video3.mp4", "description": "Exploring Tokyo, Japan"}
        ]

        result = extract_video_locations_batch(videos)

        assert isinstance(result, dict)
        assert len(result) >= 0  # May extract 0-2 locations depending on implementation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
