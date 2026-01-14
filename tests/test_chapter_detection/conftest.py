"""Fixtures for chapter detection tests."""

import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_config():
    """Create mock config for chapter detection tests."""
    config = Mock()
    config.matching = Mock()
    config.matching.chapter_detection = Mock(
        enabled=True,
        use_validation_pass=True,
        use_boundary_refinement=True,
        default_strategy='topic',
        auto_detect_content_type=True,
        max_chunk_chars=6000,
        chunk_overlap_segments=5,
        min_chapter_confidence=0.5,
        min_chapter_segments=3,
        max_chapters=20,
    )
    config.matching.location_matching = Mock(enabled=True)
    config.gemini_api_key = "test_key"
    config.cache = Mock(cache_dir="/tmp/test_cache")
    config.cache_dir = "/tmp/test_cache"
    return config


@pytest.fixture
def sample_segments():
    """Sample segments for testing."""
    return [
        {"index": 0, "text": "Welcome to Paris, the city of lights.", "start": 0.0, "end": 3.0},
        {"index": 1, "text": "The Eiffel Tower is the most iconic landmark.", "start": 3.0, "end": 6.0},
        {"index": 2, "text": "Let's explore the Louvre museum.", "start": 6.0, "end": 9.0},
        {"index": 3, "text": "Now we travel to Tokyo, Japan.", "start": 9.0, "end": 12.0},
        {"index": 4, "text": "Shibuya crossing is incredibly busy.", "start": 12.0, "end": 15.0},
        {"index": 5, "text": "The temples are beautiful and peaceful.", "start": 15.0, "end": 18.0},
    ]


@pytest.fixture
def single_topic_segments():
    """Segments that should be one chapter."""
    return [
        {"index": 0, "text": "Introduction to Python programming.", "start": 0.0, "end": 3.0},
        {"index": 1, "text": "Python is a versatile language.", "start": 3.0, "end": 6.0},
        {"index": 2, "text": "You can use it for web development.", "start": 6.0, "end": 9.0},
        {"index": 3, "text": "Data science is another popular use.", "start": 9.0, "end": 12.0},
    ]


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = Mock()
    return client


@pytest.fixture
def mock_llm_response_two_chapters():
    """Mock LLM response with two chapters."""
    response = Mock()
    response.parsed_data = [
        {
            "start_segment_idx": 0,
            "end_segment_idx": 2,
            "title": "Paris Tour",
            "topics": ["paris", "france", "landmarks"],
            "location_name": "Paris",
            "location_type": "city",
            "visual_keywords": ["Eiffel Tower", "Louvre"],
            "context_keywords": ["culture", "art"],
            "boundary_reasoning": "Clear transition to Tokyo at segment 3",
            "confidence": "high"
        },
        {
            "start_segment_idx": 3,
            "end_segment_idx": 5,
            "title": "Tokyo Adventure",
            "topics": ["tokyo", "japan", "temples"],
            "location_name": "Tokyo",
            "location_type": "city",
            "visual_keywords": ["Shibuya", "temples"],
            "context_keywords": ["busy", "peaceful"],
            "boundary_reasoning": "End of transcript",
            "confidence": "high"
        }
    ]
    response.text = ""
    return response
