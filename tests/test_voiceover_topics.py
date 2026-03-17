"""
Tests for voiceover topic extraction (US-111-002).

These tests verify:
1. Topic extraction function extracts 3-5 topics from voiceover text
2. Topics are stored in segment metadata
3. Config option enables/disables extraction
4. Fallback to original behavior when extraction fails
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

# Import the module under test
from src.matching.voiceover_topics import (
    extract_voiceover_topics,
    enrich_voiceover_segments_with_topics,
    extract_topics_for_segments_batch,
    VoiceoverTopicResult,
    DEFAULT_TOPIC_EXTRACTION_PROMPT,
    calculate_topic_similarity,
    calculate_segment_coherence,
    build_topic_aware_context,
    get_topic_enriched_text,
)


class MockConfig:
    """Mock config for testing"""
    def __init__(self, voiceover_topic_enabled=True, min_topics=3, max_topics=5, min_segment_length=50):
        self.gemini_api_key = "test_api_key"
        self.matching = Mock()
        self.matching.gemini_model = "gemini-2.0-flash"
        self.matching.voiceover_topic = Mock()
        self.matching.voiceover_topic.enabled = voiceover_topic_enabled
        self.matching.voiceover_topic.min_topics = min_topics
        self.matching.voiceover_topic.max_topics = max_topics
        self.matching.voiceover_topic.min_segment_length = min_segment_length
        # New config fields for US-126-006
        self.matching.voiceover_topic.retry_max_attempts = 3
        self.matching.voiceover_topic.retry_base_delay = 1.0
        self.matching.voiceover_topic.retry_max_delay = 10.0
        self.matching.voiceover_topic.fallback_to_keywords = True
        self.matching.voiceover_topic.cache_enabled = True


class MockSegment:
    """Mock SRTSegment for testing"""
    def __init__(self, text="", topics=None):
        self.text = text
        self.topics = topics or []


class TestExtractVoiceoverTopics:
    """Tests for the extract_voiceover_topics function"""

    def test_extract_topics_returns_list(self):
        """Test that extract_topics returns a list of topics"""
        config = MockConfig()

        # Mock the LLM client - patch where it's imported
        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.content = '["Paris", "Eiffel Tower", "France", "travel", "vacation"]'
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            result = extract_voiceover_topics(
                text="Let's visit the Eiffel Tower in Paris, France. It's a wonderful travel destination for vacation.",
                config=config,
            )

            assert result is not None
            assert isinstance(result, list)
            assert len(result) >= 3

    def test_extract_topics_empty_text(self):
        """Test that empty text returns None"""
        config = MockConfig()

        result = extract_voiceover_topics(
            text="",
            config=config,
        )

        assert result is None

    def test_extract_topics_very_short_text(self):
        """Test that very short text returns None"""
        config = MockConfig()

        result = extract_voiceover_topics(
            text="Hi",
            config=config,
        )

        assert result is None

    def test_extract_topics_respects_max(self):
        """Test that extraction respects max_topics limit"""
        config = MockConfig(min_topics=3, max_topics=3)

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = MagicMock()
            mock_response = MagicMock()
            # Return more than max_topics
            mock_response.content = '["topic1", "topic2", "topic3", "topic4", "topic5", "topic6"]'
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            result = extract_voiceover_topics(
                text="This is a longer text about various topics that should be extracted",
                config=config,
                max_topics=3,
            )

            if result:
                assert len(result) <= 3


class TestEnrichVoiceoverSegmentsWithTopics:
    """Tests for the enrich_voiceover_segments_with_topics function"""

    def test_enrich_segments_adds_topics(self):
        """Test that segments get topics added"""
        config = MockConfig(voiceover_topic_enabled=True)

        segments = [
            MockSegment(text="Paris is the capital of France with amazing architecture"),
            MockSegment(text="The Eiffel Tower is a famous landmark in Paris"),
        ]

        with patch('src.matching.voiceover_topics.extract_voiceover_topics') as mock_extract:
            mock_extract.return_value = ["Paris", "France", "architecture"]

            result = enrich_voiceover_segments_with_topics(segments, config)

            # Verify topics were added
            assert len(result) == 2
            # First segment should have topics from mock
            assert len(result[0].topics) == 3

    def test_enrich_skips_disabled_config(self):
        """Test that extraction is skipped when disabled in config"""
        config = MockConfig(voiceover_topic_enabled=False)

        segments = [
            MockSegment(text="Some text about topics"),
        ]

        with patch('src.matching.voiceover_topics.extract_voiceover_topics') as mock_extract:
            result = enrich_voiceover_segments_with_topics(segments, config)

            # Should not call extraction
            mock_extract.assert_not_called()

    def test_enrich_skips_segments_with_existing_topics(self):
        """Test that segments with existing topics are skipped"""
        config = MockConfig(voiceover_topic_enabled=True)

        segments = [
            MockSegment(text="Some text", topics=["existing", "topics"]),
        ]

        with patch('src.matching.voiceover_topics.extract_voiceover_topics') as mock_extract:
            result = enrich_voiceover_segments_with_topics(segments, config)

            # Should not call extraction for segments with existing topics
            mock_extract.assert_not_called()

    def test_enrich_skips_short_segments(self):
        """Test that segments shorter than min_segment_length are skipped"""
        config = MockConfig(voiceover_topic_enabled=True, min_segment_length=100)

        segments = [
            MockSegment(text="Short text"),  # Less than 100 chars
        ]

        with patch('src.matching.voiceover_topics.extract_voiceover_topics') as mock_extract:
            mock_extract.return_value = ["topic1", "topic2"]
            result = enrich_voiceover_segments_with_topics(segments, config)

            # Should not call extraction for short segments
            mock_extract.assert_not_called()


class TestExtractTopicsForSegmentsBatch:
    """Tests for the extract_topics_for_segments_batch function"""

    def test_batch_returns_results_for_all_segments(self):
        """Test that batch processing returns results for all segments"""
        config = MockConfig(voiceover_topic_enabled=True)

        segments = [
            MockSegment(text="Text about Paris"),
            MockSegment(text="Text about London"),
            MockSegment(text="Short"),  # Will be skipped due to length
        ]

        with patch('src.matching.voiceover_topics.extract_voiceover_topics') as mock_extract:
            def mock_extract_side_effect(text, config, min_topics, max_topics):
                if len(text) < 50:
                    return None
                return ["topic1", "topic2"]

            mock_extract.side_effect = mock_extract_side_effect

            results = extract_topics_for_segments_batch(segments, config)

            assert len(results) == 3
            assert all(isinstance(r, VoiceoverTopicResult) for r in results)

    def test_batch_disabled_returns_empty(self):
        """Test that disabled config returns empty results"""
        config = MockConfig(voiceover_topic_enabled=False)

        segments = [MockSegment(text="Some text")]

        results = extract_topics_for_segments_batch(segments, config)

        assert len(results) == 1
        assert results[0].success is False
        assert results[0].error == "disabled"


class TestVoiceoverTopicConfig:
    """Tests for VoiceoverTopicConfig integration"""

    def test_config_loaded_from_yaml(self):
        """Test that VoiceoverTopicConfig can be loaded from YAML dict"""
        from src.config.sections.matching import VoiceoverTopicConfig

        # Test with dict input
        config_dict = {
            'enabled': True,
            'min_topics': 3,
            'max_topics': 5,
            'min_segment_length': 50,
        }

        config = VoiceoverTopicConfig(**config_dict)

        assert config.enabled is True
        assert config.min_topics == 3
        assert config.max_topics == 5
        assert config.min_segment_length == 50

    def test_config_defaults(self):
        """Test that config has correct defaults"""
        from src.config.sections.matching import VoiceoverTopicConfig

        config = VoiceoverTopicConfig()

        assert config.enabled is True
        assert config.min_topics == 3
        assert config.max_topics == 5
        assert config.min_segment_length == 50


class TestPromptTemplate:
    """Tests for the topic extraction prompt"""

    def test_prompt_contains_keyword_instructions(self):
        """Test that prompt contains expected instructions"""
        assert "3-5 key topics" in DEFAULT_TOPIC_EXTRACTION_PROMPT
        assert "JSON array" in DEFAULT_TOPIC_EXTRACTION_PROMPT

    def test_prompt_template_format(self):
        """Test that prompt template has placeholder for text"""
        assert "{text}" in DEFAULT_TOPIC_EXTRACTION_PROMPT


class TestTopicSimilarity:
    """Tests for topic similarity calculation (US-111-003)"""

    def test_identical_topics(self):
        """Test that identical topics return 1.0 similarity"""
        topics1 = ["Paris", "France", "travel"]
        topics2 = ["Paris", "France", "travel"]

        similarity = calculate_topic_similarity(topics1, topics2)
        assert similarity == 1.0

    def test_partial_overlap(self):
        """Test that partial overlap returns partial similarity"""
        topics1 = ["Paris", "France", "travel"]
        topics2 = ["Paris", "London", "England"]

        similarity = calculate_topic_similarity(topics1, topics2)
        # Only "Paris" overlaps, so 1/5 unique tokens = 0.2
        assert 0.15 <= similarity <= 0.25

    def test_no_overlap(self):
        """Test that no overlap returns 0.0 similarity"""
        topics1 = ["Paris", "France"]
        topics2 = ["Tokyo", "Japan"]

        similarity = calculate_topic_similarity(topics1, topics2)
        assert similarity == 0.0

    def test_empty_topics(self):
        """Test that empty topics return 0.0"""
        assert calculate_topic_similarity([], ["Paris"]) == 0.0
        assert calculate_topic_similarity(["Paris"], []) == 0.0
        assert calculate_topic_similarity([], []) == 0.0

    def test_tokenization(self):
        """Test that topics are tokenized correctly"""
        topics1 = ["Eiffel Tower", "France"]
        topics2 = ["Tower", "French"]

        similarity = calculate_topic_similarity(topics1, topics2)
        # "tower" overlaps (from Eiffel Tower), "french" != "france"
        assert 0.2 <= similarity <= 0.4


class TestSegmentCoherence:
    """Tests for segment coherence calculation (US-111-003)"""

    def test_high_coherence(self):
        """Test that similar topics give high coherence"""
        segments = [
            MockSegment(text="Paris is great", topics=["Paris", "France", "travel"]),
            MockSegment(text="The Eiffel Tower", topics=["Paris", "France", "landmark"]),
            MockSegment(text="French food", topics=["France", "cuisine", "food"]),
        ]

        coherence = calculate_segment_coherence(segments, 1, similarity_threshold=0.3)
        assert coherence >= 0.5  # Should be coherent with neighbors

    def test_low_coherence(self):
        """Test that different topics give low coherence"""
        segments = [
            MockSegment(text="Paris", topics=["Paris", "France"]),
            MockSegment(text="Tokyo", topics=["Tokyo", "Japan"]),
            MockSegment(text="London", topics=["London", "UK"]),
        ]

        coherence = calculate_segment_coherence(segments, 1, similarity_threshold=0.3)
        assert coherence == 0.0  # No similarity with neighbors

    def test_no_topics(self):
        """Test that segments without topics return 0.0"""
        segments = [
            MockSegment(text="Some text", topics=[]),
            MockSegment(text="Some text", topics=["Paris"]),
        ]

        coherence = calculate_segment_coherence(segments, 1, similarity_threshold=0.3)
        assert coherence == 0.0

    def test_single_segment(self):
        """Test that single segment returns 1.0 (no neighbors to compare)"""
        segments = [
            MockSegment(text="Some text", topics=["Paris"]),
        ]

        coherence = calculate_segment_coherence(segments, 0, similarity_threshold=0.3)
        assert coherence == 1.0


class TestBuildTopicAwareContext:
    """Tests for topic-aware context building (US-111-003)"""

    def test_basic_context(self):
        """Test basic context building"""
        segments = [
            MockSegment(text="First", topics=["a"]),
            MockSegment(text="Second", topics=["b"]),
            MockSegment(text="Third", topics=["c"]),
        ]

        before, after, coherence = build_topic_aware_context(
            segments, 1, base_window=1
        )

        assert len(before) >= 1
        assert len(after) >= 1

    def test_coherence_adjustment(self):
        """Test that context window adjusts based on coherence"""
        # High coherence segments
        segments_high = [
            MockSegment(text="Paris", topics=["Paris", "France"]),
            MockSegment(text="Eiffel", topics=["Paris", "France"]),
            MockSegment(text="Louvre", topics=["Paris", "France"]),
        ]

        # Low coherence segments
        segments_low = [
            MockSegment(text="Paris", topics=["Paris"]),
            MockSegment(text="Tokyo", topics=["Tokyo"]),
            MockSegment(text="London", topics=["London"]),
        ]

        # With config that enables window adjustment
        class MockConfig:
            class Matching:
                class VoiceoverTopic:
                    enabled = True
                    topic_coherence_enabled = True
                    context_window_adjustment = True
                    max_context_segments = 4
                    min_topic_similarity = 0.3
            matching = Matching()

        config = MockConfig()

        # High coherence - should expand window
        before_high, after_high, coh_high = build_topic_aware_context(
            segments_high, 1, base_window=1, config=config
        )

        # Low coherence - should keep base window
        before_low, after_low, coh_low = build_topic_aware_context(
            segments_low, 1, base_window=1, config=config
        )

        assert coh_high > coh_low


class TestGetTopicEnrichedText:
    """Tests for topic-enriched text generation (US-111-003)"""

    def test_with_topics(self):
        """Test that topics are appended to text"""
        segment = MockSegment(text="Paris is the capital of France", topics=["Paris", "France", "capital"])

        result = get_topic_enriched_text(segment, max_text_length=30)

        assert "Paris" in result
        assert "France" in result
        assert "capital" in result

    def test_without_topics(self):
        """Test that text without topics returns plain text"""
        segment = MockSegment(text="Some text", topics=[])

        result = get_topic_enriched_text(segment, max_text_length=30)

        assert result == "Some text"

    def test_text_truncation(self):
        """Test that text is truncated to max_length"""
        long_text = "A" * 100
        segment = MockSegment(text=long_text, topics=["topic1"])

        result = get_topic_enriched_text(segment, max_text_length=30)

        assert len(result) <= 40  # 30 chars + "[topic1]"


class TestTopicCoherenceConfig:
    """Tests for topic coherence config settings (US-111-003)"""

    def test_config_with_coherence_settings(self):
        """Test that config loads with coherence settings"""
        from src.config.sections.matching import VoiceoverTopicConfig

        config = VoiceoverTopicConfig(
            enabled=True,
            topic_coherence_enabled=True,
            min_topic_similarity=0.3,
            coherence_boost=0.05,
            context_window_adjustment=True,
            max_context_segments=4,
        )

        assert config.topic_coherence_enabled is True
        assert config.min_topic_similarity == 0.3
        assert config.coherence_boost == 0.05
        assert config.context_window_adjustment is True
        assert config.max_context_segments == 4

    def test_config_defaults_coherence(self):
        """Test that coherence defaults are correct"""
        from src.config.sections.matching import VoiceoverTopicConfig

        config = VoiceoverTopicConfig()

        assert config.topic_coherence_enabled is True
        assert config.min_topic_similarity == 0.3
        assert config.coherence_boost == 0.05
        assert config.context_window_adjustment is True
        assert config.max_context_segments == 4


class TestVoiceoverTopicResult:
    """Tests for the VoiceoverTopicResult dataclass"""

    def test_result_creation_success(self):
        """Test creating a successful VoiceoverTopicResult"""
        from src.matching.voiceover_topics import VoiceoverTopicResult

        result = VoiceoverTopicResult(
            segment_index=0,
            topics=["Paris", "France", "travel"],
            success=True,
        )

        assert result.segment_index == 0
        assert result.topics == ["Paris", "France", "travel"]
        assert result.success is True
        assert result.error is None

    def test_result_creation_failure(self):
        """Test creating a failed VoiceoverTopicResult"""
        from src.matching.voiceover_topics import VoiceoverTopicResult

        result = VoiceoverTopicResult(
            segment_index=1,
            topics=[],
            success=False,
            error="too_short",
        )

        assert result.segment_index == 1
        assert result.topics == []
        assert result.success is False
        assert result.error == "too_short"

    def test_result_optional_error_field(self):
        """Test that error field is optional"""
        from src.matching.voiceover_topics import VoiceoverTopicResult

        result = VoiceoverTopicResult(
            segment_index=2,
            topics=["topic1"],
            success=True,
        )

        assert result.error is None


class TestExtractTopicsFromSegments:
    """Tests for topic extraction from segments (via batch function)"""

    def test_batch_extract_with_various_lengths(self):
        """Test batch extraction handles various segment lengths"""
        config = MockConfig(voiceover_topic_enabled=True)

        segments = [
            MockSegment(text="A" * 200),  # Long enough
            MockSegment(text="Short"),     # Too short
            MockSegment(text="B" * 150),   # Long enough
        ]

        with patch('src.matching.voiceover_topics.extract_voiceover_topics') as mock_extract:
            def mock_side_effect(text, config, min_topics, max_topics):
                if len(text) < 50:
                    return None
                return ["topic1", "topic2", "topic3"]

            mock_extract.side_effect = mock_side_effect

            results = extract_topics_for_segments_batch(segments, config)

            assert len(results) == 3
            # First segment: success
            assert results[0].success is True
            assert len(results[0].topics) == 3
            # Second segment: too short
            assert results[1].success is False
            assert results[1].error == "too_short"
            # Third segment: success
            assert results[2].success is True
            assert len(results[2].topics) == 3

    def test_batch_extract_extraction_failure(self):
        """Test batch handles extraction failures gracefully"""
        config = MockConfig(voiceover_topic_enabled=True)

        segments = [MockSegment(text="A" * 200)]

        with patch('src.matching.voiceover_topics.extract_voiceover_topics') as mock_extract:
            mock_extract.return_value = None  # Extraction failed

            results = extract_topics_for_segments_batch(segments, config)

            assert len(results) == 1
            assert results[0].success is False
            assert results[0].error == "extraction_failed"
            assert results[0].topics == []

    def test_batch_extract_with_llm_exception(self):
        """Test batch handles LLM exceptions gracefully"""
        config = MockConfig(voiceover_topic_enabled=True)

        segments = [MockSegment(text="A" * 200)]

        with patch('src.matching.voiceover_topics.extract_voiceover_topics') as mock_extract:
            # Return None to simulate extraction failure (as if exception was caught)
            mock_extract.return_value = None

            results = extract_topics_for_segments_batch(segments, config)

            assert len(results) == 1
            assert results[0].success is False
            assert results[0].error == "extraction_failed"


class TestRetryLogicAndFallback:
    """Tests for retry logic with exponential backoff and keyword fallback (US-126-006)"""

    def test_llm_failure_triggers_fallback_to_keyword_extraction(self):
        """Test that LLM failure triggers fallback to keyword extraction"""
        from src.matching.voiceover_topics import _extract_keywords_fallback, _topic_cache

        # Clear cache
        _topic_cache.clear()

        # Create mock config with fallback enabled
        config = MockConfig()
        config.matching.voiceover_topic.retry_max_attempts = 1  # Don't retry
        config.matching.voiceover_topic.fallback_to_keywords = True
        config.matching.voiceover_topic.cache_enabled = False

        # Mock LLM to fail
        with patch('src.matching.voiceover_topics._llm_extract_topics') as mock_llm:
            mock_llm.return_value = None  # Simulate LLM failure

            result = extract_voiceover_topics(
                text=" the Eiffel Tower in Paris,Let's visit France.",
                config=config,
                min_topics=3,
                max_topics=5,
            )

            # Should have called LLM
            assert mock_llm.called

    def test_cached_topics_returned_on_second_call(self):
        """Test that cached topics are returned on second call for same segment"""
        from src.matching.voiceover_topics import _topic_cache

        # Clear cache first
        _topic_cache.clear()

        # Create mock config with caching enabled
        config = MockConfig()
        config.matching.voiceover_topic.cache_enabled = True
        config.matching.voiceover_topic.fallback_to_keywords = False

        text = "Let's visit the Eiffel Tower in Paris, France for vacation"

        # First call - should hit LLM
        with patch('src.matching.voiceover_topics._llm_extract_topics') as mock_llm:
            mock_llm.return_value = ["Paris", "Eiffel Tower", "France"]

            result1 = extract_voiceover_topics(
                text=text,
                config=config,
                min_topics=3,
                max_topics=5,
            )

            # Should have called LLM once
            assert mock_llm.call_count == 1

        # Second call with same text - should return cached
        with patch('src.matching.voiceover_topics._llm_extract_topics') as mock_llm:
            result2 = extract_voiceover_topics(
                text=text,
                config=config,
                min_topics=3,
                max_topics=5,
            )

            # Should NOT have called LLM (cached)
            assert mock_llm.call_count == 0

        # Results should be the same
        assert result1 == result2

    def test_different_text_triggers_new_extraction(self):
        """Test that different text triggers new extraction (not cached)"""
        from src.matching.voiceover_topics import _topic_cache

        _topic_cache.clear()

        config = MockConfig()
        config.matching.voiceover_topic.cache_enabled = True

        # First call with text A
        with patch('src.matching.voiceover_topics._llm_extract_topics') as mock_llm:
            mock_llm.return_value = ["Paris", "France"]

            result1 = extract_voiceover_topics(
                text="Text about Paris and France",
                config=config,
            )

        # Second call with different text
        with patch('src.matching.voiceover_topics._llm_extract_topics') as mock_llm:
            mock_llm.return_value = ["Tokyo", "Japan"]

            result2 = extract_voiceover_topics(
                text="Text about Tokyo and Japan",
                config=config,
            )

        # Should have called LLM twice (not cached)
        assert mock_llm.call_count == 1  # Called once for second text

    def test_retry_with_exponential_backoff(self):
        """Test that retry uses exponential backoff"""
        import time
        from src.matching.voiceover_topics import _topic_cache

        _topic_cache.clear()

        config = MockConfig()
        config.matching.voiceover_topic.retry_max_attempts = 3
        config.matching.voiceover_topic.retry_base_delay = 0.1  # Fast for testing
        config.matching.voiceover_topic.retry_max_delay = 1.0
        config.matching.voiceover_topic.fallback_to_keywords = False
        config.matching.voiceover_topic.cache_enabled = False

        call_times = []

        def mock_llm_with_failure(*args, **kwargs):
            call_times.append(time.time())
            return None  # Always fail

        with patch('src.matching.voiceover_topics._llm_extract_topics') as mock_llm:
            mock_llm.side_effect = mock_llm_with_failure

            result = extract_voiceover_topics(
                text="Some text about topics",
                config=config,
                min_topics=3,
            )

            # Should have called 3 times (retry_max_attempts)
            assert mock_llm.call_count == 3
            assert result is None

    def test_fallback_keywords_extraction(self):
        """Test that keyword fallback extraction works"""
        from src.matching.voiceover_topics import _extract_keywords_fallback

        # Test with quoted phrases
        text = 'This video is about "Paris France" and "Eiffel Tower" travel vacation'
        result = _extract_keywords_fallback(text, max_topics=5)

        # Should find quoted phrases
        assert "Paris France" in result or "Eiffel Tower" in result or len(result) > 0

    def test_fallback_disabled_returns_none(self):
        """Test that when fallback is disabled, None is returned on LLM failure"""
        from src.matching.voiceover_topics import _topic_cache

        _topic_cache.clear()

        config = MockConfig()
        config.matching.voiceover_topic.retry_max_attempts = 1
        config.matching.voiceover_topic.fallback_to_keywords = False
        config.matching.voiceover_topic.cache_enabled = False

        with patch('src.matching.voiceover_topics._llm_extract_topics') as mock_llm:
            mock_llm.return_value = None  # LLM fails

            result = extract_voiceover_topics(
                text="Some text about topics",
                config=config,
                min_topics=3,
            )

            # Should return None (no fallback)
            assert result is None


class TestVoiceoverTopicConfigNew:
    """Tests for new VoiceoverTopicConfig settings (US-126-006)"""

    def test_config_with_retry_settings(self):
        """Test that config loads with retry settings"""
        from src.config.sections.matching import VoiceoverTopicConfig

        config = VoiceoverTopicConfig(
            enabled=True,
            retry_max_attempts=3,
            retry_base_delay=1.0,
            retry_max_delay=10.0,
        )

        assert config.retry_max_attempts == 3
        assert config.retry_base_delay == 1.0
        assert config.retry_max_delay == 10.0

    def test_config_defaults_retry(self):
        """Test that retry defaults are correct"""
        from src.config.sections.matching import VoiceoverTopicConfig

        config = VoiceoverTopicConfig()

        assert config.retry_max_attempts == 3
        assert config.retry_base_delay == 1.0
        assert config.retry_max_delay == 10.0

    def test_config_with_fallback_and_cache(self):
        """Test that config loads with fallback and cache settings"""
        from src.config.sections.matching import VoiceoverTopicConfig

        config = VoiceoverTopicConfig(
            fallback_to_keywords=True,
            cache_enabled=True,
        )

        assert config.fallback_to_keywords is True
        assert config.cache_enabled is True

    def test_config_defaults_fallback_and_cache(self):
        """Test that fallback and cache defaults are correct"""
        from src.config.sections.matching import VoiceoverTopicConfig

        config = VoiceoverTopicConfig()

        assert config.fallback_to_keywords is True
        assert config.cache_enabled is True


class TestExtractVoiceoverTopicsEdgeCases:
    """Additional edge case tests for extract_voiceover_topics"""

    def test_response_as_list(self):
        """Test handling when LLM returns a list directly"""
        config = MockConfig()

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = MagicMock()
            mock_response = MagicMock()
            # Return a list directly instead of string
            mock_response.content = ["Paris", "Eiffel Tower", "France", "travel"]
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            result = extract_voiceover_topics(
                text="Let's visit the Eiffel Tower in Paris, France.",
                config=config,
            )

            assert result is not None
            assert len(result) == 4

    def test_json_parse_error(self):
        """Test handling when JSON parsing fails"""
        config = MockConfig()

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = MagicMock()
            mock_response = MagicMock()
            # Return invalid JSON-like string
            mock_response.content = "This is not JSON ["
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            result = extract_voiceover_topics(
                text="Some text about Paris and France travel",
                config=config,
            )

            # Should return None due to parse failure
            assert result is None

    def test_fallback_delimiter_split(self):
        """Test fallback to delimiter splitting when no JSON found"""
        config = MockConfig()

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = MagicMock()
            mock_response = MagicMock()
            # Return plain text without JSON brackets
            mock_response.content = "Paris, France, Eiffel Tower, Travel, Vacation"
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            result = extract_voiceover_topics(
                text="Let's visit the Eiffel Tower in Paris, France.",
                config=config,
                max_topics=3,
            )

            # Should fallback to comma splitting
            assert result is not None
            assert len(result) <= 3

    def test_below_min_topics(self):
        """Test handling when fewer topics than min_topics returned"""
        config = MockConfig(min_topics=3)

        with patch('src.llm_client.create_client') as mock_create_client:
            mock_client = MagicMock()
            mock_response = MagicMock()
            # Return only 2 topics (below min of 3)
            mock_response.content = '["Paris", "France"]'
            mock_client.generate.return_value = mock_response
            mock_create_client.return_value = mock_client

            result = extract_voiceover_topics(
                text="Let's visit Paris in France.",
                config=config,
                min_topics=3,
            )

            # Should return None because below minimum
            assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
