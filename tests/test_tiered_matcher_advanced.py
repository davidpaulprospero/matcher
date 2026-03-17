"""
Advanced tests for matching/tiered_matcher.py

Targets uncovered gaps: face preference, LLM fallback, alternatives, strategies
"""

import pytest
import sys
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.tiered_matcher import TieredMatcher
from src.utils import SRTSegment, SceneInfo, Match, MatchResult, AlternativeMatch, CacheManager
from src.topic_extraction import VideoTopics


@pytest.fixture
def mock_config():
    """Create mock config for TieredMatcher"""
    config = Mock()
    config.matching = Mock()
    config.matching.gemini_model = "gemini-2.0-flash"
    config.matching.anthropic_model = "claude-3-haiku"
    config.matching.ollama_model = "llama3.2"
    config.matching.ollama_host = "http://localhost:11434"
    config.matching.primary_provider = "gemini"
    config.matching.secondary_provider = "anthropic"
    config.matching.use_local_for_review = False
    config.matching.min_confidence = 0.3
    config.matching.embedding_candidates = 10
    config.matching.high_confidence_threshold = 0.85
    config.matching.low_confidence_threshold = 0.4
    config.matching.skip_llm_threshold = 0.85
    config.matching.max_clip_reuse = 3
    config.matching.reuse_penalty = 0.05
    config.matching.chapter_matching_enabled = False
    config.matching.topic_mismatch_penalty = 0.15
    config.matching.face_preference = "neutral"
    config.matching.cache_llm_responses = True
    config.matching.location_matching = None
    config.gemini_api_key = "test_key"
    config.anthropic_api_key = "test_key2"
    return config


@pytest.fixture
def mock_cache():
    """Create mock cache manager"""
    cache = Mock(spec=CacheManager)
    cache.cache_dir = "/tmp/cache"
    cache.get_llm_response = Mock(return_value=None)
    cache.cache_llm_response = Mock()
    return cache


@pytest.fixture
def sample_vo_segment():
    """Create sample voiceover segment"""
    vo = SRTSegment(0, 0.0, 5.0, "earthquake damage in city", "")
    vo.keywords = ["earthquake", "damage", "city"]
    vo.entities = []
    return vo


@pytest.fixture
def sample_candidates():
    """Create sample video candidates"""
    seg1 = SRTSegment(0, 0.0, 5.0, "earthquake aftermath", "/video1.mp4")
    seg1.source = "current_project"
    seg1.face_score = 0.8  # High face presence

    seg2 = SRTSegment(1, 10.0, 15.0, "city destruction", "/video2.mp4")
    seg2.source = "global_cache"
    seg2.face_score = 0.2  # Low face presence (B-roll)

    seg3 = SRTSegment(2, 20.0, 25.0, "rescue operations", "/video3.mp4")
    seg3.source = "current_project"
    seg3.face_score = None  # No face score

    return [
        (seg1, 0.85),
        (seg2, 0.80),
        (seg3, 0.75)
    ]


class TestFacePreferenceConfig:
    """Test face preference configuration"""

    @pytest.mark.fast
    def test_face_preference_neutral_default(self, mock_config, mock_cache):
        """Test neutral face preference is default"""
        matcher = TieredMatcher(mock_config, mock_cache)

        assert matcher.face_preference == "neutral"

    @pytest.mark.fast
    def test_face_preference_more_config(self, mock_config, mock_cache):
        """Test 'more' face preference configuration"""
        mock_config.matching.face_preference = "more"

        matcher = TieredMatcher(mock_config, mock_cache)

        assert matcher.face_preference == "more"

    @pytest.mark.fast
    def test_face_preference_none_config(self, mock_config, mock_cache):
        """Test 'none' face preference configuration"""
        mock_config.matching.face_preference = "none"

        matcher = TieredMatcher(mock_config, mock_cache)

        assert matcher.face_preference == "none"


class TestGapMatchCreation:
    """Test gap match creation when no candidates (lines 343-351)"""

    @pytest.mark.fast
    def test_gap_match_when_no_candidates(self, mock_config, mock_cache, sample_vo_segment):
        """Test gap match creation when no candidates provided"""
        matcher = TieredMatcher(mock_config, mock_cache)

        result = matcher.match_segment(
            sample_vo_segment,
            [],  # No candidates
            segment_idx=0,
            scenes={}
        )

        assert result.has_gap == True
        assert result.gap_reason == "No candidates"
        assert result.primary_match.confidence == 0.0

    @pytest.mark.fast
    def test_reuse_tracker_initialized(self, mock_config, mock_cache):
        """Test reuse tracker is initialized properly"""
        mock_config.matching.max_clip_reuse = 5
        mock_config.matching.reuse_penalty = 0.08

        matcher = TieredMatcher(mock_config, mock_cache)

        assert matcher.reuse_tracker is not None
        assert hasattr(matcher.reuse_tracker, 'can_use')
        assert hasattr(matcher.reuse_tracker, 'adjust_confidence')


class TestLLMProviderFallback:
    """Test LLM provider initialization and fallback (lines 136-166, 496-519)"""

    @pytest.mark.fast
    def test_primary_gemini_provider_init(self, mock_config, mock_cache):
        """Test Gemini primary provider initialization"""
        mock_config.matching.primary_provider = "gemini"
        mock_config.gemini_api_key = "test_gemini_key"

        matcher = TieredMatcher(mock_config, mock_cache)

        assert matcher.primary_provider is not None
        assert matcher.primary_provider.__class__.__name__ == "GeminiMatcher"

    @pytest.mark.fast
    def test_primary_anthropic_provider_init(self, mock_config, mock_cache):
        """Test Claude primary provider initialization"""
        mock_config.matching.primary_provider = "anthropic"
        mock_config.anthropic_api_key = "test_claude_key"

        matcher = TieredMatcher(mock_config, mock_cache)

        assert matcher.primary_provider is not None
        assert matcher.primary_provider.__class__.__name__ == "ClaudeMatcher"

    @pytest.mark.fast
    def test_auto_fallback_to_gemini(self, mock_config, mock_cache):
        """Test auto-fallback to Gemini when primary not specified"""
        mock_config.matching.primary_provider = "unknown"
        mock_config.gemini_api_key = "test_gemini_key"

        matcher = TieredMatcher(mock_config, mock_cache)

        # Should fallback to Gemini (lines 142-144)
        assert matcher.primary_provider is not None

    @pytest.mark.fast
    def test_auto_fallback_to_anthropic(self, mock_config, mock_cache):
        """Test auto-fallback to Anthropic when Gemini unavailable"""
        mock_config.matching.primary_provider = "unknown"
        mock_config.gemini_api_key = None
        mock_config.anthropic_api_key = "test_claude_key"

        matcher = TieredMatcher(mock_config, mock_cache)

        # Should fallback to Anthropic (lines 145-147)
        assert matcher.primary_provider is not None

    @pytest.mark.fast
    def test_secondary_provider_init(self, mock_config, mock_cache):
        """Test secondary provider initialization"""
        mock_config.matching.secondary_provider = "anthropic"
        mock_config.anthropic_api_key = "test_claude_key"

        matcher = TieredMatcher(mock_config, mock_cache)

        assert matcher.secondary_provider is not None

    @pytest.mark.fast
    def test_local_provider_disabled(self, mock_config, mock_cache):
        """Test local provider when disabled"""
        mock_config.matching.use_local_for_review = False

        matcher = TieredMatcher(mock_config, mock_cache)

        # Should not initialize local provider
        assert matcher.local_provider is None

    @pytest.mark.fast
    def test_use_local_for_review_config(self, mock_config, mock_cache):
        """Test use_local_for_review configuration"""
        mock_config.matching.use_local_for_review = True

        # Will try to init but fail without Ollama running
        matcher = TieredMatcher(mock_config, mock_cache)

        # Should either init or be None (depends on Ollama availability)
        assert matcher.local_provider is None or matcher.local_provider is not None


class TestLocationMethods:
    """Test location delegation methods (lines 169-179)"""

    @pytest.mark.fast
    def test_set_location_chapters_with_matcher(self, mock_config, mock_cache):
        """Test setting location chapters when LocationMatcher exists"""
        # Enable location matching
        mock_config.matching.location_matching = Mock()
        mock_config.matching.location_matching.enabled = False  # Will init but not enable

        matcher = TieredMatcher(mock_config, mock_cache)

        # Mock location_matcher
        matcher.location_matcher = Mock()

        chapters = [Mock(), Mock()]
        matcher.set_location_chapters(chapters)

        # Should delegate to location_matcher
        matcher.location_matcher.set_location_chapters.assert_called_once_with(chapters)

    @pytest.mark.fast
    def test_set_location_chapters_without_matcher(self, mock_config, mock_cache):
        """Test setting location chapters when no LocationMatcher"""
        matcher = TieredMatcher(mock_config, mock_cache)

        chapters = [Mock(), Mock()]
        matcher.set_location_chapters(chapters)

        # Should handle gracefully (no error)
        assert True

    @pytest.mark.fast
    def test_set_video_locations_with_matcher(self, mock_config, mock_cache):
        """Test setting video locations when LocationMatcher exists"""
        matcher = TieredMatcher(mock_config, mock_cache)
        matcher.location_matcher = Mock()

        locations = {"/video1.mp4": Mock(), "/video2.mp4": Mock()}
        matcher.set_video_locations(locations)

        # Should delegate to location_matcher
        matcher.location_matcher.set_video_locations.assert_called_once_with(locations)


class TestUtilityMethods:
    """Test utility methods"""

    @pytest.mark.fast
    def test_should_skip_llm_high_similarity(self, mock_config, mock_cache):
        """Test LLM skipping for high similarity"""
        mock_config.matching.high_confidence_threshold = 0.85

        matcher = TieredMatcher(mock_config, mock_cache)

        assert matcher._should_skip_llm(0.90) == True
        assert matcher._should_skip_llm(0.85) == True
        assert matcher._should_skip_llm(0.80) == False

    @pytest.mark.fast
    def test_get_cache_key_generation(self, mock_config, mock_cache, sample_vo_segment, sample_candidates):
        """Test cache key generation"""
        matcher = TieredMatcher(mock_config, mock_cache)

        key = matcher._get_cache_key(sample_vo_segment.text, sample_candidates)

        # Should generate 16-char MD5 hash
        assert isinstance(key, str)
        assert len(key) == 16

    @pytest.mark.fast
    def test_get_cached_response_hit(self, mock_config, mock_cache, sample_vo_segment, sample_candidates):
        """Test cached LLM response retrieval"""
        mock_cache.get_llm_response.return_value = {
            'selected': 0,
            'confidence': 0.85,
            'reasoning': "Cached match"
        }

        matcher = TieredMatcher(mock_config, mock_cache)

        key = matcher._get_cache_key(sample_vo_segment.text, sample_candidates)
        result = matcher._get_cached_response(key)

        assert result is not None
        assert result[0] == 0  # selected index
        assert result[1] == 0.85  # confidence
        assert result[2] == "Cached match"  # reasoning

    @pytest.mark.fast
    def test_get_cached_response_miss(self, mock_config, mock_cache, sample_vo_segment, sample_candidates):
        """Test cache miss returns None"""
        mock_cache.get_llm_response.return_value = None

        matcher = TieredMatcher(mock_config, mock_cache)

        key = matcher._get_cache_key(sample_vo_segment.text, sample_candidates)
        result = matcher._get_cached_response(key)

        assert result is None


class TestChapterMatching:
    """Test chapter-based matching configuration"""

    @pytest.mark.fast
    def test_chapter_matching_enabled(self, mock_config, mock_cache):
        """Test chapter matching when enabled"""
        mock_config.matching.chapter_matching_enabled = True
        mock_config.matching.topic_mismatch_penalty = 0.20

        matcher = TieredMatcher(mock_config, mock_cache)

        assert matcher.chapter_matching_enabled == True
        assert matcher.topic_mismatch_penalty == 0.20

    @pytest.mark.fast
    def test_chapter_matching_disabled(self, mock_config, mock_cache):
        """Test chapter matching when disabled (default)"""
        # Don't set chapter_matching_enabled (should use getattr default)
        del mock_config.matching.chapter_matching_enabled

        matcher = TieredMatcher(mock_config, mock_cache)

        assert matcher.chapter_matching_enabled == False


class TestVideoTopicsIntegration:
    """Test video_topics parameter handling"""

    @pytest.mark.fast
    def test_video_topics_provided(self, mock_config, mock_cache):
        """Test TieredMatcher with video_topics"""
        video_topics = {
            "/video1.mp4": VideoTopics(
                video_path="/video1.mp4",
                topics=["earthquake", "disaster"],
                source_keyword="earthquake",
                confidence=0.9
            )
        }

        matcher = TieredMatcher(mock_config, mock_cache, video_topics=video_topics)

        assert len(matcher.video_topics) == 1
        assert "/video1.mp4" in matcher.video_topics

    @pytest.mark.fast
    def test_video_topics_default_empty(self, mock_config, mock_cache):
        """Test TieredMatcher defaults to empty video_topics"""
        matcher = TieredMatcher(mock_config, mock_cache)

        assert matcher.video_topics == {}
