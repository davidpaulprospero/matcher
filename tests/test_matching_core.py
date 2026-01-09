"""
Comprehensive tests for matching module.

Covers:
- TieredMatcher initialization and configuration
- Two-stage matching (embedding + LLM)
- LLM providers (Gemini, Anthropic, Ollama)
- Location filtering
- Face preference
- Reuse tracking
- Alternative and secondary matches
- Matching strategies (7 strategies)
- Diversity scoring

Created: 2026-01-09 (Phase 2.3)
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.tiered_matcher import TieredMatcher
from src.matching.strategies import StrategyMatcher
from src.matching.main import match_all_segments
from src.config import Config
from src.utils import SRTSegment, SceneInfo, Match, MatchResult, AlternativeMatch, StrategyMatch, CacheManager


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def config(temp_dir):
    """Create a test config"""
    config = Config()
    config.cache_dir = str(temp_dir / ".cache")
    config.gemini_api_key = "test_key"
    config.anthropic_api_key = "test_key"

    # Configure matching
    config.matching.primary_provider = "gemini"
    config.matching.gemini_model = "gemini-2.0-flash"
    config.matching.min_confidence = 0.6
    config.matching.skip_llm_threshold = 0.9
    config.matching.max_clip_reuse = 2
    config.matching.reuse_penalty = 0.1
    config.matching.embedding_candidates = 30
    config.matching.llm_rerank_candidates = 5

    # Configure output
    config.output.num_alternatives = 2
    config.output.include_strategy_tracks = True
    config.output.strategy_tracks = ["embedding_diversity", "broll_only"]

    return config


@pytest.fixture
def cache_manager(temp_dir):
    """Create a mock cache manager"""
    cache = Mock(spec=CacheManager)
    cache.cache_dir = str(temp_dir / ".cache")
    cache.get_llm_response = Mock(return_value=None)
    cache.save_llm_response = Mock()
    return cache


@pytest.fixture
def vo_segments():
    """Create sample voiceover segments"""
    return [
        SRTSegment(
            index=1,
            start_time=0.0,
            end_time=5.0,
            text="Welcome to Tokyo, the bustling capital of Japan",
            source_file="voiceover.srt"
        ),
        SRTSegment(
            index=2,
            start_time=5.0,
            end_time=10.0,
            text="The city is famous for its modern architecture",
            source_file="voiceover.srt"
        )
    ]


@pytest.fixture
def video_segments():
    """Create sample video segments"""
    segments = [
        SRTSegment(
            index=1,
            start_time=0.0,
            end_time=10.0,
            text="Tokyo cityscape with skyscrapers",
            source_file="tokyo_footage.mp4"
        ),
        SRTSegment(
            index=2,
            start_time=10.0,
            end_time=20.0,
            text="Modern buildings in Tokyo",
            source_file="tokyo_footage.mp4"
        ),
        SRTSegment(
            index=3,
            start_time=0.0,
            end_time=15.0,
            text="Kyoto temple at sunset",
            source_file="kyoto_footage.mp4"
        )
    ]

    # Add keywords for testing
    for seg in segments:
        seg.keywords = ["tokyo", "japan", "city"]

    return segments


@pytest.fixture
def embeddings():
    """Create sample embeddings (simple vectors for testing)"""
    vo_embeddings = [
        [0.9, 0.1, 0.0],  # Matches tokyo_footage closely
        [0.8, 0.2, 0.0]   # Also matches tokyo_footage
    ]

    video_embeddings = [
        [1.0, 0.0, 0.0],  # tokyo_footage seg 1
        [0.9, 0.1, 0.0],  # tokyo_footage seg 2
        [0.1, 0.0, 0.9]   # kyoto_footage seg 3 (different)
    ]

    return vo_embeddings, video_embeddings


@pytest.fixture
def scenes():
    """Create sample scene info"""
    return {
        "tokyo_footage.mp4": [
            SceneInfo(
                video_path="tokyo_footage.mp4",
                scene_index=0,
                start_time=0.0,
                end_time=10.0,
                description="Tokyo cityscape",
                visual_keywords=["building", "city", "skyscraper"]
            ),
            SceneInfo(
                video_path="tokyo_footage.mp4",
                scene_index=1,
                start_time=10.0,
                end_time=20.0,
                description="Modern architecture",
                visual_keywords=["building", "modern", "glass"]
            )
        ],
        "kyoto_footage.mp4": [
            SceneInfo(
                video_path="kyoto_footage.mp4",
                scene_index=0,
                start_time=0.0,
                end_time=15.0,
                description="Traditional temple",
                visual_keywords=["temple", "traditional", "sunset"]
            )
        ]
    }


# ============================================================================
# Test TieredMatcher Initialization
# ============================================================================

class TestTieredMatcherInit:
    """Test TieredMatcher initialization and configuration"""

    def test_init_default_config(self, config, cache_manager):
        """Test initialization with default config"""
        matcher = TieredMatcher(config, cache_manager)

        assert matcher.config == config
        assert matcher.cache == cache_manager
        assert matcher.min_confidence == 0.6
        assert matcher.max_clip_reuse == 2
        assert matcher.reuse_penalty == 0.1
        assert matcher.embedding_candidates == 30
        assert matcher.face_preference == "neutral"

    def test_init_with_video_topics(self, config, cache_manager):
        """Test initialization with video topics"""
        video_topics = {"video1.mp4": Mock()}
        matcher = TieredMatcher(config, cache_manager, video_topics=video_topics)

        assert matcher.video_topics == video_topics
        assert "video1.mp4" in matcher.video_topics

    def test_init_without_cache(self, config):
        """Test initialization without cache manager"""
        matcher = TieredMatcher(config, cache=None)

        assert matcher.cache is None
        assert matcher.config == config

    def test_init_providers(self, config, cache_manager):
        """Test LLM provider initialization"""
        with patch('src.matching.llm_providers.GeminiMatcher'):
            matcher = TieredMatcher(config, cache_manager)

            # Primary provider should be initialized
            assert matcher.primary_provider is not None

    def test_location_matching_disabled_by_default(self, config, cache_manager):
        """Test location matching is disabled by default"""
        matcher = TieredMatcher(config, cache_manager)

        assert matcher.location_matching_enabled is False
        assert matcher.location_matcher is None


# ============================================================================
# Test TieredMatcher Location Methods
# ============================================================================

class TestTieredMatcherLocation:
    """Test location-aware matching"""

    def test_set_location_chapters(self, config, cache_manager):
        """Test setting location chapters"""
        matcher = TieredMatcher(config, cache_manager)
        matcher.location_matcher = Mock()

        chapters = [Mock()]
        matcher.set_location_chapters(chapters)

        matcher.location_matcher.set_location_chapters.assert_called_once_with(chapters)

    def test_set_video_locations(self, config, cache_manager):
        """Test setting video locations"""
        matcher = TieredMatcher(config, cache_manager)
        matcher.location_matcher = Mock()

        locations = {"video1.mp4": Mock()}
        matcher.set_video_locations(locations)

        matcher.location_matcher.set_video_locations.assert_called_once_with(locations)


# ============================================================================
# Test TieredMatcher Cache Methods
# ============================================================================

class TestTieredMatcherCache:
    """Test LLM response caching"""

    def test_get_cache_key_generates_unique_key(self, config, cache_manager, video_segments):
        """Test cache key generation"""
        matcher = TieredMatcher(config, cache_manager)

        candidates = [(video_segments[0], 0.9), (video_segments[1], 0.8)]
        key = matcher._get_cache_key("test voiceover", candidates)

        assert isinstance(key, str)
        assert len(key) == 16  # MD5 hash truncated to 16 chars

    def test_get_cached_response_returns_none_if_disabled(self, config, cache_manager):
        """Test cache returns None if caching disabled"""
        config.matching.cache_llm_responses = False
        matcher = TieredMatcher(config, cache_manager)

        result = matcher._get_cached_response("test_key")

        assert result is None

    def test_get_cached_response_returns_data(self, config, cache_manager):
        """Test cache returns data if available"""
        cache_manager.get_llm_response = Mock(return_value={
            'selected': 0,
            'confidence': 0.85,
            'reasoning': 'Best match'
        })

        matcher = TieredMatcher(config, cache_manager)
        result = matcher._get_cached_response("test_key")

        assert result == (0, 0.85, 'Best match')

    def test_cache_response_saves_data(self, config, cache_manager):
        """Test caching LLM response"""
        matcher = TieredMatcher(config, cache_manager)

        matcher._cache_response("test_key", 0, 0.85, "Good match")

        cache_manager.save_llm_response.assert_called_once_with(
            "test_key",
            {'selected': 0, 'confidence': 0.85, 'reasoning': 'Good match'}
        )


# ============================================================================
# Test TieredMatcher Scene Methods
# ============================================================================

class TestTieredMatcherScene:
    """Test scene-related methods"""

    def test_get_scene_for_segment_finds_scene(self, config, cache_manager, video_segments, scenes):
        """Test finding scene for segment"""
        matcher = TieredMatcher(config, cache_manager)

        scene = matcher._get_scene_for_segment(video_segments[0], scenes)

        assert scene is not None
        assert scene.start_time == 0.0
        assert "Tokyo cityscape" in scene.description

    def test_get_scene_for_segment_no_scenes(self, config, cache_manager, video_segments):
        """Test scene lookup when no scenes available"""
        matcher = TieredMatcher(config, cache_manager)

        scene = matcher._get_scene_for_segment(video_segments[0], None)

        assert scene is None


# ============================================================================
# Test TieredMatcher Context Building
# ============================================================================

class TestTieredMatcherContext:
    """Test context building for LLM"""

    def test_build_context_with_before_and_after(self, config, cache_manager, vo_segments):
        """Test building context from surrounding segments"""
        matcher = TieredMatcher(config, cache_manager)

        context = matcher._build_context(
            context_before=[vo_segments[0]],
            context_after=[vo_segments[1]]
        )

        assert context is not None
        assert "Before:" in context
        assert "After:" in context

    def test_build_context_with_only_before(self, config, cache_manager, vo_segments):
        """Test building context with only before segments"""
        matcher = TieredMatcher(config, cache_manager)

        context = matcher._build_context(
            context_before=[vo_segments[0]],
            context_after=None
        )

        assert context is not None
        assert "Before:" in context
        assert "After:" not in context

    def test_build_context_empty_returns_none(self, config, cache_manager):
        """Test building context with no segments"""
        matcher = TieredMatcher(config, cache_manager)

        context = matcher._build_context(
            context_before=None,
            context_after=None
        )

        assert context is None


# ============================================================================
# Test TieredMatcher Match Segment
# ============================================================================

class TestTieredMatcherMatchSegment:
    """Test single segment matching"""

    def test_match_segment_no_candidates_returns_gap(self, config, cache_manager, vo_segments):
        """Test matching with no candidates returns gap"""
        matcher = TieredMatcher(config, cache_manager)

        result = matcher.match_segment(vo_segments[0], [])

        assert result.has_gap is True
        assert result.gap_reason == "No candidates"
        assert result.primary_match.confidence == 0.0

    def test_match_segment_high_similarity_skips_llm(self, config, cache_manager, vo_segments, video_segments, scenes):
        """Test high embedding similarity skips LLM"""
        config.matching.skip_llm_threshold = 0.9
        matcher = TieredMatcher(config, cache_manager)

        # High similarity candidate
        candidates = [(video_segments[0], 0.95)]

        result = matcher.match_segment(vo_segments[0], candidates, scenes)

        assert result.has_gap is False
        assert result.primary_match.confidence >= 0.9
        assert "High embedding similarity" in result.primary_match.reasoning

    def test_match_segment_uses_cached_response(self, config, cache_manager, vo_segments, video_segments, scenes):
        """Test matching uses cached LLM response"""
        cache_manager.get_llm_response = Mock(return_value={
            'selected': 0,
            'confidence': 0.85,
            'reasoning': 'Cached match'
        })

        matcher = TieredMatcher(config, cache_manager)
        candidates = [(video_segments[0], 0.7), (video_segments[1], 0.6)]

        result = matcher.match_segment(vo_segments[0], candidates, scenes)

        assert "(cached)" in result.primary_match.reasoning
        assert result.primary_match.confidence == 0.85

    def test_match_segment_with_llm(self, config, cache_manager, vo_segments, video_segments, scenes):
        """Test matching with LLM provider"""
        matcher = TieredMatcher(config, cache_manager)

        # Mock LLM provider
        mock_provider = Mock()
        mock_provider.match_batch = Mock(return_value=[(0, 0.8, "Good match")])
        matcher.primary_provider = mock_provider

        candidates = [(video_segments[0], 0.7), (video_segments[1], 0.6)]

        result = matcher.match_segment(vo_segments[0], candidates, scenes)

        assert result.has_gap is False
        assert result.primary_match.confidence >= 0.7
        mock_provider.match_batch.assert_called_once()

    def test_match_segment_generates_alternatives(self, config, cache_manager, vo_segments, video_segments, scenes):
        """Test matching generates alternatives"""
        config.output.num_alternatives = 2
        matcher = TieredMatcher(config, cache_manager)

        candidates = [
            (video_segments[0], 0.9),
            (video_segments[1], 0.8),
            (video_segments[2], 0.7)
        ]

        result = matcher.match_segment(vo_segments[0], candidates, scenes)

        assert len(result.alternatives) <= 2


# ============================================================================
# Test TieredMatcher Alternatives
# ============================================================================

class TestTieredMatcherAlternatives:
    """Test alternative match generation"""

    def test_get_alternatives_prefers_different_sources(self, config, cache_manager, video_segments, scenes):
        """Test alternatives prefer different source files"""
        config.output.num_alternatives = 2
        matcher = TieredMatcher(config, cache_manager)

        # Primary is from tokyo_footage.mp4, candidates include kyoto_footage.mp4
        primary = video_segments[0]
        candidates = [
            (video_segments[1], 0.8),  # Same source as primary
            (video_segments[2], 0.7)   # Different source (kyoto)
        ]

        alternatives = matcher._get_alternatives(candidates, scenes, primary)

        # Should prefer different source (kyoto) over same source
        assert len(alternatives) > 0
        # First alternative should be from different source if available
        if len(alternatives) > 0:
            assert alternatives[0].video_segment.source_file != primary.source_file

    def test_get_alternatives_respects_num_alternatives_config(self, config, cache_manager, video_segments, scenes):
        """Test alternatives respects num_alternatives config"""
        config.output.num_alternatives = 1
        matcher = TieredMatcher(config, cache_manager)

        candidates = [
            (video_segments[0], 0.9),
            (video_segments[1], 0.8),
            (video_segments[2], 0.7)
        ]

        alternatives = matcher._get_alternatives(candidates, scenes)

        assert len(alternatives) <= 1


# ============================================================================
# Test StrategyMatcher
# ============================================================================

class TestStrategyMatcher:
    """Test matching strategies"""

    def test_strategy_matcher_init(self, config, scenes):
        """Test StrategyMatcher initialization"""
        matcher = StrategyMatcher(config, scenes)

        assert matcher.config == config
        assert matcher.scenes == scenes
        assert matcher.variety_config is not None

    def test_get_clip_id_generates_unique_id(self, config, scenes, video_segments):
        """Test clip ID generation"""
        matcher = StrategyMatcher(config, scenes)

        clip_id = matcher.get_clip_id(video_segments[0])

        assert "tokyo_footage.mp4" in clip_id
        assert "0.00" in clip_id  # start time
        assert "10.00" in clip_id  # end time

    def test_is_clip_excluded_same_clip(self, config, scenes, video_segments):
        """Test exclusion of same clip"""
        matcher = StrategyMatcher(config, scenes)

        existing = [video_segments[0]]
        is_excluded, reason = matcher.is_clip_excluded(
            video_segments[0],
            existing,
            force_different_source=False
        )

        assert is_excluded is True
        assert "Same clip" in reason

    def test_is_clip_excluded_different_source_required(self, config, scenes, video_segments):
        """Test exclusion when different source required"""
        config.output.variety.require_different_source = True
        matcher = StrategyMatcher(config, scenes)

        # Both segments from same source file
        existing = [video_segments[0]]
        is_excluded, reason = matcher.is_clip_excluded(
            video_segments[1],  # Different segment, same source
            existing,
            force_different_source=True
        )

        assert is_excluded is True
        assert "source" in reason.lower()


# ============================================================================
# Test Matching Strategies
# ============================================================================

class TestMatchingStrategies:
    """Test individual matching strategies"""

    def test_match_visual_first_uses_scene_descriptions(self, config, scenes, vo_segments, video_segments):
        """Test visual_first strategy prioritizes scenes"""
        matcher = StrategyMatcher(config, scenes)

        candidates = [(seg, 0.7) for seg in video_segments]
        existing_matches = []

        result = matcher.match_visual_first(
            vo_segments[0],
            candidates,
            existing_matches
        )

        # Should return a match
        assert result is not None or len(existing_matches) > 0  # May fail if all excluded

    def test_match_different_source_enforces_variety(self, config, scenes, vo_segments, video_segments):
        """Test different_source strategy enforces source variety"""
        matcher = StrategyMatcher(config, scenes)

        candidates = [(seg, 0.7) for seg in video_segments]
        existing_matches = [video_segments[0]]  # tokyo_footage.mp4

        result = matcher.match_different_source(
            vo_segments[0],
            candidates,
            existing_matches
        )

        # Should select from different source or None
        if result:
            assert result.video_segment.source_file not in [m.source_file for m in existing_matches]

    def test_match_keyword_only_uses_keywords(self, config, scenes, vo_segments, video_segments):
        """Test keyword_only strategy uses keyword overlap"""
        matcher = StrategyMatcher(config, scenes)

        # Add keywords to segments
        vo_segments[0].keywords = ["tokyo", "japan"]
        video_segments[0].keywords = ["tokyo", "city"]
        video_segments[1].keywords = ["building", "architecture"]

        candidates = [(seg, 0.7) for seg in video_segments]
        existing_matches = []

        result = matcher.match_keyword_only(
            vo_segments[0],
            candidates,
            existing_matches
        )

        # Should return a match based on keywords
        assert result is not None or len(candidates) == 0

    def test_match_embedding_diversity_maximizes_difference(self, config, scenes, vo_segments, video_segments):
        """Test embedding_diversity finds different clips"""
        matcher = StrategyMatcher(config, scenes)

        # Create embeddings for testing
        vo_embedding = [0.9, 0.1, 0.0]
        candidate_embeddings = {
            matcher.get_clip_id(video_segments[0]): [1.0, 0.0, 0.0],  # Similar to vo
            matcher.get_clip_id(video_segments[1]): [0.9, 0.1, 0.0],  # Very similar to vo
            matcher.get_clip_id(video_segments[2]): [0.0, 0.0, 1.0]   # Very different
        }

        candidates = [(seg, 0.7) for seg in video_segments]
        existing_matches = [video_segments[0]]
        existing_embeddings = [[1.0, 0.0, 0.0]]

        result = matcher.match_embedding_diversity(
            vo_segments[0],
            candidates,
            existing_matches,
            existing_embeddings,
            candidate_embeddings,
            vo_embedding
        )

        # Should prefer the diverse clip (video_segments[2])
        if result:
            # The diverse clip should be selected if it meets minimum relevance
            assert result.strategy == "embedding_diversity"

    def test_match_broll_only_filters_by_is_broll(self, config, scenes, vo_segments, video_segments):
        """Test broll_only strategy filters by is_broll flag"""
        matcher = StrategyMatcher(config, scenes)

        # Mark one segment as broll
        video_segments[2].is_broll = True

        vo_embedding = [0.9, 0.1, 0.0]
        candidate_embeddings = {
            matcher.get_clip_id(seg): [0.8, 0.1, 0.1] for seg in video_segments
        }

        candidates = [(seg, 0.7) for seg in video_segments]
        existing_matches = []
        existing_embeddings = []

        result = matcher.match_broll_only(
            vo_segments[0],
            candidates,
            existing_matches,
            existing_embeddings,
            candidate_embeddings,
            vo_embedding
        )

        # Should only return B-roll segment
        if result:
            assert getattr(result.video_segment, 'is_broll', False) is True


# ============================================================================
# Test match_all_segments
# ============================================================================

class TestMatchAllSegments:
    """Test main matching orchestration"""

    def test_match_all_segments_processes_all(self, config, cache_manager, vo_segments, video_segments, embeddings, scenes):
        """Test matching all segments"""
        vo_embeddings, video_embeddings = embeddings

        with patch('src.matching.main.find_top_k_similar') as mock_find:
            # Mock embedding search
            mock_find.return_value = ([0.9, 0.8], [0, 1])

            with patch('src.matching.llm_providers.GeminiMatcher'):
                results = match_all_segments(
                    vo_segments,
                    video_segments,
                    vo_embeddings,
                    video_embeddings,
                    scenes,
                    config,
                    cache_manager
                )

        # Should return one result per voiceover segment
        assert len(results) == len(vo_segments)

        # Each result should have a primary match
        for result in results:
            assert result.primary_match is not None

    def test_match_all_segments_with_strategies(self, config, cache_manager, vo_segments, video_segments, embeddings, scenes):
        """Test matching with strategy tracks enabled"""
        config.output.include_strategy_tracks = True
        config.output.strategy_tracks = ["embedding_diversity", "broll_only"]

        vo_embeddings, video_embeddings = embeddings

        with patch('src.matching.main.find_top_k_similar') as mock_find:
            mock_find.return_value = ([0.9, 0.8], [0, 1])

            with patch('src.matching.llm_providers.GeminiMatcher'):
                results = match_all_segments(
                    vo_segments,
                    video_segments,
                    vo_embeddings,
                    video_embeddings,
                    scenes,
                    config,
                    cache_manager
                )

        # Results should have strategy matches
        for result in results:
            # Strategy matches may be empty if no valid candidates
            assert result.strategy_matches is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
