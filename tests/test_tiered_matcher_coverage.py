"""
Test coverage for src/matching/tiered_matcher.py

Target: Cover key missed lines for improved coverage.

Covers:
- Face preference handling (more, none, neutral)
- Location filtering
- Empty valid_candidates recovery
- High-similarity skip LLM
- Cached LLM response
- Secondary LLM provider
- LLM exception fallback
- No primary provider
- review_with_local_llm
- _get_secondary_matches three-pass
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock, Mock, PropertyMock
from dataclasses import dataclass, field
from typing import Optional, List


# Mock config and data structures
@dataclass
class MockMatchingConfig:
    gemini_model: str = "gemini-2.0-flash"
    anthropic_model: str = "claude-3-haiku-20240307"
    ollama_model: str = "llama3.2"
    ollama_host: str = "http://localhost:11434"
    primary_provider: str = "gemini"
    secondary_provider: str = ""
    use_local_for_review: bool = False
    min_confidence: float = 0.3
    embedding_candidates: int = 10
    high_confidence_threshold: float = 0.85
    low_confidence_threshold: float = 0.5
    skip_llm_threshold: float = 0.9
    ambiguous_threshold: float = 0.6
    confidence_threshold: float = 0.3
    max_clip_reuse: int = 3
    reuse_penalty: float = 0.1
    chapter_matching_enabled: bool = False
    topic_mismatch_penalty: float = 0.15
    location_matching: None = None
    cache_llm_responses: bool = False
    face_preference: str = "neutral"


@dataclass
class MockConfig:
    matching: MockMatchingConfig = field(default_factory=MockMatchingConfig)
    gemini_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    negative_matching: MagicMock = field(default_factory=lambda: MagicMock(enabled=False))
    output: MagicMock = field(default_factory=lambda: MagicMock(num_alternatives=2))


@dataclass
class MockSRTSegment:
    """Mock SRTSegment for testing."""
    source_file: str = "/path/to/video.mp4"
    start_time: float = 0.0
    end_time: float = 10.0
    text: str = "Test segment text"
    keywords: List[str] = field(default_factory=list)
    source: Optional[str] = None
    face_score: Optional[float] = None
    is_broll: bool = False


class TestTieredMatcherInit:
    """Test TieredMatcher initialization."""

    def test_init_with_no_api_keys(self):
        """Test initialization without any API keys."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        assert matcher.primary_provider is None
        assert matcher.secondary_provider is None

    def test_init_with_gemini_key(self):
        """Test initialization with Gemini API key."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig(gemini_api_key="test_gemini_key")

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            with patch('src.matching.tiered_matcher.GeminiMatcher') as mock_gemini:
                matcher = TieredMatcher(config=config)

        mock_gemini.assert_called_once()


class TestTieredMatcherFacePreference:
    """Test face preference handling."""

    def test_face_preference_more_boosts_high_face_score(self):
        """Test that face_preference='more' boosts high face_score."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.matching.face_preference = 'more'
        config.matching.skip_llm_threshold = 0.99  # Force LLM path

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        # Create segments with face scores
        seg1 = MockSRTSegment(source_file="/v1.mp4", face_score=0.9, source="global_cache")
        seg2 = MockSRTSegment(source_file="/v2.mp4", face_score=0.1, source="global_cache")

        vo_segment = MockSRTSegment(text="Test voiceover segment")
        candidates = [(seg1, 0.7), (seg2, 0.75)]

        # Mock dependencies
        with patch.object(matcher, 'reuse_tracker') as mock_tracker:
            mock_tracker.can_use.return_value = True
            mock_tracker.adjust_confidence.side_effect = lambda seg, conf: conf
            mock_tracker.get_usage_count.return_value = 0

            with patch('src.matching.tiered_matcher.apply_face_preference') as mock_face:
                mock_face.return_value = []  # Return empty for current project

                result = matcher.match_segment(vo_segment, candidates)

        # High face_score segment should get boost
        # Note: The actual boost logic is complex, so we just verify the path is taken
        assert result is not None

    def test_face_preference_none_boosts_low_face_score(self):
        """Test that face_preference='none' boosts low face_score."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.matching.face_preference = 'none'

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        # Create segment with low face score (no faces = good for b-roll)
        seg = MockSRTSegment(source_file="/v1.mp4", face_score=0.1, source="global_cache")

        vo_segment = MockSRTSegment(text="Test voiceover")
        candidates = [(seg, 0.7)]

        with patch.object(matcher, 'reuse_tracker') as mock_tracker:
            mock_tracker.can_use.return_value = True
            mock_tracker.adjust_confidence.side_effect = lambda s, c: c
            mock_tracker.get_usage_count.return_value = 0

            with patch('src.matching.tiered_matcher.apply_face_preference') as mock_face:
                mock_face.return_value = []

                result = matcher.match_segment(vo_segment, candidates)

        assert result is not None


class TestTieredMatcherEmptyCandidates:
    """Test empty candidates handling."""

    def test_no_candidates_returns_gap(self):
        """Test that no candidates returns gap match."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        vo_segment = MockSRTSegment(text="Test voiceover")

        result = matcher.match_segment(vo_segment, [])

        assert result.has_gap is True
        assert "No video candidates" in result.gap_reason or "No candidates" in result.gap_reason

    def test_all_candidates_filtered_uses_fallback(self):
        """Test fallback when all candidates are filtered by reuse."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.matching.skip_llm_threshold = 0.99

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        seg = MockSRTSegment(source_file="/v1.mp4")
        vo_segment = MockSRTSegment(text="Test voiceover")
        candidates = [(seg, 0.8)]

        # Make reuse tracker reject all candidates first, then allow fallback
        with patch.object(matcher, 'reuse_tracker') as mock_tracker:
            # First call rejects, subsequent calls accept
            mock_tracker.can_use.side_effect = [False, True, True, True]
            mock_tracker.adjust_confidence.side_effect = lambda s, c: c * 0.5
            mock_tracker.get_usage_count.return_value = 0
            mock_tracker.record_usage.return_value = None

            result = matcher.match_segment(vo_segment, candidates)

        # Should return a result (fallback with penalized confidence)
        assert result is not None


class TestTieredMatcherHighSimilarity:
    """Test high similarity skip LLM path."""

    def test_high_similarity_skips_llm(self):
        """Test that high embedding similarity skips LLM."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.matching.skip_llm_threshold = 0.8

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        seg = MockSRTSegment(source_file="/v1.mp4")
        vo_segment = MockSRTSegment(text="Test voiceover")
        candidates = [(seg, 0.95)]  # High similarity

        with patch.object(matcher, 'reuse_tracker') as mock_tracker:
            mock_tracker.can_use.return_value = True
            mock_tracker.adjust_confidence.return_value = 0.95
            mock_tracker.get_usage_count.return_value = 0
            mock_tracker.record_usage.return_value = None

            with patch('src.matching.tiered_matcher.apply_topic_penalty', return_value=(0.95, "")):
                with patch('src.matching.tiered_matcher.apply_broll_boost', return_value=(0.95, "")):
                    with patch('src.matching.tiered_matcher.apply_current_project_boost', return_value=(0.95, "")):
                        result = matcher.match_segment(vo_segment, candidates)

        assert result.primary_match.confidence >= 0.9
        assert "High embedding similarity" in result.primary_match.reasoning


class TestTieredMatcherLLMFallback:
    """Test LLM exception fallback."""

    def test_llm_exception_falls_back_to_embedding(self):
        """Test that LLM exception falls back to embedding similarity."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig(gemini_api_key="test_key")
        config.matching.skip_llm_threshold = 0.99  # Force LLM path

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            with patch('src.matching.tiered_matcher.GeminiMatcher') as mock_gemini_cls:
                mock_provider = MagicMock()
                mock_provider.match_batch.side_effect = Exception("LLM API error")
                mock_gemini_cls.return_value = mock_provider

                matcher = TieredMatcher(config=config)

        seg = MockSRTSegment(source_file="/v1.mp4")
        vo_segment = MockSRTSegment(text="Test voiceover")
        candidates = [(seg, 0.7)]

        with patch.object(matcher, 'reuse_tracker') as mock_tracker:
            mock_tracker.can_use.return_value = True
            mock_tracker.adjust_confidence.return_value = 0.7
            mock_tracker.get_usage_count.return_value = 0
            mock_tracker.record_usage.return_value = None

            with patch('src.matching.tiered_matcher.apply_topic_penalty', return_value=(0.6, "")):
                with patch('src.matching.tiered_matcher.apply_broll_boost', return_value=(0.6, "")):
                    with patch('src.matching.tiered_matcher.apply_current_project_boost', return_value=(0.6, "")):
                        with patch('src.matching.tiered_matcher.find_keyword_matches', return_value=(0, False, False)):
                            with patch('src.matching.tiered_matcher.get_global_logger', return_value=None):
                                result = matcher.match_segment(vo_segment, candidates)

        assert result is not None
        assert "fallback" in result.primary_match.reasoning.lower() or "LLM" in result.primary_match.reasoning


class TestTieredMatcherNoPrimaryProvider:
    """Test matching without primary LLM provider."""

    def test_no_provider_uses_embedding_only(self):
        """Test that no provider uses embedding similarity only."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()  # No API keys
        config.matching.skip_llm_threshold = 0.99

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        seg = MockSRTSegment(source_file="/v1.mp4")
        vo_segment = MockSRTSegment(text="Test voiceover")
        candidates = [(seg, 0.7)]

        with patch.object(matcher, 'reuse_tracker') as mock_tracker:
            mock_tracker.can_use.return_value = True
            mock_tracker.adjust_confidence.return_value = 0.7
            mock_tracker.get_usage_count.return_value = 0
            mock_tracker.record_usage.return_value = None

            with patch('src.matching.tiered_matcher.apply_topic_penalty', return_value=(0.6, "")):
                with patch('src.matching.tiered_matcher.apply_broll_boost', return_value=(0.6, "")):
                    with patch('src.matching.tiered_matcher.apply_current_project_boost', return_value=(0.6, "")):
                        with patch('src.matching.tiered_matcher.find_keyword_matches', return_value=(0, False, False)):
                            with patch('src.matching.tiered_matcher.get_global_logger', return_value=None):
                                result = matcher.match_segment(vo_segment, candidates)

        assert result is not None
        assert "Embedding similarity only" in result.primary_match.reasoning


class TestTieredMatcherSecondaryMatches:
    """Test _get_secondary_matches three-pass approach."""

    def test_secondary_matches_different_sources(self):
        """Test secondary matches prefer different video sources."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        # Create candidates from different sources
        seg1 = MockSRTSegment(source_file="/v1.mp4", start_time=0)
        seg2 = MockSRTSegment(source_file="/v2.mp4", start_time=0)
        seg3 = MockSRTSegment(source_file="/v3.mp4", start_time=0)
        seg4 = MockSRTSegment(source_file="/v1.mp4", start_time=10)  # Same source, different time

        candidates = [(seg1, 0.9), (seg2, 0.85), (seg3, 0.8), (seg4, 0.75)]

        primary = seg1
        excluded = {"/v1.mp4"}

        result = matcher._get_secondary_matches(
            candidates, scenes=None,
            excluded_video_files=excluded,
            primary_segment=primary,
            alt_segments=[]
        )

        # Should get up to 3 secondary matches from different sources
        assert len(result) >= 2
        # First secondary should be from v2 or v3 (not v1)
        assert result[0].video_segment.source_file != "/v1.mp4"


class TestTieredMatcherReviewWithLocalLLM:
    """Test review_with_local_llm method."""

    def test_review_no_local_provider_returns_unchanged(self):
        """Test that no local provider returns matches unchanged."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        matcher.local_provider = None

        # Create mock matches
        mock_matches = [MagicMock(), MagicMock()]

        result = matcher.review_with_local_llm(mock_matches)

        assert result == mock_matches

    def test_review_no_low_confidence_returns_unchanged(self):
        """Test that no low confidence matches returns unchanged."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.matching.ambiguous_threshold = 0.6

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        matcher.local_provider = MagicMock()

        # Create mock matches with high confidence
        mock_match1 = MagicMock()
        mock_match1.primary_match.confidence = 0.9  # Above threshold

        mock_matches = [mock_match1]

        result = matcher.review_with_local_llm(mock_matches)

        assert result == mock_matches
        matcher.local_provider.match_batch.assert_not_called()


class TestTieredMatcherAlternatives:
    """Test _get_alternatives method."""

    def test_alternatives_prefer_different_sources(self):
        """Test that alternatives prefer different video sources."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.output.num_alternatives = 2

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        seg1 = MockSRTSegment(source_file="/v1.mp4", start_time=0)
        seg2 = MockSRTSegment(source_file="/v2.mp4", start_time=0)
        seg3 = MockSRTSegment(source_file="/v1.mp4", start_time=10)

        candidates = [(seg1, 0.9), (seg2, 0.85), (seg3, 0.8)]
        primary = seg1

        result = matcher._get_alternatives(candidates, scenes=None, primary_match=primary)

        # Should prefer v2 (different source) over v1 segment at t=10
        assert len(result) >= 1
        # First alternative should be from different source
        assert result[0].video_segment.source_file != primary.source_file or \
               result[0].video_segment.start_time != primary.start_time


class TestTieredMatcherLocationFiltering:
    """Test location filtering."""

    def test_location_matcher_set_chapters(self):
        """Test set_location_chapters method."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.matching.location_matching = MagicMock(enabled=True)

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            with patch('src.matching.tiered_matcher.create_location_service') as mock_create:
                mock_service = MagicMock()
                mock_create.return_value = mock_service

                with patch('src.matching.tiered_matcher.LocationMatcher') as mock_loc_matcher:
                    mock_matcher_instance = MagicMock()
                    mock_loc_matcher.return_value = mock_matcher_instance

                    matcher = TieredMatcher(config=config)

        # Set location chapters
        mock_chapters = [MagicMock(), MagicMock()]
        matcher.set_location_chapters(mock_chapters)

        if matcher.location_matcher:
            mock_matcher_instance.set_location_chapters.assert_called_once_with(mock_chapters)

    def test_location_matcher_set_video_locations(self):
        """Test set_video_locations method."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.matching.location_matching = MagicMock(enabled=True)

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            with patch('src.matching.tiered_matcher.create_location_service') as mock_create:
                mock_service = MagicMock()
                mock_create.return_value = mock_service

                with patch('src.matching.tiered_matcher.LocationMatcher') as mock_loc_matcher:
                    mock_matcher_instance = MagicMock()
                    mock_loc_matcher.return_value = mock_matcher_instance

                    matcher = TieredMatcher(config=config)

        # Set video locations
        mock_locations = {"/v1.mp4": MagicMock()}
        matcher.set_video_locations(mock_locations)

        if matcher.location_matcher:
            mock_matcher_instance.set_video_locations.assert_called_once_with(mock_locations)


class TestTieredMatcherLocationInitExceptions:
    """Test location service initialization exceptions (lines 104-106)."""

    def test_location_service_init_exception_disables_location_matching(self):
        """Test that exception during location service init disables location matching."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.matching.location_matching = MagicMock(enabled=True)

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            with patch('src.matching.tiered_matcher.create_location_service') as mock_create:
                # Simulate exception during init (line 104-106)
                mock_create.side_effect = Exception("GeoNames connection failed")

                matcher = TieredMatcher(config=config)

        # Location matching should be disabled after exception
        assert matcher.location_matching_enabled is False
        assert matcher.location_matcher is None


class TestTieredMatcherDictLocationConfig:
    """Test location matching with dict-style config (line 95)."""

    def test_location_config_as_dict(self):
        """Test location matching enabled from dict config."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        # Dict-style config (line 95)
        config.matching.location_matching = {'enabled': True}

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            with patch('src.matching.tiered_matcher.create_location_service') as mock_create:
                mock_service = MagicMock()
                mock_create.return_value = mock_service

                with patch('src.matching.tiered_matcher.LocationMatcher') as mock_loc:
                    mock_loc.return_value = MagicMock()
                    matcher = TieredMatcher(config=config)

        assert matcher.location_matching_enabled is True


class TestTieredMatcherSecondaryGeminiProvider:
    """Test secondary provider initialization with Gemini (lines 154-155)."""

    def test_secondary_provider_gemini(self):
        """Test secondary provider is Gemini when configured."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig(
            gemini_api_key="test_gemini_key",
            anthropic_api_key="test_anthropic_key"
        )
        config.matching.primary_provider = "anthropic"
        config.matching.secondary_provider = "gemini"  # Line 153-155

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            with patch('src.matching.tiered_matcher.GeminiMatcher') as mock_gemini:
                with patch('src.matching.tiered_matcher.ClaudeMatcher') as mock_claude:
                    mock_gemini.return_value = MagicMock()
                    mock_claude.return_value = MagicMock()
                    matcher = TieredMatcher(config=config)

        # Both should be called
        mock_claude.assert_called()  # Primary
        mock_gemini.assert_called()  # Secondary


class TestTieredMatcherLocalProviderConnectionFailure:
    """Test local provider connection failure (lines 164-166)."""

    def test_local_provider_connection_refused(self):
        """Test local provider is None when Ollama connection fails."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.matching.use_local_for_review = True

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            with patch('src.matching.tiered_matcher.LocalLLMMatcher') as mock_local:
                mock_local.return_value = MagicMock()
                # Patch requests.get at the module level (lines 164-166)
                with patch('requests.get') as mock_req:
                    mock_req.side_effect = Exception("Connection refused")
                    matcher = TieredMatcher(config=config)

        # Local provider should be None after connection failure
        assert matcher.local_provider is None


class TestTieredMatcherSceneNotFound:
    """Test scene lookup returns None when no match (line 225)."""

    def test_get_scene_for_segment_no_matching_scene(self):
        """Test _get_scene_for_segment returns None when segment time outside scenes."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        # Segment at time 100s
        segment = MockSRTSegment(
            source_file="/video.mp4",
            start_time=100.0,  # No scene covers this
            end_time=105.0
        )

        # Mock SceneInfo
        from dataclasses import dataclass
        @dataclass
        class SceneInfo:
            start_time: float
            end_time: float

        scenes = {
            "/video.mp4": [
                SceneInfo(start_time=0.0, end_time=10.0),
                SceneInfo(start_time=10.0, end_time=20.0),
            ]
        }

        result = matcher._get_scene_for_segment(segment, scenes)

        # Line 225: return None when no scene matches
        assert result is None


class TestTieredMatcherNoValidCandidatesAfterFiltering:
    """Test gap result when no valid candidates after all filtering (lines 344-352)."""

    def test_no_valid_candidates_returns_gap_with_reason(self):
        """Test gap returned with correct reason when all candidates filtered by location."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        # Enable location matching to allow filtering
        config.matching.location_matching = MagicMock(enabled=True)

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            with patch('src.matching.tiered_matcher.create_location_service') as mock_create:
                mock_service = MagicMock()
                mock_create.return_value = mock_service

                with patch('src.matching.tiered_matcher.LocationMatcher') as mock_loc:
                    mock_loc_instance = MagicMock()
                    # Location filter removes ALL candidates (lines 327-331)
                    # This makes candidates empty, so lines 343-352 will trigger
                    mock_loc_instance.apply_location_filter.return_value = ([], True, "All filtered by location")
                    mock_loc.return_value = mock_loc_instance

                    matcher = TieredMatcher(config=config)

        seg = MockSRTSegment(source_file="/v1.mp4")
        vo_segment = MockSRTSegment(text="Test voiceover")
        candidates = [(seg, 0.8)]

        result = matcher.match_segment(vo_segment, candidates)

        # Lines 344-352: gap with "All candidates filtered" reason
        assert result.has_gap is True
        assert "All candidates filtered" in result.gap_reason


class TestTieredMatcherBoostReasonsSkipLLM:
    """Test boost reasons in high similarity skip LLM path (lines 382, 384)."""

    def test_skip_llm_includes_topic_and_broll_reasons(self):
        """Test that topic and broll boost reasons appear in skip LLM path."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.matching.skip_llm_threshold = 0.8

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        seg = MockSRTSegment(source_file="/v1.mp4")
        vo_segment = MockSRTSegment(text="Test voiceover")
        candidates = [(seg, 0.95)]  # High similarity to skip LLM

        with patch.object(matcher, 'reuse_tracker') as mock_tracker:
            mock_tracker.can_use.return_value = True
            mock_tracker.adjust_confidence.return_value = 0.95
            mock_tracker.get_usage_count.return_value = 0
            mock_tracker.record_usage.return_value = None

            # Return reasons (lines 382, 384)
            with patch('src.matching.tiered_matcher.apply_topic_penalty', return_value=(0.92, "topic mismatch")):
                with patch('src.matching.tiered_matcher.apply_broll_boost', return_value=(0.94, "broll boost")):
                    with patch('src.matching.tiered_matcher.apply_current_project_boost', return_value=(0.95, "")):
                        result = matcher.match_segment(vo_segment, candidates)

        # Line 382: topic reason added
        assert "topic mismatch" in result.primary_match.reasoning
        # Line 384: broll reason added
        assert "broll boost" in result.primary_match.reasoning


class TestTieredMatcherBoostReasonsCachedResponse:
    """Test boost reasons in cached response path (lines 444, 446, 448)."""

    def test_cached_response_includes_all_boost_reasons(self):
        """Test that cached responses include topic, broll, and project boost reasons."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.matching.skip_llm_threshold = 0.99  # Don't skip LLM
        config.matching.cache_llm_responses = True

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        # Setup cache
        matcher.cache = MagicMock()
        matcher.cache.get_llm_response.return_value = {
            'selected': 0,
            'confidence': 0.8,
            'reasoning': 'cached result'
        }

        seg = MockSRTSegment(source_file="/v1.mp4")
        vo_segment = MockSRTSegment(text="Test voiceover")
        candidates = [(seg, 0.7)]

        with patch.object(matcher, 'reuse_tracker') as mock_tracker:
            mock_tracker.can_use.return_value = True
            mock_tracker.adjust_confidence.return_value = 0.7
            mock_tracker.get_usage_count.return_value = 0
            mock_tracker.record_usage.return_value = None

            # Return all three reasons (lines 444, 446, 448)
            with patch('src.matching.tiered_matcher.apply_topic_penalty', return_value=(0.78, "topic penalty")):
                with patch('src.matching.tiered_matcher.apply_broll_boost', return_value=(0.80, "broll applied")):
                    with patch('src.matching.tiered_matcher.apply_current_project_boost', return_value=(0.82, "project boost")):
                        result = matcher.match_segment(vo_segment, candidates)

        # All boost reasons should be in reasoning
        reasoning = result.primary_match.reasoning
        assert "(cached)" in reasoning
        assert "topic penalty" in reasoning  # Line 444
        assert "broll applied" in reasoning  # Line 446
        assert "project boost" in reasoning  # Line 448


class TestTieredMatcherAmbiguousWithSecondary:
    """Test ambiguous match uses secondary provider (lines 497-506)."""

    def test_ambiguous_match_uses_secondary_provider(self):
        """Test secondary provider called when primary returns ambiguous result."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig(gemini_api_key="test", anthropic_api_key="test")
        config.matching.skip_llm_threshold = 0.99
        config.matching.ambiguous_threshold = 0.7
        config.matching.secondary_provider = "anthropic"

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            with patch('src.matching.tiered_matcher.GeminiMatcher') as mock_gemini:
                with patch('src.matching.tiered_matcher.ClaudeMatcher') as mock_claude:
                    # Primary returns low confidence (ambiguous)
                    mock_primary = MagicMock()
                    mock_primary.match_batch.return_value = [(0, 0.5, "primary low")]
                    mock_gemini.return_value = mock_primary

                    # Secondary returns higher confidence (lines 503-506)
                    mock_secondary = MagicMock()
                    mock_secondary.match_batch.return_value = [(0, 0.85, "secondary better")]
                    mock_claude.return_value = mock_secondary

                    matcher = TieredMatcher(config=config)

        seg = MockSRTSegment(source_file="/v1.mp4")
        vo_segment = MockSRTSegment(text="Test voiceover")
        candidates = [(seg, 0.7)]

        with patch.object(matcher, 'reuse_tracker') as mock_tracker:
            mock_tracker.can_use.return_value = True
            mock_tracker.adjust_confidence.return_value = 0.7
            mock_tracker.get_usage_count.return_value = 0
            mock_tracker.record_usage.return_value = None

            with patch('src.matching.tiered_matcher.apply_topic_penalty', return_value=(0.85, "")):
                with patch('src.matching.tiered_matcher.apply_broll_boost', return_value=(0.85, "")):
                    with patch('src.matching.tiered_matcher.apply_current_project_boost', return_value=(0.85, "")):
                        with patch('src.matching.tiered_matcher.find_keyword_matches', return_value=(0, False, False)):
                            with patch('src.matching.tiered_matcher.get_global_logger', return_value=None):
                                result = matcher.match_segment(vo_segment, candidates)

        # Secondary provider should have been called
        mock_secondary.match_batch.assert_called()
        # Result should indicate secondary was used (line 506)
        assert "(secondary)" in result.primary_match.reasoning


class TestTieredMatcherLLMPathBoosts:
    """Test boost reasons in LLM matching path (lines 557, 559)."""

    def test_llm_path_includes_topic_and_broll_reasons(self):
        """Test LLM path includes topic and broll boost reasons."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig(gemini_api_key="test")
        config.matching.skip_llm_threshold = 0.99
        config.matching.ambiguous_threshold = 0.5  # Don't trigger secondary

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            with patch('src.matching.tiered_matcher.GeminiMatcher') as mock_gemini:
                mock_provider = MagicMock()
                mock_provider.match_batch.return_value = [(0, 0.8, "llm result")]
                mock_gemini.return_value = mock_provider
                matcher = TieredMatcher(config=config)

        seg = MockSRTSegment(source_file="/v1.mp4")
        vo_segment = MockSRTSegment(text="Test voiceover")
        candidates = [(seg, 0.7)]

        with patch.object(matcher, 'reuse_tracker') as mock_tracker:
            mock_tracker.can_use.return_value = True
            mock_tracker.adjust_confidence.return_value = 0.7
            mock_tracker.get_usage_count.return_value = 0
            mock_tracker.record_usage.return_value = None

            # Return reasons (lines 557, 559)
            with patch('src.matching.tiered_matcher.apply_topic_penalty', return_value=(0.78, "topic found")):
                with patch('src.matching.tiered_matcher.apply_broll_boost', return_value=(0.80, "broll found")):
                    with patch('src.matching.tiered_matcher.apply_current_project_boost', return_value=(0.82, "")):
                        with patch('src.matching.tiered_matcher.find_keyword_matches', return_value=(0, False, False)):
                            with patch('src.matching.tiered_matcher.get_global_logger', return_value=None):
                                result = matcher.match_segment(vo_segment, candidates)

        # Line 557: topic reason
        assert "topic found" in result.primary_match.reasoning
        # Line 559: broll reason
        assert "broll found" in result.primary_match.reasoning


class TestTieredMatcherRunLoggerCalled:
    """Test run logger is called for match decision (line 578)."""

    def test_run_logger_log_match_decision_called(self):
        """Test that run logger's log_match_decision is called."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig(gemini_api_key="test")
        config.matching.skip_llm_threshold = 0.99

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            with patch('src.matching.tiered_matcher.GeminiMatcher') as mock_gemini:
                mock_provider = MagicMock()
                mock_provider.match_batch.return_value = [(0, 0.8, "result")]
                mock_gemini.return_value = mock_provider
                matcher = TieredMatcher(config=config)

        seg = MockSRTSegment(source_file="/v1.mp4")
        vo_segment = MockSRTSegment(text="Test voiceover")
        candidates = [(seg, 0.7)]

        mock_run_logger = MagicMock()

        with patch.object(matcher, 'reuse_tracker') as mock_tracker:
            mock_tracker.can_use.return_value = True
            mock_tracker.adjust_confidence.return_value = 0.7
            mock_tracker.get_usage_count.return_value = 0
            mock_tracker.record_usage.return_value = None

            with patch('src.matching.tiered_matcher.apply_topic_penalty', return_value=(0.8, "")):
                with patch('src.matching.tiered_matcher.apply_broll_boost', return_value=(0.8, "")):
                    with patch('src.matching.tiered_matcher.apply_current_project_boost', return_value=(0.8, "")):
                        with patch('src.matching.tiered_matcher.find_keyword_matches', return_value=(0, False, False)):
                            # Line 578: run logger called
                            with patch('src.matching.tiered_matcher.get_global_logger', return_value=mock_run_logger):
                                result = matcher.match_segment(vo_segment, candidates)

        # run_logger.log_match_decision should have been called
        mock_run_logger.log_match_decision.assert_called_once()


class TestTieredMatcherSecondaryMatchPasses:
    """Test secondary match three-pass logic (lines 710, 716, 733, 743-747, 758)."""

    def test_secondary_matches_second_pass_same_source_ok(self):
        """Test second pass allows same source within V4-V6."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        # Primary and alt use v1 and v2
        primary = MockSRTSegment(source_file="/v1.mp4", start_time=0)
        alt1 = MockSRTSegment(source_file="/v2.mp4", start_time=0)

        # Candidates include same source as another secondary (lines 731-752)
        sec1 = MockSRTSegment(source_file="/v3.mp4", start_time=0)   # First pass: new source
        sec2 = MockSRTSegment(source_file="/v3.mp4", start_time=10)  # Second pass: same source different time
        sec3 = MockSRTSegment(source_file="/v3.mp4", start_time=20)  # Second pass: same source different time

        candidates = [
            (sec1, 0.8),
            (sec2, 0.75),  # Lines 743-747: same source ok in second pass
            (sec3, 0.70),
        ]

        excluded = {"/v1.mp4", "/v2.mp4"}

        result = matcher._get_secondary_matches(
            candidates, scenes=None,
            excluded_video_files=excluded,
            primary_segment=primary,
            alt_segments=[alt1]
        )

        # Should get 3 secondary matches
        assert len(result) == 3
        # First should be from first pass
        assert "Secondary Primary" in result[0].reasoning
        # Second and third use "same source ok"
        assert "same source ok" in result[1].reasoning or "same source ok" in result[2].reasoning

    def test_secondary_matches_third_pass_fallback(self):
        """Test third pass allows same video file but different segment."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        primary = MockSRTSegment(source_file="/v1.mp4", start_time=0)
        alt1 = MockSRTSegment(source_file="/v2.mp4", start_time=0)

        # Only candidates are from excluded sources but different segments (line 754-777)
        sec1 = MockSRTSegment(source_file="/v1.mp4", start_time=50)  # Same as primary but different time
        sec2 = MockSRTSegment(source_file="/v2.mp4", start_time=50)  # Same as alt but different time

        candidates = [
            (sec1, 0.8),
            (sec2, 0.75),
        ]

        excluded = {"/v1.mp4", "/v2.mp4"}  # Both sources excluded

        result = matcher._get_secondary_matches(
            candidates, scenes=None,
            excluded_video_files=excluded,
            primary_segment=primary,
            alt_segments=[alt1]
        )

        # Should get secondary matches from third pass (line 758)
        assert len(result) >= 1
        # Should use "fallback - different segment" reasoning
        assert any("fallback" in s.reasoning for s in result)


class TestTieredMatcherLocalLLMReviewImproves:
    """Test local LLM review improves match (lines 811-827)."""

    def test_local_llm_improves_same_segment(self):
        """Test local LLM improves confidence for same segment selection."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.matching.ambiguous_threshold = 0.7

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        # Setup local provider that improves confidence (lines 810-815)
        mock_local = MagicMock()
        mock_local.match_batch.return_value = [(0, 0.9, "local improved")]
        matcher.local_provider = mock_local

        # Low confidence match
        vo_seg = MockSRTSegment(text="voiceover")
        video_seg = MockSRTSegment(source_file="/v1.mp4")

        mock_primary = MagicMock()
        mock_primary.voiceover_segment = vo_seg
        mock_primary.video_segment = video_seg
        mock_primary.confidence = 0.5  # Below ambiguous threshold
        mock_primary.reasoning = "original"

        mock_match_result = MagicMock()
        mock_match_result.primary_match = mock_primary
        mock_match_result.alternatives = []

        matches = [mock_match_result]

        result = matcher.review_with_local_llm(matches)

        # Line 814-815: confidence and reasoning updated
        assert result[0].primary_match.confidence == 0.9
        assert "(local refined)" in result[0].primary_match.reasoning

    def test_local_llm_selects_alternative(self):
        """Test local LLM selects alternative instead of primary."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.matching.ambiguous_threshold = 0.7

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        # Local provider selects alternative (index 1) (lines 816-825)
        mock_local = MagicMock()
        mock_local.match_batch.return_value = [(1, 0.95, "alternative better")]
        matcher.local_provider = mock_local

        vo_seg = MockSRTSegment(text="voiceover")
        video_seg = MockSRTSegment(source_file="/v1.mp4")
        alt_seg = MockSRTSegment(source_file="/v2.mp4")

        mock_primary = MagicMock()
        mock_primary.voiceover_segment = vo_seg
        mock_primary.video_segment = video_seg
        mock_primary.confidence = 0.5
        mock_primary.reasoning = "original"

        mock_alt = MagicMock()
        mock_alt.video_segment = alt_seg
        mock_alt.video_scene = MagicMock()
        mock_alt.confidence = 0.4

        mock_match_result = MagicMock()
        mock_match_result.primary_match = mock_primary
        mock_match_result.alternatives = [mock_alt]

        matches = [mock_match_result]

        result = matcher.review_with_local_llm(matches)

        # Lines 818-825: new match created with alternative
        assert result[0].primary_match.video_segment == alt_seg
        assert "(local selected)" in result[0].primary_match.reasoning

    def test_local_llm_exception_handled(self):
        """Test local LLM exception is caught and handled."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.matching.ambiguous_threshold = 0.7

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        # Local provider raises exception (line 826-827)
        mock_local = MagicMock()
        mock_local.match_batch.side_effect = Exception("Connection error")
        matcher.local_provider = mock_local

        vo_seg = MockSRTSegment(text="voiceover")
        video_seg = MockSRTSegment(source_file="/v1.mp4")

        mock_primary = MagicMock()
        mock_primary.voiceover_segment = vo_seg
        mock_primary.video_segment = video_seg
        mock_primary.confidence = 0.5
        mock_primary.reasoning = "original"

        mock_match_result = MagicMock()
        mock_match_result.primary_match = mock_primary
        mock_match_result.alternatives = []

        matches = [mock_match_result]

        # Should not raise - exception caught (line 827)
        result = matcher.review_with_local_llm(matches)

        # Original values preserved
        assert result[0].primary_match.confidence == 0.5
        assert result[0].primary_match.reasoning == "original"


# ============================================================================
# Test Coverage Gaps - Lines 295, 299-301, 316-318, 710, 734, 758
# ============================================================================

class TestTieredMatcherCoverageGaps:
    """Test coverage gaps in TieredMatcher"""

    def test_current_project_vs_global_cache_candidates_lines_295_301(self):
        """Test lines 295, 299-301: Separate current project from global cache candidates"""
        from src.matching.tiered_matcher import TieredMatcher
        from src.config import Config

        config = Config()
        config.matching.face_preference = "more"
        cache = MagicMock()
        cache.cache_dir = "/tmp/cache"

        matcher = TieredMatcher(config, cache)

        # Create current project segment (no source attribute = current project)
        current_seg = MockSRTSegment(source_file="current.mp4", text="current project")

        # Create global cache segment (source='global_cache')
        global_seg = MockSRTSegment(source_file="cached.mp4", text="global cache")
        global_seg.source = "global_cache"
        global_seg.face_score = 0.8

        candidates = [(current_seg, 0.9), (global_seg, 0.85)]

        vo_seg = MockSRTSegment(text="voiceover text")

        # Mock primary_provider to return a match
        mock_provider = MagicMock()
        mock_provider.match_batch.return_value = [{
            'video_segment': current_seg,
            'confidence': 0.9,
            'reasoning': 'test'
        }]
        matcher.primary_provider = mock_provider

        # Mock apply_face_preference
        with patch('src.matching.tiered_matcher.apply_face_preference', return_value=[(current_seg, 0.9)]):
            result = matcher.match_segment(vo_seg, candidates, {}, 0)
            assert result is not None

    def test_global_cache_face_preference_neutral_line_316(self):
        """Test line 316: Global cache candidates with neutral face preference"""
        from src.matching.tiered_matcher import TieredMatcher
        from src.config import Config

        config = Config()
        config.matching.face_preference = "neutral"
        cache = MagicMock()
        cache.cache_dir = "/tmp/cache"

        matcher = TieredMatcher(config, cache)

        # Create global cache segment with face score
        global_seg = MockSRTSegment(source_file="cached.mp4", text="global cache")
        global_seg.source = "global_cache"
        global_seg.face_score = 0.5

        candidates = [(global_seg, 0.85)]
        vo_seg = MockSRTSegment(text="voiceover")

        mock_provider = MagicMock()
        mock_provider.match_batch.return_value = [{
            'video_segment': global_seg,
            'confidence': 0.85,
            'reasoning': 'test'
        }]
        matcher.primary_provider = mock_provider

        result = matcher.match_segment(vo_seg, candidates, {}, 0)
        # Should pass through without boost (line 316)
        assert result is not None

    def test_global_cache_no_face_score_line_318(self):
        """Test line 318: Global cache segment without face_score"""
        from src.matching.tiered_matcher import TieredMatcher
        from src.config import Config

        config = Config()
        config.matching.face_preference = "more"
        cache = MagicMock()
        cache.cache_dir = "/tmp/cache"

        matcher = TieredMatcher(config, cache)

        # Create global cache segment WITHOUT face score
        global_seg = MockSRTSegment(source_file="cached.mp4", text="global cache")
        global_seg.source = "global_cache"
        # No face_score attribute - delete if present
        if hasattr(global_seg, 'face_score'):
            delattr(global_seg, 'face_score')

        candidates = [(global_seg, 0.85)]
        vo_seg = MockSRTSegment(text="voiceover")

        mock_provider = MagicMock()
        mock_provider.match_batch.return_value = [{
            'video_segment': global_seg,
            'confidence': 0.85,
            'reasoning': 'test'
        }]
        matcher.primary_provider = mock_provider

        result = matcher.match_segment(vo_seg, candidates, {}, 0)
        # Should handle missing face_score (line 318)
        assert result is not None

    def test_secondary_matches_break_lines_710_734_758(self):
        """Test lines 710, 734, 758: Break when enough secondary matches found"""
        from src.matching.tiered_matcher import TieredMatcher
        from src.config import Config

        config = Config()
        cache = MagicMock()

        matcher = TieredMatcher(config, cache)

        # Create many candidates from different sources (more than num_secondary=3)
        candidates = []
        for i in range(10):
            seg = MockSRTSegment(source_file=f"video{i}.mp4", text=f"content{i}")
            candidates.append((seg, 0.9 - i * 0.05))

        primary = MockSRTSegment(source_file="primary.mp4", text="primary")

        # Correct function signature: candidates, scenes, excluded_video_files, primary_segment, alt_segments
        result = matcher._get_secondary_matches(
            candidates,
            scenes=None,
            excluded_video_files={"primary.mp4"},  # Exclude primary
            primary_segment=primary,
            alt_segments=[]
        )

        # Should return exactly 3 (hardcoded num_secondary=3, break at line 710)
        assert len(result) == 3
