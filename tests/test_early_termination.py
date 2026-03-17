"""
Tests for early termination on obvious high-confidence matches.

US-008: Add early termination for obvious high-confidence matches
- Skip LLM when embedding sim >0.9 AND matched_keywords >=3 AND same entity found
- Return boosted confidence (min 0.92) with reason 'obvious_match_early_termination'
- Log when early termination is triggered with match details
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MockSegment:
    """Mock segment for testing"""
    text: str = "test text"
    source_file: str = "/path/to/video.mp4"
    start_time: float = 0.0
    end_time: float = 10.0
    keywords: Optional[List[str]] = None
    entities: Optional[List] = None

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.entities is None:
            self.entities = []


@dataclass
class MockConfig:
    """Mock config for testing"""
    matching: MagicMock = None

    def __post_init__(self):
        if self.matching is None:
            self.matching = MagicMock()
            # Default values
            self.matching.obvious_match_enabled = True
            self.matching.obvious_match_min_similarity = 0.9
            self.matching.obvious_match_min_keywords = 3
            self.matching.obvious_match_min_confidence = 0.92
            self.matching.skip_llm_threshold = 0.85
            self.matching.high_confidence_threshold = 0.85
            self.matching.min_confidence = 0.7
            self.matching.embedding_candidates = 50
            self.matching.max_clip_reuse = 1
            self.matching.reuse_penalty = 0.5
            self.matching.chapter_matching_enabled = False
            self.matching.topic_mismatch_penalty = 0.15
            self.matching.cache_llm_responses = False
            self.matching.primary_provider = "gemini"
            self.matching.secondary_provider = "anthropic"
            self.matching.use_local_for_review = False
            self.matching.gemini_model = "gemini-2.0-flash"
            self.matching.anthropic_model = "claude-3-haiku"
            self.matching.ollama_model = "llama3.2"
            self.matching.ambiguous_threshold = 0.6
            self.matching.confidence_threshold = 0.5
            self.matching.location_matching = None
            self.matching.adaptive_threshold_enabled = False
            self.matching.broll_boost = 0.1

        # Additional config attributes
        self.gemini_api_key = None
        self.anthropic_api_key = None
        self.global_cache = None
        self.negative_matching = MagicMock()
        self.negative_matching.enabled = False
        self.output = MagicMock()
        self.output.num_alternatives = 2


class TestCheckObviousMatchFunction:
    """Tests for the check_obvious_match() function"""

    def setup_method(self):
        """Set up test fixtures"""
        # Import here to avoid import errors during collection
        from src.matching.tiered_matcher import TieredMatcher
        self.TieredMatcher = TieredMatcher

    @pytest.mark.fast
    def test_function_exists(self):
        """check_obvious_match function should exist on TieredMatcher"""
        config = MockConfig()
        matcher = self.TieredMatcher(config=config)
        assert hasattr(matcher, 'check_obvious_match')
        assert callable(matcher.check_obvious_match)

    @pytest.mark.fast
    def test_returns_none_when_disabled(self):
        """Should return None when obvious_match_enabled is False"""
        config = MockConfig()
        config.matching.obvious_match_enabled = False
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(text="New York City skyline", keywords=["new", "york", "city", "skyline"])
        video_segment = MockSegment(
            text="Beautiful New York City skyline view",
            keywords=["new", "york", "city", "skyline", "beautiful"],
            entities=[{"text": "New York City", "type": "GPE"}]
        )

        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.95,
            matched_keywords=["new", "york", "city", "skyline"]
        )

        assert result is None

    @pytest.mark.fast
    def test_returns_none_when_similarity_too_low(self):
        """Should return None when similarity is below threshold"""
        config = MockConfig()
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="New York City",
            keywords=["new", "york", "city", "skyline"],
            entities=[{"text": "New York City", "type": "GPE"}]
        )
        video_segment = MockSegment(
            text="New York City skyline",
            keywords=["new", "york", "city", "skyline"],
            entities=[{"text": "New York City", "type": "GPE"}]
        )

        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.85,  # Below 0.9 threshold
            matched_keywords=["new", "york", "city", "skyline"]
        )

        assert result is None

    @pytest.mark.fast
    def test_returns_none_when_too_few_keywords(self):
        """Should return None when fewer than 3 keywords match"""
        config = MockConfig()
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="New York City",
            entities=[{"text": "New York City", "type": "GPE"}]
        )
        video_segment = MockSegment(
            text="New York City skyline",
            entities=[{"text": "New York City", "type": "GPE"}]
        )

        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.95,
            matched_keywords=["new", "york"]  # Only 2 keywords
        )

        assert result is None

    @pytest.mark.fast
    def test_returns_none_when_no_matching_entity(self):
        """Should return None when no matching named entity"""
        config = MockConfig()
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="Beautiful sunset over the ocean",
            keywords=["beautiful", "sunset", "ocean", "view"],
            entities=[{"text": "Los Angeles", "type": "GPE"}]
        )
        video_segment = MockSegment(
            text="Sunset at the beach",
            keywords=["beautiful", "sunset", "ocean", "view", "beach"],
            entities=[{"text": "Miami", "type": "GPE"}]  # Different entity
        )

        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.95,
            matched_keywords=["beautiful", "sunset", "ocean", "view"]
        )

        assert result is None

    @pytest.mark.fast
    def test_returns_none_when_no_entities_in_voiceover(self):
        """Should return None when voiceover has no entities"""
        config = MockConfig()
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="Beautiful sunset",
            keywords=["beautiful", "sunset", "ocean", "view"],
            entities=[]  # No entities
        )
        video_segment = MockSegment(
            text="Sunset at the beach",
            keywords=["beautiful", "sunset", "ocean", "view"],
            entities=[{"text": "Miami", "type": "GPE"}]
        )

        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.95,
            matched_keywords=["beautiful", "sunset", "ocean", "view"]
        )

        assert result is None

    @pytest.mark.fast
    def test_returns_none_when_no_entities_in_video(self):
        """Should return None when video has no entities"""
        config = MockConfig()
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="Beautiful sunset in Miami",
            keywords=["beautiful", "sunset", "miami", "view"],
            entities=[{"text": "Miami", "type": "GPE"}]
        )
        video_segment = MockSegment(
            text="Sunset at the beach",
            keywords=["beautiful", "sunset", "miami", "view"],
            entities=[]  # No entities
        )

        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.95,
            matched_keywords=["beautiful", "sunset", "miami", "view"]
        )

        assert result is None


class TestObviousMatchDetection:
    """Tests for successful obvious match detection"""

    def setup_method(self):
        from src.matching.tiered_matcher import TieredMatcher
        self.TieredMatcher = TieredMatcher

    @pytest.mark.fast
    def test_detects_obvious_match_with_all_conditions_met(self):
        """Should detect obvious match when all three conditions are met"""
        config = MockConfig()
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="The Eiffel Tower in Paris France is beautiful",
            keywords=["eiffel", "tower", "paris", "france", "beautiful"],
            entities=[{"text": "Eiffel Tower", "type": "FAC"}, {"text": "Paris", "type": "GPE"}]
        )
        video_segment = MockSegment(
            text="Eiffel Tower Paris at night beautiful view",
            keywords=["eiffel", "tower", "paris", "night", "beautiful"],
            entities=[{"text": "Eiffel Tower", "type": "FAC"}, {"text": "Paris", "type": "GPE"}]
        )

        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.95,
            matched_keywords=["eiffel", "tower", "paris", "beautiful"]
        )

        assert result is not None
        confidence, reasoning, entities = result
        assert confidence >= 0.92
        assert "obvious_match_early_termination" in reasoning
        assert len(entities) >= 1

    @pytest.mark.fast
    def test_returns_boosted_confidence_minimum_0_92(self):
        """Should return at least 0.92 confidence for obvious matches"""
        config = MockConfig()
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="New York City skyline Manhattan",
            keywords=["new", "york", "city", "skyline", "manhattan"],
            entities=[{"text": "New York City", "type": "GPE"}, {"text": "Manhattan", "type": "GPE"}]
        )
        video_segment = MockSegment(
            text="New York City skyline view Manhattan",
            keywords=["new", "york", "city", "skyline", "manhattan"],
            entities=[{"text": "New York City", "type": "GPE"}, {"text": "Manhattan", "type": "GPE"}]
        )

        # Test with similarity at threshold (0.9)
        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.90,
            matched_keywords=["new", "york", "city", "skyline"]
        )

        assert result is not None
        confidence, _, _ = result
        assert confidence >= 0.92

    @pytest.mark.fast
    def test_uses_higher_similarity_when_above_minimum(self):
        """Should use the similarity score if it's above minimum confidence"""
        config = MockConfig()
        config.matching.obvious_match_min_confidence = 0.92
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="Tokyo Japan skyline view",
            keywords=["tokyo", "japan", "skyline", "view"],
            entities=[{"text": "Tokyo", "type": "GPE"}, {"text": "Japan", "type": "GPE"}]
        )
        video_segment = MockSegment(
            text="Tokyo Japan skyline beautiful",
            keywords=["tokyo", "japan", "skyline", "beautiful"],
            entities=[{"text": "Tokyo", "type": "GPE"}, {"text": "Japan", "type": "GPE"}]
        )

        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.98,  # Higher than minimum
            matched_keywords=["tokyo", "japan", "skyline"]
        )

        assert result is not None
        confidence, _, _ = result
        assert confidence == 0.98

    @pytest.mark.fast
    def test_reasoning_includes_match_details(self):
        """Should include similarity, keyword count, and entities in reasoning"""
        config = MockConfig()
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="London England Big Ben clock",
            keywords=["london", "england", "big", "ben", "clock"],
            entities=[{"text": "London", "type": "GPE"}, {"text": "Big Ben", "type": "FAC"}]
        )
        video_segment = MockSegment(
            text="Big Ben London clock tower",
            keywords=["london", "big", "ben", "clock", "tower"],
            entities=[{"text": "London", "type": "GPE"}, {"text": "Big Ben", "type": "FAC"}]
        )

        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.93,
            matched_keywords=["london", "big", "ben", "clock"]
        )

        assert result is not None
        _, reasoning, _ = result
        assert "0.93" in reasoning or "sim=" in reasoning
        assert "keywords=" in reasoning
        assert "entities=" in reasoning

    @pytest.mark.fast
    def test_matched_entities_returned(self):
        """Should return the list of matched entities"""
        config = MockConfig()
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="Sydney Opera House Australia",
            keywords=["sydney", "opera", "house", "australia"],
            entities=[{"text": "Sydney Opera House", "type": "FAC"}, {"text": "Australia", "type": "GPE"}]
        )
        video_segment = MockSegment(
            text="Sydney Opera House at night",
            keywords=["sydney", "opera", "house", "night"],
            entities=[{"text": "Sydney Opera House", "type": "FAC"}, {"text": "Sydney", "type": "GPE"}]
        )

        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.91,
            matched_keywords=["sydney", "opera", "house"]
        )

        assert result is not None
        _, _, entities = result
        assert len(entities) >= 1
        # Should find "Sydney Opera House" as matching entity
        entity_lower = [e.lower() for e in entities]
        assert any("sydney" in e for e in entity_lower)


class TestObviousMatchConfigOptions:
    """Tests for config options controlling obvious match behavior"""

    def setup_method(self):
        from src.matching.tiered_matcher import TieredMatcher
        self.TieredMatcher = TieredMatcher

    @pytest.mark.fast
    def test_respects_custom_min_similarity(self):
        """Should use custom min_similarity from config"""
        config = MockConfig()
        config.matching.obvious_match_min_similarity = 0.95  # Higher threshold
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="Golden Gate Bridge San Francisco",
            keywords=["golden", "gate", "bridge", "san", "francisco"],
            entities=[{"text": "Golden Gate Bridge", "type": "FAC"}]
        )
        video_segment = MockSegment(
            text="Golden Gate Bridge view",
            keywords=["golden", "gate", "bridge", "view"],
            entities=[{"text": "Golden Gate Bridge", "type": "FAC"}]
        )

        # 0.92 is below the custom 0.95 threshold
        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.92,
            matched_keywords=["golden", "gate", "bridge"]
        )

        assert result is None  # Should not trigger with lower similarity

    @pytest.mark.fast
    def test_respects_custom_min_keywords(self):
        """Should use custom min_keywords from config"""
        config = MockConfig()
        config.matching.obvious_match_min_keywords = 5  # Higher threshold
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="Statue of Liberty New York",
            keywords=["statue", "liberty", "new", "york"],
            entities=[{"text": "Statue of Liberty", "type": "FAC"}]
        )
        video_segment = MockSegment(
            text="Statue of Liberty view",
            keywords=["statue", "liberty", "view"],
            entities=[{"text": "Statue of Liberty", "type": "FAC"}]
        )

        # Only 3 keywords match, below the custom 5 threshold
        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.95,
            matched_keywords=["statue", "liberty", "view"]
        )

        assert result is None  # Should not trigger with fewer keywords

    @pytest.mark.fast
    def test_respects_custom_min_confidence(self):
        """Should use custom min_confidence from config"""
        config = MockConfig()
        config.matching.obvious_match_min_confidence = 0.95  # Higher minimum
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="Grand Canyon Arizona landscape",
            keywords=["grand", "canyon", "arizona", "landscape"],
            entities=[{"text": "Grand Canyon", "type": "LOC"}, {"text": "Arizona", "type": "GPE"}]
        )
        video_segment = MockSegment(
            text="Grand Canyon view Arizona",
            keywords=["grand", "canyon", "arizona", "view"],
            entities=[{"text": "Grand Canyon", "type": "LOC"}, {"text": "Arizona", "type": "GPE"}]
        )

        # Similarity is 0.91, below custom min_confidence of 0.95
        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.91,
            matched_keywords=["grand", "canyon", "arizona"]
        )

        assert result is not None
        confidence, _, _ = result
        assert confidence >= 0.95  # Should be boosted to at least min_confidence


class TestObviousMatchConfidenceCap:
    """Tests for confidence capping in obvious match detection (US-117-006)"""

    def setup_method(self):
        from src.matching.tiered_matcher import TieredMatcher
        self.TieredMatcher = TieredMatcher

    @pytest.mark.fast
    def test_confidence_capped_at_0_98(self):
        """Confidence should be capped at 0.98 even with high similarity"""
        config = MockConfig()
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="Python programming tutorial coding",
            keywords=["python", "programming", "tutorial", "coding"],
            entities=[{"text": "Python", "type": "TECH"}]
        )
        video_segment = MockSegment(
            text="Python programming tutorial coding guide",
            keywords=["python", "programming", "tutorial", "coding", "guide"],
            entities=[{"text": "Python", "type": "TECH"}]
        )

        # With similarity = 0.97, raw_confidence = 0.97 (above 0.92)
        # boosted = min(0.98, 0.97 + 0.03) = min(0.98, 1.00) = 0.98
        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.97,
            matched_keywords=["python", "programming", "tutorial"]
        )

        assert result is not None
        confidence, reasoning, _ = result
        assert confidence == 0.98  # Capped at 0.98
        assert confidence >= 0.92  # Still above minimum

    @pytest.mark.fast
    def test_confidence_boost_applies_plus_0_03(self):
        """Confidence should be boosted by +0.03, capped at 0.98"""
        config = MockConfig()
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="JavaScript tutorial web development",
            keywords=["javascript", "tutorial", "web", "development"],
            entities=[{"text": "JavaScript", "type": "TECH"}]
        )
        video_segment = MockSegment(
            text="JavaScript tutorial web development coding",
            keywords=["javascript", "tutorial", "web", "development", "coding"],
            entities=[{"text": "JavaScript", "type": "TECH"}]
        )

        # similarity = 0.93, raw_confidence = max(0.92, 0.93) = 0.93
        # boosted = min(0.98, 0.93 + 0.03) = min(0.98, 0.96) = 0.96
        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.93,
            matched_keywords=["javascript", "tutorial", "web"]
        )

        assert result is not None
        confidence, reasoning, _ = result
        assert abs(confidence - 0.96) < 0.01  # 0.93 + 0.03 = 0.96 (below cap)
        assert "obvious_match_early_termination" in reasoning

    @pytest.mark.fast
    def test_matches_below_min_confidence_not_early_terminated(self):
        """Matches below obvious_match_min_confidence should not be early-terminated"""
        config = MockConfig()
        config.matching.obvious_match_min_confidence = 0.92
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="Random video content here",
            keywords=["random", "video", "content", "here"],
            entities=[{"text": "Random", "type": "MISC"}]
        )
        video_segment = MockSegment(
            text="Another random video content",
            keywords=["another", "random", "video", "content"],
            entities=[{"text": "Random", "type": "MISC"}]
        )

        # With similarity = 0.85 (below min_confidence of 0.92)
        # Should NOT be an obvious match
        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.85,  # Below min_confidence
            matched_keywords=["random", "video", "content"]
        )

        # Should return None because similarity is below min_similarity threshold
        assert result is None

    @pytest.mark.fast
    def test_confidence_meets_minimum_0_92(self):
        """Confidence should be at least 0.92 for obvious matches"""
        config = MockConfig()
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="Machine learning AI algorithms",
            keywords=["machine", "learning", "ai", "algorithms"],
            entities=[{"text": "Machine Learning", "type": "TECH"}]
        )
        video_segment = MockSegment(
            text="Machine learning AI algorithms tutorial",
            keywords=["machine", "learning", "ai", "algorithms", "tutorial"],
            entities=[{"text": "Machine Learning", "type": "TECH"}]
        )

        # similarity = 0.90 (at min_similarity threshold)
        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.90,
            matched_keywords=["machine", "learning", "ai"]
        )

        assert result is not None
        confidence, reasoning, _ = result
        assert confidence >= 0.92
        assert "obvious_match_early_termination" in reasoning


class TestObviousMatchVaryingMargin:
    """Tests for obvious match with varying margins between top candidates"""

    def setup_method(self):
        from src.matching.tiered_matcher import TieredMatcher
        self.TieredMatcher = TieredMatcher

    @pytest.mark.fast
    def test_small_margin_between_candidates(self):
        """Should handle small margin between top candidates"""
        config = MockConfig()
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="Beach sunset ocean waves",
            keywords=["beach", "sunset", "ocean", "waves"],
            entities=[{"text": "Ocean", "type": "LOC"}]
        )
        video_segment = MockSegment(
            text="Beach sunset ocean waves view",
            keywords=["beach", "sunset", "ocean", "waves", "view"],
            entities=[{"text": "Ocean", "type": "LOC"}]
        )

        # Small margin - similarity = 0.91
        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.91,
            matched_keywords=["beach", "sunset", "ocean"]
        )

        assert result is not None
        confidence, reasoning, _ = result
        # raw = max(0.92, 0.91) = 0.92, boosted = min(0.98, 0.92 + 0.03) = 0.95
        assert abs(confidence - 0.95) < 0.01
        assert "obvious_match_early_termination" in reasoning

    @pytest.mark.fast
    def test_large_margin_between_candidates(self):
        """Should handle large margin between top candidates"""
        config = MockConfig()
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="Mountain peak snow landscape",
            keywords=["mountain", "peak", "snow", "landscape"],
            entities=[{"text": "Mountain", "type": "LOC"}]
        )
        video_segment = MockSegment(
            text="Mountain peak snow landscape beautiful",
            keywords=["mountain", "peak", "snow", "landscape", "beautiful"],
            entities=[{"text": "Mountain", "type": "LOC"}]
        )

        # Large margin - similarity = 0.99
        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.99,
            matched_keywords=["mountain", "peak", "snow", "landscape"]
        )

        assert result is not None
        confidence, reasoning, _ = result
        # raw = max(0.92, 0.99) = 0.99, boosted = min(0.98, 0.99 + 0.03) = 0.98
        assert confidence == 0.98  # Capped
        assert "obvious_match_early_termination" in reasoning


class TestObviousMatchAdaptiveThreshold:
    """Tests for obvious match with adaptive threshold enabled"""

    def setup_method(self):
        from src.matching.tiered_matcher import TieredMatcher
        self.TieredMatcher = TieredMatcher

    @pytest.mark.fast
    def test_works_with_adaptive_threshold_enabled(self):
        """Obvious match should work with adaptive_threshold_enabled"""
        config = MockConfig()
        config.matching.adaptive_threshold_enabled = True
        config.matching.adaptive_threshold_min_boost = 0.05
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="Forest trees nature hiking",
            keywords=["forest", "trees", "nature", "hiking"],
            entities=[{"text": "Forest", "type": "LOC"}]
        )
        video_segment = MockSegment(
            text="Forest trees nature hiking trail",
            keywords=["forest", "trees", "nature", "hiking", "trail"],
            entities=[{"text": "Forest", "type": "LOC"}]
        )

        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.94,
            matched_keywords=["forest", "trees", "nature", "hiking"]
        )

        assert result is not None
        confidence, reasoning, _ = result
        assert confidence >= 0.92
        assert "obvious_match_early_termination" in reasoning

    @pytest.mark.fast
    def test_obvious_match_integration_with_adaptive(self):
        """Obvious match should integrate properly with adaptive threshold"""
        config = MockConfig()
        config.matching.adaptive_threshold_enabled = True
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="City skyline night lights",
            keywords=["city", "skyline", "night", "lights"],
            entities=[{"text": "City", "type": "GPE"}]
        )
        video_segment = MockSegment(
            text="City skyline night lights view",
            keywords=["city", "skyline", "night", "lights", "view"],
            entities=[{"text": "City", "type": "GPE"}]
        )

        # This should still work as an obvious match
        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.92,
            matched_keywords=["city", "skyline", "night"]
        )

        assert result is not None
        confidence, reasoning, _ = result
        # raw = max(0.92, 0.92) = 0.92, boosted = min(0.98, 0.92 + 0.03) = 0.95
        assert abs(confidence - 0.95) < 0.01


class TestObviousMatchEntityMatching:
    """Tests for entity matching in obvious match detection"""

    def setup_method(self):
        from src.matching.tiered_matcher import TieredMatcher
        self.TieredMatcher = TieredMatcher

    @pytest.mark.fast
    def test_case_insensitive_entity_matching(self):
        """Should match entities case-insensitively"""
        config = MockConfig()
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="PARIS FRANCE is beautiful",
            keywords=["paris", "france", "beautiful", "city"],
            entities=[{"text": "PARIS", "type": "GPE"}]  # Uppercase
        )
        video_segment = MockSegment(
            text="paris france view",
            keywords=["paris", "france", "beautiful", "view"],
            entities=[{"text": "Paris", "type": "GPE"}]  # Title case
        )

        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.92,
            matched_keywords=["paris", "france", "beautiful"]
        )

        assert result is not None

    @pytest.mark.fast
    def test_handles_string_entities(self):
        """Should handle entities stored as plain strings"""
        config = MockConfig()
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="Berlin Germany wall",
            keywords=["berlin", "germany", "wall", "history"],
            entities=["Berlin", "Germany"]  # String entities
        )
        video_segment = MockSegment(
            text="Berlin Wall Germany",
            keywords=["berlin", "germany", "wall"],
            entities=["Berlin", "Germany"]  # String entities
        )

        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.93,
            matched_keywords=["berlin", "germany", "wall"]
        )

        assert result is not None

    @pytest.mark.fast
    def test_handles_dict_entities(self):
        """Should handle entities stored as dicts with 'text' key"""
        config = MockConfig()
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="Rome Italy Colosseum ancient",
            keywords=["rome", "italy", "colosseum", "ancient"],
            entities=[
                {"text": "Rome", "type": "GPE"},
                {"text": "Colosseum", "type": "FAC"}
            ]
        )
        video_segment = MockSegment(
            text="Colosseum Rome ancient history",
            keywords=["rome", "colosseum", "ancient", "history"],
            entities=[
                {"text": "Colosseum", "type": "FAC"},
                {"text": "Rome", "type": "GPE"}
            ]
        )

        result = matcher.check_obvious_match(
            vo_segment, video_segment,
            similarity=0.94,
            matched_keywords=["rome", "colosseum", "ancient"]
        )

        assert result is not None


class TestObviousMatchLogging:
    """Tests for logging behavior of obvious match detection"""

    def setup_method(self):
        from src.matching.tiered_matcher import TieredMatcher
        self.TieredMatcher = TieredMatcher

    @pytest.mark.fast
    def test_logs_info_when_obvious_match_detected(self):
        """Should log info message when obvious match is detected"""
        config = MockConfig()
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="Dubai UAE skyline towers",
            keywords=["dubai", "uae", "skyline", "towers"],
            entities=[{"text": "Dubai", "type": "GPE"}, {"text": "UAE", "type": "GPE"}]
        )
        video_segment = MockSegment(
            text="Dubai skyline towers view",
            keywords=["dubai", "skyline", "towers", "view"],
            entities=[{"text": "Dubai", "type": "GPE"}]
        )

        with patch('src.matching.tiered_matcher.logger') as mock_logger:
            result = matcher.check_obvious_match(
                vo_segment, video_segment,
                similarity=0.95,
                matched_keywords=["dubai", "skyline", "towers"]
            )

            assert result is not None
            # Verify info was logged
            mock_logger.info.assert_called()
            # Check log message content
            log_call = mock_logger.info.call_args[0][0]
            assert "Obvious match detected" in log_call
            assert "skipping LLM" in log_call


class TestMatchSegmentIntegration:
    """Integration tests for obvious match in match_segment method"""

    def setup_method(self):
        from src.matching.tiered_matcher import TieredMatcher
        self.TieredMatcher = TieredMatcher

    def _create_full_mock_config(self):
        """Create a more complete mock config for match_segment integration tests"""
        config = MockConfig()
        # Ensure ReuseTracker gets proper int values (not MagicMock)
        config.matching.max_clip_reuse = 1
        config.matching.reuse_penalty = 0.5
        config.matching.max_source_file_reuse = 0  # Important: must be int
        config.matching.source_file_penalty = 0.05
        config.matching.face_preference = 'neutral'
        return config

    @pytest.mark.fast
    def test_obvious_match_skips_llm_call(self):
        """Should skip LLM when obvious match is detected"""
        config = self._create_full_mock_config()
        config.matching.obvious_match_enabled = True
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="Mount Fuji Japan beautiful snow peak",
            keywords=["mount", "fuji", "japan", "beautiful", "snow"],
            entities=[{"text": "Mount Fuji", "type": "LOC"}, {"text": "Japan", "type": "GPE"}]
        )
        video_segment = MockSegment(
            text="Mount Fuji Japan snow covered peak",
            keywords=["mount", "fuji", "japan", "snow", "peak"],
            entities=[{"text": "Mount Fuji", "type": "LOC"}, {"text": "Japan", "type": "GPE"}]
        )

        candidates = [(video_segment, 0.95)]

        # Mock the LLM provider to verify it's not called
        matcher.primary_provider = MagicMock()
        matcher.secondary_provider = None
        matcher.local_provider = None

        result = matcher.match_segment(vo_segment, candidates)

        # LLM should not be called for obvious match
        matcher.primary_provider.match_batch.assert_not_called()

        # Result should have the obvious match reasoning
        assert result.primary_match.confidence >= 0.92
        assert "obvious_match" in result.primary_match.reasoning

    @pytest.mark.fast
    def test_obvious_match_disabled_uses_normal_path(self):
        """Should use normal matching path when obvious match is disabled"""
        config = self._create_full_mock_config()
        config.matching.obvious_match_enabled = False
        config.matching.skip_llm_threshold = 0.85
        matcher = self.TieredMatcher(config=config)

        vo_segment = MockSegment(
            text="Cairo Egypt pyramids ancient",
            keywords=["cairo", "egypt", "pyramids", "ancient"],
            entities=[{"text": "Cairo", "type": "GPE"}, {"text": "Egypt", "type": "GPE"}]
        )
        video_segment = MockSegment(
            text="Cairo Egypt pyramids view",
            keywords=["cairo", "egypt", "pyramids", "view"],
            entities=[{"text": "Cairo", "type": "GPE"}, {"text": "Egypt", "type": "GPE"}]
        )

        candidates = [(video_segment, 0.95)]

        result = matcher.match_segment(vo_segment, candidates)

        # Should use high embedding similarity path, not obvious match
        assert "obvious_match" not in result.primary_match.reasoning


class TestDefaultConfigValues:
    """Tests for default config values"""

    @pytest.mark.fast
    def test_config_defaults_exist(self):
        """Config should have default values for obvious match settings"""
        from src.config.sections.matching import MatchingConfig

        config = MatchingConfig()

        assert hasattr(config, 'obvious_match_enabled')
        assert hasattr(config, 'obvious_match_min_similarity')
        assert hasattr(config, 'obvious_match_min_keywords')
        assert hasattr(config, 'obvious_match_min_confidence')

    @pytest.mark.fast
    def test_config_default_values(self):
        """Config defaults should match acceptance criteria"""
        from src.config.sections.matching import MatchingConfig

        config = MatchingConfig()

        assert config.obvious_match_enabled is True
        assert config.obvious_match_min_similarity == 0.9
        assert config.obvious_match_min_keywords == 3
        assert config.obvious_match_min_confidence == 0.92
