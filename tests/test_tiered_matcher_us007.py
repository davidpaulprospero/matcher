"""
Tests for US-007: Add tiered matcher strategy fallback tests

Acceptance Criteria:
- AC1: Test TieredMatcher selects correct strategy based on available resources (covered in test_tiered_matcher_advanced.py)
- AC2: Test TieredMatcher falls back from embedding to keyword when index unavailable (covered in test_fallback_matching.py)
- AC3: Test TieredMatcher handles face preference scoring at exact 0.3 threshold (covered in test_face_detection.py)
- AC4: Test TieredMatcher B-roll detection with conflicting signals (silent + faces)
- AC5: Test TieredMatcher memory exhaustion handling with >1000 candidates
"""

import pytest
from unittest.mock import MagicMock, patch, Mock
from dataclasses import dataclass, field
from typing import Optional, List


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
    scene_index: int = 0


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
    broll_boost: float = 0.2


@dataclass
class MockConfig:
    matching: MockMatchingConfig = field(default_factory=MockMatchingConfig)
    gemini_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    negative_matching: MagicMock = field(default_factory=lambda: MagicMock(enabled=False))
    output: MagicMock = field(default_factory=lambda: MagicMock(num_alternatives=2))


# ============================================================================
# AC4: B-roll detection with conflicting signals (silent + faces)
# ============================================================================

class TestBrollConflictingSignalsUS007:
    """
    AC4: Test TieredMatcher B-roll detection with conflicting signals.

    Scenarios:
    - Segment marked is_broll=True by silent detection, but face_score >= 0.3
    - Segment has face_score < 0.3 but is_broll=False (silent detection didn't run)
    - Segment has both is_broll=True and face_score < 0.3 (consistent signals)
    """

    def test_broll_true_but_high_face_score_conflicting_signal(self):
        """Test B-roll detection when is_broll=True but face_score indicates faces present."""
        from src.matching.scoring import apply_broll_boost

        config = MockConfig()

        # Segment marked as B-roll by silent detection (word_count < threshold)
        # but face detection found faces (face_score >= 0.3)
        seg = MockSRTSegment(
            source_file="/video.mp4",
            is_broll=True,  # Silent detection marked as B-roll
            face_score=0.8  # But faces are present (conflicting signal)
        )

        # apply_broll_boost should still apply boost because is_broll=True
        # The boost is based on is_broll flag, not face_score
        confidence = 0.7
        adjusted, reason = apply_broll_boost(confidence, seg, config)

        # Should apply boost (B-roll boost based on is_broll attribute)
        assert adjusted > confidence
        assert "broll" in reason.lower() or "B-roll" in reason

    def test_low_face_score_but_broll_false_no_boost(self):
        """Test no B-roll boost when face_score < 0.3 but is_broll=False."""
        from src.matching.scoring import apply_broll_boost

        config = MockConfig()

        # Low face score (no faces) but is_broll wasn't set by pipeline
        # This could happen if scene detection didn't run
        seg = MockSRTSegment(
            source_file="/video.mp4",
            is_broll=False,  # Not marked as B-roll
            face_score=0.1   # Low face score (no faces)
        )

        confidence = 0.7
        adjusted, reason = apply_broll_boost(confidence, seg, config)

        # Should NOT apply boost (is_broll is False)
        assert adjusted == confidence
        assert reason == ""

    def test_consistent_broll_signals_applies_boost(self):
        """Test B-roll boost with consistent signals (is_broll=True, face_score < 0.3)."""
        from src.matching.scoring import apply_broll_boost

        config = MockConfig()

        # Both signals agree: B-roll video
        seg = MockSRTSegment(
            source_file="/video.mp4",
            is_broll=True,
            face_score=0.1  # No faces detected
        )

        confidence = 0.7
        adjusted, reason = apply_broll_boost(confidence, seg, config)

        # Should apply boost
        assert adjusted > confidence
        assert "broll" in reason.lower() or "B-roll" in reason

    def test_broll_boost_respects_ceiling(self):
        """Test B-roll boost doesn't exceed 1.0."""
        from src.matching.scoring import apply_broll_boost

        config = MockConfig()
        config.matching.broll_boost = 0.5  # Large boost

        seg = MockSRTSegment(
            source_file="/video.mp4",
            is_broll=True,
            face_score=0.1
        )

        # High initial confidence
        confidence = 0.9
        adjusted, reason = apply_broll_boost(confidence, seg, config)

        # Should not exceed 1.0
        assert adjusted <= 1.0

    def test_face_preference_none_with_conflicting_broll(self):
        """Test face_preference='none' interaction with B-roll conflicts."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.matching.face_preference = 'none'  # Prefer no faces

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        # Segment with conflicting signals
        seg = MockSRTSegment(
            source_file="/video.mp4",
            source="global_cache",
            is_broll=True,
            face_score=0.8  # Faces present despite is_broll
        )

        vo_segment = MockSRTSegment(text="Test voiceover")
        candidates = [(seg, 0.7)]

        with patch.object(matcher, 'reuse_tracker') as mock_tracker:
            mock_tracker.can_use.return_value = True
            mock_tracker.adjust_confidence.side_effect = lambda s, c: c
            mock_tracker.get_usage_count.return_value = 0

            with patch('src.matching.tiered_matcher.apply_face_preference') as mock_face:
                mock_face.return_value = []

                result = matcher.match_segment(vo_segment, candidates)

        # Should return a result even with conflicting signals
        assert result is not None


class TestBrollExact03ThresholdUS007:
    """
    AC3: Verify face preference scoring at exact 0.3 threshold.

    Already covered in test_face_detection.py, but adding specific tiered matcher context.
    """

    def test_face_score_exactly_03_not_broll(self):
        """Test face_score exactly at 0.3 is NOT classified as B-roll."""
        from src.face_detection import is_broll_scene

        # At exactly 0.3, face_score is NOT less than threshold
        # So it should NOT be B-roll
        assert is_broll_scene(0.3, threshold=0.3) is False

    def test_face_score_just_below_03_is_broll(self):
        """Test face_score just below 0.3 IS classified as B-roll."""
        from src.face_detection import is_broll_scene

        # Just below threshold should be B-roll
        assert is_broll_scene(0.29, threshold=0.3) is True
        assert is_broll_scene(0.299, threshold=0.3) is True

    def test_face_score_just_above_03_not_broll(self):
        """Test face_score just above 0.3 is NOT classified as B-roll."""
        from src.face_detection import is_broll_scene

        # Just above threshold should not be B-roll
        assert is_broll_scene(0.31, threshold=0.3) is False
        assert is_broll_scene(0.301, threshold=0.3) is False


# ============================================================================
# AC5: Memory exhaustion handling with >1000 candidates
# ============================================================================

class TestMemoryExhaustionHandlingUS007:
    """
    AC5: Test TieredMatcher memory exhaustion handling with >1000 candidates.

    The system should:
    - Not crash with large candidate lists
    - Limit candidate processing to embedding_candidates config
    - Handle edge cases gracefully
    """

    def test_large_candidate_list_1000_candidates(self):
        """Test TieredMatcher handles 1000+ candidates without crash."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.matching.embedding_candidates = 10  # Only process top 10

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        # Create 1000 candidates
        candidates = []
        for i in range(1000):
            seg = MockSRTSegment(
                source_file=f"/video{i % 100}.mp4",  # 100 different sources
                text=f"Content for segment {i}",
                start_time=float(i * 10),
                end_time=float(i * 10 + 10)
            )
            # Decreasing similarity scores
            similarity = max(0.3, 0.95 - (i * 0.0005))
            candidates.append((seg, similarity))

        vo_segment = MockSRTSegment(text="Test voiceover segment")

        with patch.object(matcher, 'reuse_tracker') as mock_tracker:
            mock_tracker.can_use.return_value = True
            mock_tracker.adjust_confidence.side_effect = lambda s, c: c
            mock_tracker.get_usage_count.return_value = 0
            mock_tracker.record_usage.return_value = None

            # Should complete without memory error
            result = matcher.match_segment(vo_segment, candidates)

        assert result is not None
        assert result.primary_match is not None

    def test_large_candidate_list_5000_candidates(self):
        """Test TieredMatcher handles 5000 candidates (stress test)."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.matching.embedding_candidates = 10
        config.matching.skip_llm_threshold = 0.95  # High threshold to test full path

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        # Create 5000 candidates
        candidates = []
        for i in range(5000):
            seg = MockSRTSegment(
                source_file=f"/video{i % 200}.mp4",
                text=f"Segment {i}",
                start_time=float(i * 5),
                end_time=float(i * 5 + 5)
            )
            similarity = max(0.2, 0.9 - (i * 0.0001))
            candidates.append((seg, similarity))

        vo_segment = MockSRTSegment(text="Test voiceover")

        with patch.object(matcher, 'reuse_tracker') as mock_tracker:
            mock_tracker.can_use.return_value = True
            mock_tracker.adjust_confidence.side_effect = lambda s, c: c
            mock_tracker.get_usage_count.return_value = 0
            mock_tracker.record_usage.return_value = None

            # Should complete without memory error
            result = matcher.match_segment(vo_segment, candidates)

        assert result is not None

    def test_secondary_matches_limits_processing(self):
        """Test _get_secondary_matches limits candidates to prevent memory issues."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        # Create 500 candidates
        candidates = []
        for i in range(500):
            seg = MockSRTSegment(
                source_file=f"/video{i}.mp4",
                text=f"Content {i}"
            )
            candidates.append((seg, 0.9 - i * 0.001))

        primary = MockSRTSegment(source_file="/primary.mp4")

        result = matcher._get_secondary_matches(
            candidates,
            scenes=None,
            excluded_video_files={"/primary.mp4"},
            primary_segment=primary,
            alt_segments=[]
        )

        # Should return at most 3 (num_secondary limit)
        assert len(result) <= 3

    def test_alternatives_limits_processing(self):
        """Test _get_alternatives limits candidates."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()
        config.output.num_alternatives = 2

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        # Create 100 candidates
        candidates = []
        for i in range(100):
            seg = MockSRTSegment(
                source_file=f"/video{i}.mp4",
                text=f"Content {i}"
            )
            candidates.append((seg, 0.95 - i * 0.005))

        primary = MockSRTSegment(source_file="/primary.mp4")

        result = matcher._get_alternatives(candidates, scenes=None, primary_match=primary)

        # Should return at most num_alternatives
        assert len(result) <= 2

    def test_confidence_variance_calculation_large_pool(self):
        """Test confidence variance calculation with large candidate pool."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        # Create 200 candidates with varying similarities
        candidates = []
        for i in range(200):
            seg = MockSRTSegment(source_file=f"/v{i}.mp4")
            # Create varied similarities
            import math
            similarity = 0.5 + 0.4 * math.sin(i / 10.0)
            candidates.append((seg, similarity))

        variance = matcher._calculate_confidence_variance(candidates, top_n=10)

        # Should return valid variance (not NaN, not None)
        assert variance is not None
        assert not math.isnan(variance)
        assert variance >= 0


# ============================================================================
# AC1 & AC2: Strategy selection (verification tests)
# These are already covered in other test files but verify integration
# ============================================================================

class TestStrategySelectionUS007:
    """
    AC1: Test TieredMatcher selects correct strategy based on available resources.

    Already extensively covered in:
    - test_tiered_matcher_advanced.py::TestLLMProviderFallback
    - test_tiered_matcher_coverage.py::TestTieredMatcherNoPrimaryProvider

    These tests verify the integration points.
    """

    def test_no_api_keys_uses_embedding_strategy(self):
        """Test embedding-only strategy when no API keys available."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig()  # No API keys

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            matcher = TieredMatcher(config=config)

        # No LLM providers should be initialized
        assert matcher.primary_provider is None
        assert matcher.secondary_provider is None

    def test_gemini_key_uses_gemini_strategy(self):
        """Test Gemini strategy when Gemini API key available."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig(gemini_api_key="test_gemini_key")
        config.matching.primary_provider = "gemini"

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            with patch('src.matching.tiered_matcher.GeminiMatcher') as mock_gemini:
                mock_gemini.return_value = MagicMock()
                matcher = TieredMatcher(config=config)

        mock_gemini.assert_called_once()

    def test_fallback_to_anthropic_when_gemini_unavailable(self):
        """Test fallback to Anthropic when Gemini key not available."""
        from src.matching.tiered_matcher import TieredMatcher

        config = MockConfig(anthropic_api_key="test_anthropic_key")
        config.matching.primary_provider = "gemini"  # Request Gemini but no key
        config.gemini_api_key = None

        with patch('src.matching.tiered_matcher.get_config', return_value=config):
            with patch('src.matching.tiered_matcher.ClaudeMatcher') as mock_claude:
                mock_claude.return_value = MagicMock()
                matcher = TieredMatcher(config=config)

        # Should fall back to Claude
        mock_claude.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
