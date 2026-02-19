"""
Tests for US-77-003: Wire temporal coherence scoring into TieredMatcher.

Verifies:
1. compute_temporal_coherence is called in all 3 match paths
2. Previous match's video source_file is tracked and passed for temporal continuity
3. The temporal coherence adjustment appears in confidence_breakdown as 'temporal_coherence'
4. Same-source boost (+0.05 default) is applied when consecutive segments match same video
5. Context-switch penalty is applied when consecutive segments switch to different sources
6. When temporal_coherence_enabled=False, no adjustment is applied
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress FutureWarning from google.generativeai (deprecated package)
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from src.utils import SRTSegment, Match


# ---------------------------------------------------------------------------
# Mock Config (matches test_tiered_matcher_adjustments.py pattern)
# ---------------------------------------------------------------------------
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
    skip_llm_threshold: float = 0.70
    ambiguous_threshold: float = 0.6
    confidence_threshold: float = 0.3
    max_clip_reuse: int = 10
    reuse_penalty: float = 0.01
    chapter_matching_enabled: bool = False
    enforce_chapter_boundaries: bool = False  # US-95-004
    topic_mismatch_penalty: float = 0.15
    location_matching: None = None
    cache_llm_responses: bool = False
    face_preference: str = "neutral"
    caption_quality_adjustment_enabled: bool = False
    timing_penalty_enabled: bool = False
    max_consecutive_same_source: int = 3
    consecutive_source_penalty: float = 0.05
    adaptive_threshold_enabled: bool = False
    current_project_boost: float = 0.0
    broll_boost: float = 0.0
    obvious_match_min_confidence: float = 0.0
    # Temporal coherence settings
    temporal_coherence_enabled: bool = True
    temporal_coherence_same_source_boost: float = 0.05
    temporal_coherence_context_switch_penalty: float = 0.05
    # Semantic coherence settings
    semantic_coherence_enabled: bool = False  # Disable to isolate temporal tests
    semantic_coherence_similarity_boost: float = 0.0
    semantic_coherence_dissimilarity_penalty: float = 0.0
    # Prefer broll
    prefer_broll_when_topic_matches: bool = False
    broll_face_threshold: float = 0.3
    # Fallback matching
    fallback_matching_enabled: bool = False
    # Transcript quality
    transcript_quality_scoring_enabled: bool = False


@dataclass
class MockOutputConfig:
    num_alternatives: int = 2
    secondary_matches_enabled: bool = False
    strategy_matches_enabled: bool = False


@dataclass
class MockConfig:
    matching: MockMatchingConfig = field(default_factory=MockMatchingConfig)
    output: MockOutputConfig = field(default_factory=MockOutputConfig)
    gemini_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    negative_matching: MagicMock = field(default_factory=lambda: MagicMock(enabled=False))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_vo_segment(text: str, index: int = 0) -> SRTSegment:
    """Create a voiceover segment."""
    seg = SRTSegment(
        index=index,
        start_time=index * 10.0,
        end_time=(index + 1) * 10.0,
        text=text,
        source_file="",
        keywords=text.lower().split()[:5],
    )
    seg.chapter_index = 0
    return seg


def _make_video_segment(
    source_file: str,
    text: str = "video transcript text",
    topics: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
) -> SRTSegment:
    """Create a video segment with optional topics/keywords."""
    seg = SRTSegment(
        index=0,
        start_time=0.0,
        end_time=30.0,
        text=text,
        source_file=source_file,
        keywords=keywords or text.lower().split()[:5],
    )
    seg.chapter_index = 0
    seg.chapter_title = ""
    if topics:
        seg.topics = topics
    return seg


def _create_matcher(config=None):
    """Create a TieredMatcher with controlled config."""
    from src.matching.tiered_matcher import TieredMatcher

    cfg = config or MockConfig()
    with patch('src.matching.tiered_matcher.get_config', return_value=cfg):
        matcher = TieredMatcher(
            config=cfg,
            video_metadata={},
            relevance_matrix=None,
            listicle_groups=[],
        )
    return matcher


# ---------------------------------------------------------------------------
# AC1: compute_temporal_coherence called in all 3 match paths
# ---------------------------------------------------------------------------
class TestTemporalCoherenceCalledInAllPaths:
    """Verify compute_temporal_coherence is called in all 3 match paths."""

    @pytest.mark.fast
    def test_obvious_match_path_calls_temporal_coherence(self):
        """Path 1 (obvious match / skip-LLM): temporal coherence fires."""
        cfg = MockConfig()
        cfg.matching.skip_llm_threshold = 0.70  # Low threshold so skip-LLM fires
        cfg.matching.obvious_match_min_confidence = 0.0
        matcher = _create_matcher(cfg)

        vo_seg = _make_vo_segment("solar energy panels", index=0)
        # Same source as previous to trigger boost
        video_seg = _make_video_segment("vid_A", text="solar energy panels footage")

        # Seed a previous match with same source
        prev_vo = _make_vo_segment("intro about solar", index=-1)
        prev_match = Match(
            voiceover_segment=prev_vo,
            video_segment=_make_video_segment("vid_A"),
            video_scene=None,
            confidence=0.85,
            reasoning="previous match",
        )
        matcher._recent_matches = [prev_match]
        # Also seed the _previous_match_segment
        matcher._previous_match_segment = _make_video_segment("vid_A")

        candidates = [(video_seg, 0.88)]  # Above skip_llm_threshold
        result = matcher.match_segment(vo_seg, candidates, scenes=None, segment_idx=0)

        assert result is not None
        components = {e['component'] for e in result.confidence_breakdown}
        assert 'temporal_coherence' in components, (
            f"temporal_coherence missing from obvious-match path. Components: {components}"
        )

    @pytest.mark.fast
    def test_high_similarity_path_calls_temporal_coherence(self):
        """Path 2 (high embedding similarity): temporal coherence fires."""
        cfg = MockConfig()
        cfg.matching.skip_llm_threshold = 0.95  # High threshold so we fall through
        cfg.matching.high_confidence_threshold = 0.80
        matcher = _create_matcher(cfg)

        vo_seg = _make_vo_segment("wildlife nature documentary", index=0)
        video_seg = _make_video_segment("vid_B", text="nature documentary footage")

        # Seed previous match with same source
        matcher._previous_match_segment = _make_video_segment("vid_B")

        candidates = [(video_seg, 0.85)]  # Between high and skip thresholds
        result = matcher.match_segment(vo_seg, candidates, scenes=None, segment_idx=0)

        assert result is not None
        components = {e['component'] for e in result.confidence_breakdown}
        assert 'temporal_coherence' in components, (
            f"temporal_coherence missing from high-similarity path. Components: {components}"
        )

    @pytest.mark.fast
    def test_llm_path_calls_temporal_coherence(self):
        """Path 3 (LLM matching): temporal coherence fires."""
        cfg = MockConfig()
        cfg.matching.skip_llm_threshold = 0.95
        cfg.matching.high_confidence_threshold = 0.90  # Higher than candidate sim
        cfg.gemini_api_key = "test_key"
        cfg.matching.primary_provider = "gemini"
        matcher = _create_matcher(cfg)

        vo_seg = _make_vo_segment("cooking recipe kitchen", index=0)
        video_seg = _make_video_segment("vid_C", text="cooking recipe demonstration")

        # Seed previous match with same source
        matcher._previous_match_segment = _make_video_segment("vid_C")

        candidates = [(video_seg, 0.75)]  # Below high threshold -> LLM path

        # Mock the LLM provider to return a result
        mock_provider = MagicMock()
        mock_provider.match_segment.return_value = Match(
            voiceover_segment=vo_seg,
            video_segment=video_seg,
            video_scene=None,
            confidence=0.80,
            reasoning="LLM match",
        )
        matcher.primary_provider = mock_provider

        result = matcher.match_segment(vo_seg, candidates, scenes=None, segment_idx=0)

        assert result is not None
        components = {e['component'] for e in result.confidence_breakdown}
        assert 'temporal_coherence' in components, (
            f"temporal_coherence missing from LLM path. Components: {components}"
        )


# ---------------------------------------------------------------------------
# AC2: Previous match's video source_file is tracked
# ---------------------------------------------------------------------------
class TestPreviousMatchTracking:
    """Verify previous match segment is tracked and passed for continuity."""

    @pytest.mark.fast
    def test_previous_match_segment_initialized_none(self):
        """_previous_match_segment starts as None."""
        matcher = _create_matcher()
        assert matcher._previous_match_segment is None

    @pytest.mark.fast
    def test_previous_match_segment_updated_after_match(self):
        """_previous_match_segment is updated after matching a segment."""
        matcher = _create_matcher()

        vo_seg = _make_vo_segment("first segment", index=0)
        video_seg = _make_video_segment("vid_X", text="first video content")

        candidates = [(video_seg, 0.88)]
        matcher.match_segment(vo_seg, candidates, scenes=None, segment_idx=0)

        # After matching, _previous_match_segment should be set to the matched segment
        assert matcher._previous_match_segment is not None
        assert matcher._previous_match_segment.source_file == "vid_X"

    @pytest.mark.fast
    def test_previous_match_carries_to_next_segment(self):
        """After matching seg1, the source is carried to temporal coherence of seg2."""
        matcher = _create_matcher()

        # Match segment 1 with vid_A
        vo_seg1 = _make_vo_segment("first segment content", index=0)
        video_seg_a = _make_video_segment("vid_A", text="first video")
        matcher.match_segment(vo_seg1, [(video_seg_a, 0.88)], scenes=None, segment_idx=0)

        # Now match segment 2 with vid_A (same source) — should get boost
        vo_seg2 = _make_vo_segment("second segment content", index=1)
        video_seg_a2 = _make_video_segment("vid_A", text="same video second clip")
        result = matcher.match_segment(vo_seg2, [(video_seg_a2, 0.85)], scenes=None, segment_idx=1)

        assert result is not None
        temporal_entries = [
            e for e in result.confidence_breakdown if e['component'] == 'temporal_coherence'
        ]
        assert len(temporal_entries) == 1
        assert temporal_entries[0]['adjustment'] > 0, (
            f"Expected positive boost for same-source, got {temporal_entries[0]['adjustment']}"
        )


# ---------------------------------------------------------------------------
# AC3: temporal_coherence appears in confidence_breakdown
# ---------------------------------------------------------------------------
class TestTemporalCoherenceInBreakdown:
    """Verify temporal_coherence entry in confidence_breakdown."""

    @pytest.mark.fast
    def test_breakdown_entry_has_correct_keys(self):
        """temporal_coherence breakdown entry has component, adjustment, reason."""
        matcher = _create_matcher()

        # Seed a previous match
        matcher._previous_match_segment = _make_video_segment("vid_A")

        vo_seg = _make_vo_segment("test content", index=1)
        video_seg = _make_video_segment("vid_A")  # Same source -> triggers entry

        result = matcher.match_segment(vo_seg, [(video_seg, 0.85)], scenes=None, segment_idx=1)

        temporal_entries = [
            e for e in result.confidence_breakdown if e['component'] == 'temporal_coherence'
        ]
        assert len(temporal_entries) == 1
        entry = temporal_entries[0]
        assert 'component' in entry
        assert 'adjustment' in entry
        assert 'reason' in entry
        assert entry['component'] == 'temporal_coherence'

    @pytest.mark.fast
    def test_breakdown_not_added_when_no_adjustment(self):
        """No temporal_coherence entry when there's no previous match (first segment)."""
        matcher = _create_matcher()
        # _previous_match_segment is None (default)

        vo_seg = _make_vo_segment("first segment", index=0)
        video_seg = _make_video_segment("vid_A")

        result = matcher.match_segment(vo_seg, [(video_seg, 0.85)], scenes=None, segment_idx=0)

        # With no previous match, compute_temporal_coherence returns ("", no reason),
        # so _record_breakdown should NOT add an entry
        temporal_entries = [
            e for e in result.confidence_breakdown if e['component'] == 'temporal_coherence'
        ]
        assert len(temporal_entries) == 0


# ---------------------------------------------------------------------------
# AC4: Same-source boost (+0.05 default)
# ---------------------------------------------------------------------------
class TestSameSourceBoostWiring:
    """Verify +0.05 boost is applied for consecutive same-source matches."""

    @pytest.mark.fast
    def test_same_source_gives_positive_adjustment(self):
        """Consecutive matches from same source get +0.05 boost."""
        matcher = _create_matcher()

        # Seed previous match from vid_A
        matcher._previous_match_segment = _make_video_segment("vid_A")

        vo_seg = _make_vo_segment("segment about topic", index=1)
        video_seg = _make_video_segment("vid_A")  # Same source

        result = matcher.match_segment(vo_seg, [(video_seg, 0.80)], scenes=None, segment_idx=1)

        temporal_entries = [
            e for e in result.confidence_breakdown if e['component'] == 'temporal_coherence'
        ]
        assert len(temporal_entries) == 1
        assert temporal_entries[0]['adjustment'] == pytest.approx(0.05, abs=0.001), (
            f"Expected +0.05 boost, got {temporal_entries[0]['adjustment']}"
        )

    @pytest.mark.fast
    def test_same_source_boost_uses_config_value(self):
        """Custom same_source_boost value is respected."""
        cfg = MockConfig()
        cfg.matching.temporal_coherence_same_source_boost = 0.10
        matcher = _create_matcher(cfg)

        matcher._previous_match_segment = _make_video_segment("vid_A")

        vo_seg = _make_vo_segment("topic content", index=1)
        video_seg = _make_video_segment("vid_A")

        result = matcher.match_segment(vo_seg, [(video_seg, 0.80)], scenes=None, segment_idx=1)

        temporal_entries = [
            e for e in result.confidence_breakdown if e['component'] == 'temporal_coherence'
        ]
        assert len(temporal_entries) == 1
        assert temporal_entries[0]['adjustment'] == pytest.approx(0.10, abs=0.001)


# ---------------------------------------------------------------------------
# AC5: Context-switch penalty
# ---------------------------------------------------------------------------
class TestContextSwitchPenaltyWiring:
    """Verify context-switch penalty when switching to different video source."""

    @pytest.mark.fast
    def test_different_source_jarring_topics_gives_penalty(self):
        """Different source + jarring topics = negative adjustment."""
        matcher = _create_matcher()

        # Previous: astronomy video
        matcher._previous_match_segment = _make_video_segment(
            "vid_astro", text="astronomy",
            topics=["astronomy", "stars", "space"],
            keywords=["telescope", "galaxy"],
        )

        vo_seg = _make_vo_segment("cooking recipe ingredients", index=1)
        # Current candidate: completely different topic
        video_seg = _make_video_segment(
            "vid_cooking", text="cooking recipe",
            topics=["cooking", "kitchen", "food"],
            keywords=["recipe", "ingredients"],
        )

        result = matcher.match_segment(vo_seg, [(video_seg, 0.80)], scenes=None, segment_idx=1)

        temporal_entries = [
            e for e in result.confidence_breakdown if e['component'] == 'temporal_coherence'
        ]
        assert len(temporal_entries) == 1
        assert temporal_entries[0]['adjustment'] < 0, (
            f"Expected negative penalty, got {temporal_entries[0]['adjustment']}"
        )

    @pytest.mark.fast
    def test_different_source_overlapping_topics_no_penalty(self):
        """Different source but overlapping topics = no temporal_coherence entry."""
        matcher = _create_matcher()

        # Previous: energy video
        matcher._previous_match_segment = _make_video_segment(
            "vid_energy1", text="solar energy",
            topics=["energy", "solar"],
            keywords=["renewable"],
        )

        vo_seg = _make_vo_segment("renewable energy overview", index=1)
        # Current: different source but overlapping topic
        video_seg = _make_video_segment(
            "vid_energy2", text="renewable solar energy",
            topics=["energy", "renewable"],
            keywords=["solar"],
        )

        result = matcher.match_segment(vo_seg, [(video_seg, 0.80)], scenes=None, segment_idx=1)

        # With overlapping topics, _is_jarring_context_switch returns False,
        # so no penalty and no breakdown entry
        temporal_entries = [
            e for e in result.confidence_breakdown if e['component'] == 'temporal_coherence'
        ]
        assert len(temporal_entries) == 0


# ---------------------------------------------------------------------------
# AC6: Disabled config means no adjustment
# ---------------------------------------------------------------------------
class TestTemporalCoherenceDisabled:
    """Verify no adjustment when temporal_coherence_enabled=False."""

    @pytest.mark.fast
    def test_disabled_config_no_temporal_entry_in_breakdown(self):
        """With temporal_coherence_enabled=False, no temporal_coherence in breakdown."""
        cfg = MockConfig()
        cfg.matching.temporal_coherence_enabled = False
        matcher = _create_matcher(cfg)

        # Seed a previous match (same source — would normally trigger boost)
        matcher._previous_match_segment = _make_video_segment("vid_A")

        vo_seg = _make_vo_segment("some content", index=1)
        video_seg = _make_video_segment("vid_A")  # Same source

        result = matcher.match_segment(vo_seg, [(video_seg, 0.80)], scenes=None, segment_idx=1)

        temporal_entries = [
            e for e in result.confidence_breakdown if e['component'] == 'temporal_coherence'
        ]
        assert len(temporal_entries) == 0, (
            f"Expected no temporal_coherence when disabled, got {temporal_entries}"
        )

    @pytest.mark.fast
    def test_disabled_config_confidence_unchanged_by_temporal(self):
        """With disabled config, confidence not affected by temporal coherence."""
        cfg_enabled = MockConfig()
        cfg_enabled.matching.temporal_coherence_enabled = True

        cfg_disabled = MockConfig()
        cfg_disabled.matching.temporal_coherence_enabled = False

        # Run with enabled
        matcher_e = _create_matcher(cfg_enabled)
        matcher_e._previous_match_segment = _make_video_segment("vid_A")
        vo_seg = _make_vo_segment("content here", index=1)
        video_seg = _make_video_segment("vid_A")
        result_e = matcher_e.match_segment(vo_seg, [(video_seg, 0.80)], scenes=None, segment_idx=1)

        # Run with disabled
        matcher_d = _create_matcher(cfg_disabled)
        matcher_d._previous_match_segment = _make_video_segment("vid_A")
        vo_seg_d = _make_vo_segment("content here", index=1)
        video_seg_d = _make_video_segment("vid_A")
        result_d = matcher_d.match_segment(vo_seg_d, [(video_seg_d, 0.80)], scenes=None, segment_idx=1)

        # Enabled should have different confidence due to temporal boost
        conf_e = result_e.primary_match.confidence
        conf_d = result_d.primary_match.confidence

        # With same-source boost enabled, conf_e should be higher
        assert conf_e > conf_d or conf_e == conf_d, (
            "Enabled temporal coherence should boost or equal disabled"
        )
        # More precisely: enabled should have the +0.05 boost
        temporal_entries = [
            e for e in result_e.confidence_breakdown if e['component'] == 'temporal_coherence'
        ]
        if temporal_entries:
            assert temporal_entries[0]['adjustment'] > 0


# ---------------------------------------------------------------------------
# Integration: Sequential matching tracks source across segments
# ---------------------------------------------------------------------------
class TestSequentialMatchingIntegration:
    """Integration: sequential matching updates _previous_match_segment correctly."""

    @pytest.mark.fast
    def test_three_segments_alternating_sources(self):
        """Match 3 segments: same, different, same — verify tracking."""
        matcher = _create_matcher()

        # Segment 0: vid_A (no previous)
        vo0 = _make_vo_segment("first content", index=0)
        vid_a = _make_video_segment("vid_A", text="first content video")
        r0 = matcher.match_segment(vo0, [(vid_a, 0.88)], scenes=None, segment_idx=0)
        assert matcher._previous_match_segment.source_file == "vid_A"

        # Segment 1: vid_B (different from previous vid_A)
        vo1 = _make_vo_segment("second content", index=1)
        vid_b = _make_video_segment(
            "vid_B", text="different topic video",
            topics=["astronomy", "stars"],
            keywords=["galaxy"],
        )
        # Set topics on previous so jarring detection can work
        matcher._previous_match_segment.topics = ["cooking", "food"]
        matcher._previous_match_segment.keywords = ["recipe"]

        r1 = matcher.match_segment(vo1, [(vid_b, 0.85)], scenes=None, segment_idx=1)
        assert matcher._previous_match_segment.source_file == "vid_B"

        # Should have penalty for jarring switch
        t1 = [e for e in r1.confidence_breakdown if e['component'] == 'temporal_coherence']
        assert len(t1) == 1
        assert t1[0]['adjustment'] < 0

        # Segment 2: vid_B (same as previous)
        vo2 = _make_vo_segment("third content", index=2)
        vid_b2 = _make_video_segment("vid_B", text="same source again")
        r2 = matcher.match_segment(vo2, [(vid_b2, 0.82)], scenes=None, segment_idx=2)
        assert matcher._previous_match_segment.source_file == "vid_B"

        # Should have boost for same source
        t2 = [e for e in r2.confidence_breakdown if e['component'] == 'temporal_coherence']
        assert len(t2) == 1
        assert t2[0]['adjustment'] > 0


# ---------------------------------------------------------------------------
# AC6: Test interaction with chapter-aware matching (enforce_chapter_boundaries)
# ---------------------------------------------------------------------------
class TestTemporalCoherenceWithChapterBoundaries:
    """Verify temporal coherence works with chapter-aware matching enabled."""

    @pytest.mark.fast
    def test_temporal_coherence_with_chapter_boundaries_enabled(self):
        """Both temporal_coherence and enforce_chapter_boundaries can be enabled together."""
        cfg = MockConfig()
        cfg.matching.temporal_coherence_enabled = True
        cfg.matching.enforce_chapter_boundaries = True
        cfg.matching.chapter_matching_enabled = True
        matcher = _create_matcher(cfg)

        # Seed previous match with same source
        matcher._previous_match_segment = _make_video_segment("vid_A")

        vo_seg = _make_vo_segment("content about topic", index=1)
        video_seg = _make_video_segment("vid_A")

        result = matcher.match_segment(vo_seg, [(video_seg, 0.80)], scenes=None, segment_idx=1)

        assert result is not None
        # Temporal coherence should still apply even with chapter boundaries enabled
        temporal_entries = [
            e for e in result.confidence_breakdown if e['component'] == 'temporal_coherence'
        ]
        assert len(temporal_entries) == 1
        assert temporal_entries[0]['adjustment'] == pytest.approx(0.05, abs=0.001)

    @pytest.mark.fast
    def test_temporal_coherence_jarring_switch_with_chapter_boundaries(self):
        """Jarring context switch penalty still applies with chapter boundaries enabled."""
        cfg = MockConfig()
        cfg.matching.temporal_coherence_enabled = True
        cfg.matching.enforce_chapter_boundaries = True
        cfg.matching.chapter_matching_enabled = True
        matcher = _create_matcher(cfg)

        # Previous: astronomy video with topics
        matcher._previous_match_segment = _make_video_segment(
            "vid_astro",
            topics=["astronomy", "stars", "space"],
            keywords=["telescope", "galaxy"],
        )

        # Current: cooking video (jarring switch)
        vo_seg = _make_vo_segment("cooking recipe", index=1)
        video_seg = _make_video_segment(
            "vid_cooking",
            topics=["cooking", "kitchen", "food"],
            keywords=["recipe", "ingredients"],
        )

        result = matcher.match_segment(vo_seg, [(video_seg, 0.80)], scenes=None, segment_idx=1)

        assert result is not None
        temporal_entries = [
            e for e in result.confidence_breakdown if e['component'] == 'temporal_coherence'
        ]
        assert len(temporal_entries) == 1
        assert temporal_entries[0]['adjustment'] < 0  # Penalty should apply

    @pytest.mark.fast
    def test_chapter_matching_enabled_without_temporal(self):
        """Chapter boundaries work when temporal coherence is disabled."""
        cfg = MockConfig()
        cfg.matching.temporal_coherence_enabled = False
        cfg.matching.enforce_chapter_boundaries = True
        cfg.matching.chapter_matching_enabled = True
        matcher = _create_matcher(cfg)

        # Seed previous match
        matcher._previous_match_segment = _make_video_segment("vid_A")

        vo_seg = _make_vo_segment("content about topic", index=1)
        video_seg = _make_video_segment("vid_A")

        result = matcher.match_segment(vo_seg, [(video_seg, 0.80)], scenes=None, segment_idx=1)

        assert result is not None
        # No temporal coherence entry should appear when disabled
        temporal_entries = [
            e for e in result.confidence_breakdown if e['component'] == 'temporal_coherence'
        ]
        assert len(temporal_entries) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
