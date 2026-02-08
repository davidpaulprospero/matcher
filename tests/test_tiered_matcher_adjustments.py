"""
Integration test: TieredMatcher apply_all_adjustments end-to-end (US-75-008).

Verifies the complete adjustment chain works through TieredMatcher.match_segment,
including all adjustments wired in US-75-002 through US-75-007.

Creates a realistic matching scenario with chapters, video metadata, tags,
and listicle groups so that all adjustments fire and appear in confidence_breakdown.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from src.utils import SRTSegment, Match


# ---------------------------------------------------------------------------
# Mock Config
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
    skip_llm_threshold: float = 0.70  # Low threshold so high-similarity path fires
    ambiguous_threshold: float = 0.6
    confidence_threshold: float = 0.3
    max_clip_reuse: int = 10
    reuse_penalty: float = 0.01
    chapter_matching_enabled: bool = True  # Enable chapter-related adjustments
    topic_mismatch_penalty: float = 0.15
    location_matching: None = None
    cache_llm_responses: bool = False
    face_preference: str = "neutral"
    # Caption quality fields
    caption_quality_adjustment_enabled: bool = False  # Disable to avoid noise
    # Timing penalty fields
    timing_penalty_enabled: bool = False
    # Consecutive source fields
    max_consecutive_same_source: int = 3
    consecutive_source_penalty: float = 0.05
    # Adaptive threshold
    adaptive_threshold_enabled: bool = False
    # Project boost
    current_project_boost: float = 0.0
    # B-roll boost
    broll_boost: float = 0.0
    # Obvious match
    obvious_match_min_confidence: float = 0.0  # Allow full adjustment range


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
def _make_vo_segment(text: str, index: int = 0, chapter_index: int = 0) -> SRTSegment:
    """Create a voiceover segment with chapter_index set."""
    seg = SRTSegment(
        index=index,
        start_time=index * 10.0,
        end_time=(index + 1) * 10.0,
        text=text,
        source_file="",
        keywords=text.lower().split()[:5],
    )
    seg.chapter_index = chapter_index
    return seg


def _make_video_segment(
    source_file: str,
    text: str,
    index: int = 0,
    chapter_index: int = 0,
    chapter_title: str = "",
    tags: Optional[List[str]] = None,
) -> SRTSegment:
    """Create a video segment with metadata attributes set."""
    seg = SRTSegment(
        index=index,
        start_time=0.0,
        end_time=30.0,
        text=text,
        source_file=source_file,
        keywords=text.lower().split()[:5],
    )
    seg.chapter_index = chapter_index
    seg.chapter_title = chapter_title
    if tags:
        seg.tags = tags
    return seg


@dataclass
class ListicleGroup:
    start_segment_idx: int
    end_segment_idx: int
    group_id: str = "G1"
    label: str = ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestTieredMatcherAdjustmentsIntegration:
    """Integration tests for the complete adjustment chain in TieredMatcher."""

    def _create_matcher(
        self,
        config=None,
        video_metadata=None,
        relevance_matrix=None,
        listicle_groups=None,
    ):
        """Create a TieredMatcher with controlled config."""
        from src.matching.tiered_matcher import TieredMatcher

        cfg = config or MockConfig()
        with patch('src.matching.tiered_matcher.get_config', return_value=cfg):
            matcher = TieredMatcher(
                config=cfg,
                video_metadata=video_metadata or {},
                relevance_matrix=relevance_matrix,
                listicle_groups=listicle_groups or [],
            )
        return matcher

    @pytest.mark.fast
    def test_full_adjustment_chain_all_components_present(self):
        """
        Integration test: match_segment with full video metadata triggers all adjustments.

        Verifies confidence_breakdown contains title_relevance, description_relevance,
        tag_keyword_boost, chapter_topic_match, and chapter_source_consistency.
        """
        # Shared keywords between voiceover and video metadata
        shared_text = "renewable energy solar panels climate change"

        video_metadata = {
            "vid_001": {
                "title": f"Guide to renewable energy solar panels",
                "description": f"How renewable energy and solar panels help climate change",
                "tags": ["renewable", "energy", "solar", "panels", "climate"],
            },
        }

        # relevance_matrix [vo_chapter 0][vid_chapter 0] = 0.8 -> triggers cross_chapter_relevance
        relevance_matrix = [[0.8]]

        # Listicle group covering segment index 1 (not 0, so it needs a previous match)
        listicle_groups = [ListicleGroup(start_segment_idx=0, end_segment_idx=5)]

        matcher = self._create_matcher(
            video_metadata=video_metadata,
            relevance_matrix=relevance_matrix,
            listicle_groups=listicle_groups,
        )

        # Create voiceover segment with overlapping keywords
        vo_seg = _make_vo_segment(shared_text, index=1, chapter_index=0)

        # Create video segment from same source with chapter info
        video_seg = _make_video_segment(
            source_file="vid_001",
            text="transcript about renewable energy solar technology",
            chapter_index=0,
            chapter_title="renewable energy solar panels overview",
        )

        # Seed a previous match (same source, same chapter) for source_consistency + listicle
        prev_vo = _make_vo_segment("previous segment about solar", index=0, chapter_index=0)
        prev_match = Match(
            voiceover_segment=prev_vo,
            video_segment=video_seg,
            video_scene=None,
            confidence=0.85,
            reasoning="previous match",
        )
        matcher._recent_matches = [prev_match]

        # Seed chapter source counts (6 unique sources triggers coherence penalty)
        matcher._chapter_source_counts = {
            0: {"src_a", "src_b", "src_c", "src_d", "src_e", "src_f"}
        }

        # Candidates: (video_segment, similarity) — high enough to skip LLM
        candidates = [(video_seg, 0.88)]

        result = matcher.match_segment(
            vo_segment=vo_seg,
            candidates=candidates,
            scenes=None,
            segment_idx=1,
        )

        # Verify result structure
        assert result is not None
        assert result.primary_match is not None
        breakdown = result.confidence_breakdown

        # Extract component names from breakdown
        components = {entry['component'] for entry in breakdown}

        # --- Core acceptance criteria ---
        # title_relevance should fire (shared keywords in title)
        assert 'title_relevance' in components, (
            f"title_relevance missing from breakdown. Components: {components}"
        )

        # description_relevance should fire (shared keywords in description)
        assert 'description_relevance' in components, (
            f"description_relevance missing from breakdown. Components: {components}"
        )

        # tag_keyword_boost should fire (shared keywords in tags)
        assert 'tag_keyword_boost' in components, (
            f"tag_keyword_boost missing from breakdown. Components: {components}"
        )

        # chapter_topic_match should fire (chapter_matching_enabled=True, chapter_title set)
        assert 'chapter_topic_match' in components, (
            f"chapter_topic_match missing from breakdown. Components: {components}"
        )

        # chapter_source_consistency should fire (same source, same chapter, prev match)
        assert 'chapter_source_consistency' in components, (
            f"chapter_source_consistency missing from breakdown. Components: {components}"
        )

        # Verify final confidence differs from base similarity
        base_similarity = 0.88
        final_confidence = result.primary_match.confidence
        assert final_confidence != base_similarity, (
            f"Final confidence ({final_confidence}) should differ from base similarity ({base_similarity})"
        )

    @pytest.mark.fast
    def test_chapter_coherence_penalty_in_breakdown(self):
        """Verify chapter_coherence_penalty appears when source count exceeds threshold."""
        shared_text = "renewable energy solar panels climate"

        video_metadata = {
            "vid_001": {
                "title": "solar guide",
                "description": "energy solar overview",
                "tags": ["solar"],
            },
        }

        matcher = self._create_matcher(video_metadata=video_metadata)

        vo_seg = _make_vo_segment(shared_text, index=0, chapter_index=0)
        video_seg = _make_video_segment(
            source_file="vid_001",
            text="solar energy transcript",
            chapter_index=0,
            chapter_title="solar energy",
        )

        # 6 unique sources in chapter 0 -> exceeds threshold of 5, triggers penalty
        matcher._chapter_source_counts = {
            0: {"s1", "s2", "s3", "s4", "s5", "s6"}
        }

        candidates = [(video_seg, 0.85)]
        result = matcher.match_segment(vo_seg, candidates, scenes=None, segment_idx=0)

        components = {e['component'] for e in result.confidence_breakdown}
        assert 'chapter_coherence_penalty' in components, (
            f"chapter_coherence_penalty missing. Components: {components}"
        )

        # Verify the penalty is negative
        coherence_entries = [
            e for e in result.confidence_breakdown if e['component'] == 'chapter_coherence_penalty'
        ]
        assert len(coherence_entries) == 1
        assert coherence_entries[0]['adjustment'] < 0, (
            f"Coherence penalty should be negative, got {coherence_entries[0]['adjustment']}"
        )

    @pytest.mark.fast
    def test_cross_chapter_relevance_in_breakdown(self):
        """Verify cross_chapter_relevance appears when relevance_matrix has scores."""
        # relevance_matrix[0][0] = 0.7 -> boost = 0.7 * 0.1 = 0.07
        relevance_matrix = [[0.7]]

        matcher = self._create_matcher(relevance_matrix=relevance_matrix)

        vo_seg = _make_vo_segment("wildlife nature documentary", index=0, chapter_index=0)
        video_seg = _make_video_segment(
            source_file="vid_002",
            text="nature documentary footage",
            chapter_index=0,
            chapter_title="wildlife documentary",
        )

        candidates = [(video_seg, 0.82)]
        result = matcher.match_segment(vo_seg, candidates, scenes=None, segment_idx=0)

        components = {e['component'] for e in result.confidence_breakdown}
        assert 'cross_chapter_relevance' in components, (
            f"cross_chapter_relevance missing. Components: {components}"
        )

        # Verify boost is positive
        cross_entries = [
            e for e in result.confidence_breakdown if e['component'] == 'cross_chapter_relevance'
        ]
        assert len(cross_entries) == 1
        assert cross_entries[0]['adjustment'] > 0, (
            f"Cross-chapter relevance should be positive, got {cross_entries[0]['adjustment']}"
        )

    @pytest.mark.fast
    def test_adjustment_values_within_documented_ranges(self):
        """All adjustment values are within documented ranges (no unbounded boosts)."""
        shared_text = "renewable energy solar panels climate change"

        video_metadata = {
            "vid_001": {
                "title": "Guide to renewable energy solar panels",
                "description": "How renewable energy and solar panels help climate change",
                "tags": ["renewable", "energy", "solar", "panels", "climate"],
            },
        }
        relevance_matrix = [[0.9]]

        matcher = self._create_matcher(
            video_metadata=video_metadata,
            relevance_matrix=relevance_matrix,
        )

        # Seed chapter source counts so coherence penalty fires
        matcher._chapter_source_counts = {
            0: {"s1", "s2", "s3", "s4", "s5", "s6", "s7"}
        }

        vo_seg = _make_vo_segment(shared_text, index=0, chapter_index=0)
        video_seg = _make_video_segment(
            source_file="vid_001",
            text="transcript about renewable energy and solar",
            chapter_index=0,
            chapter_title="renewable energy solar overview",
        )

        candidates = [(video_seg, 0.85)]
        result = matcher.match_segment(vo_seg, candidates, scenes=None, segment_idx=0)

        # Documented ranges per adjustment type:
        RANGES = {
            'title_relevance': (-0.001, 0.09),        # max +0.08
            'description_relevance': (-0.001, 0.07),   # max +0.06
            'tag_keyword_boost': (-0.001, 0.09),       # max +0.08
            'chapter_topic_match': (-0.06, 0.11),      # -0.05 to +0.10
            'chapter_source_consistency': (-0.001, 0.04),  # max +0.03
            'chapter_coherence_penalty': (-0.11, 0.001),   # max -0.10
            'cross_chapter_relevance': (-0.001, 0.11),     # max relevance * 0.1
            'listicle_consistency': (-0.001, 0.05),        # max +0.04
            'topic_penalty': (-0.20, 0.001),               # max -0.15
            'broll_boost': (-0.001, 0.20),                 # depends on config
            'caption_quality': (-0.20, 0.20),              # depends on config
            'timing_penalty': (-0.20, 0.001),              # depends on config
            'project_boost': (-0.001, 0.20),               # depends on config
            'consecutive_source_penalty': (-0.20, 0.001),  # penalty
        }

        for entry in result.confidence_breakdown:
            comp = entry['component']
            adj = entry['adjustment']
            if comp in RANGES:
                lo, hi = RANGES[comp]
                assert lo <= adj <= hi, (
                    f"{comp} adjustment {adj} out of range [{lo}, {hi}]"
                )

        # Final confidence must be within [0, 1]
        assert 0.0 <= result.primary_match.confidence <= 1.0

    @pytest.mark.fast
    def test_listicle_consistency_in_breakdown(self):
        """Verify listicle_consistency boost fires when conditions are met."""
        listicle_groups = [ListicleGroup(start_segment_idx=0, end_segment_idx=5)]

        matcher = self._create_matcher(listicle_groups=listicle_groups)

        # Segment at index 1 (not boundary) in group [0..5]
        vo_seg = _make_vo_segment("energy solar climate", index=1, chapter_index=0)
        video_seg = _make_video_segment(
            source_file="vid_001",
            text="solar energy climate footage",
            chapter_index=0,
            chapter_title="solar climate",
        )

        # Seed a previous match from same source, in same listicle group
        prev_vo = _make_vo_segment("intro about solar", index=0, chapter_index=0)
        prev_match = Match(
            voiceover_segment=prev_vo,
            video_segment=video_seg,
            video_scene=None,
            confidence=0.80,
            reasoning="previous",
        )
        matcher._recent_matches = [prev_match]

        candidates = [(video_seg, 0.82)]
        result = matcher.match_segment(vo_seg, candidates, scenes=None, segment_idx=1)

        components = {e['component'] for e in result.confidence_breakdown}
        assert 'listicle_consistency' in components, (
            f"listicle_consistency missing. Components: {components}"
        )

        listicle_entries = [
            e for e in result.confidence_breakdown if e['component'] == 'listicle_consistency'
        ]
        assert len(listicle_entries) == 1
        assert listicle_entries[0]['adjustment'] > 0, (
            f"Listicle consistency boost should be positive, got {listicle_entries[0]['adjustment']}"
        )

    @pytest.mark.fast
    def test_confidence_differs_from_base_with_multiple_adjustments(self):
        """End-to-end: final confidence must differ from base embedding similarity."""
        shared_text = "renewable energy solar panels climate"

        video_metadata = {
            "vid_001": {
                "title": "renewable energy solar guide",
                "description": "solar panels for climate",
                "tags": ["renewable", "solar"],
            },
        }
        relevance_matrix = [[0.5]]
        listicle_groups = [ListicleGroup(start_segment_idx=0, end_segment_idx=3)]

        matcher = self._create_matcher(
            video_metadata=video_metadata,
            relevance_matrix=relevance_matrix,
            listicle_groups=listicle_groups,
        )

        # Seed previous match for consistency + listicle
        vo_seg = _make_vo_segment(shared_text, index=1, chapter_index=0)
        video_seg = _make_video_segment(
            source_file="vid_001",
            text="renewable energy solar transcript",
            chapter_index=0,
            chapter_title="renewable energy solar",
        )
        prev_vo = _make_vo_segment("intro renewable solar", index=0, chapter_index=0)
        prev_match = Match(
            voiceover_segment=prev_vo,
            video_segment=video_seg,
            video_scene=None,
            confidence=0.80,
            reasoning="prev",
        )
        matcher._recent_matches = [prev_match]

        # Seed chapter source counts to trigger coherence penalty
        matcher._chapter_source_counts = {
            0: {"s1", "s2", "s3", "s4", "s5", "s6"}
        }

        base_sim = 0.80
        candidates = [(video_seg, base_sim)]
        result = matcher.match_segment(vo_seg, candidates, scenes=None, segment_idx=1)

        # With title/desc/tag boosts + chapter adjustments, confidence must differ
        final = result.primary_match.confidence
        assert final != base_sim, (
            f"Final confidence ({final}) unchanged from base ({base_sim}) — "
            f"adjustments may not be wired. Breakdown: {result.confidence_breakdown}"
        )

        # Verify we have multiple adjustment components
        components = {e['component'] for e in result.confidence_breakdown}
        assert len(components) >= 3, (
            f"Expected at least 3 active adjustments, got {len(components)}: {components}"
        )
