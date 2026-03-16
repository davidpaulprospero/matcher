"""
Tests for chapter-level source diversity enforcement (US-77-007).

Verifies that the post-processing pass in TieredMatcher detects chapters
where all segments map to a single video source and swaps lowest-confidence
segments to alternatives from different sources.
"""
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import SRTSegment, Match, AlternativeMatch, MatchResult


pytestmark = pytest.mark.unit


def _make_vo_segment(index: int, chapter_index: int = 0) -> SRTSegment:
    """Create a voiceover segment with a chapter_index attribute."""
    seg = SRTSegment(
        index=index,
        start_time=float(index * 10),
        end_time=float(index * 10 + 9),
        text=f"Voiceover segment {index}",
        source_file=""
    )
    seg.chapter_index = chapter_index
    return seg


def _make_video_segment(index: int, source_file: str = "video_A") -> SRTSegment:
    return SRTSegment(
        index=index,
        start_time=0.0,
        end_time=10.0,
        text=f"Video segment from {source_file}",
        source_file=source_file
    )


def _make_match_result(
    vo_index: int,
    chapter_index: int,
    source_file: str,
    confidence: float,
    alt_sources: Optional[List[str]] = None
) -> MatchResult:
    """Create a MatchResult with primary match and optional alternatives."""
    vo_seg = _make_vo_segment(vo_index, chapter_index)
    vid_seg = _make_video_segment(vo_index, source_file)

    primary = Match(
        voiceover_segment=vo_seg,
        video_segment=vid_seg,
        video_scene=None,
        confidence=confidence,
        reasoning="test match",
        confidence_breakdown=[]
    )

    alternatives = []
    if alt_sources:
        for i, alt_src in enumerate(alt_sources):
            alt_vid = _make_video_segment(vo_index * 100 + i, alt_src)
            alt = AlternativeMatch(
                video_segment=alt_vid,
                video_scene=None,
                confidence=confidence - 0.05 * (i + 1),
                reasoning=f"alt from {alt_src}"
            )
            alternatives.append(alt)

    return MatchResult(
        primary_match=primary,
        alternatives=alternatives,
        confidence_breakdown=[]
    )


def _make_matcher(min_source_diversity: int = 2, enabled: bool = True):
    """Create a TieredMatcher with mocked config."""
    matcher = MagicMock()
    matcher.config = MagicMock()
    matcher.config.matching = MagicMock()
    chapter_grouping = MagicMock()
    chapter_grouping.enabled = enabled
    chapter_grouping.min_source_diversity = min_source_diversity
    matcher.config.matching.chapter_grouping = chapter_grouping

    # Import the actual method and bind it
    from src.matching.tiered_matcher import TieredMatcher
    matcher.enforce_chapter_source_diversity = TieredMatcher.enforce_chapter_source_diversity.__get__(matcher)
    return matcher


class TestChapterDiversityEnforcementTriggered:
    """Test that all-same-source chapters trigger re-matching."""

    @pytest.mark.fast
    def test_all_same_source_triggers_swap(self):
        """Chapter with >4 segments all from same source triggers diversity swap."""
        matcher = _make_matcher(min_source_diversity=2)

        # 5 segments in chapter 0, all from video_A, with alt from video_B
        results = [
            _make_match_result(i, chapter_index=0, source_file="video_A",
                             confidence=0.7 + i * 0.02, alt_sources=["video_B"])
            for i in range(5)
        ]

        matcher.enforce_chapter_source_diversity(results)

        # At least one segment should now be from video_B
        sources = {r.primary_match.video_segment.source_file for r in results}
        assert "video_B" in sources, "Expected at least one swap to video_B"
        assert len(sources) >= 2, "Expected at least 2 unique sources"

    @pytest.mark.fast
    def test_lowest_confidence_swapped_first(self):
        """The lowest-confidence segment should be swapped first."""
        matcher = _make_matcher(min_source_diversity=2)

        # 5 segments: index 0 has lowest confidence (0.50)
        results = [
            _make_match_result(0, chapter_index=0, source_file="video_A",
                             confidence=0.50, alt_sources=["video_B"]),
            _make_match_result(1, chapter_index=0, source_file="video_A",
                             confidence=0.80, alt_sources=["video_B"]),
            _make_match_result(2, chapter_index=0, source_file="video_A",
                             confidence=0.85, alt_sources=["video_B"]),
            _make_match_result(3, chapter_index=0, source_file="video_A",
                             confidence=0.90, alt_sources=["video_B"]),
            _make_match_result(4, chapter_index=0, source_file="video_A",
                             confidence=0.95, alt_sources=["video_B"]),
        ]

        matcher.enforce_chapter_source_diversity(results)

        # Segment 0 (lowest confidence) should be the one swapped
        assert results[0].primary_match.video_segment.source_file == "video_B"
        # Higher confidence ones should stay
        assert results[4].primary_match.video_segment.source_file == "video_A"

    @pytest.mark.fast
    def test_diversity_recheck_in_breakdown(self):
        """Swapped segments should have 'diversity_recheck' in confidence_breakdown."""
        matcher = _make_matcher(min_source_diversity=2)

        results = [
            _make_match_result(i, chapter_index=0, source_file="video_A",
                             confidence=0.7 + i * 0.02, alt_sources=["video_B"])
            for i in range(5)
        ]

        matcher.enforce_chapter_source_diversity(results)

        # Find the swapped segment(s) and check breakdown
        swapped = [r for r in results if r.primary_match.video_segment.source_file == "video_B"]
        assert len(swapped) >= 1

        for r in swapped:
            breakdown_components = [b['component'] for b in r.primary_match.confidence_breakdown]
            assert 'diversity_recheck' in breakdown_components, \
                f"Expected 'diversity_recheck' in breakdown, got {breakdown_components}"

            # Also check the MatchResult-level breakdown
            result_components = [b['component'] for b in r.confidence_breakdown]
            assert 'diversity_recheck' in result_components


class TestChapterDiversityNotTriggered:
    """Test that chapters with adequate diversity are left untouched."""

    @pytest.mark.fast
    def test_diverse_chapter_untouched(self):
        """Chapter already having 2+ sources is not modified."""
        matcher = _make_matcher(min_source_diversity=2)

        # 5 segments: mixed sources already
        results = [
            _make_match_result(0, chapter_index=0, source_file="video_A", confidence=0.80),
            _make_match_result(1, chapter_index=0, source_file="video_B", confidence=0.85),
            _make_match_result(2, chapter_index=0, source_file="video_A", confidence=0.75),
            _make_match_result(3, chapter_index=0, source_file="video_B", confidence=0.90),
            _make_match_result(4, chapter_index=0, source_file="video_A", confidence=0.70),
        ]

        # Record original sources
        original_sources = [r.primary_match.video_segment.source_file for r in results]

        matcher.enforce_chapter_source_diversity(results)

        # Should be unchanged
        current_sources = [r.primary_match.video_segment.source_file for r in results]
        assert current_sources == original_sources

    @pytest.mark.fast
    def test_small_chapter_untouched(self):
        """Chapter with <=4 segments is not checked (even if all same source)."""
        matcher = _make_matcher(min_source_diversity=2)

        # Only 4 segments — below the >4 threshold
        results = [
            _make_match_result(i, chapter_index=0, source_file="video_A",
                             confidence=0.80, alt_sources=["video_B"])
            for i in range(4)
        ]

        original_sources = [r.primary_match.video_segment.source_file for r in results]

        matcher.enforce_chapter_source_diversity(results)

        current_sources = [r.primary_match.video_segment.source_file for r in results]
        assert current_sources == original_sources

    @pytest.mark.fast
    def test_no_chapter_index_untouched(self):
        """Segments without chapter_index are not grouped or modified."""
        matcher = _make_matcher(min_source_diversity=2)

        # Segments with no chapter_index
        results = []
        for i in range(6):
            r = _make_match_result(i, chapter_index=0, source_file="video_A",
                                  confidence=0.80, alt_sources=["video_B"])
            # Remove chapter_index
            r.primary_match.voiceover_segment.chapter_index = -1
            results.append(r)

        original_sources = [r.primary_match.video_segment.source_file for r in results]

        matcher.enforce_chapter_source_diversity(results)

        current_sources = [r.primary_match.video_segment.source_file for r in results]
        assert current_sources == original_sources


class TestChapterDiversityConfigDisable:
    """Test that min_source_diversity=1 disables the check."""

    @pytest.mark.fast
    def test_min_diversity_1_disables(self):
        """Setting min_source_diversity=1 skips the entire check."""
        matcher = _make_matcher(min_source_diversity=1)

        results = [
            _make_match_result(i, chapter_index=0, source_file="video_A",
                             confidence=0.80, alt_sources=["video_B"])
            for i in range(6)
        ]

        original_sources = [r.primary_match.video_segment.source_file for r in results]

        matcher.enforce_chapter_source_diversity(results)

        current_sources = [r.primary_match.video_segment.source_file for r in results]
        assert current_sources == original_sources

    @pytest.mark.fast
    def test_chapter_grouping_disabled(self):
        """Setting chapter_grouping.enabled=False skips the check."""
        matcher = _make_matcher(min_source_diversity=2, enabled=False)

        results = [
            _make_match_result(i, chapter_index=0, source_file="video_A",
                             confidence=0.80, alt_sources=["video_B"])
            for i in range(6)
        ]

        original_sources = [r.primary_match.video_segment.source_file for r in results]

        matcher.enforce_chapter_source_diversity(results)

        current_sources = [r.primary_match.video_segment.source_file for r in results]
        assert current_sources == original_sources

    @pytest.mark.fast
    def test_no_alternatives_no_crash(self):
        """If no alternatives exist, the method doesn't crash — just skips."""
        matcher = _make_matcher(min_source_diversity=2)

        # 5 segments, all same source, NO alternatives
        results = [
            _make_match_result(i, chapter_index=0, source_file="video_A",
                             confidence=0.80, alt_sources=None)
            for i in range(5)
        ]

        # Should not raise
        matcher.enforce_chapter_source_diversity(results)

        # All still video_A since no alternatives available
        sources = {r.primary_match.video_segment.source_file for r in results}
        assert sources == {"video_A"}


class TestChapterDiversityConfigField:
    """Test that the config field exists and has correct default."""

    @pytest.mark.fast
    def test_config_field_exists(self):
        """ChapterGroupingConfig has min_source_diversity field."""
        from src.config.sections.matching import ChapterGroupingConfig
        config = ChapterGroupingConfig()
        assert hasattr(config, 'min_source_diversity')

    @pytest.mark.fast
    def test_config_default_value(self):
        """Default min_source_diversity is 2."""
        from src.config.sections.matching import ChapterGroupingConfig
        config = ChapterGroupingConfig()
        assert config.min_source_diversity == 2

    @pytest.mark.fast
    def test_config_value_1_disables(self):
        """min_source_diversity=1 means no minimum (disabled)."""
        from src.config.sections.matching import ChapterGroupingConfig
        config = ChapterGroupingConfig(min_source_diversity=1)
        assert config.min_source_diversity == 1
