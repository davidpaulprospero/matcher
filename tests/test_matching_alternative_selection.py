"""
Tests for alternative_selection module (V2-V6 track generation).

Covers AlternativeSelector.get_alternatives (V2-V3) and
AlternativeSelector.get_secondary_matches (V4-V6).
"""

import pytest
from dataclasses import dataclass, field
from typing import List, Optional

from src.matching.alternative_selection import (
    AlternativeSelectionConfig,
    AlternativeSelector,
)
from src.utils import AlternativeMatch, SRTSegment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seg(source: str, start: float = 0.0, end: float = 10.0, text: str = "") -> SRTSegment:
    """Create a minimal SRTSegment for testing."""
    return SRTSegment(
        index=0,
        start_time=start,
        end_time=end,
        text=text,
        source_file=source,
    )


def _candidates(*specs) -> list:
    """Build a candidate list from (source, similarity) or (source, similarity, start) tuples."""
    result = []
    for spec in specs:
        if len(spec) == 2:
            source, sim = spec
            start = 0.0
        else:
            source, sim, start = spec
        result.append((_seg(source, start=start), sim))
    return result


# ---------------------------------------------------------------------------
# AlternativeSelectionConfig
# ---------------------------------------------------------------------------

class TestAlternativeSelectionConfig:
    def test_defaults(self):
        cfg = AlternativeSelectionConfig()
        assert cfg.num_alternatives == 2
        assert cfg.num_secondary == 3

    def test_custom_values(self):
        cfg = AlternativeSelectionConfig(num_alternatives=4, num_secondary=5)
        assert cfg.num_alternatives == 4
        assert cfg.num_secondary == 5


# ---------------------------------------------------------------------------
# AlternativeSelector construction
# ---------------------------------------------------------------------------

class TestAlternativeSelectorConstruction:
    def test_from_config(self):
        cfg = AlternativeSelectionConfig(num_alternatives=3, num_secondary=4)
        selector = AlternativeSelector(cfg)
        assert selector.config.num_alternatives == 3
        assert selector.config.num_secondary == 4

    def test_from_output_config(self):
        """from_output_config reads num_alternatives from an output config object."""
        @dataclass
        class FakeOutputConfig:
            num_alternatives: int = 5

        selector = AlternativeSelector.from_output_config(FakeOutputConfig())
        assert selector.config.num_alternatives == 5
        # num_secondary is always fixed at 3
        assert selector.config.num_secondary == 3

    def test_from_output_config_missing_attr(self):
        """Falls back to default=2 when output config lacks num_alternatives."""
        selector = AlternativeSelector.from_output_config(object())
        assert selector.config.num_alternatives == 2


# ---------------------------------------------------------------------------
# get_alternatives (V2-V3)
# ---------------------------------------------------------------------------

class TestGetAlternatives:
    """Tests for V2-V3 alternative selection."""

    def _selector(self, num_alternatives: int = 2) -> AlternativeSelector:
        return AlternativeSelector(AlternativeSelectionConfig(num_alternatives=num_alternatives))

    # --- AC: alternatives do not duplicate the primary match ---

    def test_alternatives_exclude_primary_source(self):
        """Alternatives should prefer different sources from the primary match."""
        selector = self._selector()
        primary = _seg("videoA.mp4")
        candidates = _candidates(
            ("videoA.mp4", 0.9),
            ("videoB.mp4", 0.85),
            ("videoC.mp4", 0.8),
        )

        alts = selector.get_alternatives(candidates, scenes=None, primary_match=primary)

        sources = [a.video_segment.source_file for a in alts]
        assert "videoA.mp4" not in sources
        assert len(alts) == 2
        assert set(sources) == {"videoB.mp4", "videoC.mp4"}

    def test_alternatives_no_primary_still_works(self):
        """When primary_match is None, all candidates are eligible."""
        selector = self._selector()
        candidates = _candidates(("videoA.mp4", 0.9), ("videoB.mp4", 0.85))

        alts = selector.get_alternatives(candidates, scenes=None, primary_match=None)
        assert len(alts) == 2

    def test_alternatives_exclude_primary_source_in_first_pass(self):
        """First pass skips primary source; fallback fills remaining slots."""
        selector = self._selector(num_alternatives=3)
        primary = _seg("videoA.mp4", start=5.0)

        # Only source A available – first pass yields nothing, fallback fills
        candidates = [
            (_seg("videoA.mp4", start=0.0), 0.9),
            (_seg("videoA.mp4", start=10.0), 0.85),
            (_seg("videoA.mp4", start=15.0), 0.80),
        ]

        alts = selector.get_alternatives(candidates, scenes=None, primary_match=primary)

        # Fallback deduplicates within alternatives (source+start) but picks from same source
        assert len(alts) == 3
        # All come from fallback so get 0.9x penalty
        for alt in alts:
            assert alt.confidence < 0.9 * 1.01  # all penalised

    # --- AC: confidence above minimum threshold ---

    def test_alternatives_retain_candidate_confidence(self):
        """V2-V3 alternatives from first pass preserve original confidence."""
        selector = self._selector()
        candidates = _candidates(("videoB.mp4", 0.92), ("videoC.mp4", 0.88))

        alts = selector.get_alternatives(
            candidates, scenes=None, primary_match=_seg("videoA.mp4")
        )

        assert alts[0].confidence == 0.92
        assert alts[1].confidence == 0.88

    def test_fallback_alternatives_have_reduced_confidence(self):
        """Fallback alternatives (same source) get 0.9x penalty."""
        selector = self._selector(num_alternatives=2)
        primary = _seg("videoA.mp4")

        # Only videoA candidates → all go through fallback pass
        candidates = [
            (_seg("videoA.mp4", start=10.0), 0.80),
            (_seg("videoA.mp4", start=20.0), 0.70),
        ]

        alts = selector.get_alternatives(candidates, scenes=None, primary_match=primary)
        assert len(alts) == 2
        assert alts[0].confidence == pytest.approx(0.80 * 0.9)
        assert alts[1].confidence == pytest.approx(0.70 * 0.9)

    # --- AC: fewer candidates than requested ---

    def test_fewer_candidates_than_requested(self):
        """Should return as many as available without error."""
        selector = self._selector(num_alternatives=5)
        candidates = _candidates(("videoB.mp4", 0.9), ("videoC.mp4", 0.85))

        alts = selector.get_alternatives(
            candidates, scenes=None, primary_match=_seg("videoA.mp4")
        )
        assert len(alts) == 2  # only 2 available, not 5

    def test_empty_candidates(self):
        """Empty candidate list returns empty alternatives."""
        selector = self._selector()
        alts = selector.get_alternatives([], scenes=None, primary_match=_seg("videoA.mp4"))
        assert alts == []

    def test_single_candidate_same_as_primary(self):
        """Single candidate matching primary source goes to fallback pass."""
        selector = self._selector(num_alternatives=1)
        primary = _seg("videoA.mp4", start=0.0)
        candidates = [(_seg("videoA.mp4", start=10.0), 0.75)]

        alts = selector.get_alternatives(candidates, scenes=None, primary_match=primary)
        assert len(alts) == 1
        # Fallback penalty applied
        assert alts[0].confidence == pytest.approx(0.75 * 0.9)

    # --- Diversity: prefer different sources ---

    def test_prefers_diverse_sources(self):
        """First pass selects from different sources before falling back."""
        selector = self._selector(num_alternatives=2)
        primary = _seg("videoA.mp4")

        candidates = _candidates(
            ("videoB.mp4", 0.95),
            ("videoB.mp4", 0.93),  # same source
            ("videoC.mp4", 0.80),  # different source
        )

        alts = selector.get_alternatives(candidates, scenes=None, primary_match=primary)
        sources = [a.video_segment.source_file for a in alts]
        assert sources == ["videoB.mp4", "videoC.mp4"]

    # --- Reasoning strings ---

    def test_reasoning_includes_source_info(self):
        selector = self._selector()
        candidates = _candidates(("videoB.mp4", 0.9))
        alts = selector.get_alternatives(
            candidates, scenes=None, primary_match=_seg("videoA.mp4")
        )
        assert "different source" in alts[0].reasoning.lower() or "Alternative" in alts[0].reasoning

    # --- get_scene_fn callback ---

    def test_get_scene_fn_called(self):
        """Scene callback is invoked for each selected alternative."""
        selector = self._selector(num_alternatives=1)
        calls = []

        def fake_scene_fn(seg, scenes):
            calls.append(seg.source_file)
            return "fake_scene"

        candidates = _candidates(("videoB.mp4", 0.9))
        alts = selector.get_alternatives(
            candidates, scenes={"some": []}, primary_match=_seg("videoA.mp4"),
            get_scene_fn=fake_scene_fn,
        )
        assert len(calls) == 1
        assert alts[0].video_scene == "fake_scene"

    def test_no_scene_fn_gives_none(self):
        selector = self._selector(num_alternatives=1)
        candidates = _candidates(("videoB.mp4", 0.9))
        alts = selector.get_alternatives(
            candidates, scenes=None, primary_match=_seg("videoA.mp4"),
            get_scene_fn=None,
        )
        assert alts[0].video_scene is None


# ---------------------------------------------------------------------------
# get_secondary_matches (V4-V6)
# ---------------------------------------------------------------------------

class TestGetSecondaryMatches:
    """Tests for V4-V6 secondary match selection."""

    def _selector(self, num_secondary: int = 3) -> AlternativeSelector:
        return AlternativeSelector(AlternativeSelectionConfig(num_secondary=num_secondary))

    # --- AC: V4-V6 differ from primary selections ---

    def test_secondary_excludes_v1_v3_sources_first_pass(self):
        """First pass only picks sources not in excluded_video_files (V1-V3)."""
        selector = self._selector()
        excluded = {"videoA.mp4", "videoB.mp4", "videoC.mp4"}
        candidates = _candidates(
            ("videoA.mp4", 0.95),
            ("videoD.mp4", 0.90),
            ("videoE.mp4", 0.85),
            ("videoF.mp4", 0.80),
        )

        secondary = selector.get_secondary_matches(
            candidates, scenes=None, excluded_video_files=excluded,
        )

        sources = [s.video_segment.source_file for s in secondary]
        assert "videoA.mp4" not in sources
        assert set(sources) == {"videoD.mp4", "videoE.mp4", "videoF.mp4"}

    def test_secondary_do_not_duplicate_primary_segment(self):
        """Exact primary segment must not appear in secondary matches."""
        selector = self._selector(num_secondary=2)
        primary = _seg("videoA.mp4", start=5.0)
        excluded = {"videoA.mp4"}

        # Force third pass (no candidates outside excluded)
        candidates = [
            (_seg("videoA.mp4", start=5.0), 0.9),   # exact primary segment
            (_seg("videoA.mp4", start=15.0), 0.85),
            (_seg("videoA.mp4", start=25.0), 0.80),
        ]

        secondary = selector.get_secondary_matches(
            candidates, scenes=None, excluded_video_files=excluded,
            primary_segment=primary,
        )

        keys = [(s.video_segment.source_file, s.video_segment.start_time) for s in secondary]
        assert ("videoA.mp4", 5.0) not in keys

    # --- AC: diversity scoring ---

    def test_first_pass_diversity_different_sources(self):
        """First pass picks candidates from unique sources outside V1-V3."""
        selector = self._selector()
        excluded = {"videoA.mp4"}
        candidates = _candidates(
            ("videoB.mp4", 0.90),
            ("videoB.mp4", 0.88),  # same source
            ("videoC.mp4", 0.85),
            ("videoD.mp4", 0.80),
        )

        secondary = selector.get_secondary_matches(
            candidates, scenes=None, excluded_video_files=excluded,
        )

        sources = [s.video_segment.source_file for s in secondary]
        # First pass should pick one from B, C, D (all different)
        assert sources == ["videoB.mp4", "videoC.mp4", "videoD.mp4"]

    def test_second_pass_allows_same_source_within_v4_v6(self):
        """Second pass lets V4-V6 share sources when unique sources exhausted."""
        selector = self._selector(num_secondary=3)
        excluded = {"videoA.mp4"}

        # Only 2 unique sources available → need second pass for slot 3
        candidates = _candidates(
            ("videoB.mp4", 0.90, 0.0),
            ("videoC.mp4", 0.85, 0.0),
            ("videoB.mp4", 0.80, 10.0),  # same source, different segment
        )

        secondary = selector.get_secondary_matches(
            candidates, scenes=None, excluded_video_files=excluded,
        )

        assert len(secondary) == 3
        # Third match is from second pass (same source ok)
        assert secondary[2].confidence == pytest.approx(0.80 * 0.95)

    def test_third_pass_allows_same_video_as_primary(self):
        """Third pass allows V1-V3 sources but different segments."""
        selector = self._selector(num_secondary=1)
        primary = _seg("videoA.mp4", start=0.0)
        excluded = {"videoA.mp4"}

        # No candidates outside excluded → forces third pass
        candidates = [
            (_seg("videoA.mp4", start=10.0), 0.85),
        ]

        secondary = selector.get_secondary_matches(
            candidates, scenes=None, excluded_video_files=excluded,
            primary_segment=primary,
        )

        assert len(secondary) == 1
        assert secondary[0].confidence == pytest.approx(0.85 * 0.85)
        assert secondary[0].video_segment.start_time == 10.0

    # --- AC: confidence thresholds per pass ---

    def test_first_pass_full_confidence(self):
        """First-pass secondary matches retain full candidate confidence."""
        selector = self._selector(num_secondary=1)
        candidates = _candidates(("videoD.mp4", 0.91))

        secondary = selector.get_secondary_matches(
            candidates, scenes=None, excluded_video_files={"videoA.mp4"},
        )

        assert secondary[0].confidence == 0.91

    def test_second_pass_0_95_penalty(self):
        """Second-pass matches get 0.95x penalty."""
        selector = self._selector(num_secondary=2)
        excluded = {"videoA.mp4"}

        candidates = [
            (_seg("videoB.mp4", start=0.0), 0.90),
            (_seg("videoB.mp4", start=10.0), 0.80),  # same source → second pass
        ]

        secondary = selector.get_secondary_matches(
            candidates, scenes=None, excluded_video_files=excluded,
        )

        assert secondary[0].confidence == 0.90           # first pass
        assert secondary[1].confidence == pytest.approx(0.80 * 0.95)  # second pass

    def test_third_pass_0_85_penalty(self):
        """Third-pass matches get 0.85x penalty."""
        selector = self._selector(num_secondary=1)
        excluded = {"videoA.mp4"}
        primary = _seg("videoA.mp4", start=0.0)

        candidates = [(_seg("videoA.mp4", start=20.0), 1.0)]

        secondary = selector.get_secondary_matches(
            candidates, scenes=None, excluded_video_files=excluded,
            primary_segment=primary,
        )

        assert secondary[0].confidence == pytest.approx(1.0 * 0.85)

    # --- AC: fewer candidates than requested ---

    def test_fewer_candidates_returns_partial(self):
        selector = self._selector(num_secondary=3)
        candidates = _candidates(("videoD.mp4", 0.9))

        secondary = selector.get_secondary_matches(
            candidates, scenes=None, excluded_video_files={"videoA.mp4"},
        )

        assert len(secondary) == 1

    def test_empty_candidates(self):
        selector = self._selector()
        secondary = selector.get_secondary_matches(
            [], scenes=None, excluded_video_files={"videoA.mp4"},
        )
        assert secondary == []

    # --- Reasoning labels ---

    def test_reasoning_labels_position(self):
        """First secondary is 'Secondary Primary', rest are 'Secondary Alt N'."""
        selector = self._selector()
        candidates = _candidates(
            ("videoD.mp4", 0.9),
            ("videoE.mp4", 0.85),
            ("videoF.mp4", 0.80),
        )

        secondary = selector.get_secondary_matches(
            candidates, scenes=None, excluded_video_files={"videoA.mp4"},
        )

        assert "Secondary Primary" in secondary[0].reasoning
        assert "Secondary Alt 1" in secondary[1].reasoning
        assert "Secondary Alt 2" in secondary[2].reasoning

    # --- alt_segments exclusion in third pass ---

    def test_alt_segments_excluded_from_third_pass(self):
        """Segments used by V2-V3 are excluded from third-pass secondary matches."""
        selector = self._selector(num_secondary=1)
        primary = _seg("videoA.mp4", start=0.0)
        alt_seg = _seg("videoA.mp4", start=10.0)
        excluded = {"videoA.mp4"}

        candidates = [
            (_seg("videoA.mp4", start=10.0), 0.95),  # same as alt_seg → excluded
            (_seg("videoA.mp4", start=20.0), 0.85),   # different segment → allowed
        ]

        secondary = selector.get_secondary_matches(
            candidates, scenes=None, excluded_video_files=excluded,
            primary_segment=primary, alt_segments=[alt_seg],
        )

        assert len(secondary) == 1
        assert secondary[0].video_segment.start_time == 20.0

    # --- get_scene_fn callback ---

    def test_scene_fn_called_for_secondary(self):
        calls = []

        def fake_scene_fn(seg, scenes):
            calls.append(seg.source_file)
            return "scene_data"

        selector = self._selector(num_secondary=1)
        candidates = _candidates(("videoD.mp4", 0.9))

        secondary = selector.get_secondary_matches(
            candidates, scenes={"x": []}, excluded_video_files={"videoA.mp4"},
            get_scene_fn=fake_scene_fn,
        )

        assert len(calls) == 1
        assert secondary[0].video_scene == "scene_data"
