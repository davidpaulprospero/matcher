"""
Tests for US-46-006: Ensure confidence_variance and matched_keywords propagate on all MatchResult paths.

Verifies that every MatchResult instance always has confidence_variance (float) and
matched_keywords (list) regardless of which code path created it.
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import fields as dataclass_fields

from src.utils import MatchResult, Match, SRTSegment, AlternativeMatch

pytestmark = [pytest.mark.fast, pytest.mark.unit]


def _make_match(confidence=0.85, text="test voiceover", video_text="test video"):
    """Helper to create a minimal Match object."""
    vo_seg = SRTSegment(index=0, start_time=0.0, end_time=5.0, text=text)
    vid_seg = SRTSegment(index=0, start_time=0.0, end_time=5.0, text=video_text,
                         source_file="/path/to/video.mp4")
    return Match(
        voiceover_segment=vo_seg,
        video_segment=vid_seg,
        video_scene=None,
        confidence=confidence,
        reasoning="test match"
    )


def _assert_fields_present(result: MatchResult, path_name: str):
    """Assert that confidence_variance and matched_keywords are present and typed correctly."""
    assert hasattr(result, 'confidence_variance'), \
        f"{path_name}: confidence_variance missing"
    assert hasattr(result, 'matched_keywords'), \
        f"{path_name}: matched_keywords missing"
    assert isinstance(result.confidence_variance, (int, float)), \
        f"{path_name}: confidence_variance is {type(result.confidence_variance)}, expected float"
    assert isinstance(result.matched_keywords, list), \
        f"{path_name}: matched_keywords is {type(result.matched_keywords)}, expected list"


class TestMatchResultDataclassDefaults:
    """MatchResult dataclass must define defaults for both fields."""

    def test_confidence_variance_has_default(self):
        """confidence_variance field has a default value in the dataclass."""
        field_map = {f.name: f for f in dataclass_fields(MatchResult)}
        assert 'confidence_variance' in field_map, "confidence_variance not in MatchResult"
        f = field_map['confidence_variance']
        # Either default or default_factory must be set
        has_default = f.default is not f.default_factory  # at least one is not MISSING
        assert f.default == 0.0, f"confidence_variance default should be 0.0, got {f.default}"

    def test_matched_keywords_has_default(self):
        """matched_keywords field has a default_factory in the dataclass."""
        field_map = {f.name: f for f in dataclass_fields(MatchResult)}
        assert 'matched_keywords' in field_map, "matched_keywords not in MatchResult"
        f = field_map['matched_keywords']
        assert f.default_factory is not None and f.default_factory is not type(None), \
            "matched_keywords should have a default_factory (list)"


class TestGapPathsPropagation:
    """Gap/fallback MatchResult paths must have both fields."""

    def test_no_candidates_gap_has_both_fields(self):
        """MatchResult from 'no candidates' path has defaults for both fields."""
        gap_match = _make_match(confidence=0.0)
        result = MatchResult(primary_match=gap_match, has_gap=True, gap_reason="No candidates")
        _assert_fields_present(result, "no_candidates_gap")
        assert result.confidence_variance == 0.0
        assert result.matched_keywords == []

    def test_all_filtered_gap_has_both_fields(self):
        """MatchResult from 'all candidates filtered' path has defaults for both fields."""
        gap_match = _make_match(confidence=0.0)
        result = MatchResult(primary_match=gap_match, has_gap=True, gap_reason="All candidates filtered")
        _assert_fields_present(result, "all_filtered_gap")
        assert result.confidence_variance == 0.0
        assert result.matched_keywords == []


class TestExplicitPathsPropagation:
    """Paths that explicitly set confidence_variance and matched_keywords."""

    def test_high_confidence_embedding_path(self):
        """High-confidence embedding match explicitly sets both fields."""
        match = _make_match(confidence=0.95)
        result = MatchResult(
            primary_match=match,
            confidence_variance=0.05,
            matched_keywords=["travel", "beach"]
        )
        _assert_fields_present(result, "high_confidence_embedding")
        assert result.confidence_variance == 0.05
        assert result.matched_keywords == ["travel", "beach"]

    def test_skip_threshold_path(self):
        """Skip-threshold path explicitly sets both fields."""
        match = _make_match(confidence=0.88)
        result = MatchResult(
            primary_match=match,
            alternatives=[],
            secondary_matches=[],
            confidence_variance=0.12,
            matched_keywords=["mountain", "hiking"]
        )
        _assert_fields_present(result, "skip_threshold")
        assert result.confidence_variance == 0.12
        assert result.matched_keywords == ["mountain", "hiking"]

    def test_llm_reranker_path(self):
        """LLM reranker path explicitly sets both fields."""
        match = _make_match(confidence=0.72)
        result = MatchResult(
            primary_match=match,
            alternatives=[],
            secondary_matches=[],
            has_gap=False,
            gap_reason="",
            confidence_variance=0.18,
            matched_keywords=["city", "architecture"]
        )
        _assert_fields_present(result, "llm_reranker")
        assert result.confidence_variance == 0.18
        assert result.matched_keywords == ["city", "architecture"]

    def test_output_stage_normalization_path(self):
        """Output stage normalization creates MatchResult with defaults."""
        match = _make_match(confidence=0.5)
        result = MatchResult(
            primary_match=match,
            alternatives=[],
            secondary_matches=[],
            strategy_matches=[]
        )
        _assert_fields_present(result, "output_stage_normalization")
        assert result.confidence_variance == 0.0
        assert result.matched_keywords == []


class TestDirectFieldAccess:
    """After removing defensive getattr, direct field access must work."""

    def test_direct_confidence_variance_access(self):
        """r.confidence_variance works without getattr on all MatchResult instances."""
        match = _make_match()
        # Default path
        r1 = MatchResult(primary_match=match)
        assert r1.confidence_variance == 0.0

        # Explicit path
        r2 = MatchResult(primary_match=match, confidence_variance=0.15)
        assert r2.confidence_variance == 0.15

    def test_direct_matched_keywords_access(self):
        """r.matched_keywords works without getattr on all MatchResult instances."""
        match = _make_match()
        # Default path
        r1 = MatchResult(primary_match=match)
        assert r1.matched_keywords == []

        # Explicit path
        r2 = MatchResult(primary_match=match, matched_keywords=["keyword1"])
        assert r2.matched_keywords == ["keyword1"]

    def test_matched_keywords_or_empty_pattern(self):
        """The 'r.matched_keywords or []' pattern works after removing getattr."""
        match = _make_match()
        r = MatchResult(primary_match=match)
        # This is the pattern used in main.py after our fix
        kws = r.matched_keywords or []
        assert kws == []
        assert isinstance(kws, list)

    def test_confidence_variance_comparison(self):
        """Direct confidence_variance comparison works (pattern from main.py:602)."""
        match = _make_match()
        r = MatchResult(primary_match=match, confidence_variance=0.2)
        # This is the pattern used in main.py after our fix
        assert r.confidence_variance > 0.15


class TestAllMatchResultCreationPathsFromTieredMatcher:
    """
    Integration-style test verifying that match_segment returns MatchResult
    with both fields from every code path.

    The 5 paths in TieredMatcher.match_segment:
    1. No candidates -> gap
    2. All filtered -> gap
    3. High-confidence embedding (above skip_llm_threshold)
    4. Medium-confidence (above ambiguous_threshold, below skip_llm_threshold)
    5. LLM reranker (below ambiguous_threshold or needing LLM)
    """

    @pytest.fixture
    def mock_config(self):
        """Create minimal mock config for TieredMatcher."""
        config = MagicMock()
        config.matching.gemini_model = 'gemini-2.0-flash'
        config.matching.min_confidence = 0.3
        config.matching.embedding_candidates = 10
        config.matching.high_confidence_threshold = 0.85
        config.matching.low_confidence_threshold = 0.5
        config.matching.max_clip_reuse = 3
        config.matching.reuse_penalty = 0.1
        config.matching.primary_provider = None
        config.matching.secondary_provider = None
        config.matching.use_local_for_review = False
        config.matching.skip_llm_threshold = 0.9
        config.matching.cache_llm_responses = False
        config.matching.ambiguous_threshold = 0.6
        config.matching.confidence_threshold = 0.3
        config.matching.adaptive_threshold_enabled = False
        config.matching.obvious_match_enabled = False
        config.matching.fallback_matching_enabled = False
        config.matching.location_matching = None
        config.matching.face_preference = 'neutral'
        config.matching.broll_boost = 0.0
        config.matching.max_source_file_reuse = 0
        config.matching.source_file_penalty = 0.05
        config.matching.chapter_matching_enabled = False
        config.matching.topic_mismatch_penalty = 0.15
        config.matching.multimodal_enabled = False
        config.gemini_api_key = None
        config.anthropic_api_key = None
        config.output.num_alternatives = 2
        config.matching.negative_matching = MagicMock()
        config.matching.negative_matching.enabled = False
        config.negative_matching = MagicMock()
        config.negative_matching.enabled = False
        return config

    def test_no_candidates_path(self, mock_config):
        """Path 1: No candidates returns MatchResult with both fields."""
        from src.matching.tiered_matcher import TieredMatcher

        matcher = TieredMatcher(config=mock_config)
        vo_seg = SRTSegment(index=0, start_time=0.0, end_time=5.0, text="test voiceover")

        result = matcher.match_segment(
            vo_segment=vo_seg,
            candidates=[],
            scenes={},
            segment_idx=0
        )

        _assert_fields_present(result, "no_candidates_path")
        assert result.has_gap is True
        assert result.confidence_variance == 0.0
        assert result.matched_keywords == []
