"""
Sprint 4 Quality Verification Test

Verifies that quality-related modules can be imported cleanly.
Focus area: quality (match confidence, scoring, matching strategies)

This test file verifies:
1. src/matching/scoring.py imports cleanly
2. src/matching/tiered_matcher.py imports cleanly
3. src/matching/strategies.py imports cleanly
4. Key classes and functions are accessible
"""

import sys
from pathlib import Path

import pytest

# Ensure src is in path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

pytestmark = pytest.mark.unit


class TestScoringModuleImports:
    """Test that src/matching/scoring.py imports cleanly."""

    def test_scoring_module_imports(self):
        """scoring.py can be imported without errors."""
        from src.matching import scoring
        assert scoring is not None

    def test_apply_duration_penalty_function_exists(self):
        """apply_duration_penalty function is accessible."""
        from src.matching.scoring import apply_duration_penalty
        assert callable(apply_duration_penalty)

    def test_apply_topic_penalty_function_exists(self):
        """apply_topic_penalty function is accessible."""
        from src.matching.scoring import apply_topic_penalty
        assert callable(apply_topic_penalty)

    def test_apply_broll_boost_function_exists(self):
        """apply_broll_boost function is accessible."""
        from src.matching.scoring import apply_broll_boost
        assert callable(apply_broll_boost)

    def test_apply_current_project_boost_function_exists(self):
        """apply_current_project_boost function is accessible."""
        from src.matching.scoring import apply_current_project_boost
        assert callable(apply_current_project_boost)


class TestTieredMatcherModuleImports:
    """Test that src/matching/tiered_matcher.py imports cleanly."""

    def test_tiered_matcher_module_imports(self):
        """tiered_matcher.py can be imported without errors."""
        from src.matching import tiered_matcher
        assert tiered_matcher is not None

    def test_tiered_matcher_class_exists(self):
        """TieredMatcher class is accessible."""
        from src.matching.tiered_matcher import TieredMatcher
        assert TieredMatcher is not None

    def test_tiered_matcher_is_class(self):
        """TieredMatcher is a class."""
        from src.matching.tiered_matcher import TieredMatcher
        assert isinstance(TieredMatcher, type)

    def test_tiered_matcher_has_match_method(self):
        """TieredMatcher has a match method."""
        from src.matching.tiered_matcher import TieredMatcher
        assert hasattr(TieredMatcher, 'match') or hasattr(TieredMatcher, 'match_segment')


class TestStrategiesModuleImports:
    """Test that src/matching/strategies.py imports cleanly."""

    def test_strategies_module_imports(self):
        """strategies.py can be imported without errors."""
        from src.matching import strategies
        assert strategies is not None

    def test_strategy_matcher_class_exists(self):
        """StrategyMatcher class is accessible."""
        from src.matching.strategies import StrategyMatcher
        assert StrategyMatcher is not None

    def test_strategy_matcher_is_class(self):
        """StrategyMatcher is a class."""
        from src.matching.strategies import StrategyMatcher
        assert isinstance(StrategyMatcher, type)


class TestMatchingPackageImports:
    """Test that src/matching package imports work correctly."""

    def test_matching_package_imports(self):
        """src.matching package can be imported."""
        from src import matching
        assert matching is not None

    def test_import_from_matching_init(self):
        """Key items are importable from matching package __init__."""
        # This tests that __init__.py properly exposes the public API
        from src.matching import scoring, strategies, tiered_matcher
        assert scoring is not None
        assert strategies is not None
        assert tiered_matcher is not None


class TestQualityRelatedImports:
    """Test imports of quality-related utilities used by matching modules."""

    def test_utils_match_classes_importable(self):
        """Match-related classes from utils are importable."""
        from src.utils import Match, AlternativeMatch, MatchResult
        assert Match is not None
        assert AlternativeMatch is not None
        assert MatchResult is not None

    def test_srt_segment_importable(self):
        """SRTSegment class is importable."""
        from src.utils import SRTSegment
        assert SRTSegment is not None

    def test_scene_info_importable(self):
        """SceneInfo class is importable."""
        from src.utils import SceneInfo
        assert SceneInfo is not None


class TestVerificationComplete:
    """Final verification marker for sprint 4."""

    def test_all_quality_modules_load(self):
        """All three quality modules can be imported in sequence."""
        from src.matching import scoring
        from src.matching import tiered_matcher
        from src.matching import strategies

        # Verify all are loaded
        assert scoring is not None
        assert tiered_matcher is not None
        assert strategies is not None

        # Sprint 4 quality verification complete
        assert True
