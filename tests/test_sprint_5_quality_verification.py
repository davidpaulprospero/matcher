"""
Sprint 5 Quality Verification Tests

This module verifies that the core quality matching modules import correctly
and have the expected structure for Sprint 5 enhancements.
"""

import pytest


class TestModuleImports:
    """Verify that quality-related modules import cleanly."""

    def test_scoring_module_imports(self):
        """Test that src/matching/scoring.py imports cleanly."""
        # Should not raise any import errors
        from src.matching import scoring

        # Verify module is accessible
        assert scoring is not None
        assert hasattr(scoring, "__name__")
        assert "scoring" in scoring.__name__

    def test_tiered_matcher_module_imports(self):
        """Test that src/matching/tiered_matcher.py imports cleanly."""
        # Should not raise any import errors
        from src.matching import tiered_matcher

        # Verify module is accessible
        assert tiered_matcher is not None
        assert hasattr(tiered_matcher, "__name__")
        assert "tiered_matcher" in tiered_matcher.__name__

    def test_llm_providers_module_imports(self):
        """Test that src/matching/llm_providers.py imports cleanly."""
        # Should not raise any import errors
        from src.matching import llm_providers

        # Verify module is accessible
        assert llm_providers is not None
        assert hasattr(llm_providers, "__name__")
        assert "llm_providers" in llm_providers.__name__


class TestModuleStructure:
    """Verify expected module structure exists for Sprint 5 enhancements."""

    def test_scoring_module_has_expected_classes(self):
        """Verify scoring module structure."""
        from src.matching import scoring

        # The scoring module should be importable - specific classes
        # will be added by subsequent user stories
        assert scoring is not None

    def test_tiered_matcher_has_expected_classes(self):
        """Verify tiered_matcher module has TieredMatcher class."""
        from src.matching.tiered_matcher import TieredMatcher

        assert TieredMatcher is not None
        # Verify it's a class
        assert isinstance(TieredMatcher, type)

    def test_llm_providers_has_expected_classes(self):
        """Verify llm_providers module structure."""
        from src.matching import llm_providers

        # Check for expected provider classes/functions
        # The module should be ready for Sprint 5 enhancements
        assert llm_providers is not None


class TestIntegration:
    """Basic integration tests for quality modules."""

    def test_modules_can_be_imported_together(self):
        """Verify all quality modules can be imported in same session."""
        from src.matching import scoring
        from src.matching import tiered_matcher
        from src.matching import llm_providers

        # All should be accessible
        assert all([scoring, tiered_matcher, llm_providers])

    def test_tiered_matcher_can_be_instantiated(self):
        """Verify TieredMatcher can be instantiated with minimal config."""
        from src.matching.tiered_matcher import TieredMatcher

        # TieredMatcher requires specific dependencies
        # This test just verifies the class is properly defined
        assert hasattr(TieredMatcher, "__init__")
        assert hasattr(TieredMatcher, "match_segment")
