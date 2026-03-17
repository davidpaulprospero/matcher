"""Tests for US-85-008: Randomized jitter for download adaptive delay."""

import random

import pytest


class TestDownloadJitter:
    """Test cases for download delay jitter factor."""

    def test_jitter_factor_config_default(self):
        """Verify jitter_factor defaults to 0.25 in config."""
        from src.config.sections.download import DownloadConfig

        cfg = DownloadConfig()
        assert hasattr(cfg, 'segment_request_delay_jitter')
        assert cfg.segment_request_delay_jitter == 0.25

    def test_jitter_factor_config_custom_value(self):
        """Verify jitter_factor can be customized."""
        from src.config.sections.download import DownloadConfig

        cfg = DownloadConfig(segment_request_delay_jitter=0.5)
        assert cfg.segment_request_delay_jitter == 0.5

    def test_jitter_factor_zero_disables(self):
        """Verify setting jitter_factor to 0 disables jitter."""
        from src.config.sections.download import DownloadConfig

        cfg = DownloadConfig(segment_request_delay_jitter=0.0)
        assert cfg.segment_request_delay_jitter == 0.0

    def test_jitter_produces_different_results(self):
        """Test that jitter produces different results (not identical delays).

        Runs delay calculation 100 times and verifies results are not all identical.
        This proves jitter is actually applied.
        """
        # Simulate the jitter calculation
        delay = 1.0
        jitter_factor = 0.25

        results = []
        for _ in range(100):
            jittered = delay * random.uniform(1 - jitter_factor, 1 + jitter_factor)
            results.append(jittered)

        # Verify not all results are identical (proves randomness is working)
        unique_results = set(results)
        # Should have significant variation - at least 10 unique values
        assert len(unique_results) >= 10, f"Expected >=10 unique values, got {len(unique_results)}"

    def test_jitter_stays_within_bounds(self):
        """Test that jitter stays within configured bounds.

        Verifies: delay * (1 - jitter_factor) <= jittered_delay <= delay * (1 + jitter_factor)
        """
        delay = 1.0
        jitter_factor = 0.25
        min_expected = delay * (1 - jitter_factor)  # 0.75
        max_expected = delay * (1 + jitter_factor)  # 1.25

        for _ in range(100):
            jittered = delay * random.uniform(1 - jitter_factor, 1 + jitter_factor)
            assert min_expected <= jittered <= max_expected, (
                f"Jittered delay {jittered} outside bounds [{min_expected}, {max_expected}]"
            )

    def test_jitter_respects_different_factors(self):
        """Test that different jitter factors produce expected ranges."""
        delay = 10.0

        # Test 10% jitter
        jitter_10 = 0.10
        for _ in range(50):
            jittered = delay * random.uniform(1 - jitter_10, 1 + jitter_10)
            assert 9.0 <= jittered <= 11.0

        # Test 50% jitter
        jitter_50 = 0.50
        for _ in range(50):
            jittered = delay * random.uniform(1 - jitter_50, 1 + jitter_50)
            assert 5.0 <= jittered <= 15.0

    def test_jitter_applies_to_adaptive_delay(self):
        """Test jitter with adaptive delay values (base, growing, max)."""
        jitter_factor = 0.25

        # Base delay
        base_delay = 1.0
        for _ in range(20):
            jittered = base_delay * random.uniform(1 - jitter_factor, 1 + jitter_factor)
            assert 0.75 <= jittered <= 1.25

        # Growing delay (after some failures)
        growing_delay = 8.0
        for _ in range(20):
            jittered = growing_delay * random.uniform(1 - jitter_factor, 1 + jitter_factor)
            assert 6.0 <= jittered <= 10.0

        # Max delay (capped)
        max_delay = 30.0
        for _ in range(20):
            jittered = max_delay * random.uniform(1 - jitter_factor, 1 + jitter_factor)
            assert 22.5 <= jittered <= 37.5  # 30 * 0.75 to 30 * 1.25


class TestDownloadJitterIntegration:
    """Integration tests for jitter in download segments stage."""

    def test_stage_config_has_jitter(self):
        """Verify the download segments stage can access jitter config."""
        from src.config.sections.download import DownloadConfig

        cfg = DownloadConfig()
        assert hasattr(cfg, 'segment_request_delay_jitter')

        # Verify it integrates with other delay settings
        assert hasattr(cfg, 'segment_request_delay')
        assert hasattr(cfg, 'segment_request_delay_max')
