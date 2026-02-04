"""
Tests for ClipBudgetTracker in src/otio/tracks.py.

Verifies proactive clip budget estimation and auto-lite mode
that disables strategy tracks when clip count exceeds thresholds.
"""

import pytest
import logging
from dataclasses import dataclass, field
from typing import List

from src.otio.tracks import ClipBudgetTracker


@dataclass
class MockOutputConfig:
    """Mock output config for budget tracker tests."""
    include_alternatives: bool = True
    num_alternatives: int = 2
    include_strategy_tracks: bool = True
    strategy_tracks: List[str] = field(default_factory=lambda: [
        "embedding_diversity",
        "broll_only",
    ])


@dataclass
class MockConfig:
    """Mock pipeline config."""
    output: MockOutputConfig = field(default_factory=MockOutputConfig)


class TestClipBudgetTrackerEstimation:
    """Test clip count estimation logic."""

    def test_estimate_basic_10_tracks(self):
        """100 segments with all 10 tracks = 1000 clips."""
        tracker = ClipBudgetTracker(
            match_count=100,
            num_alternatives=2,      # V2-V3
            num_secondary=3,         # V4-V6
            strategy_track_names=["embedding_diversity", "broll_only"],  # V7-V8
            has_entity_images=True,  # V9
            has_entity_videos=True,  # V10
        )
        # 1 + 2 + 3 + 2 + 1 + 1 = 10 tracks * 100 = 1000
        assert tracker.estimate_clips() == 1000

    def test_estimate_minimal_tracks(self):
        """50 segments with only V1 = 50 clips."""
        tracker = ClipBudgetTracker(
            match_count=50,
            num_alternatives=0,
            num_secondary=0,
            strategy_track_names=[],
        )
        assert tracker.estimate_clips() == 50

    def test_estimate_no_entity_tracks(self):
        """200 segments without entity tracks = 200 * 8 = 1600."""
        tracker = ClipBudgetTracker(
            match_count=200,
            num_alternatives=2,
            num_secondary=3,
            strategy_track_names=["embedding_diversity", "broll_only"],
            has_entity_images=False,
            has_entity_videos=False,
        )
        # 1 + 2 + 3 + 2 = 8 tracks * 200 = 1600
        assert tracker.estimate_clips() == 1600

    def test_estimate_400_segments_all_tracks(self):
        """400 segments with all 10 tracks = 4000 clips (over both thresholds)."""
        tracker = ClipBudgetTracker(
            match_count=400,
            num_alternatives=2,
            num_secondary=3,
            strategy_track_names=["embedding_diversity", "broll_only"],
            has_entity_images=True,
            has_entity_videos=True,
        )
        assert tracker.estimate_clips() == 4000


class TestClipBudgetTrackerWarning:
    """Test warning threshold behavior (>=2500, <3000)."""

    def test_warning_logged_at_threshold(self, caplog):
        """When clips >= 2500 but < 3000, log warning but don't modify config."""
        tracker = ClipBudgetTracker(
            match_count=313,  # 313 * 8 = 2504
            num_alternatives=2,
            num_secondary=3,
            strategy_track_names=["embedding_diversity", "broll_only"],
        )
        config = MockConfig()

        with caplog.at_level(logging.WARNING):
            result = tracker.check_and_adjust(config)

        # Should NOT modify config
        assert config.output.include_strategy_tracks is True
        assert result == 2504
        assert any("approaching DaVinci limit" in msg for msg in caplog.messages)
        assert any("Consider disabling V4-V8" in msg for msg in caplog.messages)
        assert len(tracker.actions_taken) == 1
        assert "Warning" in tracker.actions_taken[0]

    def test_no_warning_below_threshold(self, caplog):
        """When clips < 2500, no warning and no config changes."""
        tracker = ClipBudgetTracker(
            match_count=100,
            num_alternatives=2,
            num_secondary=3,
            strategy_track_names=["embedding_diversity", "broll_only"],
        )
        config = MockConfig()

        with caplog.at_level(logging.WARNING):
            result = tracker.check_and_adjust(config)

        assert result == 800
        assert config.output.include_strategy_tracks is True
        assert len(tracker.actions_taken) == 0


class TestClipBudgetTrackerAutoDisable:
    """Test auto-disable behavior at error threshold (>=3000)."""

    def test_auto_disables_strategy_tracks_at_error_threshold(self, caplog):
        """400 segments * 10 tracks = 4000 -> auto-disable V7-V8."""
        tracker = ClipBudgetTracker(
            match_count=400,
            num_alternatives=2,
            num_secondary=3,
            strategy_track_names=["embedding_diversity", "broll_only"],
            has_entity_images=True,
            has_entity_videos=True,
        )
        config = MockConfig()
        assert config.output.include_strategy_tracks is True

        with caplog.at_level(logging.WARNING):
            result = tracker.check_and_adjust(config)

        # Strategy tracks should be disabled
        assert config.output.include_strategy_tracks is False
        # Recalculated: 400 * (1+2+3+0+1+1) = 400 * 8 = 3200
        # Wait - that's still over 3000. The tracker only disables strategy tracks once.
        # After disabling: 400 * (1+2+3+1+1) = 400 * 8 = 3200
        assert result == 3200
        assert any("Auto-disabling strategy tracks" in msg for msg in caplog.messages)
        assert len(tracker.actions_taken) == 1
        assert "Auto-disabled" in tracker.actions_taken[0]

    def test_auto_disable_reduces_clip_count(self):
        """Verify clip count drops after auto-disable of strategy tracks."""
        tracker = ClipBudgetTracker(
            match_count=375,  # 375 * 8 = 3000 (exactly at threshold)
            num_alternatives=2,
            num_secondary=3,
            strategy_track_names=["embedding_diversity", "broll_only"],
        )
        config = MockConfig()

        initial_estimate = tracker.estimate_clips()
        assert initial_estimate == 3000  # 375 * 8

        result = tracker.check_and_adjust(config)

        # After removing 2 strategy tracks: 375 * 6 = 2250
        assert result == 2250
        assert config.output.include_strategy_tracks is False

    def test_auto_disable_exactly_at_3000(self, caplog):
        """Exactly 3000 clips triggers auto-disable (>= threshold)."""
        tracker = ClipBudgetTracker(
            match_count=375,
            num_alternatives=2,
            num_secondary=3,
            strategy_track_names=["embedding_diversity", "broll_only"],
        )
        config = MockConfig()

        with caplog.at_level(logging.WARNING):
            tracker.check_and_adjust(config)

        assert config.output.include_strategy_tracks is False

    def test_actions_taken_property(self):
        """Verify actions_taken returns a copy of the internal list."""
        tracker = ClipBudgetTracker(
            match_count=400,
            num_alternatives=2,
            num_secondary=3,
            strategy_track_names=["embedding_diversity", "broll_only"],
        )
        config = MockConfig()
        tracker.check_and_adjust(config)

        actions = tracker.actions_taken
        assert isinstance(actions, list)
        assert len(actions) == 1
        # Modifying returned list shouldn't affect internal state
        actions.clear()
        assert len(tracker.actions_taken) == 1


class TestClipBudgetTrackerEdgeCases:
    """Edge cases and boundary conditions."""

    def test_zero_matches(self):
        """Zero matches = zero clips, no warnings."""
        tracker = ClipBudgetTracker(
            match_count=0,
            num_alternatives=2,
            num_secondary=3,
            strategy_track_names=["embedding_diversity", "broll_only"],
        )
        config = MockConfig()
        result = tracker.check_and_adjust(config)
        assert result == 0
        assert config.output.include_strategy_tracks is True

    def test_single_strategy_track(self):
        """Only one strategy track still gets disabled at threshold."""
        tracker = ClipBudgetTracker(
            match_count=430,  # 430 * 7 = 3010
            num_alternatives=2,
            num_secondary=3,
            strategy_track_names=["broll_only"],  # Only 1 strategy
        )
        config = MockConfig()

        result = tracker.check_and_adjust(config)

        assert config.output.include_strategy_tracks is False
        # After: 430 * 6 = 2580
        assert result == 2580

    def test_just_below_warning(self):
        """2499 clips should not trigger any action."""
        # Need: match_count * track_count = 2499
        # With 8 tracks: 312 * 8 = 2496 (close enough below)
        tracker = ClipBudgetTracker(
            match_count=312,
            num_alternatives=2,
            num_secondary=3,
            strategy_track_names=["embedding_diversity", "broll_only"],
        )
        config = MockConfig()
        result = tracker.check_and_adjust(config)

        assert result == 2496
        assert config.output.include_strategy_tracks is True
        assert len(tracker.actions_taken) == 0
