"""
Tests for post-edit feedback learning (US-009, Sprint 22).

Covers:
- AC1: analyze_feedback() adjusts confidence weights from editor corrections
- AC2: analyze_feedback() aggregates batch corrections correctly
- AC3: analyze_feedback() handles inconsistent user edits without crash
- AC4: Feedback validation rejects corrections outside valid range
- AC5: Confidence adjustment persists across pipeline restarts
"""

import json
import pytest
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, patch
from dataclasses import asdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feedback_learning import (
    FeedbackEntry,
    FeedbackBatch,
    FeedbackAnalysis,
    FeedbackLearner,
    analyze_feedback,
    validate_feedback_entry,
    FeedbackValidationError,
)


# ============================================================================
# AC1: analyze_feedback() adjusts confidence weights from editor corrections
# ============================================================================

class TestAnalyzeFeedbackConfidenceAdjustment:
    """AC1: Test that analyze_feedback() adjusts confidence weights from corrections."""

    @pytest.mark.fast
    def test_analyze_feedback_exists_and_callable(self):
        """analyze_feedback() should be importable and callable."""
        assert callable(analyze_feedback)

    @pytest.mark.fast
    def test_kept_clip_increases_weight_factor(self):
        """Clips kept by editor should increase weight for similar matches."""
        feedback = FeedbackBatch(entries=[
            FeedbackEntry(
                segment_id="S001",
                original_confidence=0.6,
                was_kept=True,
                matched_keywords=["travel", "beach"],
                video_source="youtube"
            )
        ])

        analysis = analyze_feedback(feedback)

        # Weight adjustment should be positive (increase confidence for similar matches)
        assert analysis.confidence_adjustment > 0
        assert analysis.keyword_boosts.get("travel", 0) > 0
        assert analysis.keyword_boosts.get("beach", 0) > 0

    @pytest.mark.fast
    def test_dropped_clip_decreases_weight_factor(self):
        """Clips dropped by editor should decrease weight for similar matches."""
        feedback = FeedbackBatch(entries=[
            FeedbackEntry(
                segment_id="S002",
                original_confidence=0.8,
                was_kept=False,
                matched_keywords=["nature", "forest"],
                video_source="youtube"
            )
        ])

        analysis = analyze_feedback(feedback)

        # Weight adjustment should be negative for dropped high-confidence clips
        # (confidence was high but editor rejected - suggests overconfidence)
        assert analysis.keyword_penalties.get("nature", 0) > 0
        assert analysis.keyword_penalties.get("forest", 0) > 0

    @pytest.mark.fast
    def test_mixed_feedback_produces_balanced_adjustment(self):
        """Mixed kept/dropped feedback produces balanced adjustments."""
        feedback = FeedbackBatch(entries=[
            FeedbackEntry(segment_id="S001", original_confidence=0.7, was_kept=True,
                          matched_keywords=["travel"], video_source="youtube"),
            FeedbackEntry(segment_id="S002", original_confidence=0.7, was_kept=False,
                          matched_keywords=["travel"], video_source="youtube"),
        ])

        analysis = analyze_feedback(feedback)

        # Mixed signals should produce smaller/zero adjustment
        assert abs(analysis.confidence_adjustment) < 0.1

    @pytest.mark.fast
    def test_adjustment_magnitude_proportional_to_confidence_delta(self):
        """Adjustments should be larger for bigger confidence mismatches."""
        # High confidence dropped = large adjustment
        feedback_high_dropped = FeedbackBatch(entries=[
            FeedbackEntry(segment_id="S001", original_confidence=0.95, was_kept=False,
                          matched_keywords=["test"], video_source="youtube")
        ])

        # Low confidence dropped = small adjustment (expected behavior)
        feedback_low_dropped = FeedbackBatch(entries=[
            FeedbackEntry(segment_id="S002", original_confidence=0.3, was_kept=False,
                          matched_keywords=["test"], video_source="youtube")
        ])

        analysis_high = analyze_feedback(feedback_high_dropped)
        analysis_low = analyze_feedback(feedback_low_dropped)

        # Dropping high-confidence should have bigger impact than dropping low-confidence
        assert abs(analysis_high.confidence_adjustment) > abs(analysis_low.confidence_adjustment)


# ============================================================================
# AC2: analyze_feedback() aggregates batch corrections correctly
# ============================================================================

class TestAnalyzeFeedbackBatchAggregation:
    """AC2: Test that analyze_feedback() aggregates batch corrections correctly."""

    @pytest.mark.fast
    def test_batch_with_multiple_entries(self):
        """Should process multiple feedback entries in a batch."""
        feedback = FeedbackBatch(entries=[
            FeedbackEntry(segment_id="S001", original_confidence=0.7, was_kept=True,
                          matched_keywords=["travel"], video_source="youtube"),
            FeedbackEntry(segment_id="S002", original_confidence=0.6, was_kept=True,
                          matched_keywords=["travel"], video_source="youtube"),
            FeedbackEntry(segment_id="S003", original_confidence=0.8, was_kept=True,
                          matched_keywords=["travel"], video_source="pexels"),
        ])

        analysis = analyze_feedback(feedback)

        assert analysis.total_entries == 3
        assert analysis.kept_count == 3
        assert analysis.dropped_count == 0

    @pytest.mark.fast
    def test_batch_aggregates_keyword_statistics(self):
        """Should aggregate keyword boost/penalty across all entries."""
        feedback = FeedbackBatch(entries=[
            FeedbackEntry(segment_id="S001", original_confidence=0.7, was_kept=True,
                          matched_keywords=["travel", "beach"], video_source="youtube"),
            FeedbackEntry(segment_id="S002", original_confidence=0.6, was_kept=True,
                          matched_keywords=["travel", "mountain"], video_source="youtube"),
        ])

        analysis = analyze_feedback(feedback)

        # "travel" appears in both kept entries - should have strong boost
        assert "travel" in analysis.keyword_boosts
        # "beach" and "mountain" each appear once
        assert "beach" in analysis.keyword_boosts
        assert "mountain" in analysis.keyword_boosts

    @pytest.mark.fast
    def test_batch_calculates_source_preferences(self):
        """Should track video source preferences from feedback."""
        feedback = FeedbackBatch(entries=[
            FeedbackEntry(segment_id="S001", original_confidence=0.7, was_kept=True,
                          matched_keywords=["test"], video_source="youtube"),
            FeedbackEntry(segment_id="S002", original_confidence=0.7, was_kept=True,
                          matched_keywords=["test"], video_source="youtube"),
            FeedbackEntry(segment_id="S003", original_confidence=0.7, was_kept=False,
                          matched_keywords=["test"], video_source="pexels"),
        ])

        analysis = analyze_feedback(feedback)

        # YouTube kept 2/2, Pexels kept 0/1
        assert analysis.source_preferences.get("youtube", 0) > analysis.source_preferences.get("pexels", 0)

    @pytest.mark.fast
    def test_empty_batch_returns_neutral_analysis(self):
        """Empty batch should return neutral (no adjustments)."""
        feedback = FeedbackBatch(entries=[])

        analysis = analyze_feedback(feedback)

        assert analysis.total_entries == 0
        assert analysis.confidence_adjustment == 0.0
        assert len(analysis.keyword_boosts) == 0
        assert len(analysis.keyword_penalties) == 0

    @pytest.mark.fast
    def test_batch_calculates_average_confidence_by_outcome(self):
        """Should calculate average confidence for kept vs dropped."""
        feedback = FeedbackBatch(entries=[
            FeedbackEntry(segment_id="S001", original_confidence=0.9, was_kept=True,
                          matched_keywords=["test"], video_source="youtube"),
            FeedbackEntry(segment_id="S002", original_confidence=0.8, was_kept=True,
                          matched_keywords=["test"], video_source="youtube"),
            FeedbackEntry(segment_id="S003", original_confidence=0.4, was_kept=False,
                          matched_keywords=["test"], video_source="youtube"),
            FeedbackEntry(segment_id="S004", original_confidence=0.3, was_kept=False,
                          matched_keywords=["test"], video_source="youtube"),
        ])

        analysis = analyze_feedback(feedback)

        assert abs(analysis.avg_kept_confidence - 0.85) < 0.01  # (0.9 + 0.8) / 2
        assert abs(analysis.avg_dropped_confidence - 0.35) < 0.01  # (0.4 + 0.3) / 2


# ============================================================================
# AC3: analyze_feedback() handles inconsistent user edits without crash
# ============================================================================

class TestAnalyzeFeedbackInconsistentEdits:
    """AC3: Test that analyze_feedback() handles inconsistent edits gracefully."""

    @pytest.mark.fast
    def test_same_segment_kept_and_dropped_different_sessions(self):
        """Handle same segment being kept in one session, dropped in another."""
        feedback = FeedbackBatch(entries=[
            FeedbackEntry(segment_id="S001", original_confidence=0.7, was_kept=True,
                          matched_keywords=["travel"], video_source="youtube"),
            FeedbackEntry(segment_id="S001", original_confidence=0.7, was_kept=False,
                          matched_keywords=["travel"], video_source="youtube"),
        ])

        # Should not raise
        analysis = analyze_feedback(feedback)

        assert analysis.inconsistent_entries == 1  # One segment with conflicting decisions

    @pytest.mark.fast
    def test_handles_none_keywords_gracefully(self):
        """Should handle entries with None keywords."""
        feedback = FeedbackBatch(entries=[
            FeedbackEntry(segment_id="S001", original_confidence=0.7, was_kept=True,
                          matched_keywords=None, video_source="youtube"),
        ])

        # Should not raise
        analysis = analyze_feedback(feedback)

        assert analysis.total_entries == 1

    @pytest.mark.fast
    def test_handles_empty_keywords_list(self):
        """Should handle entries with empty keywords list."""
        feedback = FeedbackBatch(entries=[
            FeedbackEntry(segment_id="S001", original_confidence=0.7, was_kept=True,
                          matched_keywords=[], video_source="youtube"),
        ])

        # Should not raise
        analysis = analyze_feedback(feedback)

        assert analysis.total_entries == 1
        assert len(analysis.keyword_boosts) == 0

    @pytest.mark.fast
    def test_handles_none_video_source(self):
        """Should handle entries with None video source."""
        feedback = FeedbackBatch(entries=[
            FeedbackEntry(segment_id="S001", original_confidence=0.7, was_kept=True,
                          matched_keywords=["test"], video_source=None),
        ])

        # Should not raise
        analysis = analyze_feedback(feedback)

        assert analysis.total_entries == 1
        assert "unknown" in analysis.source_preferences or len(analysis.source_preferences) == 0

    @pytest.mark.fast
    def test_handles_duplicate_segment_ids(self):
        """Should handle multiple entries for same segment (e.g., from different tracks)."""
        feedback = FeedbackBatch(entries=[
            FeedbackEntry(segment_id="S001", original_confidence=0.7, was_kept=True,
                          matched_keywords=["test"], video_source="youtube"),
            FeedbackEntry(segment_id="S001", original_confidence=0.6, was_kept=True,
                          matched_keywords=["test"], video_source="pexels"),
        ])

        # Should not raise - both from same segment, both kept
        analysis = analyze_feedback(feedback)

        assert analysis.total_entries == 2

    @pytest.mark.fast
    def test_handles_extreme_confidence_values(self):
        """Should handle edge case confidence values (0.0, 1.0)."""
        feedback = FeedbackBatch(entries=[
            FeedbackEntry(segment_id="S001", original_confidence=0.0, was_kept=True,
                          matched_keywords=["test"], video_source="youtube"),
            FeedbackEntry(segment_id="S002", original_confidence=1.0, was_kept=False,
                          matched_keywords=["test"], video_source="youtube"),
        ])

        # Should not raise
        analysis = analyze_feedback(feedback)

        assert analysis.total_entries == 2


# ============================================================================
# AC4: Feedback validation rejects corrections outside valid range
# ============================================================================

class TestFeedbackValidation:
    """AC4: Test that feedback validation rejects invalid corrections."""

    @pytest.mark.fast
    def test_confidence_below_zero_rejected(self):
        """Should reject confidence values below 0."""
        with pytest.raises(FeedbackValidationError) as exc_info:
            validate_feedback_entry(FeedbackEntry(
                segment_id="S001",
                original_confidence=-0.1,
                was_kept=True,
                matched_keywords=["test"],
                video_source="youtube"
            ))

        assert "confidence" in str(exc_info.value).lower()

    @pytest.mark.fast
    def test_confidence_above_one_rejected(self):
        """Should reject confidence values above 1."""
        with pytest.raises(FeedbackValidationError) as exc_info:
            validate_feedback_entry(FeedbackEntry(
                segment_id="S001",
                original_confidence=1.1,
                was_kept=True,
                matched_keywords=["test"],
                video_source="youtube"
            ))

        assert "confidence" in str(exc_info.value).lower()

    @pytest.mark.fast
    def test_empty_segment_id_rejected(self):
        """Should reject empty segment ID."""
        with pytest.raises(FeedbackValidationError) as exc_info:
            validate_feedback_entry(FeedbackEntry(
                segment_id="",
                original_confidence=0.7,
                was_kept=True,
                matched_keywords=["test"],
                video_source="youtube"
            ))

        assert "segment_id" in str(exc_info.value).lower()

    @pytest.mark.fast
    def test_none_segment_id_rejected(self):
        """Should reject None segment ID."""
        with pytest.raises(FeedbackValidationError) as exc_info:
            validate_feedback_entry(FeedbackEntry(
                segment_id=None,
                original_confidence=0.7,
                was_kept=True,
                matched_keywords=["test"],
                video_source="youtube"
            ))

        assert "segment_id" in str(exc_info.value).lower()

    @pytest.mark.fast
    def test_valid_entry_passes_validation(self):
        """Should accept valid feedback entries."""
        entry = FeedbackEntry(
            segment_id="S001",
            original_confidence=0.7,
            was_kept=True,
            matched_keywords=["travel", "beach"],
            video_source="youtube"
        )

        # Should not raise
        validate_feedback_entry(entry)

    @pytest.mark.fast
    def test_boundary_confidence_values_accepted(self):
        """Should accept confidence exactly at 0.0 and 1.0."""
        entry_zero = FeedbackEntry(
            segment_id="S001", original_confidence=0.0, was_kept=True,
            matched_keywords=["test"], video_source="youtube"
        )
        entry_one = FeedbackEntry(
            segment_id="S002", original_confidence=1.0, was_kept=True,
            matched_keywords=["test"], video_source="youtube"
        )

        # Should not raise
        validate_feedback_entry(entry_zero)
        validate_feedback_entry(entry_one)


# ============================================================================
# AC5: Confidence adjustment persists across pipeline restarts
# ============================================================================

class TestFeedbackPersistence:
    """AC5: Test that confidence adjustments persist across restarts."""

    @pytest.mark.fast
    def test_feedback_learner_saves_to_file(self, tmp_path):
        """FeedbackLearner should save learned adjustments to file."""
        cache_dir = tmp_path / ".cache" / "feedback"
        learner = FeedbackLearner(cache_dir=cache_dir)

        feedback = FeedbackBatch(entries=[
            FeedbackEntry(segment_id="S001", original_confidence=0.7, was_kept=True,
                          matched_keywords=["travel"], video_source="youtube"),
        ])

        learner.learn(feedback)
        learner.save()

        # Check file exists
        assert (cache_dir / "learned_adjustments.json").exists()

    @pytest.mark.fast
    def test_feedback_learner_loads_from_file(self, tmp_path):
        """FeedbackLearner should restore adjustments from saved file."""
        cache_dir = tmp_path / ".cache" / "feedback"

        # Create and save learner
        learner1 = FeedbackLearner(cache_dir=cache_dir)
        feedback = FeedbackBatch(entries=[
            FeedbackEntry(segment_id="S001", original_confidence=0.7, was_kept=True,
                          matched_keywords=["travel"], video_source="youtube"),
        ])
        learner1.learn(feedback)
        learner1.save()

        # Create new learner and load
        learner2 = FeedbackLearner(cache_dir=cache_dir)
        learner2.load()

        # Should have same adjustments
        assert learner2.keyword_boosts == learner1.keyword_boosts
        assert learner2.source_preferences == learner1.source_preferences

    @pytest.mark.fast
    def test_learner_accumulates_across_sessions(self, tmp_path):
        """Learner should accumulate feedback across multiple sessions."""
        cache_dir = tmp_path / ".cache" / "feedback"

        # Session 1: Learn travel preference
        learner1 = FeedbackLearner(cache_dir=cache_dir)
        learner1.learn(FeedbackBatch(entries=[
            FeedbackEntry(segment_id="S001", original_confidence=0.7, was_kept=True,
                          matched_keywords=["travel"], video_source="youtube"),
        ]))
        learner1.save()

        # Session 2: Load and learn beach preference
        learner2 = FeedbackLearner(cache_dir=cache_dir)
        learner2.load()
        learner2.learn(FeedbackBatch(entries=[
            FeedbackEntry(segment_id="S002", original_confidence=0.7, was_kept=True,
                          matched_keywords=["beach"], video_source="youtube"),
        ]))
        learner2.save()

        # Session 3: Both should be present
        learner3 = FeedbackLearner(cache_dir=cache_dir)
        learner3.load()

        assert "travel" in learner3.keyword_boosts
        assert "beach" in learner3.keyword_boosts

    @pytest.mark.fast
    def test_learner_handles_missing_cache_file(self, tmp_path):
        """Should handle missing cache file gracefully."""
        cache_dir = tmp_path / ".cache" / "feedback"

        learner = FeedbackLearner(cache_dir=cache_dir)

        # Should not raise on load with no file
        learner.load()

        # Should have empty adjustments
        assert len(learner.keyword_boosts) == 0

    @pytest.mark.fast
    def test_learner_handles_corrupted_cache_file(self, tmp_path):
        """Should handle corrupted cache file gracefully."""
        cache_dir = tmp_path / ".cache" / "feedback"
        cache_dir.mkdir(parents=True)

        # Write corrupted file
        (cache_dir / "learned_adjustments.json").write_text("not valid json!!!")

        learner = FeedbackLearner(cache_dir=cache_dir)

        # Should not raise on load, should reset to defaults
        learner.load()

        assert len(learner.keyword_boosts) == 0

    @pytest.mark.fast
    def test_saved_adjustments_are_json_serializable(self, tmp_path):
        """Saved adjustments should be valid JSON."""
        cache_dir = tmp_path / ".cache" / "feedback"
        learner = FeedbackLearner(cache_dir=cache_dir)

        feedback = FeedbackBatch(entries=[
            FeedbackEntry(segment_id="S001", original_confidence=0.7, was_kept=True,
                          matched_keywords=["travel", "beach"], video_source="youtube"),
        ])
        learner.learn(feedback)
        learner.save()

        # Read and parse
        content = (cache_dir / "learned_adjustments.json").read_text()
        data = json.loads(content)

        assert "keyword_boosts" in data
        assert "source_preferences" in data
        assert "total_feedback_entries" in data


# ============================================================================
# Integration tests
# ============================================================================

class TestFeedbackLearningIntegration:
    """Integration tests for feedback learning workflow."""

    @pytest.mark.fast
    def test_full_workflow(self, tmp_path):
        """Test complete workflow: feedback -> analysis -> learn -> save -> load -> apply."""
        cache_dir = tmp_path / ".cache" / "feedback"

        # Step 1: Create feedback batch from post-edit analysis
        feedback = FeedbackBatch(
            project_name="test_project",
            created_at=datetime.now().isoformat(),
            entries=[
                FeedbackEntry(segment_id="S001", original_confidence=0.6, was_kept=True,
                              matched_keywords=["travel", "beach"], video_source="youtube"),
                FeedbackEntry(segment_id="S002", original_confidence=0.8, was_kept=True,
                              matched_keywords=["travel", "mountain"], video_source="youtube"),
                FeedbackEntry(segment_id="S003", original_confidence=0.9, was_kept=False,
                              matched_keywords=["generic", "footage"], video_source="pexels"),
            ]
        )

        # Step 2: Analyze feedback
        analysis = analyze_feedback(feedback)

        assert analysis.total_entries == 3
        assert analysis.kept_count == 2
        assert analysis.dropped_count == 1

        # Step 3: Learn from feedback
        learner = FeedbackLearner(cache_dir=cache_dir)
        learner.learn(feedback)

        # Step 4: Save learned adjustments
        learner.save()

        # Step 5: New session - load and verify
        new_learner = FeedbackLearner(cache_dir=cache_dir)
        new_learner.load()

        # Should have learned travel is good
        assert new_learner.keyword_boosts.get("travel", 0) > 0

        # Should have learned generic/footage is bad
        assert new_learner.keyword_penalties.get("generic", 0) > 0 or \
               new_learner.keyword_penalties.get("footage", 0) > 0

    @pytest.mark.fast
    def test_apply_adjustments_to_confidence(self, tmp_path):
        """Test applying learned adjustments to modify confidence scores."""
        cache_dir = tmp_path / ".cache" / "feedback"
        learner = FeedbackLearner(cache_dir=cache_dir)

        # Learn from feedback
        feedback = FeedbackBatch(entries=[
            FeedbackEntry(segment_id="S001", original_confidence=0.7, was_kept=True,
                          matched_keywords=["travel"], video_source="youtube"),
            FeedbackEntry(segment_id="S002", original_confidence=0.7, was_kept=True,
                          matched_keywords=["travel"], video_source="youtube"),
        ])
        learner.learn(feedback)

        # Apply adjustment to new confidence
        original_confidence = 0.6
        keywords = ["travel", "vacation"]
        source = "youtube"

        adjusted = learner.apply_adjustment(original_confidence, keywords, source)

        # Should be boosted due to travel keyword preference
        assert adjusted >= original_confidence


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
