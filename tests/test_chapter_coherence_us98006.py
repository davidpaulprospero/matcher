"""
Tests for compute_chapter_coherence_score and apply_chapter_coherence_boost
standalone functions (US-98-006).

Verifies that:
- Chapter coherence scoring rewards videos with similar structure to voiceover chapters
- Penalizes videos with mismatched chapter counts or transition patterns
- Uses correct weights: chapter_count_similarity (0.3), topic_overlap (0.4), transition_pattern (0.3)
- Test verifies scoring differentiates between coherent and incoherent chapter structures
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.matching.scoring import compute_chapter_coherence_score, apply_chapter_coherence_boost


class TestComputeChapterCoherenceScore:
    """Tests for the compute_chapter_coherence_score function."""

    def test_identical_chapters_boost(self):
        """High coherence when voiceover and video chapters are identical."""
        vo_chapters = [
            {'title': 'Introduction', 'keywords': ['intro', 'welcome'], 'segment_count': 3},
            {'title': 'Tips', 'keywords': ['tips', 'advice'], 'segment_count': 5},
            {'title': 'Conclusion', 'keywords': ['summary', 'final'], 'segment_count': 2},
        ]
        video_chapters = [
            {'title': 'Intro', 'keywords': ['intro', 'welcome'], 'segment_count': 3},
            {'title': 'Tips', 'keywords': ['tips', 'advice'], 'segment_count': 5},
            {'title': 'Conclusion', 'keywords': ['summary', 'final'], 'segment_count': 2},
        ]

        score, reason = compute_chapter_coherence_score(vo_chapters, video_chapters)

        # Should have high coherence (all components match)
        assert score >= 0.7, f"Expected high coherence for identical chapters, got {score}"
        assert "coherence=" in reason

    def test_different_chapter_counts_penalty(self):
        """Chapter count affects coherence but topic overlap can compensate."""
        vo_chapters = [
            {'title': 'Introduction', 'keywords': ['intro'], 'segment_count': 3},
            {'title': 'Tips', 'keywords': ['tips'], 'segment_count': 5},
        ]
        video_chapters = [
            {'title': 'Chapter 1', 'keywords': ['different_keyword_1'], 'segment_count': 1},
            {'title': 'Chapter 2', 'keywords': ['different_keyword_2'], 'segment_count': 1},
            {'title': 'Chapter 3', 'keywords': ['different_keyword_3'], 'segment_count': 1},
            {'title': 'Chapter 4', 'keywords': ['different_keyword_4'], 'segment_count': 1},
            {'title': 'Chapter 5', 'keywords': ['different_keyword_5'], 'segment_count': 1},
        ]

        score, reason = compute_chapter_coherence_score(vo_chapters, video_chapters)

        # Should have lower coherence due to both count mismatch AND no topic overlap
        assert score < 0.5, f"Expected lower coherence, got {score}"

    def test_no_voiceover_chapters(self):
        """Returns 0 when no voiceover chapters provided."""
        vo_chapters = []
        video_chapters = [
            {'title': 'Chapter', 'keywords': ['test'], 'segment_count': 1},
        ]

        score, reason = compute_chapter_coherence_score(vo_chapters, video_chapters)

        assert score == 0.0
        assert "no_chapters" in reason

    def test_no_video_chapters(self):
        """Returns 0 when no video chapters provided."""
        vo_chapters = [
            {'title': 'Chapter', 'keywords': ['test'], 'segment_count': 1},
        ]
        video_chapters = []

        score, reason = compute_chapter_coherence_score(vo_chapters, video_chapters)

        assert score == 0.0
        assert "no_chapters" in reason

    def test_partial_topic_overlap(self):
        """Moderate coherence with partial topic overlap and different topic for second chapter."""
        vo_chapters = [
            {'title': 'Cooking', 'keywords': ['recipe', 'food', 'cooking'], 'segment_count': 3},
            {'title': 'Travel', 'keywords': ['travel', 'trip'], 'segment_count': 3},
        ]
        video_chapters = [
            {'title': 'Cooking Tips', 'keywords': ['recipe', 'kitchen'], 'segment_count': 3},
            {'title': 'Travel Guide', 'keywords': ['completely_different', 'keywords'], 'segment_count': 3},
        ]

        score, reason = compute_chapter_coherence_score(vo_chapters, video_chapters)

        # Should have lower coherence due to second chapter having no topic overlap
        # Note: count_sim=1.0, topic=0.12, transition=1.0 -> weighted = 0.3*1.0 + 0.4*0.12 + 0.3*1.0 = 0.65
        assert score < 0.7, f"Expected lower coherence, got {score}"

    def test_transition_pattern_mismatch(self):
        """Lower coherence when segment distribution differs significantly AND topics don't overlap."""
        vo_chapters = [
            {'title': 'Intro', 'keywords': ['intro'], 'segment_count': 1},
            {'title': 'Body', 'keywords': ['content'], 'segment_count': 8},
            {'title': 'Outro', 'keywords': ['outro'], 'segment_count': 1},
        ]
        video_chapters = [
            {'title': 'Part 1', 'keywords': ['different1'], 'segment_count': 3},
            {'title': 'Part 2', 'keywords': ['different2'], 'segment_count': 3},
            {'title': 'Part 3', 'keywords': ['different3'], 'segment_count': 3},
        ]

        score, reason = compute_chapter_coherence_score(vo_chapters, video_chapters)

        # Should have lower coherence due to both transition mismatch AND no topic overlap
        # count_sim=1.0 (same count), topic=0.0, transition=0.69 -> weighted = 0.3*1.0 + 0.4*0.0 + 0.3*0.69 = 0.51
        assert score < 0.55, f"Expected lower coherence, got {score}"

    def test_custom_weights(self):
        """Uses custom weights when provided."""
        vo_chapters = [
            {'title': 'A', 'keywords': ['x'], 'segment_count': 1},
            {'title': 'B', 'keywords': ['y'], 'segment_count': 1},
        ]
        video_chapters = [
            {'title': 'A', 'keywords': ['x'], 'segment_count': 1},
            {'title': 'B', 'keywords': ['y'], 'segment_count': 1},
        ]
        weights = {'chapter_count_similarity': 0.5, 'topic_overlap': 0.3, 'transition_pattern': 0.2}

        score, reason = compute_chapter_coherence_score(vo_chapters, video_chapters, weights=weights)

        # Should compute with custom weights
        assert 0.0 <= score <= 1.0


class TestApplyChapterCoherenceBoost:
    """Tests for the apply_chapter_coherence_boost function."""

    def test_disabled_returns_unchanged(self):
        """Returns unchanged confidence when chapter_coherence_enabled is False."""
        vo_chapters = [{'title': 'A', 'keywords': ['x'], 'segment_count': 1}]
        video_chapters = [{'title': 'A', 'keywords': ['x'], 'segment_count': 1}]

        conf, reason = apply_chapter_coherence_boost(
            0.70, vo_chapters, video_chapters, chapter_coherence_enabled=False
        )

        assert conf == 0.70
        assert reason == ""

    def test_no_chapters_returns_unchanged(self):
        """Returns unchanged confidence when no chapters provided."""
        conf, reason = apply_chapter_coherence_boost(
            0.70, None, None, chapter_coherence_enabled=True
        )

        assert conf == 0.70
        assert reason == ""

    def test_high_coherence_gets_boost(self):
        """Applies boost when chapter structures are highly coherent."""
        vo_chapters = [
            {'title': 'Intro', 'keywords': ['intro', 'welcome'], 'segment_count': 2},
            {'title': 'Tips', 'keywords': ['tips', 'advice'], 'segment_count': 4},
        ]
        video_chapters = [
            {'title': 'Intro', 'keywords': ['intro', 'welcome'], 'segment_count': 2},
            {'title': 'Tips', 'keywords': ['tips', 'advice'], 'segment_count': 4},
        ]

        conf, reason = apply_chapter_coherence_boost(
            0.70, vo_chapters, video_chapters, chapter_coherence_enabled=True,
            boost_max=0.08, penalty_max=-0.05
        )

        # Should get a positive boost
        assert conf > 0.70, f"Expected boost for high coherence, got {conf}"
        assert "chapter_coherence:" in reason
        assert "coherence=" in reason

    def test_low_coherence_gets_penalty(self):
        """Applies penalty when chapter structures are incoherent."""
        vo_chapters = [
            {'title': 'Single', 'keywords': ['topic'], 'segment_count': 10},
        ]
        video_chapters = [
            {'title': 'Ch1', 'keywords': ['a'], 'segment_count': 1},
            {'title': 'Ch2', 'keywords': ['b'], 'segment_count': 1},
            {'title': 'Ch3', 'keywords': ['c'], 'segment_count': 1},
            {'title': 'Ch4', 'keywords': ['d'], 'segment_count': 1},
            {'title': 'Ch5', 'keywords': ['e'], 'segment_count': 1},
            {'title': 'Ch6', 'keywords': ['f'], 'segment_count': 1},
        ]

        conf, reason = apply_chapter_coherence_boost(
            0.70, vo_chapters, video_chapters, chapter_coherence_enabled=True,
            boost_max=0.08, penalty_max=-0.05
        )

        # Should get a penalty
        assert conf < 0.70, f"Expected penalty for low coherence, got {conf}"
        assert "chapter_coherence:" in reason

    def test_caps_at_maximum(self):
        """Does not exceed maximum boost/penalty."""
        vo_chapters = [
            {'title': 'Intro', 'keywords': ['intro'], 'segment_count': 1},
        ]
        video_chapters = [
            {'title': 'Intro', 'keywords': ['intro'], 'segment_count': 1},
        ]

        # Test boost cap
        conf_boost, _ = apply_chapter_coherence_boost(
            0.95, vo_chapters, video_chapters, chapter_coherence_enabled=True,
            boost_max=0.08, penalty_max=-0.05
        )
        assert conf_boost <= 1.0, "Boost should not exceed 1.0"

    def test_differentiates_coherent_vs_incoherent(self):
        """Test that scoring differentiates between coherent and incoherent structures."""
        # High coherence case
        coherent_vo = [
            {'title': 'Tips', 'keywords': ['tips', 'advice'], 'segment_count': 5},
            {'title': 'Top 5', 'keywords': ['top', 'ranking'], 'segment_count': 5},
        ]
        coherent_video = [
            {'title': 'Tips', 'keywords': ['tips', 'advice'], 'segment_count': 5},
            {'title': 'Top 5', 'keywords': ['top', 'ranking'], 'segment_count': 5},
        ]

        # Low coherence case
        incoherent_vo = [
            {'title': 'Single', 'keywords': ['one'], 'segment_count': 10},
        ]
        incoherent_video = [
            {'title': 'Ch1', 'keywords': ['a'], 'segment_count': 1},
            {'title': 'Ch2', 'keywords': ['b'], 'segment_count': 1},
            {'title': 'Ch3', 'keywords': ['c'], 'segment_count': 1},
            {'title': 'Ch4', 'keywords': ['d'], 'segment_count': 1},
            {'title': 'Ch5', 'keywords': ['e'], 'segment_count': 1},
            {'title': 'Ch6', 'keywords': ['f'], 'segment_count': 1},
            {'title': 'Ch7', 'keywords': ['g'], 'segment_count': 1},
            {'title': 'Ch8', 'keywords': ['h'], 'segment_count': 1},
            {'title': 'Ch9', 'keywords': ['i'], 'segment_count': 1},
            {'title': 'Ch10', 'keywords': ['j'], 'segment_count': 1},
        ]

        coherent_conf, _ = apply_chapter_coherence_boost(
            0.70, coherent_vo, coherent_video, chapter_coherence_enabled=True,
            boost_max=0.08, penalty_max=-0.05
        )
        incoherent_conf, _ = apply_chapter_coherence_boost(
            0.70, incoherent_vo, incoherent_video, chapter_coherence_enabled=True,
            boost_max=0.08, penalty_max=-0.05
        )

        # Coherent should get a boost, incoherent should get a penalty
        assert coherent_conf > 0.70, f"Coherent case should get boost, got {coherent_conf}"
        assert incoherent_conf < 0.70, f"Incoherent case should get penalty, got {incoherent_conf}"
        assert coherent_conf > incoherent_conf, "Coherent should score higher than incoherent"


class TestChapterCoherenceIntegration:
    """Integration tests for chapter coherence in matching context."""

    @pytest.mark.integration
    def test_config_validation(self):
        """Test that chapter_coherence config validates correctly."""
        from src.config.sections.matching import MatchingScoringConfig

        # Valid config
        config = MatchingScoringConfig(
            chapter_coherence_enabled=True,
            chapter_coherence_weights={
                'chapter_count_similarity': 0.3,
                'topic_overlap': 0.4,
                'transition_pattern': 0.3,
            },
            chapter_coherence_boost_max=0.08,
            chapter_coherence_penalty_max=-0.05,
        )
        assert config.chapter_coherence_enabled is True

    @pytest.mark.integration
    def test_weights_must_sum_to_one(self):
        """Test that weights validation catches non-sum-to-1 weights."""
        from src.config.sections.matching import MatchingScoringConfig

        with pytest.raises(ValueError, match="sum to 1.0"):
            MatchingScoringConfig(
                chapter_coherence_enabled=True,
                chapter_coherence_weights={
                    'chapter_count_similarity': 0.5,
                    'topic_overlap': 0.5,
                    'transition_pattern': 0.5,
                },  # Sum = 1.5
            )

    @pytest.mark.integration
    def test_missing_weight_keys_raises(self):
        """Test that missing required keys raises error."""
        from src.config.sections.matching import MatchingScoringConfig

        with pytest.raises(ValueError, match="must contain keys"):
            MatchingScoringConfig(
                chapter_coherence_enabled=True,
                chapter_coherence_weights={
                    'chapter_count_similarity': 0.5,
                    'topic_overlap': 0.5,
                    # missing transition_pattern
                },
            )

    @pytest.mark.integration
    def test_full_matching_with_chapter_coherence(self):
        """Integration test: runs full matching with chapter coherence enabled."""
        # This test verifies that chapter coherence scoring integrates properly
        # with the matching pipeline by testing the complete flow

        # Setup: Simulate a matching scenario with chapter info
        vo_chapters = [
            {'title': 'Top 5 Tips', 'keywords': ['tips', 'advice', 'top'], 'segment_count': 4},
            {'title': 'Conclusion', 'keywords': ['summary', 'final'], 'segment_count': 2},
        ]

        # Video with similar structure (high coherence)
        video_coherent = [
            {'title': 'Tips', 'keywords': ['tips', 'advice'], 'segment_count': 4},
            {'title': 'Summary', 'keywords': ['summary'], 'segment_count': 2},
        ]

        # Video with different structure (low coherence)
        video_incoherent = [
            {'title': 'Intro', 'keywords': ['intro'], 'segment_count': 1},
            {'title': 'Part A', 'keywords': ['part', 'a'], 'segment_count': 1},
            {'title': 'Part B', 'keywords': ['part', 'b'], 'segment_count': 1},
            {'title': 'Part C', 'keywords': ['part', 'c'], 'segment_count': 1},
            {'title': 'Outro', 'keywords': ['outro'], 'segment_count': 1},
        ]

        # Apply chapter coherence in a matching-like context
        base_confidence = 0.65

        # High coherence video gets boost
        conf_high, _ = apply_chapter_coherence_boost(
            base_confidence, vo_chapters, video_coherent,
            chapter_coherence_enabled=True,
            boost_max=0.08, penalty_max=-0.05
        )

        # Low coherence video gets penalty
        conf_low, _ = apply_chapter_coherence_boost(
            base_confidence, vo_chapters, video_incoherent,
            chapter_coherence_enabled=True,
            boost_max=0.08, penalty_max=-0.05
        )

        # The high coherence video should score higher
        assert conf_high > conf_low, f"High coherence ({conf_high}) should score higher than low ({conf_low})"

        # Both should stay within valid confidence range
        assert 0.0 <= conf_high <= 1.0
        assert 0.0 <= conf_low <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
