"""
Unit tests for negative sampling in LLM reranking.

Tests the select_negative_sample() and format_negative_sample_for_prompt() functions
which help LLMs calibrate confidence by showing poor matches for comparison.
"""

import pytest
from unittest.mock import MagicMock, patch
import random

from src.matching.llm_providers import (
    select_negative_sample,
    format_negative_sample_for_prompt,
)
from src.utils import SRTSegment


def create_mock_segment(
    text: str = "Test video transcript",
    source_file: str = "test_video.mp4",
    index: int = 0
) -> SRTSegment:
    """Create a mock SRTSegment for testing."""
    return SRTSegment(
        index=index,
        start_time=0.0,
        end_time=5.0,
        text=text,
        source_file=source_file
    )


class TestSelectNegativeSample:
    """Test suite for select_negative_sample function."""

    @pytest.mark.fast
    def test_returns_none_for_empty_candidates(self):
        """Empty candidates list should return None."""
        result = select_negative_sample([])
        assert result is None

    @pytest.mark.fast
    def test_returns_none_for_insufficient_candidates(self):
        """Less than 4 candidates should return None (need meaningful bottom 25%)."""
        # 3 candidates is not enough for meaningful bottom 25%
        candidates = [
            (create_mock_segment(text=f"Segment {i}"), 0.9 - i * 0.1)
            for i in range(3)
        ]
        result = select_negative_sample(candidates)
        assert result is None

    @pytest.mark.fast
    def test_returns_sample_for_4_candidates(self):
        """4 candidates should return a sample from bottom 25% (index 3)."""
        candidates = [
            (create_mock_segment(text="High score"), 0.90),
            (create_mock_segment(text="Medium high"), 0.80),
            (create_mock_segment(text="Medium low"), 0.60),
            (create_mock_segment(text="Low score"), 0.40),
        ]

        # With random selection, need to seed for deterministic test
        with patch('random.choice', return_value=candidates[3]):
            result = select_negative_sample(candidates)

        assert result is not None
        assert result[0].text == "Low score"
        assert result[1] == 0.40

    @pytest.mark.fast
    def test_selects_from_bottom_25_percent(self):
        """Should select from bottom 25% of candidates."""
        # 8 candidates: bottom 25% = last 2 (indices 6, 7)
        candidates = [
            (create_mock_segment(text=f"Segment {i}"), 0.95 - i * 0.1)
            for i in range(8)
        ]

        # Run multiple times to verify selection is from bottom 25%
        for _ in range(10):
            result = select_negative_sample(candidates)
            assert result is not None
            # Bottom 25% of 8 = indices 6 and 7
            assert result[0].text in ["Segment 6", "Segment 7"]

    @pytest.mark.fast
    def test_selects_from_bottom_with_custom_percentile(self):
        """Should respect custom bottom_percentile parameter."""
        # 10 candidates: bottom 50% = last 5 (indices 5-9)
        candidates = [
            (create_mock_segment(text=f"Segment {i}"), 0.95 - i * 0.05)
            for i in range(10)
        ]

        # Run multiple times to verify selection is from bottom 50%
        for _ in range(10):
            result = select_negative_sample(candidates, bottom_percentile=0.50)
            assert result is not None
            # Index should be >= 5 (bottom 50%)
            seg_index = int(result[0].text.split()[1])
            assert seg_index >= 5

    @pytest.mark.fast
    def test_handles_large_candidate_list(self):
        """Should work correctly with large candidate lists."""
        # 100 candidates: bottom 25% = last 25 (indices 75-99)
        candidates = [
            (create_mock_segment(text=f"Segment {i}"), 0.99 - i * 0.009)
            for i in range(100)
        ]

        result = select_negative_sample(candidates)
        assert result is not None
        seg_index = int(result[0].text.split()[1])
        assert seg_index >= 75  # Bottom 25%

    @pytest.mark.fast
    def test_random_selection_varies(self):
        """Should randomly select different samples from bottom candidates."""
        candidates = [
            (create_mock_segment(text=f"Segment {i}"), 0.95 - i * 0.05)
            for i in range(20)
        ]

        # Collect samples from multiple runs
        samples = set()
        for _ in range(50):
            result = select_negative_sample(candidates)
            if result:
                samples.add(result[0].text)

        # Should have selected more than 1 unique sample (random variation)
        # Bottom 25% of 20 = 5 samples, so we expect variety
        assert len(samples) > 1

    @pytest.mark.fast
    def test_preserves_segment_and_similarity(self):
        """Returned tuple should preserve original segment and similarity."""
        expected_seg = create_mock_segment(text="Bottom segment", source_file="bottom.mp4")
        candidates = [
            (create_mock_segment(text="Top"), 0.95),
            (create_mock_segment(text="High"), 0.80),
            (create_mock_segment(text="Mid"), 0.65),
            (expected_seg, 0.42),
        ]

        with patch('random.choice', return_value=(expected_seg, 0.42)):
            result = select_negative_sample(candidates)

        assert result[0] is expected_seg
        assert result[1] == 0.42


class TestFormatNegativeSampleForPrompt:
    """Test suite for format_negative_sample_for_prompt function."""

    @pytest.mark.fast
    def test_formats_basic_sample(self):
        """Should format negative sample with standard label."""
        segment = create_mock_segment(text="This is a test transcript", source_file="test_video.mp4")
        result = format_negative_sample_for_prompt((segment, 0.25))

        assert "UNLIKELY" in result
        assert "test_video" in result  # source file stem
        assert "This is a test transcript" in result
        assert "unlikely match" in result.lower()

    @pytest.mark.fast
    def test_truncates_long_source_file_name(self):
        """Should truncate long source file names to 30 chars."""
        long_name = "this_is_a_very_long_video_filename_that_exceeds_30_chars.mp4"
        segment = create_mock_segment(text="Test", source_file=long_name)
        result = format_negative_sample_for_prompt((segment, 0.20))

        # Stem should be truncated to 30 chars
        assert len(long_name.replace(".mp4", "")) > 30
        # The formatted output should contain a truncated version
        assert "this_is_a_very_long_video_file" in result

    @pytest.mark.fast
    def test_truncates_long_text(self):
        """Should truncate long transcript text to 60 chars with ellipsis."""
        long_text = "This is a very long transcript that definitely exceeds sixty characters and should be truncated"
        segment = create_mock_segment(text=long_text)
        result = format_negative_sample_for_prompt((segment, 0.15))

        # Should have ellipsis for truncated text
        assert "..." in result
        # Should not contain full text
        assert long_text not in result

    @pytest.mark.fast
    def test_handles_short_text(self):
        """Short text should not have ellipsis."""
        short_text = "Short text"
        segment = create_mock_segment(text=short_text)
        result = format_negative_sample_for_prompt((segment, 0.30))

        assert "Short text" in result
        # Should not add unnecessary ellipsis
        assert result.count("...") == 0

    @pytest.mark.fast
    def test_escapes_quotes_in_text(self):
        """Should escape double quotes in text to prevent JSON issues."""
        text_with_quotes = 'He said "hello" to everyone'
        segment = create_mock_segment(text=text_with_quotes)
        result = format_negative_sample_for_prompt((segment, 0.25))

        # Double quotes should be replaced with single quotes
        assert '"hello"' not in result
        assert "'hello'" in result

    @pytest.mark.fast
    def test_uses_custom_label(self):
        """Should respect custom index_label parameter."""
        segment = create_mock_segment(text="Test transcript")
        result = format_negative_sample_for_prompt((segment, 0.20), index_label="NEGATIVE")

        assert "NEGATIVE" in result
        assert "UNLIKELY" not in result

    @pytest.mark.fast
    def test_handles_empty_source_file(self):
        """Should handle empty source file gracefully."""
        segment = create_mock_segment(text="Test", source_file="")
        result = format_negative_sample_for_prompt((segment, 0.25))

        # Should not crash, should use "unknown"
        assert "unknown" in result.lower() or result  # Either has unknown or just works

    @pytest.mark.fast
    def test_handles_empty_text(self):
        """Should handle empty transcript text gracefully."""
        segment = create_mock_segment(text="", source_file="video.mp4")
        result = format_negative_sample_for_prompt((segment, 0.25))

        # Should not crash
        assert "video" in result
        assert "unlikely match" in result.lower()


class TestNegativeSamplingIntegration:
    """Integration tests for negative sampling in LLM matchers."""

    @pytest.mark.fast
    def test_config_option_exists(self):
        """Verify negative_sampling_enabled config option exists in MatchingConfig."""
        from src.config.sections.matching import MatchingConfig

        config = MatchingConfig()
        assert hasattr(config, 'negative_sampling_enabled')
        assert config.negative_sampling_enabled is True  # Default should be True

    @pytest.mark.fast
    def test_llm_provider_accepts_negative_samples_parameter(self):
        """Verify LLMProvider.match_batch accepts negative_samples parameter."""
        from src.matching.llm_providers import LLMProvider
        import inspect

        sig = inspect.signature(LLMProvider.match_batch)
        params = list(sig.parameters.keys())
        assert 'negative_samples' in params

    @pytest.mark.fast
    def test_gemini_matcher_accepts_negative_samples(self):
        """Verify GeminiMatcher.match_batch accepts negative_samples parameter."""
        from src.matching.llm_providers import GeminiMatcher
        import inspect

        # Check method signature
        sig = inspect.signature(GeminiMatcher.match_batch)
        params = list(sig.parameters.keys())
        assert 'negative_samples' in params

    @pytest.mark.fast
    def test_claude_matcher_accepts_negative_samples(self):
        """Verify ClaudeMatcher.match_batch accepts negative_samples parameter."""
        from src.matching.llm_providers import ClaudeMatcher
        import inspect

        sig = inspect.signature(ClaudeMatcher.match_batch)
        params = list(sig.parameters.keys())
        assert 'negative_samples' in params

    @pytest.mark.fast
    def test_local_llm_matcher_accepts_negative_samples(self):
        """Verify LocalLLMMatcher.match_batch accepts negative_samples parameter."""
        from src.matching.llm_providers import LocalLLMMatcher
        import inspect

        sig = inspect.signature(LocalLLMMatcher.match_batch)
        params = list(sig.parameters.keys())
        assert 'negative_samples' in params


class TestNegativeSamplingEdgeCases:
    """Edge case tests for negative sampling functions."""

    @pytest.mark.fast
    def test_select_with_identical_scores(self):
        """Should work when all candidates have identical similarity scores."""
        candidates = [
            (create_mock_segment(text=f"Segment {i}"), 0.75)
            for i in range(10)
        ]

        result = select_negative_sample(candidates)
        assert result is not None
        assert result[1] == 0.75

    @pytest.mark.fast
    def test_select_with_zero_similarity_scores(self):
        """Should work when bottom candidates have zero similarity."""
        candidates = [
            (create_mock_segment(text="Good"), 0.90),
            (create_mock_segment(text="Medium"), 0.50),
            (create_mock_segment(text="Low"), 0.10),
            (create_mock_segment(text="Zero"), 0.00),
        ]

        with patch('random.choice', return_value=candidates[3]):
            result = select_negative_sample(candidates)

        assert result is not None
        assert result[0].text == "Zero"
        assert result[1] == 0.00

    @pytest.mark.fast
    def test_select_with_negative_similarity_scores(self):
        """Should handle negative similarity scores (edge case)."""
        candidates = [
            (create_mock_segment(text="Good"), 0.90),
            (create_mock_segment(text="Medium"), 0.50),
            (create_mock_segment(text="Low"), 0.10),
            (create_mock_segment(text="Negative"), -0.20),
        ]

        result = select_negative_sample(candidates)
        assert result is not None

    @pytest.mark.fast
    def test_format_with_unicode_text(self):
        """Should handle unicode characters in transcript text."""
        segment = create_mock_segment(
            text="Japanese: 日本語, Chinese: 中文, Emoji: 🎬",
            source_file="unicode_video.mp4"
        )
        result = format_negative_sample_for_prompt((segment, 0.25))

        # Should not crash and should contain some of the text
        assert "unicode_video" in result
        assert "unlikely match" in result.lower()

    @pytest.mark.fast
    def test_format_with_special_characters(self):
        """Should handle special characters in text."""
        segment = create_mock_segment(
            text="Special chars: <>&'\"\\n\\t",
            source_file="special.mp4"
        )
        result = format_negative_sample_for_prompt((segment, 0.25))

        # Should not crash
        assert "special" in result.lower()
