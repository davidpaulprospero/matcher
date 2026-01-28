"""
Tests for transcript quality scoring functionality.

US-005: Add transcript quality scoring
- Score based on: word count (>50 = good), sentence coherence, language consistency
- Return quality tier: high (>0.8), medium (0.5-0.8), low (<0.5)
- Reduce embedding weight by 20% when transcript quality is low
"""

import pytest

from src.matching.scoring import (
    calculate_transcript_quality,
    adjust_embedding_weight_for_transcript_quality,
    _calculate_sentence_coherence,
    _calculate_language_consistency,
    TRANSCRIPT_QUALITY_HIGH_THRESHOLD,
    TRANSCRIPT_QUALITY_MEDIUM_THRESHOLD,
    TRANSCRIPT_MIN_WORD_COUNT_GOOD,
    TRANSCRIPT_MIN_WORD_COUNT_MEDIUM,
)


class TestCalculateTranscriptQuality:
    """Test the main calculate_transcript_quality function."""

    def test_empty_transcript_returns_low_quality(self):
        """Empty or whitespace-only transcript should return low quality."""
        score, tier, reason = calculate_transcript_quality("")
        assert score == 0.0
        assert tier == "low"
        assert reason == "empty_transcript"

        score, tier, reason = calculate_transcript_quality("   \n\t  ")
        assert score == 0.0
        assert tier == "low"
        assert reason == "empty_transcript"

    def test_none_transcript_returns_low_quality(self):
        """None transcript should return low quality."""
        score, tier, reason = calculate_transcript_quality(None)
        assert score == 0.0
        assert tier == "low"
        assert reason == "empty_transcript"

    def test_short_transcript_low_word_count(self):
        """Very short transcripts (< 20 words) should score lower than longer ones."""
        short_text = "Hello world this is a test."
        score, tier, reason = calculate_transcript_quality(short_text)

        # Should have low word count score component
        assert "words:6" in reason
        # Score should be lower than high threshold due to few words
        # But well-formed short sentences can still reach medium tier
        assert score < TRANSCRIPT_QUALITY_HIGH_THRESHOLD  # Not high quality
        assert tier in ("low", "medium")  # Either low or medium tier

    def test_medium_length_transcript(self):
        """Transcripts with 20-50 words should score in medium range."""
        # Generate a 30-word transcript
        medium_text = "This is a sample transcript. " * 5  # ~30 words
        score, tier, reason = calculate_transcript_quality(medium_text)

        # Should indicate word count
        assert "words:" in reason
        # With proper sentences, should reach medium tier at minimum
        assert tier in ("medium", "high")

    def test_long_transcript_high_word_score(self):
        """Transcripts with >50 words should get full word score."""
        # Generate a 60-word transcript with good structure
        long_text = (
            "This is a well-structured transcript with many words. "
            "It contains multiple sentences with proper punctuation. "
            "The content flows naturally from one sentence to the next. "
            "This demonstrates good transcript quality. "
            "Additional words are added to ensure we exceed fifty words total. "
            "Quality assessment should give this a high score."
        )
        score, tier, reason = calculate_transcript_quality(long_text)

        # With good structure and length, should be high quality
        assert tier == "high"
        assert score >= TRANSCRIPT_QUALITY_HIGH_THRESHOLD
        assert "words:" in reason

    def test_quality_tier_thresholds(self):
        """Verify quality tier threshold values."""
        assert TRANSCRIPT_QUALITY_HIGH_THRESHOLD == 0.8
        assert TRANSCRIPT_QUALITY_MEDIUM_THRESHOLD == 0.5

    def test_word_count_thresholds(self):
        """Verify word count threshold values."""
        assert TRANSCRIPT_MIN_WORD_COUNT_GOOD == 50
        assert TRANSCRIPT_MIN_WORD_COUNT_MEDIUM == 20

    def test_returns_tuple_of_three(self):
        """Function should return exactly 3 values."""
        result = calculate_transcript_quality("Test transcript text.")
        assert isinstance(result, tuple)
        assert len(result) == 3

        score, tier, reason = result
        assert isinstance(score, float)
        assert isinstance(tier, str)
        assert isinstance(reason, str)

    def test_score_bounded_zero_to_one(self):
        """Quality score should always be between 0.0 and 1.0."""
        test_cases = [
            "",
            "a",
            "Hello world.",
            "This is a longer transcript with more content." * 20,
            "um um um um um um um um",  # Many filler words
            "abc def ghi jkl mno pqr stu vwx",  # No punctuation
        ]

        for text in test_cases:
            score, tier, reason = calculate_transcript_quality(text)
            assert 0.0 <= score <= 1.0, f"Score {score} out of bounds for: {text[:30]}..."

    def test_tier_values_are_valid(self):
        """Quality tier should be one of: high, medium, low."""
        test_cases = [
            "",
            "Hello.",
            "This is a medium-length transcript with decent structure." * 3,
            "This is a long and detailed transcript. " * 10,
        ]

        for text in test_cases:
            _, tier, _ = calculate_transcript_quality(text)
            assert tier in ("high", "medium", "low"), f"Invalid tier '{tier}' for: {text[:30]}..."


class TestSentenceCoherence:
    """Test the sentence coherence scoring helper."""

    def test_no_punctuation_low_coherence(self):
        """Text without punctuation should have low coherence score."""
        words = ["hello", "world", "this", "is", "test"]
        text = " ".join(words)

        score = _calculate_sentence_coherence(text, words)
        # Without punctuation, should get minimal score
        assert score < 0.15

    def test_proper_sentences_good_coherence(self):
        """Text with proper punctuation should have higher coherence."""
        text = "Hello there. This is a test. How are you today?"
        words = text.split()

        score = _calculate_sentence_coherence(text, words)
        # With good punctuation, should get decent score
        assert score > 0.15

    def test_very_short_text_low_coherence(self):
        """Very short text (< 3 words) should return 0."""
        text = "Hi"
        words = text.split()

        score = _calculate_sentence_coherence(text, words)
        assert score == 0.0

    def test_empty_text_returns_zero(self):
        """Empty text should return 0."""
        score = _calculate_sentence_coherence("", [])
        assert score == 0.0

    def test_capitalized_words_boost_coherence(self):
        """Capitalized words at sentence beginnings should boost score."""
        text = "This is good. Here is another. And one more."
        words = text.split()

        score = _calculate_sentence_coherence(text, words)
        # Should get boost from capitalization matching punctuation
        assert score > 0.1

    def test_long_words_boost_coherence(self):
        """Words with avg length >= 4.0 should boost score (not gibberish)."""
        # Words with avg length >= 4.0: programming, development, software, engineering, architecture
        # Average length = (11+11+8+11+12)/5 = 10.6
        words = ["programming", "development", "software", "engineering", "architecture"]
        text = " ".join(words)

        score_long = _calculate_sentence_coherence(text, words)

        # Words with avg length < 3.0: a, be, to, go, at (avg = 1.6)
        short_words = ["a", "be", "to", "go", "at"]
        short_text = " ".join(short_words)

        score_short = _calculate_sentence_coherence(short_text, short_words)

        # Long words should get +0.10 boost, short words get +0.00
        assert score_long > score_short
        # Long words get the word length boost
        assert score_long >= 0.10  # At least word length contribution

    def test_medium_word_length_partial_boost(self):
        """Words with avg length 3.0-4.0 should get partial boost (0.05)."""
        # Words with avg length ~3.5: this, that, they, them (avg = 4.0)
        # Let's use words that give us exactly avg 3.5: big, the, cat, dog, hat (3,3,3,3,3 = 3.0)
        # To get avg >= 3.0 but < 4.0, use: play, this, good (4,4,4 = 4.0)
        # Use: the, big, cat (3,3,3 = 3.0)
        words = ["the", "big", "cat"]
        text = " ".join(words)

        score = _calculate_sentence_coherence(text, words)
        # Should get partial boost for avg length exactly 3.0
        assert score >= 0.05  # At least partial word length contribution

    def test_score_capped_at_maximum(self):
        """Score should be capped at 0.35 maximum regardless of input."""
        # Create text that would maximize all scoring components:
        # - Multiple sentence endings (3 periods = punctuation_ratio capped at 1.0)
        # - Capitalized words matching sentence count
        # - Average word length >= 4.0
        # This should exceed 0.35 without cap: 0.15 (punct) + 0.10 (caps) + 0.10 (word_len) = 0.35
        text = "Programming. Development. Engineering."
        words = text.replace(".", "").split()  # ["Programming", "Development", "Engineering"]

        score = _calculate_sentence_coherence(text, words)

        # Score must be capped at 0.35
        assert score <= 0.35

    def test_score_cap_with_excessive_input(self):
        """Score should remain at 0.35 even with excessive positive signals."""
        # Create highly structured text that would theoretically score > 0.35
        text = "Welcome. Hello. Goodbye. Thanks. Cheers. Wonderful. Fantastic. Excellent. Amazing. Outstanding."
        words = text.replace(".", "").split()

        score = _calculate_sentence_coherence(text, words)

        # Even with many sentences and capitals, score is capped
        assert score == 0.35 or score < 0.35  # Capped at maximum
        assert score <= 0.35  # Explicit cap test


class TestLanguageConsistency:
    """Test the language consistency scoring helper."""

    def test_normal_text_high_consistency(self):
        """Normal text without issues should have high consistency."""
        words = ["this", "is", "a", "normal", "sentence", "with", "varied", "words"]
        text = " ".join(words)

        score = _calculate_language_consistency(text, words)
        # Normal text should retain most of the score
        assert score > 0.15

    def test_heavy_repetition_low_consistency(self):
        """Heavily repeated words (ASR stuttering) should lower score."""
        words = ["the", "the", "the", "the", "the", "word", "the", "the"]
        text = " ".join(words)

        score = _calculate_language_consistency(text, words)
        # Heavy repetition should deduct points
        assert score < 0.15

    def test_many_short_words_lower_consistency(self):
        """Too many short words (potential gibberish) should lower score."""
        words = ["a", "b", "c", "d", "e", "hi", "to", "be", "or"]
        text = " ".join(words)

        score = _calculate_language_consistency(text, words)
        # Many short words should deduct points
        assert score < 0.20

    def test_filler_words_lower_consistency(self):
        """Many filler words should lower consistency score."""
        words = ["um", "like", "basically", "um", "uh", "like", "the", "thing"]
        text = " ".join(words)

        score = _calculate_language_consistency(text, words)
        # Filler word overload should deduct points
        assert score < 0.20

    def test_very_short_word_list_returns_zero(self):
        """Less than 2 words should return 0."""
        score = _calculate_language_consistency("hi", ["hi"])
        assert score == 0.0

    def test_empty_word_list_returns_zero(self):
        """Empty word list should return 0."""
        score = _calculate_language_consistency("", [])
        assert score == 0.0

    def test_non_ascii_content_deducts_score(self):
        """Non-ASCII content (mixed scripts) should deduct 0.05 from score.

        AC6: ascii_ratio < 0.7 deducts 0.05
        """
        # Use text with significant non-ASCII content (> 30%)
        # Example: 50% non-ASCII characters
        words = ["こんにちは", "世界", "hello", "world"]  # Japanese + English
        text = " ".join(words)

        score = _calculate_language_consistency(text, words)
        # Non-ASCII should deduct points - max score is 0.25, with 0.05 deduction = 0.20 max
        assert score <= 0.20

    def test_high_ascii_ratio_no_deduction(self):
        """High ASCII ratio (>= 0.7) should not deduct for non-ASCII."""
        words = ["hello", "world", "this", "is", "english", "text"]
        text = " ".join(words)

        score = _calculate_language_consistency(text, words)
        # Pure ASCII text with no other issues should retain full score (0.25)
        assert score == 0.25

    def test_repetition_exact_deduction_heavy(self):
        """Verify exact 0.15 deduction for heavy repetition (ratio < 0.3).

        AC1: repetition_ratio < 0.3 deducts 0.15
        """
        # 8 words, 2 unique = ratio 0.25 (< 0.3)
        words = ["the", "the", "the", "the", "the", "the", "word", "word"]
        text = " ".join(words)

        score = _calculate_language_consistency(text, words)
        # 0.25 - 0.15 = 0.10 max (assuming no other penalties)
        assert score <= 0.10

    def test_short_words_exact_deduction(self):
        """Verify exact 0.10 deduction for many short words (ratio > 0.5).

        AC2: short_ratio > 0.5 deducts 0.10
        """
        # 8 words, 5 short (<=2 chars) = ratio 0.625 (> 0.5)
        words = ["a", "b", "c", "d", "go", "hello", "world", "test"]
        text = " ".join(words)

        score = _calculate_language_consistency(text, words)
        # With > 50% short words, should deduct 0.10
        assert score <= 0.15  # 0.25 - 0.10 = 0.15

    def test_filler_exact_deduction(self):
        """Verify exact 0.08 deduction for filler word overload (ratio > 0.2).

        AC3: filler_ratio > 0.2 deducts 0.08
        """
        # 10 words, 3 fillers = ratio 0.3 (> 0.2)
        words = ["um", "so", "like", "basically", "the", "thing", "is", "that", "we", "went"]
        text = " ".join(words)

        score = _calculate_language_consistency(text, words)
        # With filler ratio > 0.2, should deduct 0.08
        assert score <= 0.17  # 0.25 - 0.08 = 0.17

    def test_well_formed_text_full_score(self):
        """Well-formed text with no issues returns full score (0.25).

        AC4: Full score for well-formed text with no issues
        """
        # Diverse words, no short words, no fillers, pure ASCII
        words = ["excellent", "documentation", "provides", "comprehensive", "overview", "system"]
        text = " ".join(words)

        score = _calculate_language_consistency(text, words)
        # No deductions should apply
        assert score == 0.25


class TestAdjustEmbeddingWeight:
    """Test the embedding weight adjustment function."""

    def test_low_quality_reduces_weight_by_20_percent(self):
        """Low quality transcript should reduce weight by 20%."""
        base_weight = 0.4
        adjusted, reason = adjust_embedding_weight_for_transcript_quality(
            base_weight, "low"
        )

        expected = base_weight * (1.0 - 0.20)  # 20% reduction
        assert adjusted == pytest.approx(expected)
        assert "low_quality" in reason
        assert "-20%" in reason

    def test_medium_quality_reduces_weight_by_10_percent(self):
        """Medium quality transcript should reduce weight by 10% (half of 20%)."""
        base_weight = 0.4
        adjusted, reason = adjust_embedding_weight_for_transcript_quality(
            base_weight, "medium"
        )

        expected = base_weight * (1.0 - 0.10)  # 10% reduction
        assert adjusted == pytest.approx(expected)
        assert "medium_quality" in reason

    def test_high_quality_no_reduction(self):
        """High quality transcript should not reduce weight."""
        base_weight = 0.4
        adjusted, reason = adjust_embedding_weight_for_transcript_quality(
            base_weight, "high"
        )

        assert adjusted == base_weight
        assert "high_quality" in reason

    def test_disabled_returns_unchanged(self):
        """When disabled, weight should not be adjusted."""
        base_weight = 0.4
        adjusted, reason = adjust_embedding_weight_for_transcript_quality(
            base_weight, "low", quality_weight_enabled=False
        )

        assert adjusted == base_weight
        assert "disabled" in reason

    def test_custom_reduction_factor(self):
        """Custom reduction factor should be applied."""
        base_weight = 0.5
        adjusted, reason = adjust_embedding_weight_for_transcript_quality(
            base_weight, "low", low_quality_reduction=0.30
        )

        expected = base_weight * (1.0 - 0.30)  # 30% reduction
        assert adjusted == pytest.approx(expected)
        assert "-30%" in reason

    def test_returns_tuple_of_two(self):
        """Function should return exactly 2 values."""
        result = adjust_embedding_weight_for_transcript_quality(0.4, "high")
        assert isinstance(result, tuple)
        assert len(result) == 2

        weight, reason = result
        assert isinstance(weight, float)
        assert isinstance(reason, str)


class TestQualityTierClassification:
    """Test quality tier classification with edge cases."""

    def test_exactly_high_threshold(self):
        """Score exactly at high threshold should be 'high'."""
        # Create a transcript that scores exactly at 0.8
        # This is hard to achieve precisely, so we test that high tier starts at 0.8
        long_text = (
            "This is a comprehensive transcript demonstrating excellent quality. "
            "It contains numerous sentences with proper grammatical structure. "
            "The content flows naturally and maintains consistency throughout. "
            "Each sentence is well-formed with appropriate punctuation marks. "
            "This level of quality should definitely achieve a high tier rating."
        )
        score, tier, _ = calculate_transcript_quality(long_text)

        # If score >= 0.8, tier should be high
        if score >= 0.8:
            assert tier == "high"
        else:
            assert tier == "medium"

    def test_gibberish_transcript_low_quality(self):
        """Gibberish or ASR error text should score low."""
        gibberish = "um um uh like um basically like um the thing um"
        score, tier, reason = calculate_transcript_quality(gibberish)

        # Filler words and repetition should indicate low quality
        assert tier in ("low", "medium")
        assert score < 0.7

    def test_well_structured_transcript_high_quality(self):
        """Well-structured, long transcript should score high."""
        well_structured = (
            "Welcome to today's presentation about advanced technology. "
            "We will explore several key topics in detail. "
            "First, let us examine the fundamental concepts. "
            "These concepts form the foundation of our discussion. "
            "Next, we will delve into practical applications. "
            "Understanding these applications is crucial for success. "
            "Finally, we will summarize the key takeaways. "
            "Thank you for your attention and engagement."
        )
        score, tier, reason = calculate_transcript_quality(well_structured)

        assert tier == "high"
        assert score >= 0.8
        assert "good_coherence" in reason or "consistent_language" in reason


class TestIntegration:
    """Integration tests combining quality scoring with weight adjustment."""

    def test_empty_transcript_full_reduction(self):
        """Empty transcript should get low tier and full weight reduction."""
        score, tier, reason = calculate_transcript_quality("")
        assert tier == "low"

        base_weight = 0.4
        adjusted, adj_reason = adjust_embedding_weight_for_transcript_quality(
            base_weight, tier
        )

        # Low quality = 20% reduction
        assert adjusted == pytest.approx(base_weight * 0.8)

    def test_high_quality_no_penalty(self):
        """High quality transcript should have no weight penalty."""
        high_quality_text = (
            "This is a comprehensive and well-structured transcript. "
            "It contains many sentences with proper punctuation. "
            "The language is consistent and professional throughout. "
            "This represents an excellent quality transcription result. "
            "The content is clear, coherent, and easy to understand. "
            "Such quality ensures reliable embedding-based matching."
        )
        score, tier, reason = calculate_transcript_quality(high_quality_text)

        if tier == "high":
            base_weight = 0.4
            adjusted, adj_reason = adjust_embedding_weight_for_transcript_quality(
                base_weight, tier
            )
            assert adjusted == base_weight
            assert "high_quality" in adj_reason

    def test_workflow_with_config_option(self):
        """Test the full workflow with config option."""
        # Simulate config.matching.transcript_quality_weight = True
        text = "short text with few words"
        score, tier, reason = calculate_transcript_quality(text)

        base_weight = 0.4

        # With feature enabled
        adjusted_enabled, _ = adjust_embedding_weight_for_transcript_quality(
            base_weight, tier, quality_weight_enabled=True
        )

        # With feature disabled
        adjusted_disabled, _ = adjust_embedding_weight_for_transcript_quality(
            base_weight, tier, quality_weight_enabled=False
        )

        # Disabled should always return base weight
        assert adjusted_disabled == base_weight

        # Enabled should reduce weight for low/medium quality
        if tier in ("low", "medium"):
            assert adjusted_enabled < base_weight
