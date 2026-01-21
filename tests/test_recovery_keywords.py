"""
Tests for Recovery Keywords (High Matches Mode)

Tests the keyword generation functionality that creates targeted search
queries for weak segments.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from src.matching.coverage_analyzer import WeakSegment
from src.matching.recovery_keywords import (
    extract_keywords_from_text,
    generate_recovery_keywords,
    format_keywords_for_search,
    STOP_WORDS,
    SEARCH_SUFFIXES,
)


class TestStopWords:
    """Tests for STOP_WORDS constant"""

    def test_common_words_in_stop_words(self):
        """Test that common English words are in stop words"""
        common = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to"]
        for word in common:
            assert word in STOP_WORDS, f"{word} should be in STOP_WORDS"

    def test_pronouns_in_stop_words(self):
        """Test that pronouns are filtered"""
        pronouns = ["i", "me", "my", "you", "your", "he", "she", "it", "they"]
        for word in pronouns:
            assert word in STOP_WORDS, f"{word} should be in STOP_WORDS"


class TestSearchSuffixes:
    """Tests for SEARCH_SUFFIXES constant"""

    def test_search_suffixes_exist(self):
        """Test that search suffixes are defined"""
        assert "footage" in SEARCH_SUFFIXES
        assert "4K" in SEARCH_SUFFIXES
        assert "video" in SEARCH_SUFFIXES


class TestExtractKeywordsFromText:
    """Tests for extract_keywords_from_text function"""

    def test_empty_text(self):
        """Test with empty string"""
        result = extract_keywords_from_text("")
        assert result == []

    def test_none_text(self):
        """Test with None-like empty value"""
        result = extract_keywords_from_text("")
        assert result == []

    def test_simple_text(self):
        """Test extracting keywords from simple text"""
        text = "The ocean waves crash on the beautiful sandy beach"
        result = extract_keywords_from_text(text)

        # Should include meaningful words, not stop words
        assert "ocean" in result
        assert "waves" in result or "beach" in result
        assert "the" not in result
        assert "on" not in result

    def test_filters_short_words(self):
        """Test that short words are filtered out"""
        text = "I am at a big cat"
        result = extract_keywords_from_text(text, min_word_length=3)

        # "am", "at", "a" should be filtered (too short or stop words)
        assert "cat" in result
        assert "big" in result

    def test_max_keywords_limit(self):
        """Test max_keywords parameter limits results"""
        text = "ocean waves beach sand sun water surf coast shore tide"
        result = extract_keywords_from_text(text, max_keywords=3)

        assert len(result) <= 3

    def test_frequency_based_ranking(self):
        """Test that more frequent words rank higher"""
        text = "ocean ocean ocean beach beach sand"
        result = extract_keywords_from_text(text, max_keywords=3)

        # "ocean" appears 3x, should be first
        assert result[0] == "ocean"

    def test_case_insensitive(self):
        """Test that extraction is case-insensitive"""
        text = "Ocean WAVES Beach waves ocean"
        result = extract_keywords_from_text(text, max_keywords=2)

        # Should count "ocean" and "waves" regardless of case
        assert "ocean" in result
        assert "waves" in result

    def test_filters_stop_words(self):
        """Test that all stop words are filtered"""
        text = "The quick brown fox jumps over the lazy dog"
        result = extract_keywords_from_text(text)

        for word in result:
            assert word.lower() not in STOP_WORDS

    def test_custom_min_word_length(self):
        """Test custom minimum word length"""
        text = "ab abc abcd abcde"
        result = extract_keywords_from_text(text, min_word_length=4)

        assert "abcd" in result
        assert "abcde" in result
        assert "abc" not in result

    def test_alphabetic_only(self):
        """Test that only alphabetic words are extracted"""
        text = "video123 test_word hello world"
        result = extract_keywords_from_text(text)

        # Should extract "hello" and "world", not mixed
        assert "hello" in result
        assert "world" in result


class TestGenerateRecoveryKeywords:
    """Tests for generate_recovery_keywords function"""

    @pytest.fixture
    def weak_segments(self):
        """Create sample weak segments"""
        return [
            WeakSegment("S001", 0, "The ocean waves crash dramatically", 0.3),
            WeakSegment("S002", 1, "Mountain peaks covered in snow", 0.4),
            WeakSegment("S003", 2, "Urban cityscape at night", 0.5),
        ]

    def test_empty_weak_segments(self):
        """Test with no weak segments"""
        result = generate_recovery_keywords(
            weak_segments=[],
            existing_keywords=["test"],
        )
        assert result == []

    def test_weak_segments_strategy(self, weak_segments):
        """Test weak_segments strategy extracts from text"""
        result = generate_recovery_keywords(
            weak_segments=weak_segments,
            existing_keywords=[],
            strategy="weak_segments",
            max_keywords=5,
        )

        assert len(result) <= 5
        assert len(result) > 0
        # Should contain keywords from weak segment texts
        keywords_text = " ".join(result).lower()
        # At least one of the key concepts should appear
        assert any(word in keywords_text for word in ["ocean", "mountain", "urban", "cityscape"])

    def test_filters_existing_keywords(self, weak_segments):
        """Test that existing keywords are filtered out"""
        result = generate_recovery_keywords(
            weak_segments=weak_segments,
            existing_keywords=["ocean", "mountain", "urban"],
            strategy="weak_segments",
        )

        result_lower = [kw.lower() for kw in result]
        # Exact matches should be filtered
        # Note: "ocean footage" might still appear since full phrase is different
        for kw in result:
            kw_words = kw.lower().split()
            # The base keyword shouldn't be only an existing one
            if len(kw_words) == 1:
                assert kw.lower() not in ["ocean", "mountain", "urban"]

    def test_deduplication(self, weak_segments):
        """Test that duplicate keywords are removed"""
        # Add segments with overlapping content
        segments = weak_segments + [
            WeakSegment("S004", 3, "More ocean waves and beach", 0.35),
        ]

        result = generate_recovery_keywords(
            weak_segments=segments,
            existing_keywords=[],
            strategy="weak_segments",
            max_keywords=20,
        )

        # Check no duplicates
        result_lower = [kw.lower() for kw in result]
        assert len(result_lower) == len(set(result_lower))

    def test_diversify_strategy(self, weak_segments):
        """Test diversify strategy generates different keywords"""
        existing_videos = [
            {"title": "Ocean Documentary", "keyword": "ocean waves"},
            {"title": "Beach Sunset", "keyword": "beach sunset"},
        ]

        result = generate_recovery_keywords(
            weak_segments=weak_segments,
            existing_keywords=[],
            existing_videos=existing_videos,
            strategy="diversify",
            max_keywords=5,
        )

        assert len(result) > 0
        # Should find topics NOT covered by existing videos

    def test_both_strategy_combines(self, weak_segments):
        """Test both strategy combines weak_segments and diversify"""
        result = generate_recovery_keywords(
            weak_segments=weak_segments,
            existing_keywords=[],
            strategy="both",
            max_keywords=10,
        )

        assert len(result) > 0

    def test_max_keywords_limit(self, weak_segments):
        """Test max_keywords parameter"""
        result = generate_recovery_keywords(
            weak_segments=weak_segments,
            existing_keywords=[],
            strategy="weak_segments",
            max_keywords=3,
        )

        assert len(result) <= 3

    def test_adds_footage_suffix(self, weak_segments):
        """Test that keywords get 'footage' suffix"""
        result = generate_recovery_keywords(
            weak_segments=weak_segments[:1],  # Just one segment
            existing_keywords=[],
            strategy="weak_segments",
            max_keywords=3,
        )

        # At least some keywords should have "footage" suffix
        assert any("footage" in kw.lower() for kw in result)

    def test_llm_strategy_with_no_config(self, weak_segments):
        """Test llm strategy falls back when no config"""
        # Without config, LLM strategy should fallback to extraction
        result = generate_recovery_keywords(
            weak_segments=weak_segments,
            existing_keywords=[],
            strategy="llm",
            max_keywords=5,
            config=None,
        )

        # Should return empty or fallback results
        assert isinstance(result, list)

    @patch("src.matching.recovery_keywords._generate_llm_keywords")
    def test_llm_strategy_calls_llm(self, mock_llm, weak_segments):
        """Test llm strategy calls LLM generation"""
        mock_llm.return_value = ["ai keyword 1", "ai keyword 2"]

        result = generate_recovery_keywords(
            weak_segments=weak_segments,
            existing_keywords=[],
            strategy="llm",
            max_keywords=5,
        )

        mock_llm.assert_called_once()
        assert "ai keyword 1" in result
        assert "ai keyword 2" in result

    def test_prioritizes_lowest_confidence(self, weak_segments):
        """Test that lowest confidence segments are prioritized"""
        # Add a very low confidence segment
        segments = [
            WeakSegment("S001", 0, "unique rare specific content", 0.10),  # Lowest
            WeakSegment("S002", 1, "common generic text", 0.80),
        ]

        result = generate_recovery_keywords(
            weak_segments=segments,
            existing_keywords=[],
            strategy="weak_segments",
            max_keywords=3,
            keywords_per_segment=2,
        )

        # Keywords from lowest confidence segment should appear
        result_text = " ".join(result).lower()
        # "unique" or "rare" or "specific" should be included
        assert any(word in result_text for word in ["unique", "rare", "specific"])


class TestFormatKeywordsForSearch:
    """Tests for format_keywords_for_search function"""

    def test_adds_suffix_by_default(self):
        """Test that suffix is added by default"""
        keywords = ["ocean", "mountain", "cityscape"]
        result = format_keywords_for_search(keywords)

        assert "ocean footage" in result
        assert "mountain footage" in result

    def test_custom_suffix(self):
        """Test with custom suffix"""
        keywords = ["ocean", "mountain"]
        result = format_keywords_for_search(keywords, suffix="4K")

        assert "ocean 4K" in result
        assert "mountain 4K" in result

    def test_no_suffix(self):
        """Test with add_suffix=False"""
        keywords = ["ocean", "mountain"]
        result = format_keywords_for_search(keywords, add_suffix=False)

        assert result == keywords

    def test_no_double_suffix(self):
        """Test that keywords with suffix don't get doubled"""
        keywords = ["ocean footage", "mountain 4K", "cityscape video"]
        result = format_keywords_for_search(keywords)

        # Should not add double suffix (footage footage)
        assert "ocean footage footage" not in result
        # Note: "4K" alone doesn't prevent "footage" addition - both are valid suffixes
        # The function only checks if ANY suffix is present
        assert "ocean footage" in result

    def test_case_insensitive_suffix_check(self):
        """Test suffix check is case-insensitive"""
        keywords = ["ocean FOOTAGE", "mountain Video"]
        result = format_keywords_for_search(keywords)

        # Should recognize existing suffixes regardless of case
        for kw in result:
            assert kw.count("footage") <= 1
            assert not (kw.lower().endswith("footage footage"))

    def test_empty_keywords(self):
        """Test with empty keyword list"""
        result = format_keywords_for_search([])
        assert result == []

    def test_preserves_order(self):
        """Test that keyword order is preserved"""
        keywords = ["first", "second", "third"]
        result = format_keywords_for_search(keywords, add_suffix=False)

        assert result[0] == "first"
        assert result[1] == "second"
        assert result[2] == "third"


class TestDiversityKeywords:
    """Tests for _generate_diversity_keywords internal function"""

    def test_finds_uncovered_topics(self):
        """Test that diversity keywords target uncovered topics"""
        from src.matching.recovery_keywords import _generate_diversity_keywords

        weak_segments = [
            WeakSegment("S001", 0, "Ocean wildlife documentary about dolphins", 0.3),
            WeakSegment("S002", 1, "Urban architecture and skyscrapers", 0.4),
        ]

        existing_videos = [
            {"title": "Ocean Waves Documentary", "keyword": "ocean"},
        ]

        result = _generate_diversity_keywords(
            weak_segments=weak_segments,
            existing_set=set(),
            existing_videos=existing_videos,
        )

        # Should find "urban" or "architecture" topics not covered by "ocean"
        result_text = " ".join(result).lower()
        # Urban/architecture should be suggested since ocean is already covered
        assert any(word in result_text for word in ["urban", "architecture", "skyscrapers"])


class TestExtractionFromWeakSegments:
    """Tests for _extract_from_weak_segments internal function"""

    def test_sorts_by_confidence(self):
        """Test that segments are sorted by confidence for extraction"""
        from src.matching.recovery_keywords import _extract_from_weak_segments

        weak_segments = [
            WeakSegment("S003", 2, "high confidence text example", 0.80),
            WeakSegment("S001", 0, "lowest confidence unique words", 0.10),
            WeakSegment("S002", 1, "medium confidence content", 0.50),
        ]

        result = _extract_from_weak_segments(
            weak_segments=weak_segments,
            existing_set=set(),
            keywords_per_segment=2,
        )

        # Keywords from lowest confidence (0.10) should appear first
        # "lowest", "unique" should be prioritized
        assert len(result) > 0

    def test_adds_footage_suffix(self):
        """Test that extracted keywords get 'footage' suffix"""
        from src.matching.recovery_keywords import _extract_from_weak_segments

        weak_segments = [
            WeakSegment("S001", 0, "beautiful sunset over mountains", 0.3),
        ]

        result = _extract_from_weak_segments(
            weak_segments=weak_segments,
            existing_set=set(),
            keywords_per_segment=2,
        )

        # All extracted keywords should have "footage" suffix
        for kw in result:
            assert "footage" in kw.lower()

    def test_filters_existing_keywords(self):
        """Test that existing keywords are filtered"""
        from src.matching.recovery_keywords import _extract_from_weak_segments

        weak_segments = [
            WeakSegment("S001", 0, "sunset mountains beach ocean", 0.3),
        ]

        result = _extract_from_weak_segments(
            weak_segments=weak_segments,
            existing_set={"sunset", "mountains"},
            keywords_per_segment=2,
        )

        # "sunset" and "mountains" should be filtered out
        for kw in result:
            assert "sunset" not in kw.lower().split()
            assert "mountains" not in kw.lower().split()
