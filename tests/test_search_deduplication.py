"""
Tests for search result deduplication and freshness scoring module.

Covers:
- Title normalization
- Title similarity calculation (Jaccard)
- Video ID-based deduplication (US-153-004)
- Duplicate detection by title
- Quality preservation for duplicates
- Freshness score calculation
- Integration with deduplication pipeline

US-149-005: Search result deduplication and freshness scoring
US-153-004: Enhance search result deduplication with better video ID matching
"""

import pytest
from datetime import datetime, timedelta

from src.downloader.search_deduplication import (
    normalize_title,
    calculate_title_similarity,
    find_duplicate_indices,
    deduplicate_results,
    deduplicate_by_video_id,
    get_video_id_hash,
    calculate_quality_rank,
    VideoQualityRank,
    parse_published_at,
    calculate_freshness_score,
    add_freshness_scores,
    process_search_results,
)


# ============================================================================
# Title Normalization Tests
# ============================================================================

class TestNormalizeTitle:
    def test_lowercase(self):
        assert normalize_title("HELLO WORLD") == "hello world"
        assert normalize_title("HeLLo") == "hello"

    def test_remove_special_chars(self):
        assert normalize_title("Test!@#Video") == "test video"

    def test_collapse_spaces(self):
        assert normalize_title("Test   Video") == "test video"

    def test_remove_official_video(self):
        assert normalize_title("Test Official Video") == "test"

    def test_remove_part_number(self):
        assert normalize_title("Test Part 1") == "test"
        assert normalize_title("Test Part 2") == "test"

    def test_preserve_content(self):
        assert normalize_title("Amazing Wildlife Documentary") == "amazing wildlife documentary"


class TestTitleSimilarity:
    def test_identical_titles(self):
        assert calculate_title_similarity("Test Video", "Test Video") == 1.0

    def test_similar_titles(self):
        sim = calculate_title_similarity("How to Cook Rice", "How to Cook Pasta")
        assert sim > 0.5

    def test_different_titles(self):
        sim = calculate_title_similarity("Cooking Rice", "Fixing a Car")
        assert sim < 0.3

    def test_empty_title(self):
        assert calculate_title_similarity("", "Test") == 0.0
        assert calculate_title_similarity("Test", "") == 0.0
        assert calculate_title_similarity("", "") == 0.0


class TestFindDuplicateIndices:
    def test_no_duplicates(self):
        results = [
            {"title": "Video A"},
            {"title": "Video B"},
            {"title": "Video C"},
        ]
        indices = find_duplicate_indices(results, max_similarity=0.85)
        assert indices == []

    def test_exact_duplicates(self):
        results = [
            {"title": "Same Video"},
            {"title": "Same Video"},
            {"title": "Different Video"},
        ]
        indices = find_duplicate_indices(results, max_similarity=0.85)
        assert 1 in indices  # Second occurrence should be marked
        assert 0 not in indices  # First occurrence should be kept
        assert 2 not in indices  # Different video should not be marked

    def test_similar_duplicates(self):
        results = [
            {"title": "How to Cook Rice - Part 1"},
            {"title": "How to Cook Rice Part 1"},
            {"title": "How to Cook Pasta"},
        ]
        indices = find_duplicate_indices(results, max_similarity=0.85)
        assert 1 in indices  # Should be detected as duplicate

    def test_empty_results(self):
        assert find_duplicate_indices([], max_similarity=0.85) == []
        assert find_duplicate_indices(None, max_similarity=0.85) == []

    def test_invalid_threshold(self):
        results = [{"title": "Test"}]
        assert find_duplicate_indices(results, max_similarity=0.0) == []
        assert find_duplicate_indices(results, max_similarity=1.1) == []


class TestDeduplicateResults:
    def test_deduplicate_exact_match(self):
        results = [
            {"title": "Test Video", "video_id": "abc123"},
            {"title": "Test Video", "video_id": "def456"},
            {"title": "Different Video", "video_id": "ghi789"},
        ]
        deduped = deduplicate_results(results, max_similarity=0.85)
        assert len(deduped) == 2
        # First "Test Video" should be kept
        assert any(r.get("video_id") == "abc123" for r in deduped)
        # "Different Video" should also be kept
        assert any(r.get("video_id") == "ghi789" for r in deduped)

    def test_preserve_order(self):
        results = [
            {"title": "Video A", "video_id": "1"},
            {"title": "Video B", "video_id": "2"},
            {"title": "Video A", "video_id": "3"},  # Duplicate
            {"title": "Video C", "video_id": "4"},
        ]
        deduped = deduplicate_results(results, max_similarity=0.85)
        assert deduped[0]["video_id"] == "1"
        assert deduped[1]["video_id"] == "2"
        assert deduped[2]["video_id"] == "4"


class TestParsePublishedAt:
    def test_iso_format_with_z(self):
        result = parse_published_at("2024-01-15T12:00:00Z")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_iso_format_with_offset(self):
        result = parse_published_at("2024-01-15T12:00:00+05:00")
        assert result is not None

    def test_date_only(self):
        result = parse_published_at("2024-01-15")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_none_input(self):
        assert parse_published_at(None) is None

    def test_invalid_format(self):
        assert parse_published_at("not-a-date") is None


class TestCalculateFreshnessScore:
    def test_today(self):
        now = datetime(2024, 6, 15)
        score = calculate_freshness_score("2024-06-15T10:00:00Z", min_freshness_days=365, now=now)
        assert score == 1.0

    def test_one_year_ago(self):
        now = datetime(2024, 6, 15)
        score = calculate_freshness_score("2023-06-15T10:00:00Z", min_freshness_days=365, now=now)
        assert 0.4 < score < 0.6  # Around 0.5

    def test_two_years_ago(self):
        now = datetime(2024, 6, 15)
        score = calculate_freshness_score("2022-06-15T10:00:00Z", min_freshness_days=365, now=now)
        assert score < 0.5  # Should be less than 0.5

    def test_no_published_at(self):
        score = calculate_freshness_score(None, min_freshness_days=365)
        assert score == 0.5  # Neutral score for unknown

    def test_disabled_freshness(self):
        score = calculate_freshness_score("2020-01-01", min_freshness_days=0)
        assert score == 1.0  # Disabled returns max score


class TestAddFreshnessScores:
    def test_add_scores(self):
        results = [
            {"title": "Video 1", "published_at": "2024-06-15T10:00:00Z"},
            {"title": "Video 2", "published_at": "2023-06-15T10:00:00Z"},
        ]
        now = datetime(2024, 6, 15)
        scored = add_freshness_scores(results, min_freshness_days=365, now=now)

        assert "freshness_score" in scored[0]
        assert "freshness_score" in scored[1]
        assert scored[0]["freshness_score"] > scored[1]["freshness_score"]  # Newer = higher

    def test_disabled(self):
        results = [{"title": "Video 1"}]
        scored = add_freshness_scores(results, min_freshness_days=0)
        assert "freshness_score" not in scored[0]  # No score added when disabled


class TestProcessSearchResults:
    def test_full_pipeline(self):
        results = [
            {"title": "Test Video", "video_id": "1", "published_at": "2024-06-15T10:00:00Z"},
            {"title": "Test Video", "video_id": "2", "published_at": "2024-06-14T10:00:00Z"},  # Duplicate
            {"title": "Different Video", "video_id": "3", "published_at": "2023-06-15T10:00:00Z"},
        ]
        processed = process_search_results(
            results,
            enable_deduplication=True,
            max_title_similarity=0.85,
            enable_freshness_scoring=True,
            min_freshness_days=365,
        )

        assert len(processed) == 2  # One duplicate removed
        assert "freshness_score" in processed[0]
        assert "freshness_score" in processed[1]

    def test_deduplication_only(self):
        results = [
            {"title": "Test Video", "video_id": "1"},
            {"title": "Test Video", "video_id": "2"},
        ]
        processed = process_search_results(
            results,
            enable_deduplication=True,
            enable_freshness_scoring=False,
        )
        assert len(processed) == 1

    def test_freshness_only(self):
        results = [
            {"title": "Video A", "video_id": "1", "published_at": "2024-06-15T10:00:00Z"},
            {"title": "Video B", "video_id": "2", "published_at": "2023-06-15T10:00:00Z"},
        ]
        processed = process_search_results(
            results,
            enable_deduplication=False,
            enable_freshness_scoring=True,
            min_freshness_days=365,
        )
        assert len(processed) == 2  # No duplicates removed
        assert all("freshness_score" in r for r in processed)

    def test_empty_results(self):
        assert process_search_results([]) == []
        assert process_search_results(None) == []


# ============================================================================
# Video ID Deduplication Tests (US-153-004)
# ============================================================================

class TestGetVideoIdHash:
    def test_valid_video_id(self):
        # Same video ID should produce same hash
        hash1 = get_video_id_hash("abc123DEF")
        hash2 = get_video_id_hash("abc123def")
        assert hash1 == hash2
        assert hash1 is not None

    def test_different_video_ids(self):
        hash1 = get_video_id_hash("abc123")
        hash2 = get_video_id_hash("xyz789")
        assert hash1 != hash2

    def test_none_input(self):
        assert get_video_id_hash(None) is None

    def test_empty_string(self):
        assert get_video_id_hash("") is None

    def test_whitespace(self):
        assert get_video_id_hash("  abc123  ") is not None


class TestCalculateQualityRank:
    def test_full_metadata(self):
        result = {
            'video_id': 'abc123',
            'thumbnail': 'https://example.com/thumb.jpg',
            'duration': 120,
            'channel_name': 'Test Channel',
            'view_count': 1000
        }
        rank = calculate_quality_rank(result)
        assert rank.has_thumbnail is True
        assert rank.has_duration is True
        assert rank.has_channel_name is True
        assert rank.view_count == 1000

    def test_minimal_metadata(self):
        result = {'video_id': 'abc123'}
        rank = calculate_quality_rank(result)
        assert rank.has_thumbnail is False
        assert rank.has_duration is False
        assert rank.has_channel_name is False

    def test_rank_ordering(self):
        # Full metadata should have lower (better) rank
        full = {'thumbnail': 'x', 'duration': 10, 'channel_name': 'x'}
        minimal = {}

        full_rank = calculate_quality_rank(full)
        min_rank = calculate_quality_rank(minimal)

        assert full_rank.rank < min_rank.rank


class TestDeduplicateByVideoId:
    def test_exact_video_id_duplicates(self):
        """Same video ID should be deduplicated."""
        results = [
            {'video_id': 'abc123', 'title': 'Video A'},
            {'video_id': 'abc123', 'title': 'Video A (Duplicate)'},
            {'video_id': 'xyz789', 'title': 'Video B'},
        ]
        deduped = deduplicate_by_video_id(results)
        assert len(deduped) == 2  # One duplicate removed

    def test_no_video_id(self):
        """Results without video_id should pass through."""
        results = [
            {'title': 'Video A'},
            {'title': 'Video A'},  # Title duplicate but no video_id
            {'title': 'Video B'},
        ]
        deduped = deduplicate_by_video_id(results)
        assert len(deduped) == 3  # All pass through (no ID to dedupe)

    def test_mixed_video_ids(self):
        """Some results with video_id, some without."""
        results = [
            {'video_id': 'abc123', 'title': 'Video A'},
            {'title': 'Video B'},  # No video_id
            {'video_id': 'xyz789', 'title': 'Video C'},
        ]
        deduped = deduplicate_by_video_id(results)
        assert len(deduped) == 3  # All pass through

    def test_case_insensitive(self):
        """Video ID matching should be case-insensitive."""
        results = [
            {'video_id': 'AbC123', 'title': 'Video A'},
            {'video_id': 'abc123', 'title': 'Video A Duplicate'},
        ]
        deduped = deduplicate_by_video_id(results)
        assert len(deduped) == 1  # Should dedupe (case insensitive)

    def test_quality_preservation(self):
        """Higher quality result should be kept."""
        results = [
            {'video_id': 'abc123', 'title': 'Low Quality', 'thumbnail': None},
            {'video_id': 'abc123', 'title': 'High Quality', 'thumbnail': 'https://example.com/thumb.jpg'},
        ]
        deduped = deduplicate_by_video_id(results)
        assert len(deduped) == 1
        # Should keep the one with thumbnail (higher quality)
        assert deduped[0]['title'] == 'High Quality'

    def test_view_count_tiebreaker(self):
        """Higher view count should win when quality is equal."""
        results = [
            {'video_id': 'abc123', 'title': 'Lower Views', 'view_count': 100},
            {'video_id': 'abc123', 'title': 'Higher Views', 'view_count': 1000},
        ]
        deduped = deduplicate_by_video_id(results)
        assert len(deduped) == 1
        assert deduped[0]['title'] == 'Higher Views'

    def test_source_query_tracking(self):
        """Source query should be tracked for logging."""
        results = [
            {'video_id': 'abc123', 'title': 'Video A', '_source_query': 'query1'},
            {'video_id': 'abc123', 'title': 'Video A', '_source_query': 'query2'},
        ]
        deduped = deduplicate_by_video_id(results)
        # Internal fields should be cleaned up
        assert '_source_query' not in deduped[0]

    def test_empty_list(self):
        assert deduplicate_by_video_id([]) == []

    def test_none_list(self):
        assert deduplicate_by_video_id(None) == []


class TestProcessSearchResultsWithSourceQuery:
    def test_source_query_parameter(self):
        """Source query should be passed to results."""
        results = [
            {'video_id': 'abc123', 'title': 'Test Video'},
        ]
        processed = process_search_results(
            results,
            source_query='test query'
        )
        # Should process without error
        assert len(processed) == 1

    def test_video_id_dedup_with_source(self):
        """Video ID deduplication should work with source query."""
        results = [
            {'video_id': 'abc123', 'title': 'Video 1', '_source_query': 'query1'},
            {'video_id': 'abc123', 'title': 'Video 2', '_source_query': 'query2'},
            {'video_id': 'xyz789', 'title': 'Video 3'},
        ]
        processed = process_search_results(
            results,
            enable_deduplication=True,
            source_query='main query'
        )
        assert len(processed) == 2


class TestDeduplicationEdgeCases:
    def test_quality_variant_different_thumbnails(self):
        """Same video with different thumbnail URLs - highest quality should win."""
        results = [
            {'video_id': 'abc123', 'title': 'Video', 'thumbnail': None},
            {'video_id': 'abc123', 'title': 'Video', 'thumbnail': 'https://i.ytimg.com/vi/abc123/default.jpg'},
            {'video_id': 'abc123', 'title': 'Video', 'thumbnail': 'https://i.ytimg.com/vi/abc123/hqdefault.jpg', 'view_count': 1000},
        ]
        deduped = deduplicate_by_video_id(results)
        assert len(deduped) == 1
        # Should keep the one with hqdefault (better quality due to view count)
        assert 'hqdefault' in deduped[0].get('thumbnail', '')

    def test_quality_variant_with_and_without_duration(self):
        """Same video with/without duration info."""
        results = [
            {'video_id': 'abc123', 'title': 'Video', 'duration': None},
            {'video_id': 'abc123', 'title': 'Video', 'duration': 120},
        ]
        deduped = deduplicate_by_video_id(results)
        assert len(deduped) == 1
        assert deduped[0].get('duration') == 120

    def test_multiple_duplicates_same_video(self):
        """Multiple duplicates of the same video."""
        results = [
            {'video_id': 'abc123', 'title': 'Video A'},
            {'video_id': 'abc123', 'title': 'Video B'},
            {'video_id': 'abc123', 'title': 'Video C'},
            {'video_id': 'xyz789', 'title': 'Video D'},
        ]
        deduped = deduplicate_by_video_id(results)
        assert len(deduped) == 2  # Only unique videos

    def test_preserves_order_first_occurrence(self):
        """First occurrence should be preserved when quality is equal."""
        results = [
            {'video_id': 'abc123', 'title': 'First'},
            {'video_id': 'xyz789', 'title': 'Second'},
            {'video_id': 'abc123', 'title': 'Duplicate'},  # Duplicate of first
        ]
        deduped = deduplicate_by_video_id(results)
        assert deduped[0]['title'] == 'First'
        assert deduped[1]['title'] == 'Second'

    def test_clean_internal_fields(self):
        """Internal _source_query fields should be cleaned."""
        results = [
            {'video_id': 'abc123', 'title': 'Video', '_source_query': 'test', '_dedup_hash': 'hash'},
        ]
        deduped = deduplicate_by_video_id(results)
        assert '_source_query' not in deduped[0]
        assert '_dedup_hash' not in deduped[0]
        assert 'video_id' in deduped[0]
        assert 'title' in deduped[0]
