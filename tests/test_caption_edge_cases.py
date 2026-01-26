"""
Edge case test suite for caption cache and caption parsing (US-009 Sprint 8).

Tests corrupted cache, malformed captions, unusual filenames, and boundary
conditions that could cause crashes in production.
"""

import json
import os
import tempfile
import shutil
import time
from pathlib import Path

import pytest

from src.caption_fetcher import (
    CaptionCache,
    CachedCaption,
    CaptionResult,
    CaptionSegment,
    CaptionNormalizer,
    NormalizationConfig,
)
from src.cache.base import CacheEntry


class TestCaptionEdgeCases:
    """Edge case tests for cache and caption parsing robustness."""

    def setup_method(self):
        """Create temp directory for cache tests."""
        self.temp_dir = tempfile.mkdtemp(prefix="caption_edge_test_")

    def teardown_method(self):
        """Clean up temp directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ---- Test 1: 100+ char video title truncation preserves video ID ----

    def test_long_video_title_preserves_video_id_in_cache(self):
        """100+ char video title truncation preserves video ID for matching.

        Video titles can be extremely long, but the video_id (11 chars) used
        as the cache key and in source_file must remain intact.
        """
        video_id = "dQw4w9WgXcQ"
        long_title = "A" * 150  # 150-char title

        # Store a caption with the video_id regardless of title length
        cached = CachedCaption(
            video_id=video_id,
            language="en",
            segments=[
                {
                    'index': 0,
                    'start': 0.0,
                    'end': 5.0,
                    'text': long_title[:50],  # Text could contain title
                    'source_file': video_id,
                },
            ],
            is_auto_generated=False,
            format_source='vtt',
            fetch_timestamp=time.time(),
            duration=5.0,
        )

        # Serialize and deserialize
        data = cached.to_dict()
        restored = CachedCaption.from_dict(data)

        # Video ID must survive round-trip regardless of title length
        assert restored.video_id == video_id
        assert len(restored.video_id) == 11
        assert restored.segments[0]['source_file'] == video_id

        # Cache key uses video_id, not title
        cache = CaptionCache.__new__(CaptionCache)
        cache.enabled = True
        key = cache._make_cache_key(video_id, "en")
        assert key == "dQw4w9WgXcQ_en"
        assert long_title not in key

    def test_long_video_title_in_segment_source_file(self):
        """CaptionSegment source_file with long content round-trips correctly."""
        # Even if someone accidentally puts a long path in source_file
        long_source = "C:/very/long/path/" + "subdir/" * 20 + "video.mp4"
        seg = CaptionSegment(
            index=0,
            start_time=0.0,
            end_time=5.0,
            text="Test caption",
            source_file=long_source,
        )

        d = seg.to_dict()
        assert d['source_file'] == long_source

        # Restore via CachedCaption.to_caption_result path
        cached = CachedCaption(
            video_id="dQw4w9WgXcQ",
            language="en",
            segments=[d],
            is_auto_generated=False,
            format_source='vtt',
            fetch_timestamp=time.time(),
            duration=5.0,
        )
        result = cached.to_caption_result()
        assert result.segments[0].source_file == long_source

    # ---- Test 2: Video URL with special characters ----

    def test_video_url_special_characters_parsed_correctly(self):
        """Video URL with quotes, ampersands, and special chars parsed correctly.

        YouTube URLs can contain &, =, ?, and video IDs can appear in URLs
        with query parameters. The cache key must use only the video_id.
        """
        # Video IDs with characters that might confuse URL parsers
        test_ids = [
            "abc123def45",       # Standard
            "a-b_c1D2E3f",      # Dashes and underscores (valid YouTube ID chars)
            "ABCDEFGHIJK",       # All uppercase
            "01234567890",       # All digits
        ]

        for vid_id in test_ids:
            cached = CachedCaption(
                video_id=vid_id,
                language="en",
                segments=[
                    {
                        'index': 0,
                        'start': 0.0,
                        'end': 2.0,
                        'text': f'Caption for video {vid_id}',
                        'source_file': vid_id,
                    }
                ],
                is_auto_generated=False,
                format_source='vtt',
                fetch_timestamp=time.time(),
                duration=2.0,
            )

            # Round-trip through dict serialization
            data = cached.to_dict()
            json_str = json.dumps(data)  # Must be JSON-serializable
            restored_data = json.loads(json_str)
            restored = CachedCaption.from_dict(restored_data)

            assert restored.video_id == vid_id
            assert restored.segments[0]['source_file'] == vid_id

    def test_special_chars_in_caption_text_roundtrip(self):
        """Caption text with quotes, ampersands, angle brackets survives JSON."""
        special_texts = [
            'He said "hello" & waved',
            "It's <b>bold</b> & <i>italic</i>",
            "Price: $5.00 → €4.50 (≈ ¥750)",
            "Line1\nLine2\tTabbed",
            '{"key": "value"}',  # JSON-like text
        ]

        for text in special_texts:
            seg = CaptionSegment(
                index=0, start_time=0.0, end_time=2.0,
                text=text, source_file="test_vid_123"
            )
            d = seg.to_dict()
            json_str = json.dumps(d)
            restored_d = json.loads(json_str)

            assert restored_d['text'] == text

    # ---- Test 3: Corrupted JSON cache file returns None gracefully ----

    def test_corrupted_json_cache_returns_none(self):
        """Corrupted JSON cache file returns None gracefully (no crash).

        If the index file is corrupted (invalid JSON), the cache should
        recover by starting fresh rather than crashing.
        """
        # Write corrupted JSON to index file
        index_path = Path(self.temp_dir) / "caption_cache_index.json"
        index_path.write_text("{invalid json content!!!", encoding='utf-8')

        # CaptionCache should handle this gracefully
        cache = CaptionCache.__new__(CaptionCache)
        cache.cache_dir = Path(self.temp_dir)
        cache.index_path = index_path
        cache.ttl_seconds = 0
        cache.auto_save = False
        cache.enabled = True
        cache.validation_mode = 'warn'
        cache.validation_tolerance = 0.2
        cache.max_age_days = 30
        cache._hits = 0
        cache._misses = 0
        cache._bytes_saved = 0

        # Load index - should not crash
        cache._load_index()

        # Index should be empty (fresh start)
        assert cache.index == {}

        # get_caption should return None
        result = cache.get_caption("dQw4w9WgXcQ", "en")
        assert result is None

    def test_corrupted_entry_data_returns_none(self):
        """Corrupted entry data in valid JSON index returns None gracefully."""
        cache = CaptionCache.__new__(CaptionCache)
        cache.cache_dir = Path(self.temp_dir)
        cache.index_path = Path(self.temp_dir) / "caption_cache_index.json"
        cache.ttl_seconds = 0
        cache.auto_save = False
        cache.enabled = True
        cache.validation_mode = 'warn'
        cache.validation_tolerance = 0.2
        cache.max_age_days = 30
        cache._hits = 0
        cache._misses = 0
        cache._bytes_saved = 0

        # Valid JSON structure but corrupted entry data
        cache.index = {
            "dQw4w9WgXcQ_en": {
                "data": "not a dict - this is corrupted",
                "cached_at": time.time(),
                "metadata": {}
            }
        }

        # Should return None, not crash
        result = cache.get_caption("dQw4w9WgXcQ", "en")
        assert result is None

    # ---- Test 4: Truncated cache file (partial write) ----

    def test_truncated_cache_file_detected_and_handled(self):
        """Truncated cache file (partial write) detected and handled.

        Simulates a crash during cache write that leaves a partial JSON file.
        """
        index_path = Path(self.temp_dir) / "caption_cache_index.json"

        # Write a truncated JSON (simulates interrupted write)
        truncated_json = '{"dQw4w9WgXcQ_en": {"data": {"video_id": "dQw4w9WgXcQ", "lang'
        index_path.write_text(truncated_json, encoding='utf-8')

        # CaptionCache should handle this gracefully
        cache = CaptionCache.__new__(CaptionCache)
        cache.cache_dir = Path(self.temp_dir)
        cache.index_path = index_path
        cache.ttl_seconds = 0
        cache.auto_save = False
        cache.enabled = True
        cache.validation_mode = 'warn'
        cache.validation_tolerance = 0.2
        cache.max_age_days = 30
        cache._hits = 0
        cache._misses = 0
        cache._bytes_saved = 0

        # Load should not crash - truncated JSON = JSONDecodeError
        cache._load_index()

        # Should start with empty index
        assert cache.index == {}

        # Subsequent operations should work normally
        result = cache.get_caption("dQw4w9WgXcQ", "en")
        assert result is None

    def test_empty_cache_file_handled(self):
        """Empty cache file (0 bytes) handled gracefully."""
        index_path = Path(self.temp_dir) / "caption_cache_index.json"
        index_path.write_text("", encoding='utf-8')

        cache = CaptionCache.__new__(CaptionCache)
        cache.cache_dir = Path(self.temp_dir)
        cache.index_path = index_path
        cache.ttl_seconds = 0
        cache.auto_save = False
        cache.enabled = True
        cache.validation_mode = 'warn'
        cache.validation_tolerance = 0.2
        cache.max_age_days = 30
        cache._hits = 0
        cache._misses = 0
        cache._bytes_saved = 0

        cache._load_index()
        assert cache.index == {}

    # ---- Test 5: Caption segments with 0 duration filtered out ----

    def test_zero_duration_segments_filtered_during_normalization(self):
        """Caption segments with 0 duration (start == end) filtered out.

        The CaptionNormalizer should handle segments where start_time == end_time
        by either extending them to min_segment_duration or filtering them.
        """
        normalizer = CaptionNormalizer()

        segments = [
            CaptionSegment(0, 0.0, 0.0, "Zero duration", "vid123"),   # 0 duration
            CaptionSegment(1, 5.0, 10.0, "Normal segment", "vid123"),  # Normal
            CaptionSegment(2, 15.0, 15.0, "Also zero", "vid123"),      # 0 duration
        ]

        normalized, skipped = normalizer.normalize(segments, video_id="vid123")

        # Zero-duration segments should be fixed (extended to min_segment_duration)
        # not filtered, since they have text
        assert len(normalized) == 3
        for seg in normalized:
            assert seg.end_time > seg.start_time, \
                f"Segment {seg.index} has end <= start: {seg.end_time} <= {seg.start_time}"
            # Each should have at least min_segment_duration
            duration = seg.end_time - seg.start_time
            assert duration >= 0.1 - 1e-9, \
                f"Segment {seg.index} duration {duration} below minimum 0.1"

    def test_negative_duration_segment_handled(self):
        """Segment with end_time < start_time is fixed by normalizer."""
        normalizer = CaptionNormalizer()

        segments = [
            CaptionSegment(0, 10.0, 5.0, "Backwards timing", "vid123"),  # end < start
            CaptionSegment(1, 20.0, 25.0, "Normal", "vid123"),
        ]

        normalized, skipped = normalizer.normalize(segments, video_id="vid123")

        # Backwards segment should be fixed (end extended from start)
        assert len(normalized) == 2
        for seg in normalized:
            assert seg.end_time > seg.start_time

    # ---- Test 6: Unsorted caption timestamps reordered correctly ----

    def test_unsorted_timestamps_reordered(self):
        """Unsorted caption timestamps reordered correctly by normalizer.

        Caption segments may arrive out of order from some formats.
        The normalizer must sort them by start_time.
        """
        normalizer = CaptionNormalizer()

        # Deliberately unsorted segments
        segments = [
            CaptionSegment(0, 30.0, 35.0, "Third segment", "vid123"),
            CaptionSegment(1, 0.0, 5.0, "First segment", "vid123"),
            CaptionSegment(2, 15.0, 20.0, "Second segment", "vid123"),
        ]

        normalized, skipped = normalizer.normalize(segments, video_id="vid123")

        assert len(normalized) == 3
        assert skipped == 0

        # Verify sorted order
        assert normalized[0].text == "First segment"
        assert normalized[1].text == "Second segment"
        assert normalized[2].text == "Third segment"

        # Verify monotonically increasing start times
        for i in range(1, len(normalized)):
            assert normalized[i].start_time >= normalized[i - 1].start_time, \
                f"Segment {i} start ({normalized[i].start_time}) < " \
                f"segment {i-1} start ({normalized[i-1].start_time})"

    def test_unsorted_with_reindexing(self):
        """Segments are re-indexed sequentially after sorting."""
        normalizer = CaptionNormalizer()

        segments = [
            CaptionSegment(5, 20.0, 25.0, "Was index 5", "vid123"),
            CaptionSegment(99, 0.0, 5.0, "Was index 99", "vid123"),
            CaptionSegment(42, 10.0, 15.0, "Was index 42", "vid123"),
        ]

        normalized, _ = normalizer.normalize(segments, video_id="vid123")

        # Re-indexed to 0, 1, 2
        assert normalized[0].index == 0
        assert normalized[1].index == 1
        assert normalized[2].index == 2

        # Content order matches time order
        assert normalized[0].text == "Was index 99"   # start=0.0
        assert normalized[1].text == "Was index 42"   # start=10.0
        assert normalized[2].text == "Was index 5"    # start=20.0

    # ---- Test 7: Empty segment text (whitespace only) treated as missing ----

    def test_whitespace_only_text_filtered_out(self):
        """Empty segment text (whitespace only) treated as missing.

        Segments with only spaces, tabs, or newlines as text should be
        filtered out during normalization.
        """
        normalizer = CaptionNormalizer()

        segments = [
            CaptionSegment(0, 0.0, 5.0, "Valid text", "vid123"),
            CaptionSegment(1, 5.0, 10.0, "   ", "vid123"),       # Spaces only
            CaptionSegment(2, 10.0, 15.0, "\t\n", "vid123"),     # Tab + newline
            CaptionSegment(3, 15.0, 20.0, "", "vid123"),          # Empty string
            CaptionSegment(4, 20.0, 25.0, "Also valid", "vid123"),
        ]

        normalized, skipped = normalizer.normalize(segments, video_id="vid123")

        # Only segments with actual text content should remain
        assert len(normalized) == 2
        assert normalized[0].text == "Valid text"
        assert normalized[1].text == "Also valid"
        # Whitespace segments are intentionally filtered, not counted as errors
        assert skipped == 0

    def test_text_with_leading_trailing_whitespace_preserved(self):
        """Text with leading/trailing whitespace is NOT filtered (has content)."""
        normalizer = CaptionNormalizer()

        segments = [
            CaptionSegment(0, 0.0, 5.0, "  Hello  ", "vid123"),   # Has content
            CaptionSegment(1, 5.0, 10.0, "\nWorld\n", "vid123"),   # Has content
        ]

        normalized, skipped = normalizer.normalize(segments, video_id="vid123")

        # Both segments have non-whitespace content, should be kept
        assert len(normalized) == 2
        assert "Hello" in normalized[0].text
        assert "World" in normalized[1].text

    # ---- Test 8: Additional edge cases with explicit assertions and cleanup ----

    def test_cache_entry_with_missing_fields_handled(self):
        """Cache entry dict missing expected fields returns None gracefully."""
        cache = CaptionCache.__new__(CaptionCache)
        cache.cache_dir = Path(self.temp_dir)
        cache.index_path = Path(self.temp_dir) / "caption_cache_index.json"
        cache.ttl_seconds = 0
        cache.auto_save = False
        cache.enabled = True
        cache.validation_mode = 'warn'
        cache.validation_tolerance = 0.2
        cache.max_age_days = 30
        cache._hits = 0
        cache._misses = 0
        cache._bytes_saved = 0

        # Entry with minimal/missing fields
        cache.index = {
            "partial_en": {
                "data": {
                    "video_id": "partial",
                    "language": "en",
                    # Missing: segments, is_auto_generated, format_source, fetch_timestamp
                },
                "cached_at": time.time(),
                "metadata": {}
            }
        }

        # from_dict should handle missing fields via **kwargs filtering
        # or fail gracefully
        result = cache.get_caption("partial", "en")
        # Should either return a CachedCaption with defaults or None
        # The from_dict uses cls(**{k: v ...}) which will fail on missing required fields
        # and get_caption catches exceptions, returning None
        assert result is None or isinstance(result, CachedCaption)

    def test_segment_dict_missing_timing_fields_defaults_to_zero(self):
        """Segment dict missing 'start'/'end' keys defaults to 0.0 in to_caption_result."""
        cached = CachedCaption(
            video_id="test_video_id",
            language="en",
            segments=[
                {'index': 0, 'text': 'No timing info'},  # Missing start, end, source_file
                {'text': 'Also no index'},                 # Missing index too
            ],
            is_auto_generated=False,
            format_source='vtt',
            fetch_timestamp=time.time(),
            duration=10.0,
        )

        result = cached.to_caption_result()

        # Should not crash; missing fields default gracefully
        assert len(result.segments) == 2
        assert result.segments[0].start_time == 0.0
        assert result.segments[0].end_time == 0.0
        assert result.segments[0].text == 'No timing info'
        assert result.segments[0].source_file == "test_video_id"  # Falls back to video_id

        # Second segment uses enumerate index as fallback
        assert result.segments[1].index == 1
        assert result.segments[1].text == 'Also no index'

    def test_all_segments_whitespace_returns_empty_list(self):
        """If ALL segments are whitespace-only, normalizer returns empty list."""
        normalizer = CaptionNormalizer()

        segments = [
            CaptionSegment(0, 0.0, 5.0, "   ", "vid123"),
            CaptionSegment(1, 5.0, 10.0, "\t", "vid123"),
            CaptionSegment(2, 10.0, 15.0, "\n\n", "vid123"),
        ]

        normalized, skipped = normalizer.normalize(segments, video_id="vid123")

        assert len(normalized) == 0
        assert skipped == 0  # Filtering is intentional, not an error

    def test_duplicate_timestamps_handled(self):
        """Multiple segments with identical timestamps are handled."""
        normalizer = CaptionNormalizer()

        segments = [
            CaptionSegment(0, 5.0, 10.0, "First at 5s", "vid123"),
            CaptionSegment(1, 5.0, 10.0, "Second at 5s", "vid123"),
            CaptionSegment(2, 5.0, 10.0, "Third at 5s", "vid123"),
        ]

        # Should not crash
        normalized, skipped = normalizer.normalize(segments, video_id="vid123")

        # All segments should be present (overlap handling may merge/truncate)
        assert len(normalized) >= 1
        assert skipped == 0
