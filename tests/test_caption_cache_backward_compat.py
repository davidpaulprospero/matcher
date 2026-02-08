"""Tests for backward-compatible cache deserialization of CaptionResult metadata fields (US-73-005).

Verifies that CachedCaption.from_dict() gracefully handles missing video_description,
video_chapters, and video_tags keys (old cache entries) while preserving them through
round-trips (new cache entries).
"""

import json
import time

import pytest

from src.caption_fetcher import CachedCaption, CaptionResult, CaptionSegment


def _make_segment(index: int = 0, start: float = 0.0, end: float = 5.0,
                  text: str = "Hello world", source_file: str = "vid123"):
    """Helper to create a CaptionSegment."""
    return CaptionSegment(
        index=index, start_time=start, end_time=end,
        text=text, source_file=source_file,
    )


def _make_old_cache_dict(**overrides) -> dict:
    """Create a dict simulating an old cache entry WITHOUT metadata fields."""
    base = {
        'video_id': 'abc123',
        'language': 'en',
        'segments': [{'index': 0, 'start': 0.0, 'end': 10.0,
                       'text': 'Test caption', 'source_file': 'abc123'}],
        'is_auto_generated': False,
        'format_source': 'json3',
        'fetch_timestamp': 1700000000.0,
        'duration': 10.0,
    }
    base.update(overrides)
    return base


def _make_new_cache_dict(**overrides) -> dict:
    """Create a dict simulating a new cache entry WITH metadata fields."""
    base = _make_old_cache_dict()
    base['video_description'] = 'A documentary about ocean wildlife'
    base['video_chapters'] = [
        {'title': 'Introduction', 'start_time': 0.0, 'end_time': 60.0},
        {'title': 'Deep Sea', 'start_time': 60.0, 'end_time': 180.0},
    ]
    base['video_tags'] = ['ocean', 'wildlife', 'documentary']
    base.update(overrides)
    return base


class TestCachedCaptionBackwardCompat:
    """CachedCaption.from_dict() (equivalent deserialization for CaptionResult) handles missing metadata fields gracefully."""

    def test_old_cache_without_metadata_defaults_empty(self):
        """Old cache entries without metadata fields load with empty defaults."""
        data = _make_old_cache_dict()
        cached = CachedCaption.from_dict(data)

        assert cached.video_id == 'abc123'
        assert cached.video_description == ''
        assert cached.video_chapters == []
        assert cached.video_tags == []

    def test_old_cache_no_errors_or_warnings(self, caplog):
        """Old cache entries load without errors or warnings."""
        import logging
        with caplog.at_level(logging.WARNING):
            data = _make_old_cache_dict()
            CachedCaption.from_dict(data)
        assert len(caplog.records) == 0

    def test_caption_result_deserialization_handles_missing_metadata(self):
        """CaptionResult deserialization (via CachedCaption.from_dict) gracefully handles
        missing video_description, video_chapters, and video_tags by defaulting to '' and []."""
        data = _make_old_cache_dict()
        # No video_description, video_chapters, or video_tags keys at all
        assert 'video_description' not in data
        assert 'video_chapters' not in data
        assert 'video_tags' not in data

        # Deserialize through the full path: dict -> CachedCaption -> CaptionResult
        cached = CachedCaption.from_dict(data)
        result = cached.to_caption_result()

        # CaptionResult gets safe defaults
        assert result.video_description == ''
        assert result.video_chapters == []
        assert result.video_tags == []
        # Core fields still intact
        assert result.video_id == 'abc123'
        assert len(result.segments) == 1

    def test_existing_cached_results_without_metadata_load_clean(self, caplog):
        """Existing cached caption results without metadata fields load without errors or warnings.
        Simulates loading an old cache entry created before US-70-002 metadata fields were added."""
        import logging
        data = _make_old_cache_dict()
        with caplog.at_level(logging.DEBUG):
            cached = CachedCaption.from_dict(data)
            result = cached.to_caption_result()

        # No warnings or errors at any log level
        warning_or_error = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_or_error) == 0
        # Result is valid and usable
        assert result.video_id == 'abc123'
        assert result.video_description == ''
        assert result.video_chapters == []
        assert result.video_tags == []

    def test_new_cache_with_metadata_preserves_fields(self):
        """New cache entries with metadata fields are preserved."""
        data = _make_new_cache_dict()
        cached = CachedCaption.from_dict(data)

        assert cached.video_description == 'A documentary about ocean wildlife'
        assert len(cached.video_chapters) == 2
        assert cached.video_chapters[0]['title'] == 'Introduction'
        assert cached.video_tags == ['ocean', 'wildlife', 'documentary']

    def test_partial_metadata_fields(self):
        """Cache entry with some but not all metadata fields."""
        data = _make_old_cache_dict()
        data['video_tags'] = ['nature']
        # video_description and video_chapters intentionally absent
        cached = CachedCaption.from_dict(data)

        assert cached.video_description == ''
        assert cached.video_chapters == []
        assert cached.video_tags == ['nature']


class TestCachedCaptionRoundTrip:
    """Round-trip: serialize CaptionResult -> CachedCaption -> dict -> CachedCaption -> CaptionResult."""

    def test_round_trip_with_metadata(self):
        """Full metadata survives serialize -> deserialize round-trip."""
        result = CaptionResult(
            video_id='test123',
            segments=[_make_segment(source_file='test123')],
            language='en',
            is_auto_generated=False,
            format_source='json3',
            video_description='Python tutorial for beginners',
            video_chapters=[{'title': 'Setup', 'start_time': 0.0, 'end_time': 120.0}],
            video_tags=['python', 'tutorial', 'beginner'],
        )

        # CaptionResult -> CachedCaption
        cached = CachedCaption(
            video_id=result.video_id,
            language=result.language,
            segments=[seg.to_dict() for seg in result.segments],
            is_auto_generated=result.is_auto_generated,
            format_source=result.format_source,
            fetch_timestamp=time.time(),
            duration=result.duration,
            video_description=result.video_description,
            video_chapters=result.video_chapters,
            video_tags=result.video_tags,
        )

        # CachedCaption -> dict -> CachedCaption
        data = cached.to_dict()
        restored_cached = CachedCaption.from_dict(data)

        assert restored_cached.video_description == 'Python tutorial for beginners'
        assert restored_cached.video_chapters == [{'title': 'Setup', 'start_time': 0.0, 'end_time': 120.0}]
        assert restored_cached.video_tags == ['python', 'tutorial', 'beginner']

        # CachedCaption -> CaptionResult
        restored_result = restored_cached.to_caption_result()
        assert restored_result.video_description == result.video_description
        assert restored_result.video_chapters == result.video_chapters
        assert restored_result.video_tags == result.video_tags

    def test_round_trip_without_metadata_applies_defaults(self):
        """Old cache entry (no metadata) -> CaptionResult gets empty defaults."""
        data = _make_old_cache_dict()
        cached = CachedCaption.from_dict(data)
        result = cached.to_caption_result()

        assert result.video_description == ''
        assert result.video_chapters == []
        assert result.video_tags == []
        # Core fields still work
        assert result.video_id == 'abc123'
        assert result.language == 'en'
        assert len(result.segments) == 1

    def test_to_dict_omits_empty_metadata(self):
        """CachedCaption.to_dict() omits empty metadata to save space."""
        data = _make_old_cache_dict()
        cached = CachedCaption.from_dict(data)
        serialized = cached.to_dict()

        assert 'video_description' not in serialized
        assert 'video_chapters' not in serialized
        assert 'video_tags' not in serialized

    def test_to_dict_includes_non_empty_metadata(self):
        """CachedCaption.to_dict() includes non-empty metadata fields."""
        data = _make_new_cache_dict()
        cached = CachedCaption.from_dict(data)
        serialized = cached.to_dict()

        assert serialized['video_description'] == 'A documentary about ocean wildlife'
        assert len(serialized['video_chapters']) == 2
        assert serialized['video_tags'] == ['ocean', 'wildlife', 'documentary']


class TestJSONStringCacheFormat:
    """Test dict-based and JSON-string-based cache formats."""

    def test_json_string_old_format_without_metadata(self):
        """JSON string without metadata deserializes with defaults."""
        data = _make_old_cache_dict()
        json_string = json.dumps(data)
        parsed = json.loads(json_string)
        cached = CachedCaption.from_dict(parsed)

        assert cached.video_description == ''
        assert cached.video_chapters == []
        assert cached.video_tags == []

    def test_json_string_new_format_with_metadata(self):
        """JSON string with metadata deserializes correctly."""
        data = _make_new_cache_dict()
        json_string = json.dumps(data)
        parsed = json.loads(json_string)
        cached = CachedCaption.from_dict(parsed)

        assert cached.video_description == 'A documentary about ocean wildlife'
        assert len(cached.video_chapters) == 2
        assert cached.video_tags == ['ocean', 'wildlife', 'documentary']

    def test_json_round_trip_preserves_metadata(self):
        """Full JSON string round-trip preserves metadata."""
        data = _make_new_cache_dict()
        json_string = json.dumps(data)
        parsed = json.loads(json_string)
        cached = CachedCaption.from_dict(parsed)
        re_serialized = json.dumps(cached.to_dict())
        re_parsed = json.loads(re_serialized)
        final_cached = CachedCaption.from_dict(re_parsed)

        assert final_cached.video_description == data['video_description']
        assert final_cached.video_chapters == data['video_chapters']
        assert final_cached.video_tags == data['video_tags']

    def test_json_round_trip_old_cache_no_metadata_keys(self):
        """Old cache JSON -> serialize -> no metadata keys in output."""
        data = _make_old_cache_dict()
        json_string = json.dumps(data)
        parsed = json.loads(json_string)
        cached = CachedCaption.from_dict(parsed)
        output = cached.to_dict()

        # Empty metadata should not appear in serialized form
        assert 'video_description' not in output
        assert 'video_chapters' not in output
        assert 'video_tags' not in output


class TestCaptionResultToDict:
    """CaptionResult.to_dict() includes metadata fields."""

    def test_to_dict_includes_metadata(self):
        """CaptionResult.to_dict() serializes metadata fields."""
        result = CaptionResult(
            video_id='xyz',
            segments=[_make_segment(source_file='xyz')],
            language='en',
            video_description='Great video',
            video_chapters=[{'title': 'Ch1', 'start_time': 0.0, 'end_time': 30.0}],
            video_tags=['tag1', 'tag2'],
        )
        d = result.to_dict()
        assert d['video_description'] == 'Great video'
        assert d['video_chapters'] == [{'title': 'Ch1', 'start_time': 0.0, 'end_time': 30.0}]
        assert d['video_tags'] == ['tag1', 'tag2']

    def test_to_dict_empty_metadata_defaults(self):
        """CaptionResult.to_dict() includes empty defaults for metadata."""
        result = CaptionResult(video_id='xyz', segments=[_make_segment(source_file='xyz')])
        d = result.to_dict()
        assert d['video_description'] == ''
        assert d['video_chapters'] == []
        assert d['video_tags'] == []
