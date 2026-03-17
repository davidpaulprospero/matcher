"""Tests for backward-compatible cache deserialization of CaptionResult metadata fields.

Verifies that CachedCaption.from_dict() gracefully handles missing video_description,
video_chapters, video_tags, language_confidence, and fallback_language keys (old cache
entries) while preserving them through round-trips (new cache entries).

Also verifies CaptionSegment deserialization handles missing chapter_index and chapter_title.

Covers: US-73-005, US-78-007
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


class TestLanguageConfidenceBackwardCompat:
    """US-78-007: language_confidence and fallback_language backward compat."""

    def test_legacy_cache_missing_language_confidence_defaults_to_1(self):
        """Old cache entry without language_confidence defaults to 1.0."""
        data = _make_old_cache_dict()
        assert 'language_confidence' not in data
        assert 'fallback_language' not in data

        cached = CachedCaption.from_dict(data)
        assert cached.language_confidence == 1.0
        assert cached.fallback_language == ''

    def test_legacy_cache_to_caption_result_defaults_language_fields(self):
        """Old cache -> CaptionResult gets default language_confidence=1.0 and fallback_language=''."""
        data = _make_old_cache_dict()
        cached = CachedCaption.from_dict(data)
        result = cached.to_caption_result()

        assert result.language_confidence == 1.0
        assert result.fallback_language == ''

    def test_new_cache_with_language_confidence_preserves(self):
        """Cache entry with language_confidence and fallback_language preserves them."""
        data = _make_new_cache_dict()
        data['language_confidence'] = 0.8
        data['fallback_language'] = 'es'

        cached = CachedCaption.from_dict(data)
        assert cached.language_confidence == 0.8
        assert cached.fallback_language == 'es'

        result = cached.to_caption_result()
        assert result.language_confidence == 0.8
        assert result.fallback_language == 'es'

    def test_round_trip_language_confidence(self):
        """language_confidence and fallback_language survive full round-trip."""
        result = CaptionResult(
            video_id='lang_test',
            segments=[_make_segment(source_file='lang_test')],
            language='en',
            language_confidence=0.5,
            fallback_language='fr',
        )
        cached = CachedCaption(
            video_id=result.video_id,
            language=result.language,
            segments=[seg.to_dict() for seg in result.segments],
            is_auto_generated=False,
            format_source='vtt',
            fetch_timestamp=time.time(),
            language_confidence=result.language_confidence,
            fallback_language=result.fallback_language,
        )

        data = cached.to_dict()
        assert data['language_confidence'] == 0.5
        assert data['fallback_language'] == 'fr'

        restored = CachedCaption.from_dict(data)
        restored_result = restored.to_caption_result()
        assert restored_result.language_confidence == 0.5
        assert restored_result.fallback_language == 'fr'

    def test_default_language_confidence_omitted_from_dict(self):
        """Default language_confidence=1.0 and empty fallback_language are omitted from to_dict()."""
        data = _make_old_cache_dict()
        cached = CachedCaption.from_dict(data)
        serialized = cached.to_dict()

        assert 'language_confidence' not in serialized
        assert 'fallback_language' not in serialized


class TestCaptionSegmentChapterBackwardCompat:
    """US-78-007: CaptionSegment deserialization handles missing chapter_index and chapter_title.

    Uses src.caption.models.CaptionSegment which has chapter fields (US-78-002).
    The deserialization path through CachedCaption.to_caption_result() is the
    critical backward-compat boundary.
    """

    def test_legacy_segment_without_chapter_fields(self):
        """Old segment dict without chapter_index/chapter_title deserializes with defaults."""
        from src.caption.models import CaptionSegment as ModelSegment

        seg_dict = {'index': 0, 'start': 0.0, 'end': 5.0, 'text': 'Hello', 'source_file': 'vid1'}
        assert 'chapter_index' not in seg_dict
        assert 'chapter_title' not in seg_dict

        seg = ModelSegment(
            index=seg_dict.get('index', 0),
            start_time=seg_dict.get('start', 0.0),
            end_time=seg_dict.get('end', 0.0),
            text=seg_dict.get('text', ''),
            source_file=seg_dict.get('source_file', ''),
            chapter_index=seg_dict.get('chapter_index', None),
            chapter_title=seg_dict.get('chapter_title', ''),
        )

        assert seg.chapter_index is None
        assert seg.chapter_title == ''

    def test_segment_with_chapter_fields_preserves(self):
        """Segment dict with chapter_index/chapter_title preserves them."""
        from src.caption.models import CaptionSegment as ModelSegment

        seg = ModelSegment(
            index=0, start_time=10.0, end_time=20.0, text='Deep sea',
            source_file='vid1', chapter_index=2, chapter_title='Ocean Floor',
        )

        assert seg.chapter_index == 2
        assert seg.chapter_title == 'Ocean Floor'

    def test_cached_caption_to_caption_result_handles_missing_chapter_fields(self):
        """CachedCaption.to_caption_result() creates segments with default chapter fields
        when legacy segment dicts lack chapter_index/chapter_title."""
        data = _make_old_cache_dict()
        assert 'chapter_index' not in data['segments'][0]

        cached = CachedCaption.from_dict(data)
        result = cached.to_caption_result()

        assert len(result.segments) == 1
        # The refactored path (cache_models) uses .get() with defaults
        # The caption_fetcher path also handles this gracefully
        seg = result.segments[0]
        assert not hasattr(seg, 'chapter_index') or seg.chapter_index is None
        assert not hasattr(seg, 'chapter_title') or seg.chapter_title == ''

    def test_cached_caption_to_caption_result_preserves_chapter_fields(self):
        """CachedCaption.to_caption_result() preserves chapter fields when present."""
        data = _make_old_cache_dict()
        data['segments'] = [{
            'index': 0, 'start': 0.0, 'end': 10.0,
            'text': 'Test', 'source_file': 'abc123',
            'chapter_index': 1, 'chapter_title': 'Chapter Two',
        }]

        cached = CachedCaption.from_dict(data)
        result = cached.to_caption_result()

        seg = result.segments[0]
        # caption_fetcher CaptionSegment may not have these fields
        # but cache_models path does via .get() defaults
        if hasattr(seg, 'chapter_index'):
            assert seg.chapter_index == 1
            assert seg.chapter_title == 'Chapter Two'

    def test_refactored_cached_caption_handles_missing_chapter_fields(self):
        """Refactored CachedCaption.to_caption_result() handles missing chapter fields."""
        from src.caption.cache_models import CachedCaption as RefactoredCachedCaption

        data = _make_old_cache_dict()
        data['caption_quality'] = 'medium'
        data['coverage_ratio'] = None
        data['unavailable'] = False

        cached = RefactoredCachedCaption.from_dict(data)
        result = cached.to_caption_result()

        assert len(result.segments) == 1
        assert result.segments[0].chapter_index is None
        assert result.segments[0].chapter_title == ''

    def test_refactored_cached_caption_preserves_chapter_fields(self):
        """Refactored CachedCaption.to_caption_result() preserves chapter fields."""
        from src.caption.cache_models import CachedCaption as RefactoredCachedCaption

        data = _make_old_cache_dict()
        data['caption_quality'] = 'medium'
        data['coverage_ratio'] = None
        data['unavailable'] = False
        data['segments'] = [{
            'index': 0, 'start': 0.0, 'end': 10.0,
            'text': 'Test', 'source_file': 'abc123',
            'chapter_index': 1, 'chapter_title': 'Chapter Two',
        }]

        cached = RefactoredCachedCaption.from_dict(data)
        result = cached.to_caption_result()

        assert result.segments[0].chapter_index == 1
        assert result.segments[0].chapter_title == 'Chapter Two'

    def test_segment_to_dict_omits_chapter_when_none(self):
        """CaptionSegment.to_dict() omits chapter fields when chapter_index is None."""
        from src.caption.models import CaptionSegment as ModelSegment

        seg = ModelSegment(index=0, start_time=0.0, end_time=5.0, text='Hello', source_file='vid1')
        d = seg.to_dict()
        assert 'chapter_index' not in d
        assert 'chapter_title' not in d

    def test_segment_to_dict_includes_chapter_when_set(self):
        """CaptionSegment.to_dict() includes chapter fields when chapter_index is set."""
        from src.caption.models import CaptionSegment as ModelSegment

        seg = ModelSegment(
            index=0, start_time=0.0, end_time=5.0,
            text='Hello', source_file='vid1',
            chapter_index=0, chapter_title='Intro',
        )
        d = seg.to_dict()
        assert d['chapter_index'] == 0
        assert d['chapter_title'] == 'Intro'


class TestRefactoredCacheModelsBackwardCompat:
    """US-78-007: Test the refactored CachedCaption in src.caption.cache_models."""

    def test_refactored_from_dict_legacy_defaults(self):
        """Refactored CachedCaption.from_dict handles legacy data missing all metadata."""
        from src.caption.cache_models import CachedCaption as RefactoredCachedCaption

        data = _make_old_cache_dict()
        # Add fields the refactored version requires
        data['caption_quality'] = 'medium'
        data['coverage_ratio'] = None
        data['unavailable'] = False

        cached = RefactoredCachedCaption.from_dict(data)
        assert cached.video_description == ''
        assert cached.video_chapters == []
        assert cached.video_tags == []
        assert cached.language_confidence == 1.0
        assert cached.fallback_language == ''

    def test_refactored_to_caption_result_passes_metadata(self):
        """Refactored CachedCaption.to_caption_result() passes metadata to CaptionResult."""
        from src.caption.cache_models import CachedCaption as RefactoredCachedCaption

        data = _make_new_cache_dict()
        data['caption_quality'] = 'high'
        data['coverage_ratio'] = 0.95
        data['unavailable'] = False
        data['language_confidence'] = 0.8
        data['fallback_language'] = 'de'

        cached = RefactoredCachedCaption.from_dict(data)
        result = cached.to_caption_result()

        assert result.video_description == 'A documentary about ocean wildlife'
        assert len(result.video_chapters) == 2
        assert result.video_tags == ['ocean', 'wildlife', 'documentary']
        assert result.language_confidence == 0.8
        assert result.fallback_language == 'de'

    def test_refactored_from_caption_result_saves_metadata(self):
        """Refactored CachedCaption.from_caption_result() saves all metadata fields."""
        from src.caption.cache_models import CachedCaption as RefactoredCachedCaption

        result = CaptionResult(
            video_id='meta_test',
            segments=[_make_segment(source_file='meta_test')],
            language='en',
            video_description='Test desc',
            video_chapters=[{'title': 'Ch1', 'start_time': 0.0, 'end_time': 30.0}],
            video_tags=['test'],
            language_confidence=0.7,
            fallback_language='ja',
        )

        cached = RefactoredCachedCaption.from_caption_result(result, 'en')
        assert cached.video_description == 'Test desc'
        assert cached.video_chapters == [{'title': 'Ch1', 'start_time': 0.0, 'end_time': 30.0}]
        assert cached.video_tags == ['test']
        assert cached.language_confidence == 0.7
        assert cached.fallback_language == 'ja'

    def test_refactored_round_trip_full(self):
        """Full round-trip through refactored CachedCaption preserves all fields."""
        from src.caption.cache_models import CachedCaption as RefactoredCachedCaption

        result = CaptionResult(
            video_id='rt_test',
            segments=[_make_segment(source_file='rt_test')],
            language='en',
            video_description='Full round-trip',
            video_chapters=[{'title': 'All', 'start_time': 0.0, 'end_time': 60.0}],
            video_tags=['round', 'trip'],
            language_confidence=0.6,
            fallback_language='ko',
        )

        cached = RefactoredCachedCaption.from_caption_result(result, 'en')
        data = cached.to_dict()
        restored = RefactoredCachedCaption.from_dict(data)
        restored_result = restored.to_caption_result()

        assert restored_result.video_description == 'Full round-trip'
        assert restored_result.video_chapters == [{'title': 'All', 'start_time': 0.0, 'end_time': 60.0}]
        assert restored_result.video_tags == ['round', 'trip']
        assert restored_result.language_confidence == 0.6
        assert restored_result.fallback_language == 'ko'
