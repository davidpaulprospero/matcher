# Archived Benchmarks

**Status:** ARCHIVED - These benchmarks reference non-existent APIs from pre-refactoring codebase.

**Archived Date:** 2026-01-29
**Archived By:** Sprint 23 US-010

## Files

| File | Issue |
|------|-------|
| `test_keyword_extraction_performance.py` | Uses non-existent `TFIDFKeywordExtractor`, `EntityExtractor` (class), `KeywordRemixer`, `SegmentProcessor` |
| `test_matching_performance.py` | Uses non-existent `EmbeddingMatcher`, old `TieredMatcher.match_all()` API, non-existent `DiversityStrategy.filter_diverse_matches()` |
| `test_otio_performance.py` | Uses old `create_timeline(matches, config, frame_rate)` signature, non-existent `PrimaryTrackBuilder._create_clip()` API |

## Why Archived (Not Fixed)

These benchmarks were written speculatively before actual implementation:
1. **Non-existent classes**: `TFIDFKeywordExtractor`, `EntityExtractor` (as class), `EmbeddingMatcher`, `KeywordRemixer`, etc.
2. **API mismatches**: Function signatures don't match any version of the actual code
3. **Missing modules**: `src.keyword_remix`, `src.topic_extraction.TopicDetector`, `src.matching.scoring.calculate_match_score()`
4. **Significant effort to fix**: Would essentially require rewriting from scratch

## Working Benchmarks

See `tests/benchmarks/` for functional performance tests:
- `test_pipeline_performance.py` - Pipeline and state operations (14 benchmarks)
- `test_transcription_performance.py` - Transcription operations (12 benchmarks)

**Total Working Benchmarks:** 26 tests, all passing

## Future Reference

If you want to add benchmarks for these areas:

### Keyword Extraction
Use `LLMKeywordExtractor` from `src.keyword_extractor`:
```python
from src.keyword_extractor import LLMKeywordExtractor

def test_keyword_extraction_speed(benchmark):
    extractor = LLMKeywordExtractor(config)
    # benchmark extraction operations
```

### Matching
Use `TieredMatcher` or `match_all_segments()` from `src.matching`:
```python
from src.matching import TieredMatcher, match_all_segments

def test_matching_speed(benchmark):
    matcher = TieredMatcher(config)
    # benchmark match operations
```

### OTIO Timeline
Use `create_timeline()` from `src.otio`:
```python
from src.otio import create_timeline

def test_timeline_creation_speed(benchmark):
    timeline = create_timeline(
        voiceover_segments=segments,
        matches=matches,
        config=config
    )
```
