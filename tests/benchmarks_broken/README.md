# Broken Benchmarks - Needs Refactoring

**Status:** These benchmark tests have API mismatches and require significant refactoring.

## Files

- `test_embedding_performance.py` - Uses old `compute_similarity()` and `EmbeddingCache` APIs
- `test_keyword_extraction_performance.py` - Uses non-existent `TFIDFKeywordExtractor`, `EntityExtractor`, etc.
- `test_matching_performance.py` - Uses old `TieredMatcher` API
- `test_otio_performance.py` - Various API mismatches with OTIO module

## Issues

These benchmarks were created before major refactoring efforts and reference:
- Functions that were renamed or removed
- Classes that were moved to different modules
- APIs that changed during LLM client unification
- Old module structure before package refactoring

## Resolution

To fix these benchmarks:

1. **Review current module APIs:**
   - `src/embeddings.py` - Check available functions (use `cosine_similarity`, not `compute_similarity`)
   - `src/keyword_extractor/` - Use `LLMKeywordExtractor` API, not old class names
   - `src/matching/` - Check current matcher implementation
   - `src/otio/` - Review refactored OTIO package structure

2. **Update imports:**
   - Replace old function names with current equivalents
   - Update class imports to match refactored module structure
   - Use correct import paths after package reorganization

3. **Verify benchmark logic:**
   - Ensure benchmarked operations are still relevant
   - Update test data to match current data structures
   - Fix any timing assertions that may be outdated

4. **Test individually:**
   ```bash
   pytest tests/benchmarks_broken/test_embedding_performance.py -v
   ```

5. **Move back when fixed:**
   ```bash
   mv tests/benchmarks_broken/test_*_performance.py tests/benchmarks/
   ```

## Alternative: Rewrite from Scratch

Given the extent of API changes, it may be faster to:
1. Review what needs benchmarking in current codebase
2. Write new benchmarks using `test_pipeline_performance.py` and `test_transcription_performance.py` as templates
3. Archive these old benchmarks for reference

## Working Benchmarks

See `tests/benchmarks/` for functional performance tests:
- `test_pipeline_performance.py` - Pipeline and state operations (14 benchmarks)
- `test_transcription_performance.py` - Transcription operations (12 benchmarks)

**Total Working Benchmarks:** 26 tests, all passing
