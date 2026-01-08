# Performance Benchmarks

Comprehensive performance benchmarks for the matcher-pipeline codebase.

## Overview

This directory contains performance benchmarks that measure:
- **Execution speed** (operations per second)
- **Memory usage** (MB per operation)
- **Latency** (seconds per request)
- **Throughput** (items processed per second)

## Running Benchmarks

### Prerequisites

Install benchmark dependencies:
```bash
pip install pytest-benchmark memory-profiler
```

### Run All Benchmarks

```bash
# Run all benchmarks
pytest tests/benchmarks/ -v --benchmark-only

# With comparison
pytest tests/benchmarks/ -v --benchmark-compare

# Save baseline
pytest tests/benchmarks/ -v --benchmark-save=baseline

# Compare against baseline
pytest tests/benchmarks/ -v --benchmark-compare=baseline
```

### Run Specific Benchmark Suites

```bash
# Keyword extraction
pytest tests/benchmarks/test_keyword_extraction_performance.py -v

# Embedding generation
pytest tests/benchmarks/test_embedding_performance.py -v

# Matching algorithms
pytest tests/benchmarks/test_matching_performance.py -v

# OTIO timeline generation
pytest tests/benchmarks/test_otio_performance.py -v
```

## Benchmark Suites

### 1. Keyword Extraction Performance

**File:** `test_keyword_extraction_performance.py`

**Benchmarks:**
- TF-IDF keyword extraction speed
- Entity extraction speed
- Topic detection speed
- LLM keyword extraction latency (requires API key)
- Keyword remixing speed
- Segment processing throughput

**Memory Profiling:**
- TF-IDF memory usage
- Entity extraction memory usage

**Typical Results:**
- TF-IDF: ~1000 segments/second
- Entity extraction: ~50ms for 10KB text
- Memory: <50MB for 100 segments

### 2. Embedding Generation Performance

**File:** `test_embedding_performance.py`

**Benchmarks:**
- Voyage AI embedding generation speed (requires API key)
- Batch embedding processing
- Cosine similarity computation
- Embedding cache read/write performance

**Memory Profiling:**
- Embedding storage memory (expected: ~4 bytes × dimensions × count)
- Similarity computation memory

**Typical Results:**
- Similarity computation: ~10,000 comparisons/second
- Cache I/O: <100ms for 1000 embeddings
- Memory: ~400MB for 1000 × 1024 embeddings

### 3. Matching Algorithm Performance

**File:** `test_matching_performance.py`

**Benchmarks:**
- Embedding-based matching speed
- Scoring algorithm speed
- Diversity filtering speed
- Location-based filtering speed
- Full matching pipeline latency

**Memory Profiling:**
- Embedding matching memory
- Match result storage memory

**Typical Results:**
- Matching: ~10 segments/second against 1000 candidates
- Scoring: ~1000 candidates/second
- Memory: <200MB for 100 segments × 1000 candidates

### 4. OTIO Timeline Generation Performance

**File:** `test_otio_performance.py`

**Benchmarks:**
- Timeline creation speed
- Track building speed
- OTIO serialization speed
- EDL export speed
- FCP7 XML generation speed
- Split timeline export speed
- Segment map generation speed

**Memory Profiling:**
- Timeline memory usage

**Typical Results:**
- Timeline creation: ~100 segments/second
- OTIO serialization: <1s for 1000 segments
- Memory: <500MB for 1000 segments

## Understanding Results

### Benchmark Output

```
test_keyword_extraction_speed
  Mean: 0.0012 seconds
  Min: 0.0010 seconds
  Max: 0.0015 seconds
  StdDev: 0.0002 seconds
  Rounds: 100
```

**Interpreting:**
- **Mean:** Average execution time (lower is better)
- **Min/Max:** Range of execution times
- **StdDev:** Consistency (lower is more consistent)
- **Rounds:** Number of iterations

### Memory Profiling Output

```
TF-IDF memory usage: 12.34 MB for 100 segments
```

**Interpreting:**
- Shows peak memory usage during operation
- Compare against baselines to detect regressions

## Performance Baselines

### Acceptable Performance Targets

| Operation | Target | Critical |
|-----------|--------|----------|
| Keyword extraction (100 segs) | <1s | <5s |
| Embedding similarity (1000) | <100ms | <500ms |
| Matching (100×1000) | <10s | <60s |
| OTIO timeline (1000 segs) | <5s | <30s |

### Memory Usage Targets

| Operation | Target | Critical |
|-----------|--------|----------|
| Keyword extraction | <50MB | <200MB |
| Embedding storage (1000) | <500MB | <1GB |
| Matching | <200MB | <1GB |
| OTIO timeline (1000) | <500MB | <2GB |

## Continuous Integration

Benchmarks are **not** run in CI by default (they're slow). To enable:

1. Tag tests with `@pytest.mark.benchmark`
2. Run in separate CI job with extended timeout
3. Compare results against baseline
4. Alert on >20% regression

## Adding New Benchmarks

### Template

```python
def test_my_operation_speed(self, fixture_data, benchmark):
    \"\"\"Benchmark my operation speed.\"\"\"
    from src.my_module import my_function

    def operation():
        return my_function(fixture_data)

    result = benchmark(operation)

    assert result is not None
    print(f"\\nProcessed {len(result)} items")

    # Calculate throughput
    items_per_second = len(result) / benchmark.stats['mean']
    print(f"Throughput: {items_per_second:.1f} items/second")
```

### Best Practices

1. **Use realistic data sizes** (100-1000 items typical)
2. **Isolate operations** (test one thing at a time)
3. **Print context** (items processed, throughput, etc.)
4. **Set assertions** (verify results are valid)
5. **Document expectations** (typical results in docstring)

## Troubleshooting

### Benchmarks Too Slow

- Reduce data size in fixtures
- Skip slow tests with `@pytest.mark.slow`
- Run subsets: `pytest tests/benchmarks/test_keyword_*.py`

### Inconsistent Results

- Close other applications
- Run multiple rounds: `--benchmark-min-rounds=10`
- Use `--benchmark-warmup=on`
- Check for background processes

### Memory Profiling Failures

- Requires `tracemalloc` module
- May not work on all platforms
- Use `@pytest.mark.skipif` for platform-specific tests

## Resources

- [pytest-benchmark docs](https://pytest-benchmark.readthedocs.io/)
- [Python profiling docs](https://docs.python.org/3/library/profile.html)
- [Memory profiling](https://docs.python.org/3/library/tracemalloc.html)
