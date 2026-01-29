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

Benchmarks are run in CI on a schedule and on benchmark file changes. The `.github/workflows/benchmarks.yml` workflow handles this automatically.

### GitHub Actions Workflow

**Location:** `.github/workflows/benchmarks.yml`

**Triggers:**
- Weekly schedule (Sundays at midnight UTC)
- Push to `main` affecting benchmark files
- Manual workflow dispatch

**Features:**
- Runs all benchmark suites
- Compares against baseline with 20% regression threshold
- Uploads results as artifacts (90-day retention)
- Stores baseline as artifact (365-day retention)
- Posts summary to GitHub Actions job log
- Creates warnings on regression detection

**Manual Workflow Options:**
- `update_baseline`: Set to `true` to update the baseline with new results
- `threshold`: Override the default 20% regression threshold

**Usage:**
```bash
# Trigger manually via GitHub CLI
gh workflow run benchmarks.yml

# Update baseline
gh workflow run benchmarks.yml -f update_baseline=true

# Custom threshold
gh workflow run benchmarks.yml -f threshold=30
```

### CI Benchmark Guidelines

**DO:**
- Run benchmarks in a separate CI job with extended timeout (10+ minutes)
- Use dedicated runners with consistent hardware for reproducible results
- Compare against baselines stored in the repository
- Alert on >20% regression (configurable threshold)
- Archive benchmark results as CI artifacts

**DON'T:**
- Block merges solely on benchmark failures (timing varies)
- Run benchmarks on every commit (too slow)
- Compare results across different hardware/runners
- Use benchmark results from shared runners (noisy neighbors)

### Running Benchmarks in CI

```yaml
# GitHub Actions example
benchmark:
  runs-on: ubuntu-latest  # Use dedicated runner for consistency
  timeout-minutes: 15
  steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: '3.11'
    - run: pip install -r requirements-dev.txt
    - name: Run benchmarks
      run: python scripts/benchmark_runner.py --output benchmark_results.json
    - name: Compare to baseline
      run: python scripts/benchmark_runner.py --compare benchmark_baseline.json --threshold 0.20
    - uses: actions/upload-artifact@v4
      with:
        name: benchmark-results
        path: benchmark_results.json
```

### Using the Benchmark Runner Script

The `scripts/benchmark_runner.py` script provides CI-friendly benchmark execution:

```bash
# Run all benchmarks and save results
python scripts/benchmark_runner.py --output results.json

# Compare results to baseline with 20% threshold
python scripts/benchmark_runner.py --compare baseline.json --threshold 0.20

# Compare against default baseline (tests/benchmarks/baseline.json)
python scripts/benchmark_runner.py --compare-baseline --threshold 0.20

# Run specific benchmark suites only
python scripts/benchmark_runner.py --suite pipeline embedding --output results.json

# Verbose output for debugging
python scripts/benchmark_runner.py --verbose

# Generate baseline from current run
python scripts/benchmark_runner.py --output baseline.json --save-baseline

# Update baseline with confirmation prompt (interactive)
python scripts/benchmark_runner.py --update-baseline

# Update baseline without confirmation (CI mode)
python scripts/benchmark_runner.py --update-baseline --yes

# View baseline history
python scripts/benchmark_runner.py --list-history
```

### Performance Baseline Tracking

Baselines are stored as JSON files with automatic history tracking:

```
tests/benchmarks/
├── baseline.json              # Current performance baseline
├── baseline_history/          # Automatic archive (last 5 baselines)
│   ├── baseline_2026-01-15_10-30-00.json
│   └── baseline_2026-01-20_14-45-30.json
└── ...
```

**Baseline History Management:**

The benchmark runner automatically manages baseline history:
- **Archiving**: Before updating, current baseline is archived to `baseline_history/`
- **Pruning**: Only the last 5 baselines are kept (configurable via `MAX_BASELINE_HISTORY`)
- **Timestamps**: Archive filenames include ISO timestamps for sorting
- **Listing**: Use `--list-history` to view all archived baselines

**Baseline JSON Structure:**

```json
{
  "version": "1.0",
  "timestamp": "2026-01-29T10:00:00Z",
  "runner": "github-actions-ubuntu-latest",
  "python_version": "3.11.5",
  "benchmarks": {
    "test_checkpoint_save_small_state": {
      "mean": 0.0045,
      "stddev": 0.0002,
      "min": 0.0040,
      "max": 0.0055
    },
    "test_state_initialization": {
      "mean": 0.0005,
      "stddev": 0.0001,
      "min": 0.0004,
      "max": 0.0007
    }
  }
}
```

### Comparison and Alerting

The benchmark runner compares each test's mean time against the baseline:

```
Benchmark Comparison Results
============================
✓ test_checkpoint_save_small_state: 0.0045s → 0.0048s (+6.7%) [OK]
✓ test_state_initialization: 0.0005s → 0.0005s (+0.0%) [OK]
⚠ test_create_1000_segments: 0.0150s → 0.0195s (+30.0%) [REGRESSION]
✓ test_config_initialization: 0.0015s → 0.0014s (-6.7%) [OK]

Summary: 3 OK, 1 REGRESSION (threshold: 20%)
Exit code: 1 (regressions detected)
```

**Threshold Configuration:**

| Scenario | Threshold | Notes |
|----------|-----------|-------|
| Strict (blocking) | 10% | For critical paths |
| Standard (warning) | 20% | Recommended default |
| Lenient (info) | 50% | For variable operations |

### Updating Baselines

When you intentionally change performance (optimization or added functionality):

```bash
# 1. Run benchmarks and verify results are expected
python scripts/benchmark_runner.py --verbose

# 2. If performance change is intentional, update baseline
python scripts/benchmark_runner.py --output tests/benchmarks/baseline.json --save-baseline

# 3. Commit the new baseline with explanation
git add tests/benchmarks/baseline.json
git commit -m "perf: Update benchmark baseline after optimization

- Improved checkpoint serialization by 30%
- Added new match result creation tests
- Baseline updated to reflect intentional changes"
```

**When to Update Baselines:**

| Scenario | Action |
|----------|--------|
| Optimization merged | Update baseline ✓ |
| New benchmark added | Update baseline ✓ |
| Hardware changed | Update baseline ✓ |
| Unintentional regression | Fix code, don't update baseline ✗ |
| Noisy/flaky results | Increase rounds, don't ignore ✗ |

### Pytest Benchmark Flags

Key flags for CI and local development:

```bash
# Standard benchmark run
pytest tests/benchmarks/ --benchmark-only

# Compare against saved baseline
pytest tests/benchmarks/ --benchmark-compare=baseline

# Save current run as new baseline
pytest tests/benchmarks/ --benchmark-save=baseline

# Increase rounds for more stable results
pytest tests/benchmarks/ --benchmark-min-rounds=10

# Enable warmup to reduce cold-start variance
pytest tests/benchmarks/ --benchmark-warmup=on

# Disable benchmarks (just run assertions)
pytest tests/benchmarks/ --benchmark-disable

# JSON output for CI parsing
pytest tests/benchmarks/ --benchmark-json=output.json

# Skip benchmarks entirely (run as regular tests)
pytest tests/benchmarks/ --benchmark-skip
```

### Handling Benchmark Failures

**Regression Detected:**

1. Check if it's a real regression or noise:
   ```bash
   pytest tests/benchmarks/test_specific.py --benchmark-min-rounds=20
   ```

2. Profile the code to find the slowdown:
   ```bash
   python -m cProfile -o profile.pstats -m pytest tests/benchmarks/test_specific.py
   snakeviz profile.pstats  # Visualize with snakeviz
   ```

3. If intentional, update baseline with explanation

**Flaky Benchmarks:**

Signs of flaky benchmarks:
- High stddev (>20% of mean)
- Inconsistent pass/fail across runs
- Different results on different hardware

Mitigation:
- Increase `--benchmark-min-rounds`
- Use `--benchmark-warmup=on`
- Consider marking as `@pytest.mark.slow` to exclude from fast CI

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
