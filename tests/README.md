# Test Suite Documentation

This directory contains the test suite for the voiceover-matcher-subtitle project.

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run fast tests only (unit tests, <100ms each)
pytest tests/ -m fast -v

# Skip slow tests
pytest tests/ -m "not slow" -v

# Skip tests requiring network
pytest tests/ -m "not requires_network" -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Test Markers

The following pytest markers are defined in `pytest.ini`:

| Marker | Description | When to Run |
|--------|-------------|-------------|
| `fast` | Pure unit tests (no I/O, no network, mocks only) | Every commit |
| `slow` | Slow tests (>5 seconds) | On-demand |
| `integration` | Integration tests requiring external resources | PR merge |
| `stress` | Stress/chaos tests | Nightly |
| `simulation` | Healing simulation tests | PR merge |
| `requires_api` | Tests requiring API keys (Gemini, etc.) | When keys available |
| `requires_network` | Tests making HTTP requests | When online |
| `flaky` | Intermittent failures (timing, network, race conditions) | Auto-retried 2x |

### Marker Statistics (Sprint 27)

| Marker | Test Count | Approx. Runtime |
|--------|------------|-----------------|
| `fast` | 11,225 | ~45 seconds |
| `integration` | 409 | Varies |
| `requires_network` | 148 | Network-dependent |
| `requires_api` | 83 | API-dependent |
| `simulation` | 43 | Variable |
| `flaky` | 30 | Auto-retried |
| `stress` | 15 | Minutes |
| `slow` | 11 | >5 seconds each |
| Unmarked | 25 | - |

**Total tests analyzed: 11,683**

*Statistics updated in Sprint 27 after completing marker assignment and fixing class-level marker inheritance in categorize_tests.py.*

## CI Workflow (Sprint 26)

The CI pipeline is configured for **marker-based test selection** to provide fast feedback on PRs while ensuring full test coverage on merge to main.

### Workflow Jobs

| Job | Trigger | Tests Run | Purpose |
|-----|---------|-----------|---------|
| `fast-tests` | Every PR update | `pytest -m fast` | Quick feedback (~45s target) |
| `full-tests` | Merge to main only | Full test suite | Complete coverage verification |
| `offline-tests` | Every PR update | `pytest -m "not requires_network"` | Fallback for network-restricted runners |
| `lint` | Every PR update | black, isort, ruff | Code formatting and style |
| `quality` | PRs only | mypy, extended ruff | Informational quality checks |

### Performance Targets

| Metric | Target | Limit |
|--------|--------|-------|
| Fast tests duration | 45 seconds | 60 seconds |
| Full suite | N/A | Depends on coverage |

### Job Summary

Each test job generates a GitHub Actions job summary with:
- Total test count
- Passed/Failed/Skipped breakdown
- Duration with performance status indicator

### Local Development

```bash
# Simulate CI fast tests
pytest tests/ -m fast --tb=short -v

# Simulate CI offline tests
pytest tests/ -m "not requires_network" --tb=short -v

# Simulate CI full suite
pytest tests/ --cov=src --cov-report=html --tb=short -v
```

### Workflow File

See `.github/workflows/tests.yml` for the complete workflow configuration.

**Key features:**
- Parallel test execution with `pytest-xdist` (`-n auto`)
- JSON report generation for metrics extraction
- Coverage uploaded to Codecov (on merge to main)
- Cached pip packages for faster installs

### Using Markers

Add markers to test functions:

```python
import pytest

@pytest.mark.fast
def test_quick_unit_test():
    assert 1 + 1 == 2

@pytest.mark.requires_network
def test_fetch_from_api():
    response = requests.get("https://api.example.com")
    assert response.ok

@pytest.mark.requires_api
def test_llm_integration():
    # Requires GEMINI_API_KEY environment variable
    pass
```

### Selecting Tests by Marker

```bash
# Run only fast tests
pytest -m fast

# Exclude slow tests
pytest -m "not slow"

# Run tests that don't need network
pytest -m "not requires_network"

# Combine markers
pytest -m "fast and not requires_api"

# Disable flaky test retries for debugging
pytest tests/test_flaky_detection.py --reruns 0 -v
```

## Parallel Test Execution (Sprint 27)

Tests can be run in parallel using `pytest-xdist` for faster execution. CI workflows use `-n auto` by default.

### Running Tests in Parallel

```bash
# Run all tests in parallel (auto-detect CPU count)
pytest tests/ -n auto

# Run with specific worker count
pytest tests/ -n 4

# Run parallel excluding serial tests (safest for complex suites)
pytest tests/ -n auto -m 'not serial'

# Run only serial tests (tests that must not run in parallel)
pytest tests/ -m serial
```

### Serial Marker

Some tests must run serially due to:
- Use of `os.chdir()` which affects global state
- Session-scoped fixtures with side effects
- Tests that modify shared external resources

These tests are marked with `@pytest.mark.serial`:

| File | Reason |
|------|--------|
| `test_config_path_resolution.py` | Uses `os.chdir()` |
| `test_utils_path_functions.py` | Uses `os.chdir()` |

### Analyzing Parallelization Safety

Use the analyzer script to identify tests that may need the serial marker:

```bash
# Run analysis and show summary
python scripts/analyze_test_parallelization.py

# Suggest files needing @pytest.mark.serial
python scripts/analyze_test_parallelization.py --suggest-serial

# Generate detailed report
python scripts/analyze_test_parallelization.py --report tests/PARALLELIZATION_REPORT.md

# Verbose output with all issues
python scripts/analyze_test_parallelization.py -v
```

### Writing Parallel-Safe Tests

**Safe Patterns:**
- Use `tmp_path` fixture for file operations (each worker gets isolated temp directory)
- Use `monkeypatch` for environment variables
- Use function-scoped fixtures (default)
- Mock external dependencies instead of using real I/O

**Unsafe Patterns (require `@pytest.mark.serial`):**
- `os.chdir()` - affects all workers
- Global mutable state modified during tests
- Session-scoped fixtures with side effects
- File locks without worker isolation

### CI Configuration

CI workflows automatically run tests in parallel:

| Job | Command | Notes |
|-----|---------|-------|
| `fast-tests` | `pytest -m fast -n auto` | Parallel for quick feedback |
| `full-tests` | `pytest -n auto` | Parallel full suite |
| `offline-tests` | `pytest -m "not requires_network" -n auto` | Parallel offline tests |

## Conditional Skip Patterns (Sprint 27)

Tests that depend on external resources use conditional `skipif` decorators instead of permanent skips. This allows tests to run when resources are available while cleanly skipping when they're not.

### Resource Detection

The `conftest.py` provides resource detection functions and reusable skip conditions:

```python
# Available skip conditions in conftest.py
from conftest import (
    SKIP_NO_GEMINI,             # Requires GEMINI_API_KEY or GOOGLE_API_KEY
    SKIP_NO_VOYAGE,             # Requires VOYAGE_API_KEY
    SKIP_NO_COOKIES,            # Requires cookies.txt for yt-dlp
    SKIP_NO_SENTENCE_TRANSFORMERS,  # Requires sentence-transformers
    SKIP_NO_OTIO,               # Requires opentimelineio
    SKIP_NO_NUMPY,              # Requires numpy
    SKIP_NO_TEST_PROJECT,       # Requires test project directory
)

# Usage
@pytest.mark.integration
@SKIP_NO_GEMINI
def test_gemini_embeddings():
    """Test Gemini embeddings - runs only with API key available."""
    pass
```

### Detection Functions

```python
# Import detection functions for custom conditions
from conftest import (
    has_gemini_api_key,         # Check GEMINI_API_KEY or GOOGLE_API_KEY
    has_voyage_api_key,         # Check VOYAGE_API_KEY
    has_cookies_file,           # Check cookies.txt in common locations
    has_sentence_transformers,  # Check sentence-transformers importable
    has_opentimelineio,         # Check opentimelineio importable
    has_numpy,                  # Check numpy importable
    has_test_project,           # Check test project directory exists
)
```

### Module-Level Constants

```python
# Module-level constants for skipif conditions
from conftest import (
    HAS_GEMINI_API,             # True if Gemini API key available
    HAS_VOYAGE_API,             # True if Voyage API key available
    HAS_COOKIES,                # True if cookies.txt found
    HAS_SENTENCE_TRANSFORMERS,  # True if sentence-transformers installed
    HAS_OTIO,                   # True if opentimelineio installed
    HAS_NUMPY,                  # True if numpy installed
)

# Custom skipif usage
@pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
def test_numpy_operations():
    import numpy as np
    # ...
```

### Skip Reason Documentation

All skipped tests should document their skip reason in the docstring:

```python
@pytest.mark.integration
@SKIP_NO_COOKIES
def test_video_download():
    """
    Test video downloading with yt-dlp.

    Requires:
        - cookies.txt file for yt-dlp authentication
        - Network access to YouTube

    Skip reason: Requires cookies.txt for yt-dlp video downloads
    """
    pass
```

## Test Organization

```
tests/
├── conftest.py          # Shared fixtures and configuration
├── test_*.py            # Module-specific test files
├── test_agents/         # Agent/healer tests
├── test_chapter_detection/
├── benchmarks/          # Performance benchmarks
└── benchmarks_broken/   # Disabled benchmarks (not discovered)
```

## Test Isolation Best Practices (Sprint 26)

Tests should be **isolated** - they shouldn't depend on or affect other tests. This section covers best practices for maintaining test isolation.

### Use `tmp_path` Instead of `tempfile.mkdtemp()`

**Preferred pattern:**
```python
def test_file_operations(tmp_path):
    """tmp_path is automatically cleaned up after test."""
    test_file = tmp_path / "data.json"
    test_file.write_text('{"key": "value"}')
    assert test_file.exists()
    # Automatic cleanup - no teardown needed
```

**Avoid this pattern:**
```python
def test_file_operations():
    """mkdtemp requires manual cleanup."""
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp()  # Creates directory that may leak
    try:
        # ... test code ...
        pass
    finally:
        shutil.rmtree(temp_dir)  # Must clean up manually
```

**For unittest.TestCase style:**
```python
class TestSomething(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)  # REQUIRED
```

### Check for Isolation Issues

Run tests with isolation checking enabled:

```bash
# Check for mkdtemp() usage (warns but doesn't fail)
pytest tests/ --check-isolation -v

# The summary will list tests using mkdtemp()
```

### Common Isolation Anti-Patterns

| Anti-Pattern | Problem | Solution |
|--------------|---------|----------|
| Hardcoded paths (`Path("E:/test")`) | Won't work on other machines | Use `tmp_path` fixture |
| `tempfile.mkdtemp()` without cleanup | Leaves files after test | Use `tmp_path` or add `tearDown` |
| Writing to project directories | Pollutes repo | Use `tmp_path` |
| Shared global state | Tests affect each other | Use fixtures with scope |
| External file dependencies | Flaky on CI | Use skip conditions or fixtures |

### When Fixed Paths Are Acceptable

Some tests need external resources. Use skip conditions:

```python
PROJECT_DIR = Path("E:/Edit Job/test/jan_test/audio__2026-01-01")

@pytest.mark.skipif(
    not PROJECT_DIR.exists(),
    reason=f"Test project not found at {PROJECT_DIR}"
)
def test_with_real_project():
    """Only runs if external project exists."""
    # Use PROJECT_DIR safely
    pass
```

### Fixture Scopes for Resource Management

| Scope | Use Case | Cleanup |
|-------|----------|---------|
| `function` (default) | Most tests | After each test |
| `class` | Shared setup within class | After all methods |
| `module` | Expensive setup for module | After module |
| `session` | One-time global setup | After all tests |

```python
@pytest.fixture(scope="module")
def expensive_resource(tmp_path_factory):
    """Shared across all tests in module."""
    return tmp_path_factory.mktemp("shared")
```

### Verifying Isolation

After running tests, verify no artifacts remain:

```bash
# Check for leftover temp directories
ls -la /tmp/pytest-* 2>/dev/null || echo "No pytest temp dirs found (good)"

# Check for modified tracked files
git status --porcelain
```

## Fixture Factories

The `conftest.py` file provides factory fixtures for creating test objects:

```python
# Create SRTSegment with custom values
segment = srt_segment_factory(text="Custom text", is_broll=True)

# Create Config with section overrides
config = config_factory(matching={'min_confidence': 0.9})

# Create PipelineState with pre-populated data
state = pipeline_state_factory(keywords=['travel', 'nature'])

# Create MatchResult with alternatives
result = match_result_factory(confidence=0.95, num_alternatives=3)
```

See `conftest.py` for full factory documentation.

## Complex Coordination Scenario Fixtures (US-010)

The `tests/fixtures/` module provides factory functions for complex test scenarios:

### Checkpoint with Populated Stages (Rule 25)

For testing `--output-only` mode which requires checkpoint with populated `stages` dict:

```python
from tests.fixtures import create_checkpoint_with_populated_stages

# Create checkpoint with default populated stage data
checkpoint = create_checkpoint_with_populated_stages()

# Customize match count and confidence
checkpoint = create_checkpoint_with_populated_stages(
    project_dir=tmp_path,
    match_count=20,
    avg_confidence=0.92,
    video_count=10
)

# Override specific stage data
checkpoint = create_checkpoint_with_populated_stages(
    match={"custom_field": "value"},
    analyze={"keywords": ["custom", "keywords"]}
)

# Use in tests
assert checkpoint["match"]["match_count"] == 20
assert len(checkpoint["match"]["matches"]) == 20
```

### Concurrent Download Escalation

For testing EscalationManager with multiple keywords under thread contention:

```python
from tests.fixtures import create_concurrent_escalation_fixture

# Create fixture with 5 concurrent keywords
fixture = create_concurrent_escalation_fixture(num_keywords=5)

# Access configuration
keywords = fixture["keywords"]  # ['test_keyword_0', ..., 'test_keyword_4']
ext_config = fixture["ext_config"]  # ExtractorArgs-compatible dict
initial_states = fixture["initial_states"]  # Per-keyword tier states

# Expected escalation results
tier2_args = fixture["expected_tier2_args"]
tier3_rotates = fixture["expected_tier3_rotate_cookies"]

# Customize escalation behavior
fixture = create_concurrent_escalation_fixture(
    num_keywords=3,
    escalation_threshold=3,  # 3 consecutive 403s to escalate
    cooldown_seconds=0.0,    # No cooldown for tests
    max_tier=3
)
```

### OTIO Timeline Fixture (Rule 17)

For testing auto-split at 3000 item threshold:

```python
from tests.fixtures import create_otio_timeline_fixture

# Create fixture for boundary testing
fixture = create_otio_timeline_fixture(3000)
assert fixture["at_threshold"] is True
assert fixture["expected_parts"] == 1

# Create fixture above threshold (should split)
fixture = create_otio_timeline_fixture(3001)
assert fixture["above_threshold"] is True
assert fixture["expected_parts"] == 2

# Create large timeline
fixture = create_otio_timeline_fixture(6500)
assert fixture["expected_parts"] == 3

# With custom configuration
fixture = create_otio_timeline_fixture(
    num_segments=1000,
    frame_rate=24.0,
    include_audio=True,
    segment_duration_frames=48  # 2 seconds at 24fps
)

# Access constants
MAX_SEGMENTS = fixture["max_segments_per_part"]  # 3000
WARNING_THRESHOLD = fixture["warning_threshold"]  # 2500
```

### B-roll Propagation Chain State (Rule 8)

For testing is_broll flag propagation from SceneDetection through Match stage:

```python
from tests.fixtures import create_broll_propagation_chain_state

# Create state with 30% B-roll ratio
state = create_broll_propagation_chain_state(
    num_scenes=10,
    broll_ratio=0.3
)

# Access scene data
scenes = state["scenes"]
text_metadata = state["text_metadata"]

# Check expected B-roll counts
assert state["expected_broll_count"] == 3
assert state["face_detected_broll"] == 1  # Half of B-roll via face detection
assert state["silent_detected_broll"] == 2  # Half via word count

# Check thresholds
assert state["thresholds"]["face_score"] == 0.3
assert state["thresholds"]["min_words"] == 10

# V8 track expectations
assert state["expected_v8_entries"] == 3

# Customize detection thresholds
state = create_broll_propagation_chain_state(
    num_scenes=20,
    broll_ratio=0.4,
    face_score_threshold=0.25,  # Stricter face detection
    min_words_threshold=5       # More scenes as "silent"
)
```

### Importing Fixtures

```python
# Import specific fixtures
from tests.fixtures import (
    create_checkpoint_with_populated_stages,
    create_concurrent_escalation_fixture,
    create_otio_timeline_fixture,
    create_broll_propagation_chain_state,
)

# Or use shorter aliases
from tests.fixtures import (
    checkpoint_with_stages,
    concurrent_escalation,
    otio_timeline,
    broll_chain_state,
)
```

## Flaky Test Handling

The project uses `pytest-rerunfailures` to handle intermittently failing tests.

### What Makes a Test Flaky?

A test is flaky if it fails intermittently without code changes. Common causes:

| Cause | Example | Solution |
|-------|---------|----------|
| **Timing/Race Conditions** | Thread scheduling, async operations | Add `@pytest.mark.flaky(reruns=2)` |
| **Network Dependencies** | External API timeouts, rate limiting | Add retry with delay: `reruns_delay=0.5` |
| **Resource Contention** | Port conflicts, file locks | Use unique resources per test |
| **CI Environment** | Slower runners, resource limits | Increase retries in CI only |

### Marking a Flaky Test

```python
import pytest

# Basic pattern - retry up to 2 times
@pytest.mark.flaky(reruns=2)
def test_sometimes_fails():
    ...

# With delay between retries (for rate-limited APIs)
@pytest.mark.flaky(reruns=2, reruns_delay=0.5)
def test_api_call():
    ...

# Platform-specific flakiness
@pytest.mark.flaky(reruns=2, condition="sys.platform == 'win32'")
def test_windows_timing():
    ...

# CI-aware retry count
import os
IS_CI = os.environ.get("CI", "false").lower() == "true"

@pytest.mark.flaky(reruns=3 if IS_CI else 1)
def test_ci_sensitive():
    ...
```

### When to Use @pytest.mark.flaky

**DO use flaky marker for:**
- Tests with known intermittent failures that are not worth fixing
- External service dependencies with occasional timeouts
- Race conditions that are inherent to the design
- Tests that fail only in CI due to resource constraints

**DON'T use flaky marker for:**
- Tests that fail consistently (fix the bug instead)
- Tests with easy-to-fix timing issues (use proper synchronization)
- Tests that mask real bugs (investigate root cause)
- Tests that fail >50% of the time (too unreliable)

### Documenting Flaky Tests

When marking a test as flaky, document the reason:

```python
@pytest.mark.flaky(reruns=2)
def test_webhook_delivery():
    """
    Test webhook delivery to external service.

    Flakiness Reason:
        External webhook endpoint occasionally returns 503 under load.

    Mitigation:
        - Increased timeout from 5s to 15s (reduced failures by 60%)
        - Added retry logic in source code (further reduced by 30%)
        - Remaining ~5% failures handled by pytest-rerunfailures

    Related: GitHub Issue #456
    """
    ...
```

### Identifying Flaky Tests

Signs a test may be flaky:
1. **CI failures without code changes** - Test passes locally but fails in CI
2. **Inconsistent failures** - Same test fails on different runs
3. **Timing-related errors** - Timeouts, "operation not completed" errors
4. **Order-dependent failures** - Fails only when run with certain other tests

Tools to help identify:
```bash
# Run tests multiple times to find flaky ones
pytest tests/test_matching.py --count=5 -x

# Check for tests with I/O that should be marked
grep -r "requests\|subprocess\|open\|time.sleep" tests/*.py
```

### Debugging Flaky Tests

```bash
# Disable retries to see failures immediately
pytest tests/test_flaky.py --reruns 0 -v

# Run repeatedly to reproduce
pytest tests/test_flaky.py --count=10 -v

# Show extra output on failure
pytest tests/test_flaky.py -v --showlocals --tb=long
```

### Flaky Test Patterns

See `tests/test_flaky_detection.py` for documented patterns:
1. **Basic flaky marker** - Simple retry configuration
2. **Flaky with condition** - Platform-specific retries
3. **Network-dependent** - API calls with retry delay
4. **Timing-sensitive** - Race conditions and async operations
5. **Resource contention** - Shared resource access
6. **CI-aware** - Different retry counts for CI vs local
7. **Infrastructure validation** - Verifying retry setup works
8. **Documentation pattern** - How to document flaky tests

## Coverage Requirements

Target: 85% overall coverage (configured in `.coveragerc`)

### Critical Modules

| Module | Purpose | Target |
|--------|---------|--------|
| `src/matching/` | Video-to-voiceover matching | 85% |
| `src/agents/` | Self-healing pipeline agents | 85% |
| `src/compilation/` | Keyword compilation pipeline | 85% |

### Check Coverage Locally

```bash
# Full coverage report
pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
open htmlcov/index.html  # View report (macOS)
start htmlcov/index.html  # View report (Windows)

# Per-module coverage
pytest tests/ --cov=src/matching --cov-report=term-missing
pytest tests/ --cov=src/agents --cov-report=term-missing
pytest tests/ --cov=src/compilation --cov-report=term-missing

# Generate coverage badge URL
python scripts/generate_coverage_badge.py --verbose

# CI also uploads to Codecov (see .github/workflows/tests.yml)
```

### Coverage Configuration

Coverage settings are in `.coveragerc`:
- `fail_under = 85` - CI fails if coverage drops below 85%
- HTML reports go to `htmlcov/`
- XML reports go to `coverage.xml`
- Excludes: legacy modules, test files, type-checking blocks

### Per-Module Coverage Breakdown

The `scripts/coverage_by_module.py` script generates per-module coverage reports with critical module highlighting and baseline comparison.

#### Quick Usage

```bash
# Generate basic report
python scripts/coverage_by_module.py

# Verbose report (show all modules)
python scripts/coverage_by_module.py --verbose

# JSON output for CI
python scripts/coverage_by_module.py --json --output report.json

# Compare to baseline
python scripts/coverage_by_module.py --compare-baseline baseline.json

# Save current as baseline
python scripts/coverage_by_module.py --save-baseline --output baseline.json

# Fail on regression (CI integration)
python scripts/coverage_by_module.py --compare-baseline baseline.json --fail-on-regression
```

#### Report Output

```
======================================================================
COVERAGE REPORT BY MODULE
======================================================================

Overall Coverage: 87.5% (target: 85.0%)
[OK] Coverage target met

----------------------------------------------------------------------
CRITICAL MODULES
----------------------------------------------------------------------
Module                                Coverage           Lines     Status
----------------------------------------------------------------------
matching                                92.50%        2356/2547     [OK]
agents                                  88.00%        1243/1413     [OK]
stages                                  85.50%        2962/3465     [OK]
compilation                             84.00%           45/53     [WARN]

----------------------------------------------------------------------
WARNINGS
----------------------------------------------------------------------
  WARNING: compilation at 84.0% (target: 85.0%)
======================================================================
```

#### Critical Modules

The script highlights these critical modules that must maintain high coverage:

| Module | Purpose | Target | Minimum |
|--------|---------|--------|---------|
| `matching/` | Video-to-voiceover matching | 85% | 80% |
| `agents/` | Self-healing pipeline agents | 85% | 80% |
| `compilation/` | Keyword compilation pipeline | 85% | 80% |
| `stages/` | Pipeline stage implementations | 85% | 80% |

#### JSON Output Format

```json
{
  "timestamp": "2026-01-29T10:30:00.000000",
  "overall_coverage": 87.5,
  "target": 85.0,
  "modules": [...],
  "critical_modules": [...],
  "warnings": [...],
  "summary": {
    "total_modules": 24,
    "below_target_count": 3,
    "critical_modules_count": 4,
    "critical_below_target": 1,
    "meets_target": true
  }
}
```

#### CI Integration

Add to your GitHub Actions workflow:

```yaml
jobs:
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements-dev.txt

      - name: Run tests with coverage
        run: pytest tests/ --cov=src --cov-report=xml

      - name: Generate coverage report
        run: python scripts/coverage_by_module.py --json --output coverage-report.json

      - name: Compare to baseline
        run: |
          python scripts/coverage_by_module.py \
            --compare-baseline coverage-baseline.json \
            --fail-on-regression

      - name: Upload coverage report
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: coverage-report.json
```

#### Updating Baseline

When coverage improves and you want to update the baseline:

```bash
# Generate fresh coverage data
pytest tests/ --cov=src --cov-report=xml

# Save new baseline
python scripts/coverage_by_module.py --save-baseline --output coverage-baseline.json

# Commit the baseline
git add coverage-baseline.json
git commit -m "chore: update coverage baseline"
```

## Assertion Helpers

The `tests/helpers/` module provides reusable assertion helpers for common validation patterns.

### Importing Helpers

```python
from tests.helpers import (
    assert_valid_otio_timeline,
    assert_valid_match_result,
    assert_checkpoint_consistent,
    assert_config_valid,
    assert_file_exists,
    assert_dir_exists,
    assert_json_structure,
)
```

### assert_valid_otio_timeline()

Validates OTIO timeline structure including tracks, clips, and media references.

```python
import opentimelineio as otio

def test_timeline_generation():
    timeline = create_timeline(matches, config)

    # Basic validation
    assert_valid_otio_timeline(timeline)

    # With requirements
    assert_valid_otio_timeline(
        timeline,
        min_tracks=2,
        require_video=True,
        require_audio=True,
        expected_duration=120.0,  # seconds
        max_clips=3000,  # Rule 17 limit
    )
```

**Validates:**
- Timeline is an opentimelineio.schema.Timeline object
- Has minimum required tracks (video and/or audio)
- All clips have valid source_range and media_reference
- Total clip count is within limits (default: 3000 per Rule 17)
- Duration matches expected value (if specified)

### assert_valid_match_result()

Validates match result dicts from the matching stage.

```python
def test_matcher_output():
    result = matcher.match(segment)

    # Basic validation with minimum confidence
    assert_valid_match_result(result, min_confidence=0.5)

    # Full validation
    assert_valid_match_result(
        result,
        min_confidence=0.7,
        require_video_file=True,
        require_timing=True,
        require_strategy=True,
        valid_strategies=["embedding", "llm", "fallback"],
    )
```

**Validates:**
- Required fields present (segment_index, confidence)
- Confidence in range [0, 1] and meets minimum threshold
- Video file path is present and non-empty
- Timing fields (start/end) are valid and ordered correctly
- Strategy is from allowed list (if specified)

### assert_checkpoint_consistent()

Validates checkpoint structure and data consistency.

```python
def test_checkpoint_save_restore():
    checkpoint = load_checkpoint(project_dir)

    # Basic validation
    assert_checkpoint_consistent(checkpoint)

    # For --output-only mode (Rule 25)
    assert_checkpoint_consistent(
        checkpoint,
        require_stages=True,  # Ensures stages dict is populated
        expected_stage="MATCH",
        min_matches=10,
    )
```

**Validates:**
- Required fields (version, last_completed_stage)
- Stage name is valid (one of VALID_STAGES)
- Stages dict is populated (required for --output-only mode)
- Match data consistency (count matches len(matches))

### assert_config_valid()

Validates Config object structure and values.

```python
def test_config_loading():
    config = load_config("config.yaml")

    # Get list of validation errors
    errors = assert_config_valid(config)
    assert not errors, f"Config errors: {errors}"

    # With requirements
    errors = assert_config_valid(
        config,
        require_api_keys=True,
        require_matching=True,
        check_constraints=True,
        allowed_errors=["GEMINI_API_KEY"],  # Ignore missing key in tests
    )
```

**Validates:**
- Required sections exist (matching, download, etc.)
- Value ranges (confidence 0-1, workers >= 1)
- Enum values (valid providers, strategies)
- Constraint relationships (min <= max)
- Uses Config.validate() if available

### Utility Assertions

```python
# File existence
assert_file_exists(output_path, description="OTIO output")
assert_dir_exists(project_dir, description="Project directory")

# JSON structure
assert_json_structure(
    data,
    required_keys=["version", "timestamp", "matches"],
    type_checks={"version": str, "matches": list},
)
```

### Best Practices

1. **Use helpers for repeated patterns** - If you're writing the same validation logic in multiple tests, add it to helpers.

2. **Provide clear error messages** - Helpers give actionable errors:
   ```
   AssertionError: Timeline has 3500 clips, exceeds limit of 3000.
   Consider using save_timeline_split() for auto-splitting.
   ```

3. **Combine with fixtures** - Use helpers with fixture factories:
   ```python
   from tests.fixtures import create_mock_state
   from tests.helpers import assert_valid_match_result

   def test_with_mock_data():
       state = create_mock_state()
       result = matcher.match(state.voiceover_segments[0])
       assert_valid_match_result(result, min_confidence=0.3)
   ```

4. **Use allowed_errors for test contexts** - Skip known issues in test environments:
   ```python
   errors = assert_config_valid(config, allowed_errors=["API_KEY"])
   ```

## CI Optimization: Test Categorization

The `scripts/categorize_tests.py` tool analyzes test marker distribution and suggests markers for unmarked tests based on I/O patterns.

### Why Categorize Tests?

| Benefit | Impact |
|---------|--------|
| **Faster feedback loop** | Run `pytest -m fast` (216 tests, ~45s) instead of full suite (11K+ tests) |
| **Selective CI runs** | Skip slow/network tests on PR drafts |
| **Resource optimization** | Don't run API-dependent tests without keys |
| **Parallel execution** | Separate fast/slow tests for concurrent jobs |

### Quick Usage

```bash
# View marker distribution report
python scripts/categorize_tests.py

# Show sample tests needing markers
python scripts/categorize_tests.py --verbose

# Export to CSV for review
python scripts/categorize_tests.py --output report.csv

# Auto-suggest mode with grouped recommendations
python scripts/categorize_tests.py --auto-suggest

# JSON output for CI integration
python scripts/categorize_tests.py --json --output report.json

# Analyze specific directory
python scripts/categorize_tests.py --path tests/test_agents
```

### Understanding the Output

The script categorizes tests into:

| Category | I/O Pattern Detected | Suggested Marker |
|----------|---------------------|------------------|
| **Pure unit tests** | No I/O, uses mocks | `@pytest.mark.fast` |
| **File I/O tests** | tempfile, shutil, subprocess | `@pytest.mark.integration` |
| **Network tests** | requests, httpx, socket | `@pytest.mark.requires_network` |
| **API tests** | *_API_KEY environment vars | `@pytest.mark.requires_api` |
| **Slow tests** | time.sleep > 2s | `@pytest.mark.slow` |

### CSV Output Format

```csv
test_file,test_name,line_number,current_markers,suggested_marker,reason,io_patterns
tests/test_cache.py,TestBaseCache::test_ttl_expiration,64,(none),integration,Found I/O pattern: tempfile...,integration:tempfile...
tests/test_core.py,test_simple_logic,10,(none),fast,No I/O patterns detected,...
```

### CI Integration

Add to your GitHub Actions workflow:

```yaml
jobs:
  analyze-markers:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Analyze test markers
        run: |
          python scripts/categorize_tests.py --json --output marker-report.json

      - name: Upload marker report
        uses: actions/upload-artifact@v4
        with:
          name: marker-report
          path: marker-report.json
```

### Adding Markers Based on Suggestions

1. Run `python scripts/categorize_tests.py --auto-suggest > suggestions.txt`
2. Review suggestions by marker type
3. Add markers to tests that match the suggested criteria
4. Re-run to verify marker distribution improved

Example before/after:

```bash
# Before
Unmarked tests: 11143
Tests with suggestions: 11255

# After adding markers to high-impact tests
Unmarked tests: 8500
fast: 2800 tests
integration: 400 tests
```

### Best Practices

1. **Start with integration markers** - Tests using tempfile/subprocess are clearly integration tests
2. **Mark fast tests gradually** - Don't add `@pytest.mark.fast` to everything at once
3. **Verify with selective runs** - After marking, test with `pytest -m fast` to confirm
4. **Update CI to use markers** - Once marked, update CI to run `-m fast` on PR updates

## Integration Test Runner

The `scripts/run_integration_tests.py` script provides isolated execution for integration tests with automatic cleanup and timeout protection.

### Quick Usage

```bash
# Run all integration tests (default: 5 min timeout per test)
python scripts/run_integration_tests.py

# Verbose output with test progress
python scripts/run_integration_tests.py -v

# Parallel execution (requires pytest-xdist)
python scripts/run_integration_tests.py --parallel 4

# Custom timeout (10 minutes per test)
python scripts/run_integration_tests.py --timeout 600

# Keep test artifacts for debugging
python scripts/run_integration_tests.py --keep-artifacts

# Filter by pattern
python scripts/run_integration_tests.py -k "test_caption"

# Export JSON report
python scripts/run_integration_tests.py --json-report results.json
```

### Features

| Feature | Description |
|---------|-------------|
| **Isolated environment** | Each run creates a fresh temp directory with cache, output, downloads subdirs |
| **Automatic cleanup** | Temp directories are removed after tests complete (unless `--keep-artifacts`) |
| **Timeout protection** | Default 5 minutes per test, prevents hanging tests from blocking CI |
| **Parallel execution** | Use `--parallel N` to run tests concurrently (requires pytest-xdist) |
| **JSON reports** | Export results with `--json-report` for CI integration |
| **Pattern matching** | Filter tests with `-k` pattern (same as pytest) |
| **Marker selection** | Default: `integration`, `requires_network`, `requires_api` |

### Environment Variables

The runner sets these environment variables for test isolation:

| Variable | Description |
|----------|-------------|
| `INTEGRATION_TEST_TEMP_DIR` | Root temp directory for the test run |
| `INTEGRATION_TEST_CACHE_DIR` | Isolated cache directory |
| `INTEGRATION_TEST_OUTPUT_DIR` | Isolated output directory |
| `INTEGRATION_TEST_DOWNLOADS_DIR` | Isolated downloads directory |
| `INTEGRATION_TEST_NO_CACHE` | Set to "1" to disable caching |

### Using in Tests

Access the isolated directories in your integration tests:

```python
import os
import pytest

@pytest.mark.integration
def test_with_isolation():
    temp_dir = os.environ.get("INTEGRATION_TEST_TEMP_DIR")
    if temp_dir:
        # Use isolated directories
        cache_dir = os.environ.get("INTEGRATION_TEST_CACHE_DIR")
        output_dir = os.environ.get("INTEGRATION_TEST_OUTPUT_DIR")
    else:
        # Fallback for direct pytest runs
        pass
```

### CI Integration

Add to GitHub Actions workflow:

```yaml
jobs:
  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements-dev.txt

      - name: Run integration tests
        run: |
          python scripts/run_integration_tests.py \
            --parallel 2 \
            --timeout 300 \
            --json-report integration-results.json
        continue-on-error: true

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: integration-results
          path: integration-results.json
```

### Troubleshooting

| Issue | Solution |
|-------|----------|
| Tests timeout | Increase `--timeout` or check for hanging network calls |
| Parallel fails | Install pytest-xdist: `pip install pytest-xdist` |
| Artifacts not found | Use `--keep-artifacts` and check temp directory |
| Permission errors | Ensure temp directory is writable |

## Linting and Code Quality

The project enforces code quality through automated linting checks that **block PR merges** on failure.

### Linting Tools

| Tool | Purpose | Fails CI |
|------|---------|----------|
| **black** | Code formatting (PEP 8 style) | Yes |
| **isort** | Import sorting (grouped, alphabetized) | Yes |
| **ruff** | Fast Python linter (replaces flake8, pylint) | Yes (for errors) |
| **mypy** | Static type checking | No (informational) |

### CI Linting Jobs

| Job | When | What | Blocking |
|-----|------|------|----------|
| `lint` | Every PR | black, isort, ruff (errors only) | **Yes** |
| `quality` | PRs only | mypy, extended ruff | No (informational) |

### Local Development

Run linting checks before committing:

```bash
# Check formatting (will show diff, not modify)
black --check --diff src/ tests/

# Auto-fix formatting
black src/ tests/

# Check import sorting
isort --check-only --diff src/ tests/

# Auto-fix imports
isort src/ tests/

# Run ruff linter (errors only - same as CI)
ruff check src/ tests/ --select=E9,F63,F7,F82

# Run full ruff (informational)
ruff check src/ tests/

# Auto-fix ruff issues
ruff check src/ tests/ --fix

# All checks in one command
black --check src/ tests/ && isort --check-only src/ tests/ && ruff check src/ tests/ --select=E9,F63,F7,F82
```

### What Blocks PRs

The `lint` job checks for **fatal errors only**:

| Error Code | Description | Example |
|------------|-------------|---------|
| **E9** | Runtime/syntax errors | Invalid syntax, undefined names |
| **F63** | String .format() issues | Missing/extra format arguments |
| **F7** | Type errors | Incorrect type annotations |
| **F82** | Undefined names | Using undefined variables |

Other ruff warnings (unused imports, line length, etc.) are reported but don't block PRs.

### Configuring Ruff

Ruff configuration is in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 120
target-version = "py39"

[tool.ruff.lint]
select = ["E", "F", "I", "W"]
ignore = ["E501"]  # Line length (handled by black)
```

### Configuring Black

Black configuration is in `pyproject.toml`:

```toml
[tool.black]
line-length = 120
target-version = ["py39", "py310", "py311"]
```

### Configuring isort

isort configuration is in `pyproject.toml`:

```toml
[tool.isort]
profile = "black"
line_length = 120
```

### Pre-commit Hook (Recommended)

For automatic linting on commit, add a pre-commit hook:

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 24.4.2
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--select=E9,F63,F7,F82, --fix]
EOF

# Install hooks
pre-commit install
```

### Troubleshooting Lint Failures

| Issue | Cause | Fix |
|-------|-------|-----|
| `black --check` fails | Code not formatted | Run `black src/ tests/` |
| `isort --check-only` fails | Imports not sorted | Run `isort src/ tests/` |
| `ruff check` E9 error | Syntax error | Fix syntax in reported file |
| `ruff check` F82 error | Undefined name | Import or define the name |

### Why Lint Errors Block PRs

Code quality tools ensure:
1. **Consistency** - All code follows the same style
2. **Readability** - Standard formatting reduces cognitive load
3. **Correctness** - Catches syntax errors before runtime
4. **Maintainability** - Clean imports and structure

The fatal error categories (E9, F63, F7, F82) catch bugs that would cause runtime failures, so blocking PRs prevents broken code from merging.
