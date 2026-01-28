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

### Marker Statistics (Sprint 23)

| Marker | Test Count | Approx. Runtime |
|--------|------------|-----------------|
| `fast` | ~2,680 | ~45 seconds |
| `integration` | ~72 | Varies (network-dependent) |
| Unmarked | ~8,800 | - |

**CI Optimization:**
- Use `pytest -m fast` for quick feedback loop during development
- Use `pytest -m "not requires_network"` for offline testing
- Full suite runs on PR merge

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
