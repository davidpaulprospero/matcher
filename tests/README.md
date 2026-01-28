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
