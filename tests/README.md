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
| `fast` | Fast unit tests (<100ms) | Every commit |
| `slow` | Slow tests (>5 seconds) | On-demand |
| `integration` | Integration tests requiring external resources | PR merge |
| `stress` | Stress/chaos tests | Nightly |
| `simulation` | Healing simulation tests | PR merge |
| `requires_api` | Tests requiring API keys (Gemini, etc.) | When keys available |
| `requires_network` | Tests making HTTP requests | When online |

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
